from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.constants import (
    FACECAM_EDGE_SCORE_MIN,
    FACECAM_MAX_AREA_FRAC,
    FACECAM_MIN_AREA_FRAC,
    LLAMA_N_CTX,
    ContentType,
)
from src.core.exceptions import AIProviderError
from src.core.workspace import BIN_DIR


def make_even(value: int) -> int:
    """Round down to the nearest even integer (FFmpeg crop/scale dimensions must be even)."""
    return value if value % 2 == 0 else value - 1


def parse_timerange_line(line: str) -> tuple[float, float]:
    """Parse a manual timerange line into (start_seconds, end_seconds).

    Accepts "MM:SS - MM:SS" or "HH:MM:SS - HH:MM:SS" (either side independently 2 or 3
    colon-parts). Only a positivity/ordering safety check is enforced here — manual mode
    intentionally bypasses ``MIN_CLIP_SECONDS`` and any other selection-config floor.
    """

    def _to_seconds(part: str) -> float:
        pieces = part.strip().split(":")
        if len(pieces) == 2:
            m, s = pieces
            h = "0"
        elif len(pieces) == 3:
            h, m, s = pieces
        else:
            raise ValueError(
                f"Invalid timestamp {part!r} in line {line!r}: expected MM:SS or HH:MM:SS"
            )
        try:
            return int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp {part!r} in line {line!r}: {e}") from e

    if "-" not in line:
        raise ValueError(f"Invalid timerange line {line!r}: expected 'START - END'")
    start_part, _, end_part = line.rpartition("-")
    start, end = _to_seconds(start_part), _to_seconds(end_part)
    if end <= start:
        raise ValueError(f"Invalid timerange line {line!r}: end must be after start")
    return start, end


def load_timerange_file(path: Path) -> list[dict[str, Any]]:
    """Read a manual timerange file into ``[{"start_time", "end_time", ...}, ...]``.

    Each line is ``START - END`` with an optional ``| CONTENT_TYPE`` suffix that pins the
    layout for that range (e.g. ``1:30 - 2:30 | JUST_CHAT``). A line without the suffix
    omits ``content_type``, so the pipeline falls back to auto detection for it. The type
    is matched case-insensitively against ``ContentType`` (PODCAST, JUST_CHAT, GAMING_SOLO,
    GAMING_COLLAB, DONATION_OVERLAY). Blank lines and ``#``-prefixed comments are skipped;
    raises ``ValueError`` on an unknown type or if the file yields no clips.
    """
    clips: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        range_part, _, type_part = line.partition("|")
        start, end = parse_timerange_line(range_part)
        clip: dict[str, Any] = {"start_time": start, "end_time": end}
        type_part = type_part.strip()
        if type_part:
            try:
                clip["content_type"] = ContentType(type_part.upper()).value
            except ValueError as e:
                raise ValueError(
                    f"Invalid content type {type_part!r} in line {line!r}: expected one of "
                    f"{', '.join(t.value for t in ContentType)}"
                ) from e
        clips.append(clip)
    if not clips:
        raise ValueError(f"No valid timerange entries found in {path}")
    return clips


def box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-Union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    thresh: float = 0.3,
) -> bool:
    """True if box a overlaps box b — IoU > thresh OR a's centre lies inside b."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw / 2, ay + ah / 2
    if bx <= acx <= bx + bw and by <= acy <= by + bh:
        return True
    return box_iou(a, b) > thresh


def extract_digits(value: object, default: str = "1080") -> str:
    """Extract leading digits from a config value like ``"1080p"`` or ``"192K"``.

    Args:
        value: The raw config string (e.g. ``"1080p"``, ``"192k"``).
        default: Fallback when no digits are found.

    Returns:
        The leading digit substring (e.g. ``"1080"``, ``"192"``).
    """
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else default


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Centre point (cx, cy) of an (x, y, w, h) box."""
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def edge_score(
    box: tuple[float, float, float, float], width: int, height: int
) -> float:
    """1 − (centre's distance to the nearest frame edge) / (min(w,h)/2).

    ~0 for an interior box (far from every border), high for one hugging any edge/corner.
    Shared by VisualAnalyzer's facecam picking and LayoutBuilder's collab-cam fallback.
    """
    cx, cy = box_center(box)
    margin = min(cx, width - cx, cy, height - cy)
    half = max(1.0, min(width, height) / 2.0)
    return max(0.0, 1.0 - margin / half)


def is_facecam_candidate(
    box: tuple[float, float, float, float], frame_area: float, width: int, height: int
) -> bool:
    """True when a box is cam-sized (not a tiny character or a full-frame face) and edge-hugging.

    Shared gate behind GAMING_SOLO/COLLAB webcam detection (VisualAnalyzer) and the Mode C
    collaborator fallback (LayoutBuilder) — both need "is this person box a real webcam?".
    """
    area_frac = (float(box[2]) * float(box[3])) / frame_area
    if not (FACECAM_MIN_AREA_FRAC <= area_frac <= FACECAM_MAX_AREA_FRAC):
        return False
    return edge_score(box, width, height) >= FACECAM_EDGE_SCORE_MIN


def center_crop(width: int, height: int, aspect: float) -> dict:
    """Static centre crop pre-shaped to a panel aspect (even-rounded)."""
    crop_h = make_even(height)
    crop_w = make_even(min(width, int(round(height * aspect))))
    crop_x = make_even((width - crop_w) // 2)
    crop_y = make_even((height - crop_h) // 2)
    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def expand_box_to_aspect(
    x: int, y: int, w: int, h: int, width: int, height: int, aspect: float
) -> dict:
    """Expand an arbitrary box to a panel aspect, centred + clamped to the frame."""
    cx = x + w / 2.0
    cy = y + h / 2.0

    if w / max(1, h) < aspect:
        new_w, new_h = h * aspect, float(h)
    else:
        new_w, new_h = float(w), w / aspect

    new_w = min(new_w, width)
    new_h = min(new_h, height)
    if new_w / new_h > aspect:
        new_w = new_h * aspect
    else:
        new_h = new_w / aspect

    crop_w = make_even(int(new_w))
    crop_h = make_even(int(new_h))
    crop_x = make_even(int(max(0, min(cx - crop_w / 2.0, width - crop_w))))
    crop_y = make_even(int(max(0, min(cy - crop_h / 2.0, height - crop_h))))
    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def clamp_crop_x(center_x: float, crop_w: int, width: int) -> int:
    """Clamp a crop's horizontal position so a ``crop_w``-wide box stays inside ``width``."""
    return make_even(int(max(0, min(center_x - crop_w / 2.0, width - crop_w))))


class SystemUtils:
    _device_logged: bool = (
        False  # so the resolved compute device is logged once, not per model load
    )

    @staticmethod
    def is_wsl() -> bool:
        """Detect if running inside Windows Subsystem for Linux."""
        if platform.system() == "Linux":
            try:
                with open("/proc/version") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        return True
            except FileNotFoundError:
                logger.debug("Not running under WSL (/proc/version not found).")
        return False

    @staticmethod
    def get_windows_username() -> str | None:
        """Execute Windows cmd to get the host username when running in WSL."""
        try:
            result = subprocess.run(
                ["cmd.exe", "/c", "echo %USERNAME%"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            username = result.stdout.strip()
            if username:
                return username
        except Exception as e:
            logger.warning(f"Failed to resolve Windows username: {e}")
        return None

    @staticmethod
    def display_path(path: Path) -> Path:
        """Return a relative path for cleaner log output, falling back to absolute if outside cwd."""
        try:
            return path.relative_to(Path.cwd())
        except ValueError:
            return path

    @staticmethod
    def escape_ffmpeg_path(path: Path | str) -> str:
        """Escape absolute paths for FFmpeg filter arguments."""
        if isinstance(path, str):
            path = Path(path)
        return str(path.resolve()).replace("\\", "/").replace(":", "\\:")

    @staticmethod
    def get_ffmpeg_path() -> str:
        """Resolve FFmpeg binary from workspace/bin, falling back to system 'ffmpeg'."""
        bin_dir = BIN_DIR.resolve()

        ffmpeg_exe = bin_dir / "ffmpeg.exe" if os.name == "nt" else bin_dir / "ffmpeg"

        if ffmpeg_exe.exists():
            return str(ffmpeg_exe)

        # Cross-check: on Linux, maybe .exe exists (shouldn't, but defensive)
        alt = bin_dir / ("ffmpeg" if os.name == "nt" else "ffmpeg.exe")
        if alt.exists():
            return str(alt)

        logger.warning(
            "FFmpeg binary not found in workspace/bin. Falling back to system 'ffmpeg'."
        )
        return "ffmpeg"

    @staticmethod
    def resolve_device(device_str: str) -> str:
        """Resolve 'auto' device string to actual 'cuda' or 'cpu'.

        Probes torch availability for CUDA. Falls back to 'cpu' if torch is not
        installed or no CUDA-capable device is found. Non-'auto' values are returned as-is.
        """
        if device_str.lower() != "auto":
            return device_str.lower()

        resolved = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                resolved = "cuda"
        except ImportError:
            logger.debug("torch not installed; compute device defaults to CPU.")

        # Log the resolved device only the first time (it's queried on every model load).
        if not SystemUtils._device_logged:
            if resolved == "cuda":
                logger.info("Compute device: CUDA GPU.")
            else:
                logger.info("Compute device: CPU (no CUDA GPU detected).")
            SystemUtils._device_logged = True
        return resolved

    @staticmethod
    def extract_youtube_id(url: str) -> str | None:
        """Extract the 11-character YouTube video ID from a watch/short/live URL.

        Returns None when the URL is not a YouTube URL or the ID cannot be parsed.
        """
        patterns = (
            r"(?:v=|/v/|youtu\.be/)([A-Za-z0-9_-]{11})",
            r"(?:/shorts/|/live/)([A-Za-z0-9_-]{11})",
        )
        for pat in patterns:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return None


class AIUtils:
    @staticmethod
    def has_credentials(api_key: str | None) -> bool:
        """True when ``api_key`` is set and is not the config.yaml.example placeholder."""
        from src.core.constants import PLACEHOLDER_API_KEY

        return bool(api_key and api_key != PLACEHOLDER_API_KEY)

    @staticmethod
    def load_whisper_model(model_size: str, device: str) -> object:
        """Load a faster-whisper ``WhisperModel`` with the shared compute-type rule.

        Int8 quantization destroys tiny/base accuracy on CPU, so those sizes force float32.
        """
        from faster_whisper import WhisperModel

        c_type = "float32" if model_size in ("tiny", "base") else "int8"
        resolved_device = SystemUtils.resolve_device(device)
        return WhisperModel(model_size, device=resolved_device, compute_type=c_type)

    @staticmethod
    def load_llama(resolved_path: str, n_gpu_layers: int) -> object:
        """Load a ``llama_cpp.Llama`` model with the shared context/verbosity defaults."""
        from llama_cpp import Llama

        return Llama(
            model_path=resolved_path,
            n_ctx=LLAMA_N_CTX,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    @staticmethod
    def resolve_llm_model_path(model_path: str) -> str:
        """Resolves local model path, downloading from HuggingFace Hub if tag exists."""
        if ":" in model_path:
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
            except ImportError as e:
                logger.error("huggingface_hub is not installed.")
                raise AIProviderError("huggingface_hub package missing.") from e

            repo_id, tag_or_filename = model_path.split(":", 1)

            try:
                if tag_or_filename.endswith(".gguf"):
                    filename = tag_or_filename
                else:
                    logger.info(
                        f"Searching HuggingFace repo '{repo_id}' for a '{tag_or_filename}' GGUF model..."
                    )
                    files = list_repo_files(repo_id)
                    matches = [
                        f
                        for f in files
                        if f.endswith(".gguf") and tag_or_filename.lower() in f.lower()
                    ]

                    if not matches:
                        raise ValueError(
                            f"No GGUF file containing '{tag_or_filename}' found in repo {repo_id}"
                        )

                    filename = matches[0]
                    if len(matches) > 1:
                        logger.warning(
                            f"Multiple matches found. Auto-selecting the first one: {filename}"
                        )

                logger.info(
                    f"Locating model {filename} from HuggingFace repo {repo_id}..."
                )
                return hf_hub_download(repo_id=repo_id, filename=filename)

            except Exception as e:
                logger.error(
                    f"Failed to download/resolve model from HuggingFace Hub: {e}"
                )
                raise AIProviderError(f"HuggingFace model download failed: {e}") from e
        else:
            if not Path(model_path).exists():
                logger.error(f"Local model path does not exist: {model_path}")
                raise AIProviderError(f"Model path does not exist: {model_path}")
            return model_path

    @staticmethod
    def parse_json_array(response_text: str) -> list[dict[str, Any]]:
        """Strip markdown, parse JSON, and unwrap json_object wrappers.

        Handles both raw ``[...]`` arrays and ``{"key": [...]}`` wrappers
        produced by OpenAI's ``json_object`` response_format mode.
        """
        # prompts is imported lazily here: core → ai is a backwards layer dependency, so keep it
        # out of the module-level import graph to avoid a cycle.
        from src.ai.prompts import strip_json_markdown

        clean_text = strip_json_markdown(response_text)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse AI response as JSON: {e}. "
                f"First 200 chars: {clean_text[:200]!r}. "
                f"Last 200 chars: {clean_text[-200:]!r}."
            )
            raise AIProviderError("Failed to parse JSON response.") from e

        # json_object mode may wrap the array as {"clips": [...]} — unwrap.
        if isinstance(parsed_data, dict):
            for v in parsed_data.values():
                if isinstance(v, list):
                    parsed_data = v
                    break
            else:
                raise AIProviderError(
                    "Parsed response is a JSON object with no array value."
                )

        if not isinstance(parsed_data, list):
            raise AIProviderError("Parsed response is not a JSON array.")

        return parsed_data
