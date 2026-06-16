import os
import platform
import subprocess

from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from src.core.exceptions import AIProviderError
from src.core.workspace import BIN_DIR


class SystemUtils:
    _device_logged: bool = False  # so the resolved compute device is logged once, not per model load

    @staticmethod
    def is_wsl() -> bool:
        """Detect if running inside Windows Subsystem for Linux."""
        if platform.system() == "Linux":
            try:
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        return True
            except FileNotFoundError:
                pass
        return False

    @staticmethod
    def get_windows_username() -> Optional[str]:
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
        bin_dir = Path(str(BIN_DIR)).resolve()

        if os.name == "nt":
            ffmpeg_exe = bin_dir / "ffmpeg.exe"
        else:
            ffmpeg_exe = bin_dir / "ffmpeg"

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
            pass

        # Log the resolved device only the first time (it's queried on every model load).
        if not SystemUtils._device_logged:
            if resolved == "cuda":
                logger.info("Compute device: CUDA GPU.")
            else:
                logger.info("Compute device: CPU (no CUDA GPU detected).")
            SystemUtils._device_logged = True
        return resolved


class AIUtils:
    @staticmethod
    def resolve_llm_model_path(model_path: str) -> str:
        """Resolves local model path, downloading from HuggingFace Hub if tag exists."""
        if ":" in model_path:
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
            except ImportError as e:
                logger.error("huggingface_hub is not installed.")
                raise ImportError("huggingface_hub package missing.") from e

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

                logger.info(f"Locating model {filename} from HuggingFace repo {repo_id}...")
                return hf_hub_download(
                    repo_id=repo_id, filename=filename
                )

            except Exception as e:
                logger.error(f"Failed to download/resolve model from HuggingFace Hub: {e}")
                raise AIProviderError(f"HuggingFace model download failed: {e}") from e
        else:
            if not Path(model_path).exists():
                logger.error(f"Local model path does not exist: {model_path}")
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            return model_path

    @staticmethod
    def parse_json_array(response_text: str) -> List[Dict[str, Any]]:
        """Strips markdown and parses the response into a JSON array, ensuring it's valid."""
        import json
        from src.ai.prompts import strip_json_markdown

        clean_text = strip_json_markdown(response_text)
        try:
            parsed_data = json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw AI response text: {clean_text}")
            raise ValueError("Failed to parse JSON response.") from e

        if not isinstance(parsed_data, list):
            raise ValueError("Parsed response is not a JSON array.")

        return parsed_data
