from __future__ import annotations

import io
import json
import os
import platform
import shutil
import threading
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.core.constants import BYTES_PER_MB
from src.core.exceptions import CacheInitError

# Global Path Constants
WORKSPACE_DIR = Path("workspace").resolve()
BIN_DIR = WORKSPACE_DIR / "bin"
FONTS_DIR = WORKSPACE_DIR / "fonts"
MODELS_DIR = WORKSPACE_DIR / "models"
AUDIOS_DIR = WORKSPACE_DIR / "audios"
VIDEOS_DIR = WORKSPACE_DIR / "videos"
SUBTITLES_DIR = WORKSPACE_DIR / "subtitles"  # .ass subtitle files only
DATA_DIR = WORKSPACE_DIR / "data"  # STT transcripts + AI/cache JSON
CLIPS_DIR = WORKSPACE_DIR / "clips"
LOGS_DIR = WORKSPACE_DIR / "logs"
TMP_DIR = WORKSPACE_DIR / "tmp"

# Global pipeline execution guard — set while any pipeline or download is active
active_pipeline_event: threading.Event = threading.Event()


def _binary_exists(name: str, bin_dir: Path = BIN_DIR) -> bool:
    """Check if a binary exists in bin_dir, handling platform extension."""
    exe = bin_dir / (name + ".exe" if os.name == "nt" else name)
    return exe.exists()


def workspace_path(video_id: str, suffix: str) -> Path:
    """Build a ``DATA_DIR / f"{video_id}{suffix}"`` path (e.g. suffix="_words.json")."""
    return DATA_DIR / f"{video_id}{suffix}"


def save_candidate_cache(video_id: str, suffix: str, candidates: list[dict], label: str) -> None:
    """Persist a ``{"video_id", "candidates": [...]}`` JSON cache for render-time reuse.

    Shared by the per-candidate word-timing and donation-scan caches (``stt_local.py``) — same
    schema, only the filename suffix and log noun differ.

    Args:
        video_id: The source video's id (filename stem).
        suffix: Filename suffix appended to ``video_id``, e.g. "_words.json".
        candidates: The per-candidate dicts to persist verbatim.
        label: Human-readable noun for the log line, e.g. "word timing data".
    """
    path = workspace_path(video_id, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps({"video_id": video_id, "candidates": candidates}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved {label} ({len(candidates)} candidate(s)): {path.name}")
    except OSError as e:
        logger.warning(f"Could not save {label} to {path.name}: {e}")


def load_candidate_cache(video_id: str, suffix: str, label: str) -> dict | None:
    """Load a candidate cache written by ``save_candidate_cache``, or None if absent/unreadable."""
    path = workspace_path(video_id, suffix)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read {label} from {path.name}: {e}")
        return None


def ensure_workspace_integrity() -> None:
    """Check ./workspace/ environment on startup and download missing assets."""
    try:
        for d in [
            WORKSPACE_DIR,
            BIN_DIR,
            FONTS_DIR,
            MODELS_DIR,
            VIDEOS_DIR,
            AUDIOS_DIR,
            SUBTITLES_DIR,
            DATA_DIR,
            TMP_DIR,
            CLIPS_DIR,
            LOGS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise CacheInitError(f"Failed to create cache directories: {e}") from e

    # Check for FFmpeg binaries
    ffmpeg_missing = not _binary_exists("ffmpeg")

    if ffmpeg_missing:
        logger.info("Downloading video processing tool (FFmpeg) to workspace/bin...")
        try:
            from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise

            ffmpeg_exe, ffprobe_exe = get_or_fetch_platform_executables_else_raise()
            shutil.copy(ffmpeg_exe, BIN_DIR)
            shutil.copy(ffprobe_exe, BIN_DIR)
        except Exception as e:
            logger.warning(
                f"Failed to download FFmpeg: {e}. Make sure FFmpeg is installed on your system."
            )

    # Check for Bun
    bun_missing = not _binary_exists("bun")

    if bun_missing:
        logger.info("Downloading Bun JavaScript runtime to workspace/bin...")
        try:
            system = platform.system().lower()
            machine = platform.machine().lower()

            if system == "windows":
                url = "https://github.com/oven-sh/bun/releases/latest/download/bun-windows-x64.zip"
                exe_name = "bun.exe"
            elif system == "darwin":
                url = (
                    "https://github.com/oven-sh/bun/releases/latest/download/bun-darwin-aarch64.zip"
                    if machine == "arm64"
                    else "https://github.com/oven-sh/bun/releases/latest/download/bun-darwin-x64.zip"
                )
                exe_name = "bun"
            else:  # Linux
                url = (
                    "https://github.com/oven-sh/bun/releases/latest/download/bun-linux-aarch64.zip"
                    if machine in ("aarch64", "arm64")
                    else "https://github.com/oven-sh/bun/releases/latest/download/bun-linux-x64.zip"
                )
                exe_name = "bun"

            with (
                urllib.request.urlopen(url, timeout=60) as response,
                zipfile.ZipFile(io.BytesIO(response.read())) as z,
            ):
                for file_name in z.namelist():
                    if file_name.endswith(exe_name):
                        target_path = BIN_DIR / exe_name
                        with z.open(file_name) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        if system != "windows":
                            os.chmod(target_path, 0o755)
                        break
        except Exception as e:
            logger.error(f"Failed to download Bun: {e}")

    # Check font
    font_file = FONTS_DIR / "Anton.ttf"
    if not font_file.exists():
        logger.info("Downloading default high-impact font to workspace/fonts...")
        try:
            font_url = "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
            urllib.request.urlretrieve(font_url, font_file)
        except Exception as e:
            logger.error(f"Failed to download default font: {e}")

    logger.info("Workspace check complete. All required tools and directories are ready.")


def run_purge_cycle(force: bool = False, specific_target: str | list[str] | None = None) -> None:
    """Run a single cache purge check sequentially, deleting stale files based on retention settings.

    Args:
        force: When True, bypass retention and dry-run settings.
        specific_target: Directory name(s) to purge (e.g. "tmp" or ["clips", "tmp"]).
            When None, all purgeable directories are processed.
    """
    from src.core.config import load_config

    config = load_config()
    cleanup_cfg = config.workspace_cleanup
    if not cleanup_cfg.enabled and not force:
        return

    targets_set: set[str] | None = None
    if isinstance(specific_target, str):
        targets_set = {specific_target}
    elif specific_target is not None:
        targets_set = set(specific_target)

    protected_dirs = set(cleanup_cfg.protected_dirs)

    if targets_set is not None:
        blocked = targets_set & protected_dirs
        if blocked:
            for d in sorted(blocked):
                logger.warning(
                    f"Workspace Cleaner - Can't clean cache directory '{d}' because it is protected."
                )
            return

    if force:
        target_str = ", ".join(sorted(targets_set)) if targets_set else "all targets"
        logger.warning(
            f"Workspace Cleaner - Starting FORCED manual cache purge for {target_str} (ignoring retention/dry-run)..."
        )
    else:
        logger.info("Workspace Cleaner - Starting scheduled cache purge cycle...")

    dry_run = False if force else cleanup_cfg.dry_run
    retention = cleanup_cfg.retention_days

    files_deleted = 0
    total_freed_bytes = 0

    # Folders to cleanup and their corresponding retention settings
    targets = [
        ("videos", retention.videos, False),
        ("audios", retention.audios, False),
        ("subtitles", retention.subtitles, False),
        ("data", retention.data, False),
        ("tmp", retention.tmp, True),
        ("clips", -1, False),  # -1 means never auto-delete unless forced
        ("logs", -1, False),  # Handled by loguru's own retention policy
    ]

    for dir_name, days, recursive in targets:
        if targets_set and dir_name not in targets_set:
            continue

        if not force and days == -1:
            logger.debug(
                f"Workspace Cleaner - Cleanup disabled for subdirectory '{dir_name}' (retention = -1)."
            )
            continue

        dir_path = WORKSPACE_DIR / dir_name
        if not dir_path.exists():
            continue

        # Check if dir_name itself or any parent matches protected folders (just in case)
        if dir_name in protected_dirs:
            continue

        # Determine threshold time
        threshold_time = datetime.now() - timedelta(days=days)

        # Get all candidate paths
        if recursive:
            paths = [p for p in dir_path.rglob("*") if p.is_file()]
        else:
            paths = [p for p in dir_path.iterdir() if p.is_file()]

        for path in paths:
            try:
                # Get file modification time
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if force or mtime < threshold_time:
                    file_size = path.stat().st_size
                    if dry_run:
                        logger.info(
                            f"Workspace Cleaner - Dry Run Would delete: {path} (Modified: {mtime})"
                        )
                    else:
                        path.unlink(missing_ok=True)
                        files_deleted += 1
                        total_freed_bytes += file_size
                        logger.debug(f"Workspace Cleaner - Deleted: {path}")
            except Exception as e:
                # Ignore errors if file is already deleted by another process
                if not isinstance(e, FileNotFoundError):
                    logger.warning(f"Workspace Cleaner - Failed to process/delete file {path}: {e}")

    if files_deleted > 0 or total_freed_bytes > 0:
        freed_mb = total_freed_bytes / BYTES_PER_MB
        logger.info(f"Workspace Cleaner - Purged {files_deleted} files, freeing {freed_mb:.2f} MB.")
    else:
        logger.info("Workspace Cleaner - Purge cycle completed. No stale files found.")
