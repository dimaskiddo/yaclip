from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import yt_dlp
from loguru import logger

from src.core.config import load_config
from src.core.exceptions import DownloadError
from src.core.utils import SystemUtils, extract_digits
from src.core.workspace import DATA_DIR, active_pipeline_event, video_output_path
from src.media.audio import AudioExtractor


def _is_transient_download_error(exc: Exception) -> bool:
    """Determine if a download exception is transient (network drop, 429, timeout) and worth retrying."""
    exc_name = type(exc).__name__
    exc_msg = str(exc).lower()

    transient_classes = {
        "URLError",
        "HTTPError",
        "timeout",
        "SocketTimeout",
        "socket.timeout",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "IncompleteRead",
        "HTTPException",
    }

    if any(cls in exc_name for cls in transient_classes):
        return True

    transient_patterns = [
        "http error 4",
        "too many requests",
        "connection",
        "timeout",
        "timed out",
        "incompleteread",
        "try again",
        "temporary failure",
        "429",
        "read timed out",
    ]
    if any(pat in exc_msg for pat in transient_patterns):
        nontransient_patterns = [
            "unplayable",
            "private",
            "removed",
            "copyright",
            "geo-block",
            "geoblock",
            "sign in",
        ]
        if not any(np in exc_msg for np in nontransient_patterns):
            return True

    return False


class YTDLLogger:
    def debug(self, msg: str) -> None:
        pass  # Ignore debug to keep logs clean

    def info(self, msg: str) -> None:
        pass  # Ignore info to keep logs clean

    def warning(self, msg: str) -> None:
        logger.warning(f"Downloader (yt-dlp): {msg}")

    def error(self, msg: str) -> None:
        logger.error(f"Downloader (yt-dlp): {msg}")


class VideoDownloader:
    """Downloads YouTube videos and extracts their audio tracks and heatmap data."""

    def __init__(self) -> None:
        self.config = load_config()

    @staticmethod
    def _build_result(
        video_path: Path,
        audio_path: str,
        title: str = "Cached",
    ) -> dict[str, str]:
        """Build the standard download return dict."""
        return {
            "video_path": str(video_path),
            "audio_path": audio_path,
            "title": title,
        }

    def _extract_metadata(self, info: dict) -> dict[str, object]:
        """Pull lightweight game/show context fields from the yt-dlp info dict.

        YouTube exposes no stable 'game' field, so we capture the signals an LLM can use to
        infer it: title, categories, top tags, and a truncated description.
        """
        description = (info.get("description") or "")[:500]
        tags = info.get("tags") or []
        return {
            "title": info.get("title", ""),
            "categories": info.get("categories") or [],
            "tags": tags[:15],
            "description": description,
            "channel": info.get("uploader") or info.get("channel") or "",
        }

    def download_video(
        self,
        url: str,
        output_dir: str,
        progress_callback: object | None = None,
        force: bool = False,
        cookies_file: str | None = None,
    ) -> dict[str, str]:
        """
        Download a video and extract its audio track dynamically pulling settings
        from config.yaml.
        """

        active_pipeline_event.set()
        try:
            logger.info(f"Starting download for {url}")
            out_dir_path = Path(output_dir).resolve()
            out_dir_path.mkdir(parents=True, exist_ok=True)

            dl_cfg = self.config.downloader

            # Robustly parse resolution and audio quality digits
            target_res = extract_digits(dl_cfg.target_resolution, default="1080")

            vid_ext = dl_cfg.video_format

            ydl_opts = {
                "format": f"bestvideo[ext=mp4][vcodec^=avc][height<={target_res}]+bestaudio[ext=m4a]/best[ext=mp4][height<={target_res}]/bestvideo[height<={target_res}]+bestaudio/best[height<={target_res}]",
                "merge_output_format": vid_ext,
                "outtmpl": str(out_dir_path / "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "logger": YTDLLogger(),
                "js_runtimes": {"bun": {}},
                "remote_components": ["ejs:github"],
                "extractor_args": {"youtube": ["player_client=ios,android,web"]},
                "overwrites": force,
                "socket_timeout": dl_cfg.retry.socket_timeout,
                "fragment_retries": dl_cfg.retry.fragment_retries,
            }

            # Optional progress hook for CLI and Gradio UI
            download_occurred = [False]

            def progress_hook(d: dict) -> None:
                if d["status"] == "downloading":
                    download_occurred[0] = True
                    percent_str = d.get("_percent_str", "N/A").strip()
                    speed = d.get("_speed_str", "N/A").strip()
                    eta = d.get("_eta_str", "N/A").strip()

                    sys.stdout.write(
                        f"\r\033[K[yt-dlp] Downloading... {percent_str} at {speed} (ETA: {eta})"
                    )
                    sys.stdout.flush()

                    if progress_callback:
                        clean_percent = (
                            re.sub(r"\x1b\[[0-9;]*m", "", percent_str)
                            .replace("%", "")
                            .strip()
                        )
                        try:
                            percent = float(clean_percent)
                        except ValueError:
                            percent = 0.0
                        progress_callback(percent, d)

                elif d["status"] == "finished":
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    logger.info("Download completed, processing file...")
                    if progress_callback:
                        progress_callback(100.0, d)

            ydl_opts["progress_hooks"] = [progress_hook]

            # Handle cookies file (manually exported browser cookies in Netscape format).
            if cookies_file and Path(cookies_file).exists():
                ydl_opts["cookiefile"] = cookies_file
                logger.info(f"Using cookies file: {cookies_file}")
            else:
                logger.info("No cookies file provided, proceeding without cookies.")

            # Single AudioExtractor instance shared by pre-flight and download paths.
            audio_extractor = AudioExtractor()

            # Pre-flight check: if uppercase video file already exists, skip download entirely.
            cached_id = SystemUtils.extract_youtube_id(url)
            if cached_id:
                cached_path = video_output_path(cached_id, vid_ext)
                if cached_path.exists() and not force:
                    logger.info(
                        f"Video already downloaded, using existing file: "
                        f"{SystemUtils.display_path(cached_path)}"
                    )
                    video_id = cached_id
                    final_video_path = cached_path
                    final_audio_path = audio_extractor.extract_audio(
                        str(final_video_path), force=False
                    )
                    return self._build_result(final_video_path, final_audio_path)

            try:
                max_attempts = dl_cfg.retry.max_attempts
                delay_seconds = dl_cfg.retry.delay_seconds

                info = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            logger.info(
                                f"Fetching video info (attempt {attempt}/{max_attempts})..."
                            )
                            info = ydl.extract_info(url, download=True)
                            break
                    except Exception as e:
                        if attempt == max_attempts or not _is_transient_download_error(
                            e
                        ):
                            logger.error(
                                f"Download failed permanently on attempt {attempt}/{max_attempts}: {e}"
                            )
                            raise DownloadError(f"Download failed: {e}") from e

                        sleep_time = delay_seconds**attempt
                        logger.warning(
                            f"Download attempt {attempt}/{max_attempts} failed with transient error: {e}. "
                            f"Retrying in {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)

                if info is None:
                    raise DownloadError(
                        "Failed to extract video info: no metadata returned."
                    )

                video_id = info.get("id", "unknown")
                flat_path = out_dir_path / f"{video_id}.{vid_ext}"
                final_video_path = video_output_path(video_id, vid_ext)

                if flat_path.exists():
                    if flat_path != final_video_path:
                        flat_path.rename(final_video_path)
                elif not final_video_path.exists():
                    logger.error(
                        f"Download completed but expected video file is missing: {final_video_path}"
                    )
                    raise DownloadError("Missing video file post-download.")

                rel_video_path = SystemUtils.display_path(final_video_path)

                if download_occurred[0]:
                    logger.info(f"Video downloaded successfully to {rel_video_path}.")
                else:
                    logger.info(
                        f"Video already downloaded, using existing file: {rel_video_path}."
                    )

                # Extract and save Heatmap data if available
                heatmap_data = info.get("heatmap")
                if heatmap_data:
                    heatmap_path = DATA_DIR / f"{video_id}_heatmap_youtube.json"
                    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                    heatmap_path.write_text(
                        json.dumps(heatmap_data, indent=2), encoding="utf-8"
                    )
                    logger.info("Replay heatmap data saved from YouTube.")

                # Capture lightweight metadata (game/show context for the LLM).
                metadata = self._extract_metadata(info)
                meta_path = DATA_DIR / f"{video_id}_metadata.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                cats = metadata.get("categories", [])
                cats_str = (
                    ", ".join(cats)
                    if isinstance(cats, list)
                    else str(cats or "unknown")
                )
                logger.info(f"Video category detected as {cats_str}.")

                # Extract audio via the shared AudioExtractor instance.
                final_audio_path = audio_extractor.extract_audio(
                    final_video_path, force=force
                )

                return self._build_result(
                    final_video_path, final_audio_path, info.get("title", "")
                )

            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise
        finally:
            active_pipeline_event.clear()
