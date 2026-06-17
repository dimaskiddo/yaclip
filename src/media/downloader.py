import json
import re
import shutil
import sys
import yt_dlp

from pathlib import Path
from typing import Dict, Optional
from loguru import logger

from src.core.config import load_config
from src.core.utils import SystemUtils
from src.core.workspace import DATA_DIR, TMP_DIR


class YTDLLogger:
    def debug(self, msg):
        pass  # Ignore debug to keep logs clean

    def info(self, msg):
        pass  # Ignore info to keep logs clean

    def warning(self, msg):
        logger.warning(f"Downloader (yt-dlp): {msg}")

    def error(self, msg):
        logger.error(f"Downloader (yt-dlp): {msg}")


class VideoDownloader:
    """Downloads YouTube videos and extracts their audio tracks and heatmap data."""

    def __init__(self) -> None:
        self.config = load_config()

    def _resolve_wsl_cookies(self, browser: str) -> Optional[str]:
        """
        Resolve and copy Windows browser cookies into Linux ./workspace/tmp/ to bypass SQLite locks.
        Returns the fake profile directory path containing the copied cookies database.
        """
        win_user = SystemUtils.get_windows_username()
        if not win_user:
            logger.error("Could not resolve Windows username for cookie mapping.")
            return None

        paths = {
            "edge": f"/mnt/c/Users/{win_user}/AppData/Local/Microsoft/Edge/User Data/Default/Network/Cookies",
            "chrome": f"/mnt/c/Users/{win_user}/AppData/Local/Google/Chrome/User Data/Default/Network/Cookies",
            "firefox": f"/mnt/c/Users/{win_user}/AppData/Roaming/Mozilla/Firefox/Profiles",
        }

        browser = browser.lower()
        if browser not in paths:
            logger.warning(f"Unsupported WSL browser cookie mapping: {browser}")
            return None

        src_path = Path(paths[browser])

        if browser == "firefox":
            if src_path.exists():
                for profile in src_path.iterdir():
                    if profile.is_dir() and profile.name.endswith(".default-release"):
                        db_path = profile / "cookies.sqlite"
                        if db_path.exists():
                            src_path = db_path
                            break
                else:
                    logger.warning("Could not find Firefox default profile cookies.")
                    return None
            else:
                logger.warning(f"Firefox profile path not found: {src_path}")
                return None
        elif not src_path.exists():
            logger.warning(f"Cookie database not found at {src_path}")
            return None

        # Fake profile structure for yt-dlp in ./workspace/tmp/
        tmp_profile = (TMP_DIR / f"wsl_{browser}_cookies").resolve()
        if browser in ("edge", "chrome"):
            tmp_db_dir = tmp_profile / "Network"
            tmp_db_dir.mkdir(parents=True, exist_ok=True)
            dest_path = tmp_db_dir / "Cookies"
        else:
            tmp_db_dir = tmp_profile
            tmp_db_dir.mkdir(parents=True, exist_ok=True)
            dest_path = tmp_db_dir / "cookies.sqlite"

        try:
            shutil.copy2(src_path, dest_path)
            logger.info(f"Copied {browser} cookie DB from {src_path} to {dest_path}")
            return str(tmp_profile)
        except Exception as e:
            logger.error(f"Failed to copy cookie database: {e}")
            return None

    def _extract_metadata(self, info: dict) -> Dict[str, object]:
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
        self, url: str, output_dir: str, progress_callback=None, force: bool = False
    ) -> Dict[str, str]:
        """
        Download a video and extract its audio track dynamically pulling settings
        from config.yaml.
        """
        from src.core.workspace import active_pipeline_event

        active_pipeline_event.set()
        try:
            logger.info(f"Starting download for {url}")
            out_dir_path = Path(output_dir).resolve()
            out_dir_path.mkdir(parents=True, exist_ok=True)

            dl_cfg = self.config.downloader
            browser = dl_cfg.browser_cookies

            # Robustly parse resolution and audio quality digits
            target_res_raw = dl_cfg.target_resolution
            res_match = re.search(r"\d+", str(target_res_raw))
            target_res = res_match.group(0) if res_match else "1080"

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
            }

            # Optional progress hook for CLI and Gradio UI
            download_occurred = [False]

            def progress_hook(d):
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

            # Handle Cookies
            if SystemUtils.is_wsl():
                logger.info("WSL detected. Resolving Windows host cookies...")
                tmp_profile = self._resolve_wsl_cookies(browser)
                if tmp_profile:
                    ydl_opts["cookiesfrombrowser"] = (browser, tmp_profile, None, None)
                else:
                    logger.warning(
                        "WSL cookie resolution failed, proceeding without cookies..."
                    )
            else:
                ydl_opts["cookiesfrombrowser"] = (browser, None, None, None)

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info("Fetching video info...")
                    info = ydl.extract_info(url, download=True)
                    video_id = info.get("id", "unknown")

                    final_video_path = out_dir_path / f"{video_id}.{vid_ext}"

                    if not final_video_path.exists():
                        logger.error(
                            f"Download completed but expected video file is missing: {final_video_path}"
                        )
                        raise FileNotFoundError("Missing video file post-download.")

                    rel_video_path = SystemUtils.display_path(final_video_path)

                    if download_occurred[0]:
                        logger.info(f"Successfully downloaded video: {rel_video_path}")
                    else:
                        logger.info(
                            f"Video file already exists, skipping download: {rel_video_path}"
                        )

                    # Extract and save Heatmap data if available
                    heatmap_data = info.get("heatmap")
                    if heatmap_data:
                        heatmap_path = DATA_DIR / f"{video_id}_heatmap_youtube.json"
                        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                        heatmap_path.write_text(
                            json.dumps(heatmap_data, indent=2), encoding="utf-8"
                        )
                        logger.info(
                            f"Saved YouTube Most Replayed heatmap data ({len(heatmap_data)} points)"
                        )

                    # Capture lightweight metadata (game/show context for the LLM).
                    metadata = self._extract_metadata(info)
                    meta_path = DATA_DIR / f"{video_id}_metadata.json"
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                    logger.info(
                        f"Saved video metadata (category={metadata.get('categories')})"
                    )

                    # Use our custom FFmpeg command to extract the audio
                    from src.media.audio import AudioExtractor

                    audio_extractor = AudioExtractor()
                    final_audio_path = audio_extractor.extract_audio(
                        final_video_path, force=force
                    )

                    return {
                        "video_path": str(final_video_path),
                        "audio_path": final_audio_path,
                        "title": info.get("title", ""),
                        "metadata": metadata,
                    }

            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise
        finally:
            active_pipeline_event.clear()
