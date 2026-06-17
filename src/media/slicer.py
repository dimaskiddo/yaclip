from __future__ import annotations

import subprocess

from loguru import logger

from src.core.utils import SystemUtils


class AudioSlicer:
    """Provides methods to slice audio segments without re-encoding."""

    def slice_audio_chunk(
        self, audio_path: str, start_time: float, end_time: float, output_path: str
    ) -> bool:
        """
        Extracts a specific time segment from an audio file efficiently without re-encoding.
        """
        ffmpeg_cmd = SystemUtils.get_ffmpeg_path()

        # Guard against inverted/zero-length ranges: a degenerate clip would make ffmpeg abort
        # ("-to value smaller than -ss"). Use duration (-t) which is robust to such edge cases.
        duration = end_time - start_time
        if duration <= 0:
            logger.warning(
                f"Skipping audio slice: non-positive duration "
                f"(start={start_time:.2f}s, end={end_time:.2f}s)."
            )
            return False

        cmd = [
            ffmpeg_cmd,
            "-y",
            "-i",
            audio_path,
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-loglevel",
            "error",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cut audio clip: {e}")
            return False
