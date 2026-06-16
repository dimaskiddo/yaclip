import re
import subprocess

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.utils import SystemUtils
from src.core.workspace import AUDIOS_DIR


class AudioExtractor:
    """Extracts audio tracks from video files."""

    def __init__(self) -> None:
        self.config = load_config()

    def extract_audio(self, video_path: str | Path, force: bool = False) -> str:
        """Extract audio track from video using local FFmpeg binary based on configuration."""
        video_path = Path(video_path)

        aud_ext = self.config.downloader.audio_format
        aud_qual = self.config.downloader.audio_quality
        output_dir = str(AUDIOS_DIR)

        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / f"{video_path.stem}.{aud_ext}"

        # Sanitize audio quality to only digits
        qual_match = re.search(r"\d+", str(aud_qual))
        clean_qual = qual_match.group(0) if qual_match else "192"

        if audio_path.exists() and not force:
            logger.info(
                f"Audio file already exists, skipping extraction: {SystemUtils.display_path(audio_path)}"
            )
            return str(audio_path)

        ffmpeg_cmd = SystemUtils.get_ffmpeg_path()

        cmd = [ffmpeg_cmd, "-y", "-i", str(video_path), "-vn"]

        if aud_ext == "mp3":
            cmd.extend(["-acodec", "libmp3lame", "-b:a", f"{clean_qual}k"])
        elif aud_ext == "aac":
            cmd.extend(["-acodec", "aac", "-b:a", f"{clean_qual}k"])
        elif aud_ext == "wav":
            # WAV uses PCM. Sample rate is parameterized from audio_quality
            try:
                sr = int(clean_qual)
                if sr < 1000:
                    if sr == 44:
                        sample_rate = 44100
                    elif sr == 48:
                        sample_rate = 48000
                    elif sr == 192:
                        sample_rate = 192000
                    else:
                        sample_rate = sr * 1000
                else:
                    sample_rate = sr
            except ValueError:
                sample_rate = 16000  # default fallback for STT

            cmd.extend(["-acodec", "pcm_s16le", "-ar", str(sample_rate)])
        else:
            cmd.extend(["-acodec", "copy"])

        cmd.append(str(audio_path))

        logger.info(f"Extracting audio with format {aud_ext.upper()}...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Audio extraction failed: {result.stderr}")
            raise RuntimeError(f"FFMPEG failed: {result.stderr}")

        logger.info(f"Successfully extracted audio: {SystemUtils.display_path(audio_path)}")
        return str(audio_path)
