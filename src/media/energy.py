from __future__ import annotations

import math
import struct
import subprocess

from loguru import logger

from src.core.config import load_config
from src.core.constants import RMS_SAMPLE_RATE
from src.core.utils import SystemUtils


class AudioEnergyAnalyzer:
    """Analyzes audio energy (RMS) to generate pseudo-heatmap spikes."""

    def __init__(self) -> None:
        self.config = load_config()

    def _stream_rms(
        self,
        cmd: list[str],
        chunk_duration_sec: float,
    ) -> list[float]:
        """Run an FFmpeg command emitting raw s16le mono PCM on stdout and return one
        RMS value per ``chunk_duration_sec`` chunk.  Shared by ``analyze_audio_energy``
        (whole-file heatmap) and ``rms_envelope`` (windowed, aligned to detection steps)."""
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"Failed to start audio analysis process: {e}")
            return []

        chunk_size = int(RMS_SAMPLE_RATE * chunk_duration_sec * 2)  # 2 bytes/sample
        chunk_size = max(2, chunk_size - (chunk_size % 2))  # keep whole 16-bit samples

        rms_values: list[float] = []
        while True:
            data = process.stdout.read(chunk_size)
            if not data:
                break
            samples = struct.unpack(f"<{len(data) // 2}h", data)
            if not samples:
                break
            rms_values.append(math.sqrt(sum(s * s for s in samples) / len(samples)))

        process.wait()
        return rms_values

    def rms_envelope(
        self,
        media_path: str,
        start: float,
        end: float,
        step_seconds: float,
    ) -> list[float]:
        """Return a per-step RMS loudness envelope for the ``[start, end]`` window of a
        media file (audio or video container — FFmpeg extracts the audio with ``-vn``).

        Used by the face tracker for audio-visual active-speaker detection: each value
        aligns to one detection step (``step_seconds = detection_interval / fps``), so the
        envelope index maps 1:1 to the visual detection index.
        """
        if end <= start or step_seconds <= 0:
            return []

        ffmpeg_cmd = SystemUtils.get_ffmpeg_path()
        cmd = [
            ffmpeg_cmd,
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", media_path,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(RMS_SAMPLE_RATE),
            "-ac", "1",
            "pipe:1",
            "-loglevel", "quiet",
        ]
        return self._stream_rms(cmd, step_seconds)

    def analyze_audio_energy(
        self, audio_path: str, chunk_duration_sec: float = 1.0
    ) -> list[dict]:
        """
        Generates a pseudo-heatmap by calculating RMS energy of audio chunks.
        Returns standard clip objects based on the loudest spikes.
        """
        ffmpeg_cmd = SystemUtils.get_ffmpeg_path()

        logger.info("Analyzing audio loudness to find the most energetic moments...")
        cmd = [
            ffmpeg_cmd,
            "-i",
            audio_path,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(RMS_SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
            "-loglevel",
            "quiet",
        ]

        rms_values = self._stream_rms(cmd, chunk_duration_sec)

        energies = [
            {"time": i * chunk_duration_sec, "rms": rms}
            for i, rms in enumerate(rms_values)
        ]

        if not energies:
            logger.warning("No audio energy data extracted.")
            return []

        logger.info(f"Processed {len(energies)} seconds of audio for energy analysis.")

        clip_cfg = self.config.clip_selection
        min_duration = clip_cfg.min_clip_duration_seconds
        max_clips = clip_cfg.max_clips

        # Sort by RMS descending to find the absolute loudest moments
        loudest = sorted(energies, key=lambda x: x["rms"], reverse=True)

        clips = []
        for peak in loudest:
            if len(clips) >= max_clips:
                break

            spike_time = peak["time"]

            # Check if this spike is already inside an existing clip to prevent overlapping identical clips
            is_overlapping = False
            for c in clips:
                if (
                    c["start_time"] - 10 <= spike_time <= c["end_time"] + 10
                ):  # 10 second safety buffer
                    is_overlapping = True
                    break

            if not is_overlapping:
                start_time = max(0.0, spike_time - (min_duration / 2.0))
                end_time = spike_time + (min_duration / 2.0)

                clips.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "title": "High Energy Moment",
                        "reasoning": f"Detected a peak loudness moment (energy score: {peak['rms']:.2f}).",
                        "score": float(peak["rms"]),
                    }
                )

        # Sort chronological
        clips.sort(key=lambda x: x["start_time"])
        return clips
