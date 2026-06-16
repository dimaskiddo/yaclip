import math
import struct
import subprocess

from typing import Dict, List
from loguru import logger

from src.core.config import load_config
from src.core.utils import SystemUtils


class AudioEnergyAnalyzer:
    """Analyzes audio energy (RMS) to generate pseudo-heatmap spikes."""

    def __init__(self) -> None:
        self.config = load_config()

    def analyze_audio_energy(
        self, audio_path: str, chunk_duration_sec: float = 1.0
    ) -> List[Dict]:
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
            "8000",
            "-ac",
            "1",
            "pipe:1",
            "-loglevel",
            "quiet",
        ]

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"Failed to start audio analysis process: {e}")
            return []

        sample_rate = 8000
        chunk_size = int(sample_rate * chunk_duration_sec * 2)  # 2 bytes per sample

        energies = []
        timestamp = 0.0

        while True:
            data = process.stdout.read(chunk_size)
            if not data:
                break

            samples = struct.unpack(f"<{len(data) // 2}h", data)
            if not samples:
                break

            # Compute RMS
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            energies.append({"time": timestamp, "rms": rms})
            timestamp += chunk_duration_sec

        process.wait()

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
