from __future__ import annotations

import math
import struct
import subprocess
import numpy as np

from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    RMS_SAMPLE_RATE,
    SPEAKER_F0_MAX_HZ,
    SPEAKER_F0_MIN_HZ,
    SPEAKER_KMEANS_ITERS,
    SPEAKER_MIN_CLUSTER_FRAC,
    SPEAKER_MIN_SEPARATION_HZ,
    SPEAKER_MIN_VOICED_FRAMES,
    SPEAKER_PITCH_CLARITY_MIN,
    SPEAKER_PITCH_FRAME_SAMPLES,
    SPEAKER_VOICE_RMS_FLOOR_FRAC,
)
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

    # ----------------------------------------------------- Speaker-Count Hint

    def estimate_speaker_count(self, audio_path: str) -> int:
        """Estimate distinct speakers in an audio slice (lightweight pitch clustering; no ML model).

        Reuses the same 8 kHz mono PCM pipe as the RMS analysis: voiced frames are pitch-tracked and
        the pitches clustered — two separated, well-populated modes ⇒ ~2 speakers, else 1. A coarse
        hint for the LLM clip-classifier (one voice over gameplay vs two people talking), not exact
        diarization.

        Returns:
            1 or 2 (1 on any failure or too little voiced audio).
        """
        samples = self._decode_mono_pcm(audio_path)
        if samples.size == 0:
            return 1
        pitches = self._voiced_frame_pitches(samples)
        return 2 if self._looks_like_two_speakers(pitches) else 1

    def _decode_mono_pcm(self, audio_path: str) -> np.ndarray:
        """Decode an audio file to a float32 mono 8 kHz sample array (empty on failure)."""
        cmd = [
            SystemUtils.get_ffmpeg_path(), "-i", audio_path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(RMS_SAMPLE_RATE), "-ac", "1",
            "pipe:1", "-loglevel", "quiet",
        ]
        try:
            out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except Exception as e:
            logger.warning(f"Speaker-count audio decode failed: {e}")
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(out, dtype=np.int16).astype(np.float32)

    @staticmethod
    def _voiced_frame_pitches(samples: np.ndarray) -> np.ndarray:
        """Per-frame fundamental frequency (Hz) for voiced frames, via autocorrelation."""
        frame = SPEAKER_PITCH_FRAME_SAMPLES
        if samples.size < frame * 2:
            return np.empty(0, dtype=np.float32)
        hop = frame // 2
        n = 1 + (samples.size - frame) // hop
        frames = np.stack([samples[i * hop : i * hop + frame] for i in range(n)])

        # Voice gate: keep frames at least half as loud as the median frame (drop near-silence).
        rms = np.sqrt((frames**2).mean(axis=1) + 1e-9)
        voiced = frames[rms >= SPEAKER_VOICE_RMS_FLOOR_FRAC * float(np.median(rms))]

        lag_min = int(RMS_SAMPLE_RATE / SPEAKER_F0_MAX_HZ)
        lag_max = int(RMS_SAMPLE_RATE / SPEAKER_F0_MIN_HZ)
        pitches: list[float] = []
        for f in voiced:
            f = f - f.mean()
            corr = np.correlate(f, f, mode="full")[f.size - 1 :]
            if corr[0] <= 0:
                continue
            band = corr[lag_min : lag_max + 1]
            if band.size == 0:
                continue
            lag = lag_min + int(np.argmax(band))
            # A clear pitch has a strong autocorrelation peak relative to zero-lag energy.
            if corr[lag] >= SPEAKER_PITCH_CLARITY_MIN * corr[0]:
                pitches.append(RMS_SAMPLE_RATE / lag)
        return np.array(pitches, dtype=np.float32)

    @staticmethod
    def _looks_like_two_speakers(pitches: np.ndarray) -> bool:
        """True when the pitch values split into two separated, well-populated clusters."""
        if pitches.size < SPEAKER_MIN_VOICED_FRAMES:
            return False
        lo, hi = float(pitches.min()), float(pitches.max())
        if hi - lo < SPEAKER_MIN_SEPARATION_HZ:
            return False
        c0, c1 = lo, hi  # seed the 1-D 2-means at the pitch extremes
        g0 = g1 = pitches
        for _ in range(SPEAKER_KMEANS_ITERS):
            g0 = pitches[np.abs(pitches - c0) <= np.abs(pitches - c1)]
            g1 = pitches[np.abs(pitches - c0) > np.abs(pitches - c1)]
            if g0.size == 0 or g1.size == 0:
                return False
            c0, c1 = float(g0.mean()), float(g1.mean())
        separated = abs(c1 - c0) >= SPEAKER_MIN_SEPARATION_HZ
        balanced = min(g0.size, g1.size) / pitches.size >= SPEAKER_MIN_CLUSTER_FRAC
        return separated and balanced
