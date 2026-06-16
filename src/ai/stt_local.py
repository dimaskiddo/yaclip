from __future__ import annotations

import gc
import json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

from pathlib import Path
from loguru import logger

from src.ai.prompts import get_language_prompt
from src.core.config import load_config
from src.core.constants import (
    STT_COMPRESSION_MAX,
    STT_LOGPROB_MIN,
    STT_NO_SPEECH_MAX,
    STT_REPEAT_TOKEN_MAX,
)
from src.core.utils import SystemUtils
from src.core.workspace import SUBTITLES_DIR


def segments_to_transcript(segments: list[dict]) -> str:
    """Flatten word-level segments into the ``[start - end] text`` lines the LLM expects."""
    return "\n".join(
        f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}" for s in segments
    )


def _norm_token(word: str) -> str:
    """Lowercase, alphanumerics-only form of a word (for repeat detection)."""
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _is_hallucinated_segment(seg: object) -> bool:
    """True when a faster-whisper segment looks like laughter / noise / looped hallucination.

    Whisper transcribes non-speech (laughter, music, reaction noise) as repeated filler tokens that
    `temperature=0` will not auto-reject. A segment is dropped when it is clearly repetitive
    (compression ratio), clearly non-speech (no_speech_prob + low avg_logprob), or a single token
    repeated many times. Complements the visual word de-dup in ``subtitles.py``.
    """
    if (getattr(seg, "compression_ratio", 0.0) or 0.0) > STT_COMPRESSION_MAX:
        return True
    no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
    avg_logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
    if no_speech > STT_NO_SPEECH_MAX and avg_logprob < STT_LOGPROB_MIN:
        return True
    tokens = [
        _norm_token(w.word) for w in (getattr(seg, "words", None) or []) if _norm_token(w.word)
    ]
    if len(tokens) > STT_REPEAT_TOKEN_MAX and len(set(tokens)) <= 1:
        return True
    return False


def save_words_cache(video_id: str, candidates: list[dict]) -> None:
    """Persist per-candidate word-level segments (absolute times) for render-time reuse.

    Schema: ``{"video_id", "candidates": [{"start", "end", "segments": [...]}]}``. Written to
    ``workspace/subtitles/{video_id}_words.json`` so the renderer can skip re-transcribing clips.
    """
    path = SUBTITLES_DIR / f"{video_id}_words.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps({"video_id": video_id, "candidates": candidates}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            f"Saved word timing data ({len(candidates)} clip(s)): "
            f"{SystemUtils.display_path(path)}"
        )
    except OSError as e:
        logger.warning(f"Could not save word timing data to {path.name}: {e}")


def load_words_cache(video_id: str) -> dict | None:
    """Load the per-candidate word cache for a video, or None if absent/unreadable."""
    path = SUBTITLES_DIR / f"{video_id}_words.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read word timing data from {path.name}: {e}")
        return None


def save_mediashare_cache(video_id: str, candidates: list[dict]) -> None:
    """Persist per-candidate donation/MediaShare scan results for render-time reuse.

    Schema: ``{"video_id", "candidates": [{"start", "end", "mediashare_present", "mediashare_box"}]}``.
    Written to ``workspace/subtitles/{video_id}_mediashare.json`` so the renderer can reuse
    the selection-phase donation scan instead of rescanning each clip window.
    """
    path = SUBTITLES_DIR / f"{video_id}_mediashare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps({"video_id": video_id, "candidates": candidates}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            f"Saved donation scan results ({len(candidates)} candidate(s)): "
            f"{SystemUtils.display_path(path)}"
        )
    except OSError as e:
        logger.warning(f"Could not save donation scan results to {path.name}: {e}")


def load_mediashare_cache(video_id: str) -> dict | None:
    """Load the per-candidate donation scan cache for a video, or None if absent/unreadable."""
    path = SUBTITLES_DIR / f"{video_id}_mediashare.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read donation scan results from {path.name}: {e}")
        return None


class LocalSTTProvider:
    """Provides STT transcription via local models (faster-whisper)."""

    def __init__(self) -> None:
        self.config = load_config()
        self.local_cfg = self.config.ai_pipeline.stt.local
        self.model_size = self.local_cfg.model_size

        if not self.model_size:
            raise ValueError(
                "Local STT mode requires 'model_size' to be defined under ai_pipeline.stt.local in config.yaml."
            )

    def _build_transcribe_kwargs(self, language: str) -> dict:
        """Shared faster-whisper decode kwargs (with hallucination-control knobs)."""
        kwargs: dict = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 1000},
            "word_timestamps": True,  # DTW alignment helps ignore music hallucinations
            "condition_on_previous_text": False,  # Prevents loop hallucinations on background noise
            "temperature": [0.0],  # Freeze sampling for deterministic output
            "no_speech_threshold": STT_NO_SPEECH_MAX,
            "log_prob_threshold": STT_LOGPROB_MIN,
            "compression_ratio_threshold": STT_COMPRESSION_MAX,
            "hallucination_silence_threshold": 2.0,
            "repetition_penalty": 1.1,
        }
        explicit = bool(language and language.lower() != "auto")
        if explicit:
            kwargs["language"] = language
        # initial_prompt = native-language primer (when a language is set) + the user vocab hint.
        # Applied even on auto-detect when stt_context is set, so names/terms still prime decoding.
        lang_primer = get_language_prompt(language) if explicit else None
        stt_context = self.config.video_processing.subtitles.stt_context.strip()
        primer = " ".join(p for p in (lang_primer, stt_context) if p)
        if primer:
            kwargs["initial_prompt"] = primer
        return kwargs

    def _transcribe_whisper(
        self, audio_path: str, model: WhisperModel | None = None
    ) -> str:
        """Run faster-whisper locally to extract transcript with timestamps."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            logger.error("faster-whisper is not installed.")
            raise ImportError("faster-whisper package missing.") from e

        # Int8 quantization destroys tiny/base accuracy on CPU, force float32 for those
        c_type = "float32" if self.model_size in ("tiny", "base") else "int8"

        device_config = SystemUtils.resolve_device(self.local_cfg.device)

        if model is None:
            logger.info("Loading local transcription model...")
            model = WhisperModel(
                self.model_size, device=device_config, compute_type=c_type
            )
        else:
            logger.info("Reusing local transcription model...")

        language = self.config.video_processing.subtitles.language
        if language and language.lower() != "auto":
            logger.info(f"Forcing local transcription language: {language}")
        kwargs = self._build_transcribe_kwargs(language)

        logger.info("Starting local transcription...")
        segments, info = model.transcribe(audio_path, **kwargs)

        transcript_lines = []
        dropped = 0
        for segment in segments:
            if _is_hallucinated_segment(segment):
                dropped += 1
                continue
            line = f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}"
            transcript_lines.append(line)
            logger.debug(line)

        if dropped:
            logger.info(f"Filtered out {dropped} non-speech segment(s) (laughter, noise, or repetition).")
        logger.info("Local transcription complete.")
        return "\n".join(transcript_lines)

    def transcribe_segments(
        self,
        audio_path: str | Path,
        model: WhisperModel | None = None,
        time_offset: float = 0.0,
    ) -> list[dict]:
        """Transcribe audio and return structured segments with word-level timestamps.

        Unlike ``transcribe`` (which returns flattened text for the LLM), this surfaces the
        per-word timing needed to burn karaoke subtitles. Timestamps are shifted by
        ``time_offset`` so they sit on the original video timeline.

        Args:
            audio_path: Path to the audio file (typically a single clip's audio range).
            model: Optional pre-loaded WhisperModel to reuse across clips.
            time_offset: Seconds to add to every timestamp (the clip's start in the source).

        Returns:
            A list of ``{text, start, end, words: [{word, start, end}]}`` segment dicts.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            logger.error("faster-whisper is not installed.")
            raise ImportError("faster-whisper package missing.") from e

        c_type = "float32" if self.model_size in ("tiny", "base") else "int8"
        device_config = SystemUtils.resolve_device(self.local_cfg.device)

        if model is None:
            logger.info("Loading local transcription model for subtitle segments...")
            model = WhisperModel(
                self.model_size, device=device_config, compute_type=c_type
            )

        language = self.config.video_processing.subtitles.language
        kwargs = self._build_transcribe_kwargs(language)

        segments, _info = model.transcribe(str(audio_path), **kwargs)

        out: list[dict] = []
        dropped = 0
        for segment in segments:
            if _is_hallucinated_segment(segment):
                dropped += 1
                continue
            words = []
            for w in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "word": w.word,
                        "start": float(w.start) + time_offset,
                        "end": float(w.end) + time_offset,
                    }
                )
            out.append(
                {
                    "text": segment.text.strip(),
                    "start": float(segment.start) + time_offset,
                    "end": float(segment.end) + time_offset,
                    "words": words,
                }
            )
        if dropped:
            logger.info(f"Filtered out {dropped} non-speech segment(s) (laughter, noise, or repetition).")
        logger.info(f"Transcribed {len(out)} spoken segment(s) with word timings.")
        return out

    def transcribe(
        self,
        audio_path: str | Path,
        force: bool = False,
        model: WhisperModel | None = None,
        cache_dir: Path | None = None,
    ) -> str:
        """Transcribes audio locally using Whisper and returns the transcript text.

        Args:
            audio_path: Path to the audio file to transcribe.
            force: If True, re-transcribe even if a cached transcript exists.
            model: Optional pre-loaded WhisperModel to reuse (avoids re-loading).
            cache_dir: Directory to save the transcript cache. Defaults to SUBTITLES_DIR.
                       Pass TMP_DIR for temporary slice chunks so they are auto-cleaned.
        """
        audio_path = Path(audio_path)
        video_id = audio_path.stem
        out_dir = cache_dir if cache_dir is not None else SUBTITLES_DIR
        out_txt = out_dir / f"{video_id}.txt"

        if out_txt.exists() and not force:
            logger.info(
                f"Transcript already exists, skipping transcription: {SystemUtils.display_path(out_txt)}"
            )
            return out_txt.read_text(encoding="utf-8")

        try:
            transcript = self._transcribe_whisper(str(audio_path), model=model)
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            out_txt.write_text(transcript, encoding="utf-8")
            logger.info(f"Saved raw transcript to {SystemUtils.display_path(out_txt)}")
            return transcript
        finally:
            if model is None:
                logger.info("Releasing transcription memory...")
                gc.collect()

