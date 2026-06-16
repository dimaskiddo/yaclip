from __future__ import annotations

import time

from pathlib import Path
from loguru import logger

from src.ai.prompts import get_language_prompt
from src.core.config import load_config
from src.core.exceptions import AIProviderError
from src.core.utils import SystemUtils
from src.core.workspace import SUBTITLES_DIR


class CloudSTTProvider:
    """Provides STT transcription via Cloud AI (Google Gemini or OpenAI Whisper)."""

    def __init__(self) -> None:
        self.config = load_config()
        self.cloud_config = self.config.ai_pipeline.stt.cloud

        if not self.cloud_config.provider:
            raise ValueError(
                "Cloud STT mode requires 'provider' to be defined in config.yaml."
            )

        self.provider = self.cloud_config.provider.lower()
        self.api_key = self.cloud_config.api_key
        self.base_url = getattr(self.cloud_config, "base_url", None)
        self.model_name = self.cloud_config.model

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError(
                f"API key for STT provider '{self.provider}' is missing or not configured."
            )

    def _transcribe_openai(self, audio_path: str) -> str:
        """Transcribes audio using OpenAI API and returns the transcript text."""
        try:
            from openai import OpenAI
        except ImportError as e:
            logger.error("openai package is not installed.")
            raise ImportError("openai package missing.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("Starting cloud transcription with OpenAI...")

        language = self.config.video_processing.subtitles.language

        from src.ai.api_client import retry_api_call

        @retry_api_call(max_retries=3)
        def _call_openai_stt(audio_file):
            kwargs = {
                "model": self.model_name,
                "file": audio_file,
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
                "temperature": 0.0,
            }

            stt_context = self.config.video_processing.subtitles.stt_context.strip()
            lang_primer = None
            if language and language.lower() != "auto":
                kwargs["language"] = language
                logger.info(f"Forcing cloud transcription language: {language}")
                lang_primer = get_language_prompt(language)
            # prompt = native-language primer + the user vocab hint (names/game/terms).
            primer = " ".join(p for p in (lang_primer, stt_context) if p)
            if primer:
                kwargs["prompt"] = primer

            return client.audio.transcriptions.create(**kwargs)

        try:
            with open(audio_path, "rb") as audio_file:
                transcript_res = _call_openai_stt(audio_file)

            transcript_lines = []
            segments = getattr(transcript_res, "segments", [])
            for segment in segments:
                start = segment["start"] if isinstance(segment, dict) else segment.start
                end = segment["end"] if isinstance(segment, dict) else segment.end
                text = segment["text"] if isinstance(segment, dict) else segment.text
                line = f"[{start:.2f} - {end:.2f}] {text.strip()}"
                transcript_lines.append(line)

            transcript = "\n".join(transcript_lines)
            logger.info("Cloud transcription complete.")
            return transcript
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {e}")
            raise AIProviderError(f"OpenAI STT failed: {e}") from e

    def _transcribe_gemini(self, audio_path: str) -> str:
        """Transcribes audio using Gemini and returns the transcript text."""
        try:
            import google.generativeai as genai
        except ImportError as e:
            logger.error("google-generativeai package is not installed.")
            raise ImportError("google-generativeai package missing.") from e

        genai.configure(api_key=self.api_key)
        logger.info("Starting cloud transcription with Gemini...")

        from src.ai.api_client import retry_api_call

        @retry_api_call(max_retries=3)
        def _upload():
            return genai.upload_file(path=audio_path)

        @retry_api_call(max_retries=3)
        def _generate(uploaded_file, prompt):
            model = genai.GenerativeModel(model_name=self.model_name)
            return model.generate_content([uploaded_file, prompt])

        uploaded_file = None
        try:
            uploaded_file = _upload()
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the uploaded file.")

            language = self.config.video_processing.subtitles.language
            lang_instruction = ""
            if language and language.lower() != "auto":
                lang_instruction = f" The primary language of the audio is '{language}'."
                # Same native-language primer used by the OpenAI/local paths, for consistency.
                lang_prompt = get_language_prompt(language)
                if lang_prompt:
                    lang_instruction += f" {lang_prompt}"
            # User vocab hint (names/game/terms) — same as OpenAI/local stt_context.
            stt_context = self.config.video_processing.subtitles.stt_context.strip()
            if stt_context:
                lang_instruction += f" Likely terms: {stt_context}."

            prompt = (
                f"Transcribe the following audio file.{lang_instruction} "
                "Output the transcript with timestamps in the format: [start_seconds - end_seconds] spoken text. "
                "Do not include any other markdown formatting outside the timestamps and spoken text."
            )

            response = _generate(uploaded_file, prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini transcription failed: {e}")
            raise AIProviderError(f"Gemini STT failed: {e}") from e
        finally:
            if uploaded_file:
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete file {uploaded_file.name}: {e}")

    def transcribe(self, audio_path: str | Path, force: bool = False) -> str:
        """Transcribes audio using the configured cloud provider and caches the transcript."""
        audio_path = Path(audio_path)
        video_id = audio_path.stem
        out_txt = SUBTITLES_DIR / f"{video_id}.txt"

        if out_txt.exists() and not force:
            logger.info(
                f"Transcript already exists, skipping cloud transcription: {SystemUtils.display_path(out_txt)}"
            )
            return out_txt.read_text(encoding="utf-8")

        if self.provider == "google":
            transcript = self._transcribe_gemini(str(audio_path))
        elif self.provider == "openai" or self.base_url:
            transcript = self._transcribe_openai(str(audio_path))
        else:
            raise NotImplementedError(
                f"Cloud STT provider '{self.provider}' is not supported."
            )

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(transcript, encoding="utf-8")
        logger.info(f"Saved raw transcript to {SystemUtils.display_path(out_txt)}")
        return transcript
