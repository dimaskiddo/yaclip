from __future__ import annotations

from loguru import logger

from src.ai.api_client import make_openai_client, retry_api_call
from src.ai.prompts import (
    build_batch_system_prompt,
    build_batch_user_prompt,
    build_single_user_prompt,
    get_system_prompt,
)
from src.core.config import load_config
from src.core.exceptions import AIProviderError
from src.core.utils import AIUtils


class CloudLLMProvider:
    """Provides LLM transcript analysis via Cloud AI (Google Gemini or OpenAI)."""

    def __init__(self) -> None:
        self.config = load_config()
        self.cloud_config = self.config.ai_pipeline.llm.cloud

        if not self.cloud_config.provider:
            raise ValueError("Cloud LLM mode requires 'provider' to be defined in config.yaml.")

        self.provider = self.cloud_config.provider.lower()
        self.api_key = self.cloud_config.api_key
        self.base_url = getattr(self.cloud_config, "base_url", None)
        self.model_name = self.cloud_config.model
        self.timeout = self.cloud_config.timeout

        if not AIUtils.has_credentials(self.api_key):
            raise ValueError(
                f"API key for LLM provider '{self.provider}' is missing or not configured."
            )

        if self.provider == "google":
            try:
                import google.generativeai as genai
            except ImportError as e:
                logger.error("google-generativeai package is not installed.")
                raise ImportError("google-generativeai package missing.") from e
            genai.configure(api_key=self.api_key)

    def _analyze_transcript_openai(
        self,
        transcript: str,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        target_clips: int = 5,
    ) -> list[dict]:
        """Uses OpenAI to analyze transcript and extract highlights."""
        client = make_openai_client(self.api_key, self.base_url, self.timeout)

        language = self.config.video_processing.subtitles.language
        system_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )
        user_prompt = build_single_user_prompt(transcript, target_clips)

        @retry_api_call(max_retries=3)
        def _call_openai_chat() -> str:
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                stream=True,
            )
            chunks: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            return "".join(chunks)

        logger.info("Sending transcript to OpenAI for clip selection...")
        try:
            response_text = _call_openai_chat()
            logger.debug(f"Raw OpenAI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(f"OpenAI found {len(parsed_clips)} clip(s) to extract.")
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read OpenAI response as JSON: {e}")
            raise AIProviderError(f"OpenAI clip selection failed: {e}") from e

    def _analyze_transcript_gemini(
        self,
        transcript: str,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        target_clips: int = 5,
    ) -> list[dict]:
        """Uses Gemini to analyze transcript and extract highlights."""
        import google.generativeai as genai

        logger.info("Sending transcript to Gemini for clip selection...")

        @retry_api_call(max_retries=3)
        def _generate(prompt: str) -> str:
            from google.api_core import retry as google_retry

            model = genai.GenerativeModel(model_name=self.model_name)
            stream = model.generate_content(
                prompt,
                stream=True,
                request_options={
                    "timeout": self.timeout,
                    "retry": google_retry.Retry(deadline=self.timeout),
                },
            )
            chunks: list[str] = []
            for chunk in stream:
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks)

        language = self.config.video_processing.subtitles.language
        system_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )
        prompt = (
            f"{system_prompt}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Analyze the transcript, identify up to {target_clips} clips, and return ONLY the requested JSON array."
        )

        try:
            response_text = _generate(prompt)
            logger.debug(f"Raw Gemini response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(f"Gemini found {len(parsed_clips)} clip(s) to extract.")
            return parsed_clips
        except Exception as e:
            logger.error(f"Gemini clip selection failed: {e}")
            raise AIProviderError(f"Gemini clip selection failed: {e}") from e

    def _analyze_batch_openai(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
    ) -> list[dict]:
        """Uses OpenAI to compare and select the best clips from candidates in a single batched call."""
        client = make_openai_client(self.api_key, self.base_url, self.timeout)

        language = self.config.video_processing.subtitles.language
        base_sys_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )

        system_prompt = build_batch_system_prompt(target_clips)

        user_prompt = build_batch_user_prompt(candidates_text, target_clips, base_sys_prompt)

        @retry_api_call(max_retries=3)
        def _call_openai_chat() -> str:
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                stream=True,
            )
            chunks: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
            return "".join(chunks)

        logger.info("Sending all candidate clips to OpenAI for batch selection...")
        try:
            response_text = _call_openai_chat()
            logger.debug(f"Raw OpenAI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(f"OpenAI selected {len(parsed_clips)} clip(s) from the batch.")
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read OpenAI batch response as JSON: {e}")
            raise AIProviderError(f"OpenAI batch clip selection failed: {e}") from e

    def _analyze_batch_gemini(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
    ) -> list[dict]:
        """Uses Gemini to compare and select the best clips from candidates in a single batched call."""
        import google.generativeai as genai

        logger.info("Sending all candidate clips to Gemini for batch selection...")

        @retry_api_call(max_retries=3)
        def _generate(prompt: str) -> str:
            from google.api_core import retry as google_retry

            model = genai.GenerativeModel(model_name=self.model_name)
            stream = model.generate_content(
                prompt,
                stream=True,
                request_options={
                    "timeout": self.timeout,
                    "retry": google_retry.Retry(deadline=self.timeout),
                },
            )
            chunks: list[str] = []
            for chunk in stream:
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks)

        language = self.config.video_processing.subtitles.language
        base_sys_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )

        system_prompt = build_batch_system_prompt(target_clips)

        prompt = (
            f"{system_prompt}\n\n"
            f"{build_batch_user_prompt(candidates_text, target_clips, base_sys_prompt)}"
        )

        try:
            response_text = _generate(prompt)
            logger.debug(f"Raw Gemini response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(f"Gemini selected {len(parsed_clips)} clip(s) from the batch.")
            return parsed_clips
        except Exception as e:
            logger.error(f"Gemini batch clip selection failed: {e}")
            raise AIProviderError(f"Gemini batch clip selection failed: {e}") from e

    def analyze_transcript(
        self,
        transcript: str,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        target_clips: int = 5,
    ) -> list[dict]:
        """Analyzes transcript using configured cloud provider and extracts highlights."""
        if self.provider == "google":
            return self._analyze_transcript_gemini(
                transcript, content_type, target_duration, target_clips
            )
        elif self.provider == "openai" or self.base_url:
            return self._analyze_transcript_openai(
                transcript, content_type, target_duration, target_clips
            )
        else:
            raise NotImplementedError(f"Cloud provider '{self.provider}' is not supported yet.")

    def analyze_batch(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
    ) -> list[dict]:
        """Compares and selects clips from candidates using configured cloud provider."""
        if self.provider == "google":
            return self._analyze_batch_gemini(
                candidates_text, target_clips, content_type, target_duration
            )
        elif self.provider == "openai" or self.base_url:
            return self._analyze_batch_openai(
                candidates_text, target_clips, content_type, target_duration
            )
        else:
            raise NotImplementedError(f"Cloud provider '{self.provider}' is not supported yet.")
