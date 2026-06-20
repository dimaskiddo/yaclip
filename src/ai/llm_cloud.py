from __future__ import annotations

from loguru import logger

from src.ai.api_client import retry_api_call
from src.ai.prompts import get_system_prompt
from src.core.config import load_config
from src.core.exceptions import AIProviderError
from src.core.utils import AIUtils


class CloudLLMProvider:
    """Provides LLM transcript analysis via Cloud AI (Google Gemini or OpenAI)."""

    def __init__(self) -> None:
        self.config = load_config()
        self.cloud_config = self.config.ai_pipeline.llm.cloud

        if not self.cloud_config.provider:
            raise ValueError(
                "Cloud LLM mode requires 'provider' to be defined in config.yaml."
            )

        self.provider = self.cloud_config.provider.lower()
        self.api_key = self.cloud_config.api_key
        self.base_url = getattr(self.cloud_config, "base_url", None)
        self.model_name = self.cloud_config.model

        if not self.api_key or self.api_key == "your-api-key-here":
            raise ValueError(
                f"API key for LLM provider '{self.provider}' is missing or not configured."
            )

    def _analyze_transcript_openai(
        self, transcript: str, content_type: str | None = "PODCAST", target_duration: int = 60, target_clips: int = 5
    ) -> list[dict]:
        """Uses OpenAI to analyze transcript and extract highlights."""
        try:
            from openai import OpenAI
        except ImportError as e:
            logger.error("openai package is not installed.")
            raise ImportError("openai package missing.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        system_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )
        user_prompt = (
            f"Transcript:\n{transcript}\n\n"
            f"Please analyze and identify up to {target_clips} highlight clips, then return the JSON array."
        )


        @retry_api_call(max_retries=3)
        def _call_openai_chat() -> object:
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

        logger.info("Sending transcript to OpenAI for clip selection...")
        try:
            response = _call_openai_chat()
            response_text = response.choices[0].message.content
            logger.debug(f"Raw OpenAI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"OpenAI found {len(parsed_clips)} clip(s) to extract."
            )
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read OpenAI response as JSON: {e}")
            raise AIProviderError(f"OpenAI clip selection failed: {e}") from e

    def _analyze_transcript_gemini(
        self, transcript: str, content_type: str | None = "PODCAST", target_duration: int = 60, target_clips: int = 5
    ) -> list[dict]:
        """Uses Gemini to analyze transcript and extract highlights."""
        try:
            import google.generativeai as genai
        except ImportError as e:
            logger.error("google-generativeai package is not installed.")
            raise ImportError("google-generativeai package missing.") from e

        genai.configure(api_key=self.api_key)
        logger.info("Sending transcript to Gemini for clip selection...")


        @retry_api_call(max_retries=3)
        def _generate(prompt: str) -> object:
            model = genai.GenerativeModel(model_name=self.model_name)
            return model.generate_content(prompt)

        system_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )
        prompt = (
            f"{system_prompt}\n\n"
            f"Transcript:\n{transcript}\n\n"
            f"Analyze the transcript, identify up to {target_clips} clips, and return ONLY the requested JSON array."
        )

        try:
            response = _generate(prompt)
            response_text = response.text
            logger.debug(f"Raw Gemini response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"Gemini found {len(parsed_clips)} clip(s) to extract."
            )
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
        try:
            from openai import OpenAI
        except ImportError as e:
            logger.error("openai package is not installed.")
            raise ImportError("openai package missing.") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        base_sys_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )

        system_prompt = (
            "You are an expert social media content curator. You are given transcripts of several candidate segments from a video. "
            f"Your task is to compare these candidates and select the best {target_clips} segments to render as final clips."
        )

        user_prompt = (
            f"Here are the candidate segments:\n\n{candidates_text}\n\n"
            f"Select the best {target_clips} clips. Return JSON array with fields as specified below.\n"
            f"{base_sys_prompt}\n\n"
            "Format response ONLY as a valid JSON array, no markdown wrappers."
        )


        @retry_api_call(max_retries=3)
        def _call_openai_chat() -> object:
            return client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )

        logger.info("Sending all candidate clips to OpenAI for batch selection...")
        try:
            response = _call_openai_chat()
            response_text = response.choices[0].message.content
            logger.debug(f"Raw OpenAI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"OpenAI selected {len(parsed_clips)} clip(s) from the batch."
            )
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
        try:
            import google.generativeai as genai
        except ImportError as e:
            logger.error("google-generativeai package is not installed.")
            raise ImportError("google-generativeai package missing.") from e

        genai.configure(api_key=self.api_key)
        logger.info("Sending all candidate clips to Gemini for batch selection...")


        @retry_api_call(max_retries=3)
        def _generate(prompt: str) -> object:
            model = genai.GenerativeModel(model_name=self.model_name)
            return model.generate_content(prompt)

        base_sys_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )

        system_prompt = (
            "You are an expert social media content curator. You are given transcripts of several candidate segments from a video. "
            f"Your task is to compare these candidates and select the best {target_clips} segments to render as final clips."
        )

        prompt = (
            f"{system_prompt}\n\n"
            f"Here are the candidate segments:\n\n{candidates_text}\n\n"
            f"Select the best {target_clips} clips. Return JSON array with fields as specified below.\n"
            f"{base_sys_prompt}\n\n"
            "Format response ONLY as a valid JSON array, no markdown wrappers."
        )

        try:
            response = _generate(prompt)
            response_text = response.text
            logger.debug(f"Raw Gemini response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"Gemini selected {len(parsed_clips)} clip(s) from the batch."
            )
            return parsed_clips
        except Exception as e:
            logger.error(f"Gemini batch clip selection failed: {e}")
            raise AIProviderError(f"Gemini batch clip selection failed: {e}") from e

    def analyze_transcript(
        self, transcript: str, content_type: str | None = "PODCAST", target_duration: int = 60, target_clips: int = 5
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
            raise NotImplementedError(
                f"Cloud provider '{self.provider}' is not supported yet."
            )

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
            raise NotImplementedError(
                f"Cloud provider '{self.provider}' is not supported yet."
            )
