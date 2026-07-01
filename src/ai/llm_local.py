from __future__ import annotations

import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama

from loguru import logger

from src.ai.prompts import (
    build_batch_system_prompt,
    build_batch_user_prompt,
    build_single_user_prompt,
    get_system_prompt,
)
from src.core.config import load_config
from src.core.constants import LLM_LOCAL_MAX_TOKENS_BATCH, LLM_LOCAL_MAX_TOKENS_SINGLE
from src.core.exceptions import AIProviderError
from src.core.utils import AIUtils


class LocalLLMProvider:
    """Provides LLM transcript analysis via local Llama model."""

    def __init__(self) -> None:
        self.config = load_config()
        self.local_cfg = self.config.ai_pipeline.llm.local
        self.model_path = self.local_cfg.model_name
        self.device_config = self.local_cfg.device.lower()
        self.n_gpu = self.local_cfg.n_gpu_layers

        if not self.model_path:
            raise ValueError(
                "Local LLM mode requires 'model_name' to be defined under ai_pipeline.llm.local in config.yaml."
            )

        self.resolved_path = AIUtils.resolve_llm_model_path(self.model_path)

    def _analyze_transcript_llama(
        self,
        transcript: str,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        target_clips: int = 5,
        llm: Llama | None = None,
    ) -> list[dict]:
        """Uses local Llama model to analyze transcript and extract highlights."""
        language = self.config.video_processing.subtitles.language
        system_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )
        user_prompt = build_single_user_prompt(transcript, target_clips)

        try:
            if llm is None:
                logger.info("Loading local AI reasoning model...")
                llm = AIUtils.load_llama(self.resolved_path, self.n_gpu)
            else:
                logger.info("Reusing local AI reasoning model...")

            logger.info("Analysing transcript locally to find the best clip moments...")
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=LLM_LOCAL_MAX_TOKENS_SINGLE,
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            response_text = output["choices"][0]["message"]["content"]
            logger.debug(f"Raw local AI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(f"Local AI found {len(parsed_clips)} clip(s) to extract.")
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read local AI response as JSON: {e}")
            raise AIProviderError(f"Local AI reasoning failed: {e}") from e

    def _analyze_batch_llama(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> list[dict]:
        """Uses local Llama model to compare and select the best clips from candidates in a single batched call."""
        language = self.config.video_processing.subtitles.language
        base_sys_prompt = get_system_prompt(
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            language=language,
        )

        system_prompt = build_batch_system_prompt(target_clips)

        user_prompt = build_batch_user_prompt(
            candidates_text, target_clips, base_sys_prompt
        )

        try:
            if llm is None:
                logger.info(
                    "Loading local AI reasoning model for batch clip selection..."
                )
                llm = AIUtils.load_llama(self.resolved_path, self.n_gpu)
            else:
                logger.info("Reusing local AI reasoning model...")

            logger.info(
                "Analysing all candidate clips locally to select the best ones..."
            )
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=LLM_LOCAL_MAX_TOKENS_BATCH,
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            response_text = output["choices"][0]["message"]["content"]
            logger.debug(f"Raw local AI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"Local AI selected {len(parsed_clips)} clip(s) from the batch."
            )
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read local AI batch response as JSON: {e}")
            raise AIProviderError(f"Local AI batch reasoning failed: {e}") from e

    def analyze_transcript(
        self,
        transcript: str,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        target_clips: int = 5,
        llm: Llama | None = None,
    ) -> list[dict]:
        """Analyzes transcript using local Llama model to extract highlights."""
        try:
            return self._analyze_transcript_llama(
                transcript, content_type, target_duration, target_clips, llm=llm
            )
        finally:
            if llm is None:
                logger.info("Releasing local AI reasoning model from memory...")
                gc.collect()

    def analyze_batch(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str | None = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> list[dict]:
        """Compares and selects clips from candidates using local Llama model."""
        try:
            return self._analyze_batch_llama(
                candidates_text, target_clips, content_type, target_duration, llm=llm
            )
        finally:
            if llm is None:
                logger.info("Releasing local AI reasoning model from memory...")
                gc.collect()
