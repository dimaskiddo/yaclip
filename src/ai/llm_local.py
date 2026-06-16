from __future__ import annotations

import gc

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from llama_cpp import Llama

from loguru import logger

from src.ai.prompts import get_system_prompt
from src.core.utils import AIUtils
from src.core.config import load_config
from src.core.exceptions import AIProviderError


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
        content_type: str = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> List[Dict]:
        """Uses local Llama model to analyze transcript and extract highlights."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            logger.error("llama-cpp-python is not installed.")
            raise ImportError("llama-cpp-python package missing.") from e

        system_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )
        user_prompt = (
            f"Transcript:\n{transcript}\n\nPlease analyze and return the JSON array."
        )

        try:
            if llm is None:
                logger.info("Loading local AI reasoning model...")
                llm = Llama(
                    model_path=self.resolved_path,
                    n_ctx=4096,
                    n_gpu_layers=self.n_gpu,
                    verbose=False,
                )
            else:
                logger.info("Reusing local AI reasoning model...")

            logger.info("Analysing transcript locally to find the best clip moments...")
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
            )

            response_text = output["choices"][0]["message"]["content"]
            logger.debug(f"Raw local AI response: {response_text}")

            parsed_clips = AIUtils.parse_json_array(response_text)
            logger.info(
                f"Local AI found {len(parsed_clips)} clip(s) to extract."
            )
            return parsed_clips
        except Exception as e:
            logger.error(f"Failed to read local AI response as JSON: {e}")
            raise AIProviderError(f"Local AI reasoning failed: {e}") from e

    def _analyze_batch_llama(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> List[Dict]:
        """Uses local Llama model to compare and select the best clips from candidates in a single batched call."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            logger.error("llama-cpp-python is not installed.")
            raise ImportError("llama-cpp-python package missing.") from e

        from src.ai.prompts import get_system_prompt

        base_sys_prompt = get_system_prompt(
            content_type=content_type, target_duration=target_duration
        )

        system_prompt = (
            "You are an expert social media content curator. You are given transcripts of several candidate segments from a video. "
            f"Your task is to compare these candidates and select the best {target_clips} segments to render as final clips."
        )

        user_prompt = (
            f"Here are the candidate segments:\n\n{candidates_text}\n\n"
            f"Please select the best {target_clips} clips. For each selected clip, output an object in a JSON array. "
            "Each object MUST contain:\n"
            "- `candidate_index`: (integer) the index of the candidate this clip is chosen from (1-based index)\n"
            "- `start_time`: (float) the start time relative to the candidate's transcript (where 0.0 is the start of that candidate's transcript)\n"
            "- `end_time`: (float) the end time relative to the candidate's transcript\n"
            "- `title`: (string, max 50 chars) a catchy title for the clip\n"
            "- `reasoning`: (string) one sentence explaining why this clip is selected and how it compares to other candidates\n\n"
            "Here are the specific requirements for clip extraction:\n"
            f"{base_sys_prompt}\n\n"
            "Format the response ONLY as a valid JSON array, with no other text or markdown wrappers."
        )

        try:
            if llm is None:
                logger.info("Loading local AI reasoning model for batch clip selection...")
                llm = Llama(
                    model_path=self.resolved_path,
                    n_ctx=4096,
                    n_gpu_layers=self.n_gpu,
                    verbose=False,
                )
            else:
                logger.info("Reusing local AI reasoning model...")

            logger.info("Analysing all candidate clips locally to select the best ones...")
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2048,
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
        content_type: str = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> List[Dict]:
        """Analyzes transcript using local Llama model to extract highlights."""
        try:
            return self._analyze_transcript_llama(
                transcript, content_type, target_duration, llm=llm
            )
        finally:
            if llm is None:
                logger.info("Releasing local AI reasoning model from memory...")
                gc.collect()

    def analyze_batch(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str = "PODCAST",
        target_duration: int = 60,
        llm: Llama | None = None,
    ) -> List[Dict]:
        """Compares and selects clips from candidates using local Llama model."""
        try:
            return self._analyze_batch_llama(
                candidates_text, target_clips, content_type, target_duration, llm=llm
            )
        finally:
            if llm is None:
                logger.info("Releasing local AI reasoning model from memory...")
                gc.collect()
