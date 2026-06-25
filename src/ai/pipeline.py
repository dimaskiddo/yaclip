from __future__ import annotations

import concurrent.futures
import gc
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.ai.api_client import retry_api_call
from src.ai.heatmap import HeatmapAnalyzer
from src.ai.llm_cloud import CloudLLMProvider
from src.ai.llm_local import LocalLLMProvider
from src.ai.prompts import (
    build_detection_evidence_block,
    build_language_instruction,
    get_system_prompt,
    strip_json_markdown,
)
from src.ai.stt_cloud import CloudSTTProvider
from src.ai.stt_local import (
    LocalSTTProvider,
    save_mediashare_cache,
    save_words_cache,
    segments_to_transcript,
)
from src.core.config import load_config
from src.core.constants import CANDIDATE_WINDOW_BUFFER, MIN_CLIP_SECONDS, ContentType
from src.core.utils import AIUtils, SystemUtils
from src.core.workspace import DATA_DIR, TMP_DIR, active_pipeline_event
from src.media.energy import AudioEnergyAnalyzer
from src.media.slicer import AudioSlicer


class AIPipeline:
    """Orchestrates AI tasks like STT and LLM reasoning by routing to the appropriate providers."""

    def __init__(self) -> None:
        self.config = load_config()

    def is_unified_gemini_possible(self) -> bool:
        """Returns True if both STT and LLM are configured to use cloud Google Gemini."""
        stt_cfg = self.config.ai_pipeline.stt
        llm_cfg = self.config.ai_pipeline.llm

        stt_provider = stt_cfg.provider.lower()
        llm_provider = llm_cfg.provider.lower()

        stt_cloud_google = (
            (stt_provider in ("cloud", "auto"))
            and stt_cfg.cloud.provider.lower() == "google"
            and bool(stt_cfg.cloud.api_key and stt_cfg.cloud.api_key != "your-api-key-here")
        )
        llm_cloud_google = (
            (llm_provider in ("cloud", "auto"))
            and llm_cfg.cloud.provider.lower() == "google"
            and bool(llm_cfg.cloud.api_key and llm_cfg.cloud.api_key != "your-api-key-here")
        )
        return stt_cloud_google and llm_cloud_google

    def run_unified_gemini(
        self,
        audio_path: str,
        content_type: str | None,
        target_duration: int,
        target_clips: int,
        detected_type: ContentType | None = None,
        detection_evidence: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        """Runs a single unified Gemini call that handles both STT and LLM analysis."""
        try:
            import google.generativeai as genai
        except ImportError as e:
            logger.error("google-generativeai package is not installed.")
            raise ImportError("google-generativeai package missing.") from e

        stt_cloud_cfg = self.config.ai_pipeline.stt.cloud
        llm_cloud_cfg = self.config.ai_pipeline.llm.cloud
        api_key = llm_cloud_cfg.api_key or stt_cloud_cfg.api_key
        model_name = llm_cloud_cfg.model or stt_cloud_cfg.model
        llm_timeout = llm_cloud_cfg.timeout

        genai.configure(api_key=api_key)
        logger.info("Starting cloud AI transcription and clip selection...")

        @retry_api_call(max_retries=3)
        def _upload() -> object:
            return genai.upload_file(path=audio_path)

        @retry_api_call(max_retries=3)
        def _generate_content(contents: object, system_prompt: str) -> str:
            from google.api_core import retry as google_retry

            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
            stream = model.generate_content(
                contents,
                stream=True,
                request_options={
                    "timeout": llm_timeout,
                    "retry": google_retry.Retry(deadline=llm_timeout),
                },
            )
            chunks: list[str] = []
            for chunk in stream:
                if chunk.text:
                    chunks.append(chunk.text)
            return "".join(chunks)

        video_id = Path(audio_path).stem
        out_txt = DATA_DIR / f"{video_id}.txt"

        uploaded_file = None
        try:
            uploaded_file = _upload()
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = genai.get_file(uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the uploaded file.")

            sub_language = self.config.video_processing.subtitles.language
            base_system_prompt = get_system_prompt(
                content_type=content_type,
                target_duration=target_duration,
                target_clips=target_clips,
                language=sub_language,
            )

            system_instruction = (
                "You are an expert social media content curator and high-accuracy transcription engine. "
                "You must perform two tasks in a single run: transcribing the audio file and extracting clips. "
                f"You MUST identify EXACTLY {target_clips} of the most engaging/viral segments — no more, no fewer."
            )

            # Prepend detection evidence when the algorithm was uncertain.
            evidence_block = ""
            if detected_type is None and detection_evidence:
                evidence_block = build_detection_evidence_block(detection_evidence)

            lang_instruction = build_language_instruction(sub_language)

            user_prompt = (
                f"{evidence_block}"
                f"Format your output ONLY as a JSON object with the following structure:\n"
                "{\n"
                '  "transcript": "Write the full timestamped transcript text here. Use format: [start_seconds - end_seconds] spoken text, one line per segment.",\n'
                '  "clips": [\n'
                "    {\n"
                '      "start_time": float,\n'
                '      "end_time": float,\n'
                '      "title": "catchy hook/bait title (max 50 chars)",\n'
                '      "caption": "short caption for social media (max 150 chars)",\n'
                '      "description": "longer description with hook + context + CTA (max 300 chars)",\n'
                '      "hashtags": "5-8 space-separated hashtags, e.g. #gaming #mlbb #shorts",\n'
                '      "content_type": "string (one of PODCAST, JUST_CHAT, GAMING_SOLO, GAMING_COLLAB)",\n'
                '      "reasoning": "one sentence explaining why this is engaging"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"{lang_instruction}\n\n"
                "Here are the specific requirements for clip extraction:\n"
                f"{base_system_prompt}"
            )

            response_text = _generate_content([uploaded_file, user_prompt], system_instruction)
            response_text = strip_json_markdown(response_text)

            data = json.loads(response_text)

            transcript = data.get("transcript", "")
            if transcript:
                out_txt.parent.mkdir(parents=True, exist_ok=True)
                out_txt.write_text(transcript, encoding="utf-8")
                logger.info(f"Saved combined transcript to {out_txt}")

            clips = data.get("clips", [])
            # Enforce target_clips: LLM may return more or fewer than requested
            if len(clips) > target_clips:
                logger.info(f"Reducing clip count from {len(clips)} to {target_clips}.")
                clips = clips[:target_clips]
            elif len(clips) < target_clips:
                logger.warning(
                    f"Unified Gemini returned {len(clips)} clips, expected {target_clips}. "
                    "Consider increasing candidate margin or check LLM response quality."
                )
            # Normalise content_type per clip (same pattern as the hybrid path)
            for cc in clips:
                if detected_type is not None:
                    cc["content_type"] = detected_type.value
                else:
                    resolved = self._normalise_base_content_type(cc.get("content_type"))
                    if resolved is not None:
                        cc["content_type"] = resolved
                    else:
                        cc.pop("content_type", None)

                    # Post-hoc validation with raw detection evidence.
                    if detection_evidence:
                        self._validate_clip_content_type(
                            cc,
                            detection_evidence,
                            clips.index(cc) + 1,
                            audio_path=audio_path,
                        )
            logger.info(f"Cloud AI found {len(clips)} clips from the video.")
            return clips

        except Exception as e:
            logger.error(f"Combined Gemini analysis failed: {e}")
            raise e
        finally:
            if uploaded_file:
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete file {uploaded_file.name}: {e}")

    def run_stt_transcription(
        self,
        audio_path: str,
        force: bool = False,
        whisper_model: object | None = None,
        cache_dir: str | Path | None = None,
    ) -> str:
        """Helper to run STT transcription based on config, automatically falling back if requested."""
        stt_cfg = self.config.ai_pipeline.stt
        provider = stt_cfg.provider.lower()

        stt_api_key = stt_cfg.cloud.api_key
        has_cloud_credentials = bool(stt_api_key and stt_api_key != "your-api-key-here")

        # Cloud STT path
        if provider == "cloud" or (provider == "auto" and has_cloud_credentials):
            try:
                cloud_provider = CloudSTTProvider()
                return cloud_provider.transcribe(audio_path, force=force)
            except Exception as e:
                if provider == "auto":
                    logger.warning(
                        f"Cloud transcription failed: {e}. Switching to local transcription."
                    )
                else:
                    raise e

        # Local STT path
        local_provider = LocalSTTProvider()
        return local_provider.transcribe(
            audio_path, force=force, model=whisper_model, cache_dir=cache_dir
        )

    def run_llm_analysis(
        self,
        transcript: str,
        content_type: str | None,
        target_duration: int,
        target_clips: int = 5,
        llama_model: object | None = None,
    ) -> list[dict[str, Any]]:
        """Helper to run LLM transcript analysis based on config, automatically falling back if requested."""
        llm_cfg = self.config.ai_pipeline.llm
        provider = llm_cfg.provider.lower()

        llm_api_key = llm_cfg.cloud.api_key
        has_cloud_credentials = bool(llm_api_key and llm_api_key != "your-api-key-here")

        # Cloud LLM path
        if provider == "cloud" or (provider == "auto" and has_cloud_credentials):
            try:
                cloud_provider = CloudLLMProvider()
                return cloud_provider.analyze_transcript(
                    transcript,
                    content_type=content_type,
                    target_duration=target_duration,
                    target_clips=target_clips,
                )
            except Exception as e:
                if provider == "auto":
                    logger.warning(f"Cloud AI analysis failed: {e}. Switching to local AI.")
                else:
                    raise e

        # Local LLM path
        local_provider = LocalLLMProvider()
        return local_provider.analyze_transcript(
            transcript,
            content_type=content_type,
            target_duration=target_duration,
            target_clips=target_clips,
            llm=llama_model,
        )

    def run_batch_llm_analysis(
        self,
        candidates_text: str,
        target_clips: int,
        content_type: str,
        target_duration: int,
        llama_model: object | None = None,
    ) -> list[dict[str, Any]]:
        """Helper to run batched LLM candidate comparisons and selection based on config."""
        llm_cfg = self.config.ai_pipeline.llm
        provider = llm_cfg.provider.lower()

        llm_api_key = llm_cfg.cloud.api_key
        has_cloud_credentials = bool(llm_api_key and llm_api_key != "your-api-key-here")

        # Cloud LLM path
        if provider == "cloud" or (provider == "auto" and has_cloud_credentials):
            try:
                cloud_provider = CloudLLMProvider()
                return cloud_provider.analyze_batch(
                    candidates_text,
                    target_clips,
                    content_type=content_type,
                    target_duration=target_duration,
                )
            except Exception as e:
                if provider == "auto":
                    logger.warning(f"Cloud AI batch analysis failed: {e}. Switching to local AI.")
                else:
                    raise e

        # Local LLM path
        local_provider = LocalLLMProvider()
        return local_provider.analyze_batch(
            candidates_text,
            target_clips,
            content_type=content_type,
            target_duration=target_duration,
            llm=llama_model,
        )

    def _load_metadata_context(self, video_id: str) -> str:
        """Build a compact 'Video metadata' header (title/category/tags) for the LLM prompt."""
        meta_path = DATA_DIR / f"{video_id}_metadata.json"
        if not meta_path.exists():
            return ""
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read video metadata: {e}")
            return ""

        categories = ", ".join(meta.get("categories") or []) or "unknown"
        tags = ", ".join((meta.get("tags") or [])[:10])
        parts = [
            f"Video: {meta.get('title', '')} | Channel: {meta.get('channel', '')} | Category: {categories}",
        ]
        if tags:
            parts.append(f"Tags: {tags}")
        parts.append("")
        return "\n".join(parts) + "\n"

    @staticmethod
    def _normalise_base_content_type(value: object) -> str | None:
        """Validate an LLM-returned content_type against the 4 base types.

        Excludes the render-time-only DONATION_OVERLAY (that promotion happens in the renderer).
        Returns the canonical enum name, or None when the value is missing/invalid.
        """
        if not isinstance(value, str):
            return None
        name = value.strip().upper()
        allowed = {
            ContentType.PODCAST.value,
            ContentType.JUST_CHAT.value,
            ContentType.GAMING_SOLO.value,
            ContentType.GAMING_COLLAB.value,
        }
        return name if name in allowed else None

    @staticmethod
    def _validate_clip_content_type(
        clip: dict[str, Any],
        detection_evidence: dict[str, object],
        candidate_index: int,
        audio_path: str = "",
    ) -> None:
        """Post-hoc LLM output validation: algorithmic signals override LLM picks.

        Used when ``detected_type`` was None (uncertain) and the LLM decided
        per-clip.  Rules:
          1. Audio ~1 speaker + LLM returned GAMING_COLLAB → impossible.
             Force to GAMING_SOLO (if gameplay) or PODCAST.
          2. Audio ≥2 speakers + gameplay_present + webcam_count≥2 → force
             GAMING_COLLAB (LLM cannot downgrade).

        Args:
            clip: A single clip dict from the LLM response.
            detection_evidence: Structured evidence from the video-level detector.
            candidate_index: 1-based index for logging.
            audio_path: Path to the audio chunk (batch path) or full audio
                (Gemini path) for speaker counting.
        """
        resolved = clip.get("content_type")
        # Must be a valid string — already normalised by _normalise_base_content_type.
        if not isinstance(resolved, str) or resolved not in {
            ContentType.PODCAST.value,
            ContentType.JUST_CHAT.value,
            ContentType.GAMING_SOLO.value,
            ContentType.GAMING_COLLAB.value,
        }:
            return

        gameplay_present = bool(detection_evidence.get("gameplay_present", False))
        webcam_count = int(detection_evidence.get("webcam_count", 0))

        if not audio_path:
            return

        # Rule 1: 1 speaker cannot be COLLAB.
        if resolved == ContentType.GAMING_COLLAB.value:
            from src.media.energy import AudioEnergyAnalyzer

            speakers = AudioEnergyAnalyzer().estimate_speaker_count(audio_path)
            if speakers < 2:
                resolved = (
                    ContentType.GAMING_SOLO.value if gameplay_present else ContentType.PODCAST.value
                )
                clip["content_type"] = resolved
                logger.info(
                    f"Clip {candidate_index} corrected from collaboration to single layout "
                    f"(audio has 1 speaker)."
                )
                return

        # Rule 2: 2+ speakers + gameplay + 2+ webcams → force COLLAB.
        if gameplay_present and webcam_count >= 2 and resolved != ContentType.GAMING_COLLAB.value:
            clip["content_type"] = ContentType.GAMING_COLLAB.value
            logger.info(
                f"Clip {candidate_index} set to collaboration layout "
                f"(gameplay with {webcam_count} webcams)."
            )

    @staticmethod
    def _enforce_duration(
        start: float,
        end: float,
        win_start: float,
        win_end: float,
        min_d: float,
        max_d: float,
    ) -> tuple[float, float]:
        """Clamp a clip to the configured [min_d, max_d] length within the candidate window.

        A too-long pick is capped at ``max_d`` (keeping the hook start); a too-short pick is grown
        symmetrically up to ``min_d``, clamped inside ``[win_start, win_end]`` (the transcribed window,
        which is widened to ≥ max_d so the floor always fits). Returns the adjusted (start, end).
        """
        if end - start > max_d:
            end = start + max_d
        if end - start < min_d:
            grow = (min_d - (end - start)) / 2.0
            start = max(win_start, start - grow)
            end = start + min_d
            if end > win_end:
                end = win_end
                start = max(win_start, end - min_d)
        return start, end

    def process_audio(
        self,
        audio_path: str,
        video_path: str | None = None,
        force: bool = False,
        detected_type: ContentType | None = None,
        detection_evidence: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Main AI Orchestrator. Evaluates config to route between Cloud and Local logic.
        Protects execution using the active_pipeline_event safety guard.

        Args:
            audio_path: Path to the extracted audio track.
            video_path: Path to the source video. When provided, each candidate window is
                visually analysed (YOLOv8n) and a text descriptor is fed to the LLM so clip
                selection considers on-screen events (mediashare, action) — not audio alone.
            force: Re-run STT even if a cached transcript exists.
            detected_type: Video-level content type from ``ContentTypeDetector.detect_content_type``.
                When confident, it's used for all clips and threaded to the LLM prompt.
                When None (uncertain), the LLM decides per-clip.
            detection_evidence: Structured evidence from ``ContentTypeDetectionResult.evidence``.
                Injected into the LLM prompt when ``detected_type`` is None (uncertain)
                so the LLM has raw detection numbers and doesn't hallucinate.
        """
        active_pipeline_event.set()
        try:
            clip_cfg = self.config.clip_selection
            mode_val = clip_cfg.mode.lower()
            strategy = clip_cfg.auto_strategy.lower()

            strategy_label = {
                "hybrid": "hybrid",
                "heatmap": "replay data",
                "ai": "full AI",
            }.get(strategy, strategy)
            logger.info(f"Running in {strategy_label} mode to find the best clips.")

            if mode_val == "manual":
                logger.info("Manual mode: skipping AI clip selection.")
                return []

            # Resolve content type for the system prompt. The video-level detector's result
            # takes precedence over the config override (the detector already read config).
            # When detected_type is None (uncertain), the LLM decides per-clip.
            if detected_type is not None:
                content_type = detected_type.value
            else:
                content_type = self.config.video_processing.content_type_override
                if not content_type or content_type.lower() == "auto":
                    content_type = None  # uncertain — LLM decides

            target_duration = clip_cfg.default_clip_duration_seconds or 60
            target_clips = clip_cfg.default_clips or 5

            # Check for unified Gemini path (Only when STT and LLM are both cloud Google)
            if self.is_unified_gemini_possible():
                logger.info("Using cloud AI for both transcription and clip selection.")
                try:
                    return self.run_unified_gemini(
                        audio_path=audio_path,
                        content_type=content_type,
                        target_duration=target_duration,
                        target_clips=target_clips,
                        detected_type=detected_type,
                        detection_evidence=detection_evidence,
                    )
                except Exception as e:
                    logger.error(
                        f"Combined Gemini pipeline failed: {e}. Falling back to separate transcription and AI steps."
                    )

            # Decoupled Pipeline
            if strategy == "ai":
                logger.info("Transcribing the full audio for AI analysis.")
                transcript = self.run_stt_transcription(audio_path, force=force)
                return self.run_llm_analysis(
                    transcript,
                    content_type=content_type,
                    target_duration=target_duration,
                    target_clips=target_clips,
                )

            video_id = Path(audio_path).stem

            # Extract Spikes (Heatmap or Energy)
            logger.info("Scanning video for most engaging moments.")
            heatmap_analyzer = HeatmapAnalyzer()
            spikes = heatmap_analyzer.analyze_heatmap(video_id)
            if not spikes:
                logger.info("No replay data available, using audio energy instead.")
                energy_analyzer = AudioEnergyAnalyzer()
                spikes = energy_analyzer.analyze_audio_energy(audio_path)

            if not spikes:
                logger.warning(
                    "No engagement spikes found. Running full audio transcription and AI analysis (this may take longer on long videos)."
                )
                transcript = self.run_stt_transcription(audio_path, force=force)
                return self.run_llm_analysis(
                    transcript,
                    content_type=content_type,
                    target_duration=target_duration,
                    target_clips=target_clips,
                )

            if strategy == "heatmap":
                logger.info("Using replay data peaks directly without AI ranking.")
                return spikes

            # Slice and Refine Spikes (Hybrid strategy)
            logger.info(f"Found {len(spikes)} candidate sections across the video.")

            # Sort spikes by score descending (highest priority first)
            spikes.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

            # Additive margin: pool = target + margin (not target × N) so STT cost stays bounded as
            # the requested clip count grows (e.g. 15 clips + 2 = 17 candidates, not 30).
            candidate_margin = clip_cfg.candidate_margin
            pool_size = target_clips + candidate_margin
            top_candidates = spikes[:pool_size]

            logger.info(
                f"Selected top {len(top_candidates)} from {len(spikes)} candidate sections for AI analysis."
            )

            slicer = AudioSlicer()

            refined_clips = []
            TMP_DIR.mkdir(parents=True, exist_ok=True)

            # Step 1: Pre-slice top candidates concurrently
            sliced_chunks = []

            # Size each candidate window to the TARGET clip length (default + margin) + buffer around
            # the spike centre — wide enough for the LLM to pick a [default, default+margin] clip and
            # for the post-map enforcement to extend a short pick up to the floor.
            clip_ceiling = (
                clip_cfg.default_clip_duration_seconds + clip_cfg.clip_length_margin_seconds
            )
            half_window = clip_ceiling / 2.0 + CANDIDATE_WINDOW_BUFFER

            def _slice_chunk(i_spike: tuple[int, dict]) -> tuple | None:
                i, spike = i_spike
                centre = (float(spike["start_time"]) + float(spike["end_time"])) / 2.0
                s_start = max(0.0, centre - half_window)
                s_end = centre + half_window
                chunk_path = str(TMP_DIR / f"audio_{video_id}_{i + 1}.aac")
                success = slicer.slice_audio_chunk(audio_path, s_start, s_end, chunk_path)
                return (i, spike, chunk_path, s_start, s_end, success)

            logger.info("Slicing audio from candidate sections.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Preserve order by mapping
                results = list(executor.map(_slice_chunk, enumerate(top_candidates)))

            for i, spike, chunk_path, s_start, s_end, success in results:
                if not success:
                    logger.error(
                        f"Failed to cut audio clip for candidate {i + 1}. Using original time bounds instead."
                    )
                    refined_clips.append(spike)
                else:
                    sliced_chunks.append((i, spike, chunk_path, s_start, s_end))

            if not sliced_chunks:
                logger.warning(
                    "Could not cut any audio clips from candidates. Using original time positions instead."
                )
                return top_candidates

            # Step 2: Transcribe all sliced chunks (STT)
            transcribed_slices = []
            whisper_model = None

            # If running STT locally, load the WhisperModel ONCE and reuse it
            stt_provider = self.config.ai_pipeline.stt.provider.lower()
            stt_api_key = self.config.ai_pipeline.stt.cloud.api_key
            has_stt_credentials = bool(stt_api_key and stt_api_key != "your-api-key-here")
            use_local_stt = (stt_provider == "local") or (
                stt_provider == "auto" and not has_stt_credentials
            )

            if use_local_stt:
                try:
                    from faster_whisper import WhisperModel

                    whisper_model_size = self.config.ai_pipeline.stt.local.model_size
                    device_config = SystemUtils.resolve_device(
                        self.config.ai_pipeline.stt.local.device
                    )
                    c_type = "float32" if whisper_model_size in ("tiny", "base") else "int8"
                    logger.info(f"Loading speech-to-text model ({whisper_model_size})...")
                    whisper_model = WhisperModel(
                        whisper_model_size, device=device_config, compute_type=c_type
                    )
                except Exception as e:
                    logger.error(f"Failed to load local transcription model: {e}")

            # Content type is per-clip now and unknown at this point, so we keep word-level timings
            # whenever subtitles are enabled (the renderer still skips subtitles per-clip for the
            # cramped GAMING_COLLAB layout). When subtitles are off entirely, transcribe text-only.
            sub_cfg = self.config.video_processing.subtitles
            subs_will_render = sub_cfg.enabled

            # Local STT keeps word-level segments (for the subtitle word cache) only when subtitles
            # will actually render; otherwise it transcribes text-only for selection.
            local_stt = LocalSTTProvider() if (use_local_stt and subs_will_render) else None
            if use_local_stt and not subs_will_render:
                logger.info("Subtitles are turned off, transcribing for AI selection only.")

            def _offset_segments(segs: list[dict], off: float) -> list[dict]:
                """Shift relative segment/word times onto the absolute video timeline."""
                return [
                    {
                        "text": s["text"],
                        "start": s["start"] + off,
                        "end": s["end"] + off,
                        "words": [
                            {"word": w["word"], "start": w["start"] + off, "end": w["end"] + off}
                            for w in s.get("words", [])
                        ],
                    }
                    for s in segs
                ]

            def _transcribe_chunk(data: tuple) -> tuple | None:
                i, spike, chunk_path, s_start, s_end = data
                logger.info(f"Transcribing candidate section {i + 1}/{len(sliced_chunks)}.")
                try:
                    if local_stt is not None:
                        # One whisper pass: relative segments → LLM text + absolute words for cache.
                        segs = local_stt.transcribe_segments(
                            chunk_path, model=whisper_model, time_offset=0.0
                        )
                        transcript = segments_to_transcript(segs)
                        abs_segs = _offset_segments(segs, s_start)
                    else:
                        transcript = self.run_stt_transcription(
                            chunk_path,
                            force=force,
                            whisper_model=whisper_model,
                            cache_dir=TMP_DIR,  # Slice transcripts → TMP_DIR for auto-cleanup
                        )
                        abs_segs = None
                    return (i, spike, chunk_path, s_start, s_end, transcript, abs_segs, None)
                except Exception as e:
                    return (i, spike, chunk_path, s_start, s_end, None, None, e)

            if use_local_stt:
                logger.info("Transcribing candidate sections with local speech-to-text.")
                stt_results = [_transcribe_chunk(data) for data in sliced_chunks]
            else:
                logger.info("Transcribing candidate sections with cloud speech-to-text.")
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    stt_results = list(executor.map(_transcribe_chunk, sliced_chunks))

            word_cache: list[dict] = []
            for i, spike, chunk_path, s_start, s_end, transcript, abs_segs, err in stt_results:
                if err:
                    logger.error(
                        f"Transcription failed for candidate {i + 1}: {err}. Using original time position instead."
                    )
                    refined_clips.append(spike)
                else:
                    transcribed_slices.append((i, spike, chunk_path, s_start, s_end, transcript))
                    if abs_segs is not None:
                        word_cache.append({"start": s_start, "end": s_end, "segments": abs_segs})

            # Persist the candidate word-segments so the renderer can skip its own STT pass.
            if word_cache:
                save_words_cache(video_id, word_cache)

            # Cleanup Whisper model memory before loading LLM (RAM safety)
            if whisper_model is not None:
                logger.debug("Released local speech-to-text model memory.")
                del whisper_model
                gc.collect()

            # Step 2b: Visual analysis on the SAME candidate windows (audio + vision aligned).
            # Produces a text descriptor per candidate so a (possibly non-multimodal) LLM can
            # weigh on-screen events. YOLO is loaded once and freed before the LLM loads.
            visual_by_index: dict[int, str] = {}
            region_enabled = self.config.video_processing.region_detection.enabled
            if video_path and region_enabled and transcribed_slices:
                from src.vision.visual_analyzer import VisualAnalyzer

                analyzer = VisualAnalyzer()
                mediashare_cache_entries: list[dict] = []

                # When the video is a known gaming collaboration, lock the webcam pair
                # so per-candidate log summaries show both webcams consistently.
                collab_cams: list = []
                if detected_type is not None and detected_type == ContentType.GAMING_COLLAB:
                    collab_cams = analyzer.detect_facecams(Path(video_path))
                    if len(collab_cams) >= 2:
                        logger.debug(
                            f"Passing {len(collab_cams)} locked webcams "
                            f"to per-candidate scene analysis."
                        )

                try:
                    for idx, (_, _, _, s_start, s_end, _) in enumerate(transcribed_slices):
                        try:
                            res = analyzer.analyze_window(
                                Path(video_path),
                                s_start,
                                s_end,
                                facecam_boxes=collab_cams if len(collab_cams) >= 2 else None,
                            )
                            visual_by_index[idx] = res["descriptor"]
                            # Collect donation scan results so the renderer can reuse them instead
                            # of rescanning each clip window (avoids the duplicate dense frame scan).
                            if self.config.video_processing.preserve_donation_overlays:
                                mediashare_cache_entries.append(
                                    {
                                        "start": s_start,
                                        "end": s_end,
                                        "mediashare_present": res.get("mediashare_present", False),
                                        "mediashare_box": res.get("mediashare_box"),
                                    }
                                )
                        except Exception as ve:
                            logger.warning(f"Scene analysis failed for candidate {idx + 1}: {ve}")
                finally:
                    analyzer.release()

                # Persist donation scan results alongside the word-timings cache so the renderer
                # can look them up per-clip without re-scanning.
                if mediashare_cache_entries:
                    save_mediashare_cache(video_id, mediashare_cache_entries)

            # Step 3: Run LLM analysis on all transcripts (Batch LLM Call)
            if transcribed_slices:
                # Construct candidate transcripts text block
                prompt_parts = []
                speaker_analyzer = AudioEnergyAnalyzer()
                for idx, (_, _, chunk_path, s_start, s_end, transcript) in enumerate(
                    transcribed_slices
                ):
                    visual_line = visual_by_index.get(idx)
                    visual_block = f"Visual context: {visual_line}\n" if visual_line else ""
                    # Lightweight audio speaker-count hint (≈1 vs ≈2+ voices).
                    speakers = speaker_analyzer.estimate_speaker_count(chunk_path)
                    audio_line = (
                        "Audio: ~2 distinct speakers (leans podcast/collab)\n"
                        if speakers >= 2
                        else "Audio: ~1 speaker (leans solo/just-chat)\n"
                    )
                    prompt_parts.append(
                        f"[Candidate {idx + 1}] Window: [{s_start:.2f}s-{s_end:.2f}s]\n"
                        f"{visual_block}"
                        f"{audio_line}"
                        f"Transcript:\n{transcript}\n"
                    )
                # Prepend video/game metadata context so the LLM knows the show/game.
                meta_block = self._load_metadata_context(Path(audio_path).stem)
                # Prepend detection evidence when algorithm was uncertain so the LLM
                # has the raw numbers, not just the per-clip natural-language descriptors.
                evidence_block = ""
                if detected_type is None and detection_evidence:
                    evidence_block = build_detection_evidence_block(detection_evidence)
                    logger.debug(
                        f"Injecting detection evidence into LLM prompt: "
                        f"{' '.join(f'{k}={v}' for k, v in detection_evidence.items())}"
                    )
                candidates_text = evidence_block + meta_block + "\n".join(prompt_parts)

                llama_model = None
                llm_provider = self.config.ai_pipeline.llm.provider.lower()
                llm_api_key = self.config.ai_pipeline.llm.cloud.api_key
                has_llm_credentials = bool(llm_api_key and llm_api_key != "your-api-key-here")
                use_local_llm = (llm_provider == "local") or (
                    llm_provider == "auto" and not has_llm_credentials
                )

                if use_local_llm:
                    try:
                        from llama_cpp import Llama

                        llm_model_name = self.config.ai_pipeline.llm.local.model_name
                        resolved_path = AIUtils.resolve_llm_model_path(llm_model_name)
                        device_config = self.config.ai_pipeline.llm.local.device.lower()
                        n_gpu = self.config.ai_pipeline.llm.local.n_gpu_layers

                        logger.info("Loading local AI model for clip selection.")
                        llama_model = Llama(
                            model_path=resolved_path,
                            n_ctx=4096,
                            n_gpu_layers=n_gpu,
                            verbose=False,
                        )
                    except Exception as e:
                        logger.error(f"Failed to load local AI reasoning model: {e}")

                try:
                    # Perform the batched LLM call
                    batch_clips = self.run_batch_llm_analysis(
                        candidates_text=candidates_text,
                        target_clips=target_clips,
                        content_type=content_type,
                        target_duration=target_duration,
                        llama_model=llama_model,
                    )

                    # Map relative timestamps back to the original video timeline
                    for cc in batch_clips:
                        try:
                            cand_idx = int(cc.get("candidate_index", 1)) - 1
                            if 0 <= cand_idx < len(transcribed_slices):
                                _, spike, chunk_path, s_start, s_end, _ = transcribed_slices[
                                    cand_idx
                                ]

                                rel_start = float(cc.get("start_time", 0.0))
                                rel_end = float(cc.get("end_time", s_end - s_start))

                                # Calculate mapped start/end times
                                mapped_start = s_start + rel_start
                                mapped_end = s_start + rel_end

                                # Boundary constraints
                                mapped_start = max(s_start, mapped_start)
                                mapped_end = min(s_end, mapped_end)

                                # Drop only truly degenerate clips (inverted/zero-length); valid but
                                # short/long picks are extended/capped to [default, default+margin].
                                if mapped_end - mapped_start < MIN_CLIP_SECONDS:
                                    logger.warning(
                                        f"Skipping invalid clip from candidate {cand_idx + 1} "
                                        f"({mapped_start:.2f}s-{mapped_end:.2f}s): clip duration too short."
                                    )
                                    continue

                                # Enforce the clip length: extend a short pick up to the target floor
                                # (default), cap a long pick at the ceiling (default + margin).
                                mapped_start, mapped_end = self._enforce_duration(
                                    mapped_start,
                                    mapped_end,
                                    s_start,
                                    s_end,
                                    clip_cfg.default_clip_duration_seconds,
                                    clip_ceiling,
                                )

                                cc["start_time"] = mapped_start
                                cc["end_time"] = mapped_end

                                # When the video-level detector was confident, all clips get that
                                # type. When uncertain (detected_type=None), the LLM decided per-clip.
                                if detected_type is not None:
                                    cc["content_type"] = detected_type.value
                                else:
                                    # Validate the LLM's per-clip pick; drop invalid values.
                                    resolved = self._normalise_base_content_type(
                                        cc.get("content_type")
                                    )
                                    if resolved is not None:
                                        cc["content_type"] = resolved
                                    else:
                                        cc.pop("content_type", None)

                                    # Post-hoc validation with raw detection evidence.
                                    if detection_evidence:
                                        self._validate_clip_content_type(
                                            cc,
                                            detection_evidence,
                                            cand_idx + 1,
                                            audio_path=str(chunk_path),
                                        )

                                if "Heatmap" not in cc.get("reasoning", ""):
                                    cc["reasoning"] += " (Extracted via Heatmap/Energy Spike)."

                                refined_clips.append(cc)
                        except Exception as parse_e:
                            logger.error(f"Failed to parse mapped clip candidate: {parse_e}")

                except Exception as e:
                    logger.error(
                        f"AI clip selection failed: {e}. Falling back to original candidate positions."
                    )
                    for _, spike, _, _, _, _ in transcribed_slices:
                        refined_clips.append(spike)
                finally:
                    # Cleanup sliced audio files (skipped: rely on scheduler cleanup for debugging)

                    # Cleanup Llama model memory (RAM safety)
                    if llama_model is not None:
                        logger.debug("Released local AI model memory.")
                        del llama_model
                        gc.collect()

            # Sort chronological and filter duplicates
            refined_clips.sort(key=lambda x: float(x["start_time"]))

            final_clips = []
            last_end = -1.0
            for c in refined_clips:
                # Skip degenerate clips (inverted/zero-length) so the saved JSON stays valid.
                if float(c["end_time"]) - float(c["start_time"]) < MIN_CLIP_SECONDS:
                    continue
                if float(c["start_time"]) > last_end - 5:  # Allow 5s overlap
                    final_clips.append(c)
                    last_end = float(c["end_time"])

            # Enforce target_clips: dedup first, then clamp to requested count
            if len(final_clips) > target_clips:
                logger.info(f"Reducing clip count from {len(final_clips)} to {target_clips}.")
                final_clips = final_clips[:target_clips]
            elif len(final_clips) < target_clips:
                logger.warning(
                    f"AI returned {len(final_clips)} clips, expected {target_clips}. "
                    "Some candidates may have been dropped during deduplication or duration filtering. "
                    "Consider increasing candidate margin."
                )

            logger.info(f"AI finished selecting {len(final_clips)} clips, ready for rendering.")
            return final_clips

        finally:
            active_pipeline_event.clear()
