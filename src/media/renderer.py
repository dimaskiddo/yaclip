from __future__ import annotations

import gc
import subprocess
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from src.ai.stt_local import LocalSTTProvider, load_mediashare_cache, load_words_cache
from src.core.config import load_config
from src.core.constants import (
    GPU_ENCODER_FAILURE_SIGNS,
    MIN_CLIP_SECONDS,
    STACK2_PANEL_ASPECT,
    STACK3_PANEL_ASPECT,
    ContentType,
)
from src.core.utils import SystemUtils, clip_stem
from src.core.workspace import SUBTITLES_DIR, TMP_DIR, audio_output_path
from src.media.audio import AudioExtractor
from src.media.ffmpeg_builder import FFmpegCommandBuilder
from src.media.slicer import AudioSlicer
from src.media.subtitles import SubtitleGenerator
from src.vision.content_type_detector import ContentTypeDetector
from src.vision.face_tracker import FaceTracker
from src.vision.frame_utils import probe_video_dims
from src.vision.layout_builder import LayoutBuilder
from src.vision.overlay_detector import OverlayDetector
from src.vision.visual_analyzer import VisualAnalyzer


def _sanitize_title(title: str) -> str:
    """Strip non-alphanumeric chars (except ``-``/``_``), collapse leading/trailing underscores."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title).strip(
        "_"
    )


def _titlecase_filename(title: str) -> str:
    """Convert a clip title to Title-Case-With-Hyphens for use as a filename segment."""
    return title.strip().replace("_", " ").title().replace(" ", "-")


class ClipRenderer:
    """Orchestrates region analysis, subtitle transcription, layout, and FFmpeg rendering.

    Runs as three memory-safe passes so heavy models never coexist in RAM:
      1. Regions   — YOLOv8n VisualAnalyzer per clip, then released.
      2. Subtitles — faster-whisper word-level transcription per clip, then released.
      3. Render    — layout + .ass + FFmpeg encode per clip.
    """

    def __init__(self) -> None:
        self.config = load_config()
        self.detector = ContentTypeDetector()
        self.tracker = FaceTracker()
        self.overlay_detector = OverlayDetector()
        self.layout_builder = LayoutBuilder()
        self.subtitle_gen = SubtitleGenerator()
        self.ffmpeg_builder = FFmpegCommandBuilder()
        self.slicer = AudioSlicer()

    def render_clips(
        self,
        video_path: Path,
        clip_proposals: list[dict],
        content_type: ContentType | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[Path]:
        """Render multiple clip proposals to vertical 9:16 MP4 files.

        Args:
            video_path: Path to the raw downloaded video file.
            clip_proposals: [{"start_time", "end_time", "title", optional "content_type"}].
            content_type: Video-level content type from the detector. When set, used as the
                type for all clips (except those promoted to DONATION_OVERLAY or overridden).
            progress_callback: Optional (fraction, description) callback fired at pass boundaries
                so a UI can display a moving progress bar.

        Returns:
            Paths to the successfully rendered clips.
        """
        output_dir = Path(self.config.video_processing.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        # Defensively drop degenerate clips (inverted/zero-length) before any slice/render so a
        # hand-edited or manual JSON cannot crash the subtitle or FFmpeg pass.
        clip_proposals = [
            c
            for c in clip_proposals
            if float(c.get("end_time", 0.0)) - float(c.get("start_time", 0.0))
            >= MIN_CLIP_SECONDS
        ]
        if not clip_proposals:
            logger.warning(
                "No valid clips to render after dropping degenerate proposals."
            )
            return []

        # Resolve the per-clip content type. Precedence: config override > explicit per-clip type
        # (from LLM / review edit) > video-level detected_type (from cli.py) > visual fallback > PODCAST.
        override = self._content_type_override()
        base_type = (
            override or content_type
        )  # video-level type from the detector (may be None)
        clip_types: list[ContentType | None] = [
            self._initial_clip_type(c, base_type) for c in clip_proposals
        ]

        # Pass 1 — visual region analysis (YOLO loaded once, released after).
        if progress_callback:
            progress_callback(0.1, "Analyzing video regions...")
        analyses = self._analyze_regions(video_path, clip_proposals, clip_types)

        # For manual mode (no LLM call) — any clip still without a type gets a visual fallback.
        clip_types = self._finalize_clip_types(
            video_path, clip_proposals, clip_types, analyses
        )

        # Promote any clip whose window contains a mediashare/donation popup to DONATION_OVERLAY
        # (mediashare_present is only known after Pass 1). Done before Pass 2 so a promoted clip is
        # not caught by the GAMING_COLLAB subtitle skip.
        clip_types = self._apply_donation_override(clip_proposals, clip_types, analyses)

        # Pass 2 — per-clip subtitle segments (whisper loaded once, released after).
        if progress_callback:
            progress_callback(0.4, "Transcribing audio...")
        segments_per_clip = self._transcribe_subtitles(
            video_path, clip_proposals, clip_types
        )

        # Pass 3 — layout + render.
        if progress_callback:
            progress_callback(0.7, "Rendering clips...")
        return self._render_pass(
            video_path,
            clip_proposals,
            clip_types,
            analyses,
            segments_per_clip,
            output_dir,
        )

    # ---------------------------------------------------------------- FFmpeg run

    def _run_render_with_fallback(
        self, build_cmd: Callable[[str], list[str]], encoder: str
    ) -> None:
        """Run the FFmpeg render; if a GPU encoder fails at runtime, rebuild with libx264 and retry.

        ``build_cmd`` is a callable ``(encoder_name) -> cmd_list`` so the command can be rebuilt
        cleanly for the CPU fallback (rather than scrubbing the GPU flags out of the array).
        """
        cmd = build_cmd(encoder)
        logger.debug(f"FFmpeg render command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            is_cpu = encoder.lower() in ("cpu", "libx264")
            is_gpu_failure = any(s in stderr.lower() for s in GPU_ENCODER_FAILURE_SIGNS)
            if is_cpu or not is_gpu_failure:
                raise  # already CPU, or a genuine (non-hardware) FFmpeg error → handled by caller
            logger.warning(
                f"GPU encoder '{encoder}' failed — retrying this clip with CPU (libx264)."
            )
            cpu_cmd = build_cmd("cpu")
            subprocess.run(cpu_cmd, capture_output=True, text=True, check=True)
            logger.info("CPU fallback render succeeded.")

    # ------------------------------------------------------------- Content Type

    def _content_type_override(self) -> ContentType | None:
        """The forced content type from ``content_type_override``, or None when set to 'auto'."""
        override = self.config.video_processing.content_type_override
        if override == "auto":
            return None
        try:
            return ContentType(override.upper())
        except ValueError:
            logger.warning(
                f"Invalid content_type_override '{override}' — ignoring, will auto-detect."
            )
            return None

    def _initial_clip_type(
        self, clip: dict, override: ContentType | None
    ) -> ContentType | None:
        """Override wins; else an explicit/LLM-assigned per-clip type; else None (resolved later)."""
        if override is not None:
            return override
        if "content_type" in clip:
            try:
                return ContentType(str(clip["content_type"]).upper())
            except ValueError:
                pass
        return None

    def _finalize_clip_types(
        self,
        video_path: Path,
        clips: list[dict],
        clip_types: list[ContentType | None],
        analyses: list[dict],
    ) -> list[ContentType]:
        """Resolve any still-unknown clip type. Only used for manual mode (no LLM call).

        When the video-level detector returned None (uncertain) AND the LLM didn't assign types
        (manual mode), fall back to the per-clip visual classifier. Inconclusive windows default
        to PODCAST. A clip that turns out GAMING_COLLAB but was region-analyzed as non-collab is
        re-analyzed with the locked facecam pair.
        """
        gaming_hint = self.detector.metadata_gaming_hint(video_path)
        out: list[ContentType] = []
        collab_pending: list[int] = []
        for idx, ctype in enumerate(clip_types):
            if ctype is None:
                ctype = self.detector.classify_from_analysis(analyses[idx], gaming_hint)
                if ctype is None:
                    ctype = ContentType.PODCAST
                    logger.info(
                        f"Clip {idx + 1} video type undetermined, using standard framing."
                    )
                else:
                    from src.core.constants import CONTENT_TYPE_HUMAN_NAMES

                    human_name = CONTENT_TYPE_HUMAN_NAMES.get(
                        ctype.value, ctype.value.lower()
                    )
                    logger.info(f"Clip {idx + 1} video type detected as {human_name}.")
                if (
                    ctype == ContentType.GAMING_COLLAB
                    and analyses[idx].get("collab_box") is None
                ):
                    collab_pending.append(idx)
            out.append(ctype)
        if collab_pending:
            self._reanalyze_collab(video_path, clips, analyses, collab_pending)
        return out

    def _reanalyze_collab(
        self,
        video_path: Path,
        clips: list[dict],
        analyses: list[dict],
        indices: list[int],
    ) -> None:
        """Re-run region analysis for late-classified COLLAB clips with the locked facecam pair."""
        analyzer = VisualAnalyzer()
        try:
            collab_cams = analyzer.detect_facecams(video_path)
            if len(collab_cams) < 2:
                logger.info(
                    "Collaboration layout needed but only 1 webcam found, using single webcam framing."
                )
                return
            stable = analyzer.detect_stable_facecam(video_path)
            for idx in indices:
                analyses[idx] = analyzer.analyze_window(
                    video_path,
                    clips[idx]["start_time"],
                    clips[idx]["end_time"],
                    facecam_override=stable,
                    facecam_boxes=collab_cams,
                    gameplay_aspect=STACK3_PANEL_ASPECT,
                )
        finally:
            analyzer.release()

    def _apply_donation_override(
        self,
        clips: list[dict],
        clip_types: list[ContentType],
        analyses: list[dict],
    ) -> list[ContentType]:
        """Promote clips with a detected mediashare/donation popup to DONATION_OVERLAY.

        Gated on ``preserve_donation_overlays`` (off → donations are shown nowhere). An explicit
        per-clip ``content_type`` on the proposal (user/LLM review edit) is respected and never
        overridden.
        """
        if not self.config.video_processing.preserve_donation_overlays:
            return clip_types

        # Build exclusion set from config (skip unknown names defensively).
        exclude_types: set[ContentType] = set()
        for name in self.config.video_processing.donation_overlay_exclude_types:
            try:
                exclude_types.add(ContentType(name))
            except ValueError:
                logger.warning(
                    f"donation_overlay_exclude_types: unknown content type '{name}' — skipped."
                )

        out = list(clip_types)
        for idx, analysis in enumerate(analyses):
            if "content_type" in clips[idx]:
                continue
            # Skip types excluded from donation routing (configurable; default: PODCAST, GAMING_COLLAB).
            if out[idx] in exclude_types:
                continue
            if analysis.get("mediashare_present"):
                out[idx] = ContentType.DONATION_OVERLAY
                logger.info(
                    f"Clip {idx + 1} donation popup detected, using donation overlay layout."
                )
        return out

    # ----------------------------------------------------------------- Pass 1

    def _analyze_regions(
        self, video_path: Path, clips: list[dict], clip_types: list[ContentType]
    ) -> list[dict]:
        """Run VisualAnalyzer on each clip window; YOLO loaded once and freed after."""

        # Use the static centred crop (_motion_region) by default track_gameplay=False
        follow = self.config.video_processing.region_detection.gameplay_follow_motion
        analyzer = VisualAnalyzer()
        analyses: list[dict] = []

        # Load the selection-phase donation scan cache (saved by pipeline step 2b) so we can skip
        # the expensive dense frame rescan per clip and reuse the already-computed results.
        ms_cache = load_mediashare_cache(video_path.stem)

        try:
            # Lock one stable facecam box for the whole video → identical top framing per clip.
            stable_facecam = analyzer.detect_stable_facecam(video_path)
            # For collab, lock the reliable 2-cam PAIR once (primary + collaborator). Both cams are
            # then excluded from the gameplay centre so neither bleeds in (no "double facecam").
            collab_cams = (
                analyzer.detect_facecams(video_path)
                if any(t == ContentType.GAMING_COLLAB for t in clip_types)
                else []
            )
            # Probe source video dimensions for debug-position logging.
            src_w, src_h, _fps, _frames = probe_video_dims(video_path)

            # Log explicit panel assignment for collaboration layout.
            if len(collab_cams) >= 2:
                pri_pos = analyzer._where(collab_cams[0], src_w, src_h)
                sec_pos = analyzer._where(collab_cams[1], src_w, src_h)
                logger.info(f"Primary webcam at {pri_pos}, assigned for Top Panel.")
                logger.info(
                    f"Secondary webcam at {sec_pos}, assigned for Bottom Panel."
                )
            for idx, clip in enumerate(clips):
                is_collab = clip_types[idx] == ContentType.GAMING_COLLAB
                # Collab centre uses the wider 3-stack panel aspect; everyone else the 2-stack one.
                aspect = STACK3_PANEL_ASPECT if is_collab else STACK2_PANEL_ASPECT

                # Reuse the selection-phase donation scan for this clip window when available.
                # Falls back to a fresh scan on cache miss (manual mode / no prior pipeline run).
                mediashare_cached = self._match_cached_mediashare(
                    ms_cache, clip["start_time"], clip["end_time"]
                )
                if mediashare_cached is not None:
                    found = "overlay found" if mediashare_cached[0] else "no overlay"
                    logger.info(
                        f"Clip {idx + 1}: reusing earlier donation scan ({found})."
                    )

                analyses.append(
                    analyzer.analyze_window(
                        video_path,
                        clip["start_time"],
                        clip["end_time"],
                        track_gameplay=follow,
                        facecam_override=stable_facecam,
                        facecam_boxes=collab_cams
                        if (is_collab and len(collab_cams) >= 2)
                        else None,
                        gameplay_aspect=aspect,
                        mediashare_cached=mediashare_cached,
                    )
                )
        finally:
            analyzer.release()
        return analyses

    @staticmethod
    def _write_clip_metadata(
        output_dir: Path,
        video_id: str,
        clip_index: int,
        title: str,
        caption: str,
        description: str,
        hashtags: str,
        total: int = 1,
    ) -> Path | None:
        """Write a TXT metadata file alongside a rendered clip MP4.

        Contains Title, Caption, Description, and Hashtags for easy copy-paste
        when uploading to YouTube Shorts, Instagram Reels, or TikTok.

        Returns the Path to the written file, or None if no text content exists.
        """
        safe_title = _sanitize_title(title)
        if not safe_title:
            safe_title = f"clip_{clip_index + 1}"

        # Manual mode with --no-metadata sets only a default Manual_<start>_<end> title and
        # leaves caption/description/hashtags empty — skip the sidecar entirely in that case
        # rather than writing a title-only file.
        if not (caption or description or hashtags):
            return None

        lines = []
        if title:
            lines.append(f"Title: {title}")
        if caption:
            lines.append(f"Caption: {caption}")
        if description:
            lines.append(f"Description: {description}")
        if hashtags:
            lines.append(f"Hashtags: {hashtags}")

        txt_subdir = output_dir / video_id.upper()
        txt_subdir.mkdir(parents=True, exist_ok=True)
        tc = _titlecase_filename(safe_title)
        txt_path = txt_subdir / f"{clip_stem(clip_index, total, tc)}.txt"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return txt_path

    @staticmethod
    def _smallest_containing_candidate(
        cache: dict | None, start: float, end: float
    ) -> dict | None:
        """Smallest cached candidate window fully containing [start, end] (±0.5s tolerance).

        Shared by ``_match_cached_mediashare`` and ``_match_cached_words`` — both caches use the
        same ``{"candidates": [{"start", "end", ...}]}`` schema and selection rule.
        """
        if not cache:
            return None
        best: dict | None = None
        for cand in cache.get("candidates", []):
            cs, ce = float(cand["start"]), float(cand["end"])
            if (
                cs - 0.5 <= start
                and end <= ce + 0.5
                and (
                    best is None
                    or (ce - cs) < (float(best["end"]) - float(best["start"]))
                )
            ):
                best = cand
        return best

    @staticmethod
    def _match_cached_mediashare(
        cache: dict | None,
        start: float,
        end: float,
    ) -> tuple[bool, tuple | None] | None:
        """Find a cached donation scan entry whose window fully contains [start, end] (±0.5s).

        Returns ``(mediashare_present, mediashare_box)`` from the smallest matching window,
        or ``None`` on cache miss so the caller falls back to a fresh scan.
        """
        best = ClipRenderer._smallest_containing_candidate(cache, start, end)
        if best is None:
            return None
        box = best.get("mediashare_box")
        if box is not None:
            box = tuple(box)  # JSON deserialises as list; callers expect tuple
        return (bool(best.get("mediashare_present", False)), box)

    # ----------------------------------------------------------------- Pass 2

    def _transcribe_subtitles(
        self, video_path: Path, clips: list[dict], clip_types: list[ContentType]
    ) -> list[list[dict]]:
        """Word-level subtitle segments per clip.

        Reuses the pipeline's per-candidate word cache (``{video_id}_words.json``) when a clip falls
        inside a cached window — avoiding a second Whisper pass entirely on a full hit. Misses (cloud
        STT, manual mode, no cache) are transcribed fresh with one shared model load. Clips whose
        subtitles are disabled (globally, or GAMING_COLLAB when ``collab_enabled`` is false) are
        skipped to save compute.
        """

        sub_cfg = self.config.video_processing.subtitles
        results: list[list[dict]] = [[] for _ in clips]
        if not sub_cfg.enabled:
            logger.info("Subtitles are turned off, skipping transcription.")
            return results

        cache = load_words_cache(video_path.stem)
        pending: list[int] = []
        for idx, clip in enumerate(clips):
            # Skip subtitles on the cramped 3-stack collab layout unless explicitly enabled.
            if (
                clip_types[idx] == ContentType.GAMING_COLLAB
                and not sub_cfg.collab_enabled
            ):
                logger.info(
                    "Subtitles skipped for collaboration layout as screen space is limited."
                )
                continue
            cand = (
                self._match_cached_words(cache, clip["start_time"], clip["end_time"])
                if cache
                else None
            )
            if cand is not None:
                results[idx] = self._slice_cached_segments(
                    cand, clip["start_time"], clip["end_time"]
                )
                logger.info(f"Clip {idx + 1} using saved word timings.")
            else:
                pending.append(idx)

        if not pending:
            logger.info("Using saved word timings from earlier transcription.")
            return results

        self._transcribe_pending(video_path, clips, pending, results)
        return results

    def _transcribe_pending(
        self,
        video_path: Path,
        clips: list[dict],
        pending: list[int],
        results: list[list[dict]],
    ) -> None:
        """Fresh word-level transcription for cache-miss clips; one shared Whisper model load."""
        audio_path = self._resolve_audio_track(video_path)
        if audio_path is None or not audio_path.exists():
            logger.warning(
                "No audio track available; pending clips will have no subtitles."
            )
            return

        provider = LocalSTTProvider()
        audio_ext = self.config.downloader.audio_format
        model = None
        try:
            from faster_whisper import WhisperModel

            device = SystemUtils.resolve_device(
                self.config.ai_pipeline.stt.local.device
            )
            size = provider.model_size
            c_type = "float32" if size in ("tiny", "base") else "int8"
            logger.info(
                f"Loading speech-to-text model for {len(pending)} clip(s) that need subtitles."
            )
            model = WhisperModel(size, device=device, compute_type=c_type)

            for idx in pending:
                clip = clips[idx]
                start_t, end_t = clip["start_time"], clip["end_time"]
                slice_path = TMP_DIR / f"audio_{video_path.stem}_{idx + 1}.{audio_ext}"
                ok = self.slicer.slice_audio_chunk(
                    str(audio_path), start_t, end_t, str(slice_path)
                )
                if not ok:
                    logger.warning(
                        f"Could not slice audio for clip {idx + 1}; no subs."
                    )
                    continue
                try:
                    results[idx] = provider.transcribe_segments(
                        slice_path, model=model, time_offset=start_t
                    )
                except Exception as e:
                    logger.warning(
                        f"Subtitle transcription failed for clip {idx + 1}: {e}"
                    )
        except Exception as e:
            logger.error(f"Subtitle transcription pass failed: {e}")
        finally:
            if model is not None:
                del model
                gc.collect()

    @staticmethod
    def _match_cached_words(cache: dict, start: float, end: float) -> dict | None:
        """Smallest cached candidate window fully containing [start, end] (±0.5s tolerance)."""
        return ClipRenderer._smallest_containing_candidate(cache, start, end)

    @staticmethod
    def _slice_cached_segments(cand: dict, start: float, end: float) -> list[dict]:
        """Filter a cached candidate's segments/words to the clip range (absolute times)."""
        lo, hi = start - 0.5, end + 0.5
        out: list[dict] = []
        for seg in cand.get("segments", []):
            words = [
                w for w in seg.get("words", []) if w["end"] >= lo and w["start"] <= hi
            ]
            if not words:
                continue
            out.append(
                {
                    "text": seg.get("text", ""),
                    "start": words[0]["start"],
                    "end": words[-1]["end"],
                    "words": words,
                }
            )
        return out

    def _resolve_audio_track(self, video_path: Path) -> Path | None:
        """Return the extracted audio track path, extracting it if missing."""
        audio_ext = self.config.downloader.audio_format
        audio_path = audio_output_path(video_path.stem, audio_ext)
        if audio_path.exists():
            return audio_path
        try:
            AudioExtractor().extract_audio(str(video_path), force=False)
        except Exception as e:
            logger.warning(f"Could not extract audio track: {e}")
        return audio_path if audio_path.exists() else None

    # ----------------------------------------------------------------- Pass 3

    def _render_pass(
        self,
        video_path: Path,
        clips: list[dict],
        clip_types: list[ContentType],
        analyses: list[dict],
        segments_per_clip: list[list[dict]],
        output_dir: Path,
    ) -> list[Path]:
        rendered: list[Path] = []
        audio_path = self._resolve_audio_track(video_path)

        video_id = video_path.stem
        clip_subdir = output_dir / video_id.upper()
        clip_subdir.mkdir(parents=True, exist_ok=True)

        total = len(clips)
        for idx, clip in enumerate(clips):
            result = self._render_one_clip(
                idx,
                clip,
                clip_types,
                total,
                analyses,
                segments_per_clip,
                video_path,
                audio_path,
                video_id,
                output_dir,
                clip_subdir,
            )
            if result is not None:
                rendered.append(result)

        return rendered

    def _build_render_cmd(
        self,
        video_path: Path,
        start_t: float,
        duration: float,
        layout_spec: dict,
        sub_arg: Path | None,
        output_path: Path,
        audio_path: Path | None,
        enc: str,
    ) -> list[str]:
        """Build the FFmpeg command for one clip (extracted from the closure in _render_pass)."""
        return self.ffmpeg_builder.build_render_command(
            video_path,
            start_t,
            duration,
            layout_spec,
            sub_arg,
            output_path,
            audio_path=audio_path,
            video_encoder=enc,
        )

    def _render_one_clip(
        self,
        idx: int,
        clip: dict,
        clip_types: list[ContentType],
        total: int,
        analyses: list[dict],
        segments_per_clip: list[list[dict]],
        video_path: Path,
        audio_path: Path | None,
        video_id: str,
        output_dir: Path,
        clip_subdir: Path,
    ) -> Path | None:
        """Render a single clip: face-tracking → overlay → layout → subs → encode."""
        start_t, end_t = clip["start_time"], clip["end_time"]
        duration = end_t - start_t
        title = clip.get("title", f"clip_{idx + 1}")
        content_type = clip_types[idx]
        safe_title = _sanitize_title(title)
        # Manual-mode default titles (e.g. "Manual_1-30_2-30") encode the timerange verbatim —
        # title-casing would mangle the hyphenated timestamps, so skip it for this one prefix.
        tc = (
            safe_title
            if safe_title.startswith("Manual_")
            else _titlecase_filename(safe_title)
        )
        output_path = clip_subdir / f"{clip_stem(idx, total, tc)}.mp4"

        logger.info(
            f"--- Rendering Clip {idx + 1}/{len(clip_types)}: {title} "
            f"({duration:.1f}s, {content_type.value}) ---"
        )

        try:
            face_data: list[dict] = []
            if content_type == ContentType.PODCAST:
                video_person_count = max(
                    (a.get("face_count", 0) for a in analyses), default=0
                )
                face_data = self.tracker.track_clip(
                    video_path,
                    start_t,
                    end_t,
                    content_type,
                    person_count=video_person_count,
                )

            overlay_data = []
            if content_type == ContentType.DONATION_OVERLAY and not analyses[idx].get(
                "mediashare_present"
            ):
                overlay_data = self.overlay_detector.detect_overlays(
                    video_path,
                    start_t,
                    end_t,
                    facecam_box=analyses[idx].get("facecam_box"),
                )

            layout_spec = self.layout_builder.build_layout(
                content_type, analyses[idx], face_data, overlay_data
            )

            ass_path = SUBTITLES_DIR / f"subs_{video_id}_{idx + 1}.ass"
            has_subs = self.subtitle_gen.generate_ass(
                segments_per_clip[idx], ass_path, clip_start=start_t
            )
            sub_arg = ass_path if has_subs else None
            encoder = self.config.video_processing.video_encoder

            logger.info(f"Encoding clip: {safe_title} ({duration:.1f}s)...")
            self._run_render_with_fallback(
                lambda enc: self._build_render_cmd(
                    video_path,
                    start_t,
                    duration,
                    layout_spec,
                    sub_arg,
                    output_path,
                    audio_path,
                    enc,
                ),
                encoder,
            )

            logger.info(f"Successfully rendered: {output_path.name}")
            txt_path = self._write_clip_metadata(
                output_dir,
                video_id,
                idx,
                title,
                clip.get("caption", ""),
                clip.get("description", ""),
                clip.get("hashtags", ""),
                total,
            )
            if txt_path:
                logger.info(f"Metadata saved: {txt_path.name}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Video encoding failed for clip '{title}': {e.stderr[-1500:] if e.stderr else e}"
            )
            return None
        except Exception as e:
            logger.exception(f"Failed to render clip {title} at {start_t}s: {e}")
            return None
