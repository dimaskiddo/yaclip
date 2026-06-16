from __future__ import annotations

import gc
import subprocess

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.constants import MIN_CLIP_SECONDS, ContentType
from src.media.ffmpeg_builder import FFmpegCommandBuilder
from src.media.slicer import AudioSlicer
from src.media.subtitles import SubtitleGenerator
from src.vision.content_type_detector import ContentTypeDetector
from src.vision.face_tracker import FaceTracker
from src.vision.layout_builder import LayoutBuilder
from src.vision.overlay_detector import OverlayDetector
from src.vision.visual_analyzer import VisualAnalyzer
from src.core.workspace import AUDIOS_DIR, TMP_DIR


class ClipRenderer:
    """Orchestrates region analysis, subtitle transcription, layout, and FFmpeg rendering.

    Runs as three memory-safe passes so heavy models never coexist in RAM:
      1. Regions  — YOLOv8n VisualAnalyzer per clip, then released.
      2. Subtitles — faster-whisper word-level transcription per clip, then released.
      3. Render   — layout + .ass + FFmpeg encode per clip.
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
        transcript_segments: list[dict] | None = None,
        content_type: ContentType | None = None,
    ) -> list[Path]:
        """Render multiple clip proposals to vertical 9:16 MP4 files.

        Args:
            video_path: Path to the raw downloaded video file.
            clip_proposals: [{"start_time", "end_time", "title", optional "content_type"}].
            transcript_segments: Unused — subtitles are re-transcribed per clip locally for
                reliable word timings (kept for signature compatibility).

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
            logger.warning("No valid clips to render after dropping degenerate proposals.")
            return []

        # Use the content type detected upstream (cli) when provided — avoids a second detection.
        base_content_type = content_type or self._resolve_base_content_type(video_path)
        clip_types = [self._clip_content_type(c, base_content_type) for c in clip_proposals]

        # Pass 1 — visual region analysis (YOLO loaded once, released after).
        analyses = self._analyze_regions(video_path, clip_proposals, clip_types)

        # Promote any clip whose window contains a mediashare/donation popup to DONATION_OVERLAY
        # (mediashare_present is only known after Pass 1). Done before Pass 2 so a promoted clip is
        # not caught by the GAMING_COLLAB subtitle skip.
        clip_types = self._apply_donation_override(clip_proposals, clip_types, analyses)

        # Pass 2 — per-clip subtitle segments (whisper loaded once, released after).
        segments_per_clip = self._transcribe_subtitles(
            video_path, clip_proposals, clip_types
        )

        # Pass 3 — layout + render.
        return self._render_pass(
            video_path, clip_proposals, clip_types, analyses, segments_per_clip, output_dir
        )

    # ------------------------------------------------------------- content type

    def _resolve_base_content_type(self, video_path: Path) -> ContentType:
        override = self.config.video_processing.content_type_override
        if override != "auto":
            try:
                return ContentType(override.upper())
            except ValueError:
                pass
        return self.detector.detect_content_type(video_path)

    def _clip_content_type(self, clip: dict, base: ContentType) -> ContentType:
        if "content_type" in clip:
            try:
                return ContentType(clip["content_type"].upper())
            except ValueError:
                pass
        return base

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
        out = list(clip_types)
        for idx, analysis in enumerate(analyses):
            if "content_type" in clips[idx]:
                continue
            # Collab videos always keep the 3-stack — the donation popup does not replace a panel.
            if out[idx] == ContentType.GAMING_COLLAB:
                continue
            if analysis.get("mediashare_present"):
                out[idx] = ContentType.DONATION_OVERLAY
                logger.info(
                    f"Clip {idx + 1}: donation overlay detected → DONATION_OVERLAY layout."
                )
        return out

    # ----------------------------------------------------------------- pass 1

    def _analyze_regions(
        self, video_path: Path, clips: list[dict], clip_types: list[ContentType]
    ) -> list[dict]:
        """Run VisualAnalyzer on each clip window; YOLO loaded once and freed after."""
        from src.core.constants import STACK2_PANEL_ASPECT, STACK3_PANEL_ASPECT
        from src.ai.stt_local import load_mediashare_cache

        # Gameplay follow-motion is opt-in; default is a static centred zoom-out (no track).
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
                    logger.info(
                        f"Clip {idx + 1}: reusing saved donation scan result "
                        f"(mediashare_present={mediashare_cached[0]}); skipping rescan."
                    )

                analyses.append(
                    analyzer.analyze_window(
                        video_path,
                        clip["start_time"],
                        clip["end_time"],
                        track_gameplay=follow,
                        facecam_override=stable_facecam,
                        facecam_boxes=collab_cams if (is_collab and len(collab_cams) >= 2) else None,
                        gameplay_aspect=aspect,
                        mediashare_cached=mediashare_cached,
                    )
                )
        finally:
            analyzer.release()
        return analyses

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
        if not cache:
            return None
        best: dict | None = None
        for cand in cache.get("candidates", []):
            cs, ce = float(cand["start"]), float(cand["end"])
            if cs - 0.5 <= start and end <= ce + 0.5:
                if best is None or (ce - cs) < (float(best["end"]) - float(best["start"])):
                    best = cand
        if best is None:
            return None
        box = best.get("mediashare_box")
        if box is not None:
            box = tuple(box)  # JSON deserialises as list; callers expect tuple
        return (bool(best.get("mediashare_present", False)), box)

    # ----------------------------------------------------------------- pass 2

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
        from src.ai.stt_local import load_words_cache

        sub_cfg = self.config.video_processing.subtitles
        results: list[list[dict]] = [[] for _ in clips]
        if not sub_cfg.enabled:
            logger.info("Subtitles disabled; skipping transcription pass.")
            return results

        cache = load_words_cache(video_path.stem)
        pending: list[int] = []
        for idx, clip in enumerate(clips):
            # Skip subtitles on the cramped 3-stack collab layout unless explicitly enabled.
            if clip_types[idx] == ContentType.GAMING_COLLAB and not sub_cfg.collab_enabled:
                logger.info(f"Clip {idx + 1}: collaborative gaming layout — subtitles are disabled for this video type.")
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
                logger.info(
                    f"Clip {idx + 1}: reusing saved word timings "
                    f"({len(results[idx])} segment(s)); skipping re-transcription."
                )
            else:
                pending.append(idx)

        if not pending:
            logger.info(
                "All clips already have word timings cached or subtitles are off. Skipping transcription."
            )
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
            logger.warning("No audio track available; pending clips will have no subtitles.")
            return

        from src.ai.stt_local import LocalSTTProvider

        provider = LocalSTTProvider()
        audio_ext = self.config.downloader.audio_format
        model = None
        try:
            from faster_whisper import WhisperModel

            from src.core.utils import SystemUtils

            device = SystemUtils.resolve_device(self.config.ai_pipeline.stt.local.device)
            size = provider.model_size
            c_type = "float32" if size in ("tiny", "base") else "int8"
            logger.info(f"Loading transcription model ({size}) for {len(pending)} clip(s) that need subtitles...")
            model = WhisperModel(size, device=device, compute_type=c_type)

            for idx in pending:
                clip = clips[idx]
                start_t, end_t = clip["start_time"], clip["end_time"]
                slice_path = TMP_DIR / f"subaud_{idx + 1}_{start_t:.1f}.{audio_ext}"
                ok = self.slicer.slice_audio_chunk(
                    str(audio_path), start_t, end_t, str(slice_path)
                )
                if not ok:
                    logger.warning(f"Could not slice audio for clip {idx + 1}; no subs.")
                    continue
                try:
                    results[idx] = provider.transcribe_segments(
                        slice_path, model=model, time_offset=start_t
                    )
                except Exception as e:
                    logger.warning(f"Subtitle transcription failed for clip {idx + 1}: {e}")
        except Exception as e:
            logger.error(f"Subtitle transcription pass failed: {e}")
        finally:
            if model is not None:
                del model
                gc.collect()

    @staticmethod
    def _match_cached_words(cache: dict, start: float, end: float) -> dict | None:
        """Smallest cached candidate window fully containing [start, end] (±0.5s tolerance)."""
        best: dict | None = None
        for cand in cache.get("candidates", []):
            cs, ce = float(cand["start"]), float(cand["end"])
            if cs - 0.5 <= start and end <= ce + 0.5:
                if best is None or (ce - cs) < (float(best["end"]) - float(best["start"])):
                    best = cand
        return best

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
        AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = AUDIOS_DIR / f"{video_path.stem}.{audio_ext}"
        if audio_path.exists():
            return audio_path
        try:
            from src.media.audio import AudioExtractor

            AudioExtractor().extract_audio(str(video_path), force=False)
        except Exception as e:
            logger.warning(f"Could not extract audio track: {e}")
        return audio_path if audio_path.exists() else None

    # ----------------------------------------------------------------- pass 3

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

        for idx, clip in enumerate(clips):
            start_t, end_t = clip["start_time"], clip["end_time"]
            duration = end_t - start_t
            title = clip.get("title", f"clip_{idx + 1}")
            content_type = clip_types[idx]
            safe_title = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in title
            ).strip("_")
            output_path = output_dir / f"{safe_title}_{start_t:.1f}_{end_t:.1f}.mp4"

            logger.info(
                f"--- Rendering Clip {idx + 1}/{len(clips)}: {title} "
                f"({duration:.1f}s, {content_type.value}) ---"
            )

            try:
                # Mode A needs smooth face-pan crops; Mode B/C use static analyzer regions.
                face_data: list[dict] = []
                if content_type in (ContentType.PODCAST, ContentType.INTERVIEW):
                    face_data = self.tracker.track_clip(
                        video_path, start_t, end_t, content_type
                    )

                # overlay_data feeds layout_builder._mediashare_box as a colour-box fallback.
                # When analyze_window already detected a popup (mediashare_present=True), the
                # layout builder uses analysis["mediashare_box"] directly and never consults
                # overlay_data. Run the expensive detect_overlays only for the edge case where a
                # clip is explicitly tagged DONATION_OVERLAY (e.g. by the user in review) but no
                # popup was auto-detected — so the layout can still attempt the colour fallback.
                overlay_data = []
                if (
                    content_type == ContentType.DONATION_OVERLAY
                    and not analyses[idx].get("mediashare_present")
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

                ass_path = TMP_DIR / f"subs_{safe_title}_{start_t:.1f}.ass"
                has_subs = self.subtitle_gen.generate_ass(
                    segments_per_clip[idx], ass_path, clip_start=start_t
                )
                sub_arg = ass_path if has_subs else None

                cmd = self.ffmpeg_builder.build_render_command(
                    video_path,
                    start_t,
                    duration,
                    layout_spec,
                    sub_arg,
                    output_path,
                    audio_path=audio_path,
                )

                logger.info(f"Encoding clip: {safe_title} ({duration:.1f}s)...")
                logger.debug(f"FFmpeg render command: {' '.join(cmd)}")
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

                logger.info(f"Successfully rendered: {output_path.name}")
                rendered.append(output_path)

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Video encoding failed for clip '{title}': {e.stderr[-1500:] if e.stderr else e}"
                )
                continue
            except Exception as e:
                logger.exception(f"Failed to render clip {title} at {start_t}s: {e}")
                continue

        return rendered
