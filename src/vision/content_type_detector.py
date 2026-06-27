from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    DONATION_ASPECT_MAX,
    DONATION_ASPECT_MIN,
    DONATION_HSV_ORANGE_LOWER,
    DONATION_HSV_ORANGE_UPPER,
    DONATION_HSV_RED_LOWER1,
    DONATION_HSV_RED_LOWER2,
    DONATION_HSV_RED_UPPER1,
    DONATION_HSV_RED_UPPER2,
    DONATION_MAX_AREA,
    DONATION_MIN_AREA,
    DONATION_PERSISTENCE_MAX_RATIO,
    GAMEPLAY_MIN_NONPERSON_MOTION,
    GAMEPLAY_MIN_OPEN_AREA_FRAC,
    GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT,
    HUD_ANALYSIS_SIZE,
    HUD_EDGE_MARGIN,
    HUD_SCORE_THRESHOLD_DEFAULT,
    HUD_SCORE_THRESHOLD_GAMING_HINT,
    HUD_SPATIAL_GRAD_THRESH,
    HUD_TEMP_STD_THRESH,
    SAMPLE_FRAME_END_FRAC,
    SAMPLE_FRAME_START_FRAC,
    ContentType,
)
from src.core.workspace import DATA_DIR
from src.vision.frame_utils import sample_frame_indices_evenly, sample_frames_timed, video_props


@dataclass
class ContentTypeDetectionResult:
    """Bundles a detection decision with the structured evidence that produced it.

    ``content_type`` is ``None`` when the algorithm could not decide; in that case
    ``evidence`` MUST be passed to the LLM so it has full context for per-clip
    classification.
    """

    content_type: ContentType | None
    evidence: dict[str, object] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """True when the algorithm made a definitive decision (not deferring to LLM)."""
        return self.content_type is not None


class ContentTypeDetector:
    """Detects the ContentType of a video by aggregating evidence across the whole video.

    Detection runs ONCE before rendering. It returns a ``ContentType`` when confident,
    or ``None`` ("uncertain") — in which case the caller defers to the selection LLM
    (which also sees per-candidate visual descriptors, audio speaker counts, transcripts,
    and game/show metadata) to decide.
    """

    def __init__(self) -> None:
        self.config = load_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_content_type(self, video_path: Path) -> ContentType | None:
        """Analyze the full video and determine its content type.

        Returns a ``ContentType`` enum value, or ``None`` when the video is
        ambiguous (the LLM resolves uncertain videos).

        This is a backwards-compatibility wrapper. For the full decision plus
        structured evidence, use ``detect_content_type_full()``.
        """
        return self.detect_content_type_full(video_path).content_type

    def detect_content_type_full(self, video_path: Path) -> ContentTypeDetectionResult:
        """Analyze the full video, return decision + structured evidence.

        The algorithm makes the primary decision from visual signals. Only
        truly ambiguous cases (no signals at all) return ``None`` — the caller
        should then pass ``evidence`` to the LLM for per-clip classification.

        Returns:
            A ``ContentTypeDetectionResult`` with ``content_type`` (or ``None``
            when uncertain) and an ``evidence`` dict of the raw detection
            numbers.
        """
        # 1. Config override
        override_val = self.config.video_processing.content_type_override
        if override_val != "auto":
            try:
                ct = ContentType(override_val.upper())
                from src.core.constants import CONTENT_TYPE_HUMAN_NAMES

                human_name = CONTENT_TYPE_HUMAN_NAMES.get(ct.value, ct.value.lower())
                logger.info(f"Video type forced by config, using {human_name} layout.")
                return ContentTypeDetectionResult(
                    content_type=ct, evidence={"override": override_val}
                )
            except ValueError:
                logger.warning(
                    f"Invalid video type override '{override_val}' in config — ignoring, will auto-detect."
                )

        logger.info("Detecting video type...")

        # 2. Sample frames from video (HUD heuristic + face count across middle 80%)
        sampled = self._sample_frames(video_path, num_samples=25)
        if not sampled:
            logger.warning("Could not read any video frames. Defaulting to PODCAST type.")
            return ContentTypeDetectionResult(
                content_type=ContentType.PODCAST,
                evidence={"fallback": "no_frames_readable"},
            )

        frames = [f[1] for f in sampled]
        h, w, _ = frames[0].shape

        # 3. Gaming hints (YouTube category + HUD score)
        gaming_hint = self.metadata_gaming_hint(video_path)
        hud_score = self._compute_hud_score(frames)
        hud_threshold = (
            HUD_SCORE_THRESHOLD_GAMING_HINT if gaming_hint else HUD_SCORE_THRESHOLD_DEFAULT
        )
        has_hud = hud_score >= hud_threshold
        logger.debug(
            f"Game HUD scan — score: {hud_score:.4f}, detected: {has_hud}, "
            f"gaming category hint: {gaming_hint}"
        )

        # 4. Reliable webcam count (filters by persistence/area/edge/separation).
        #    Much more reliable than raw YOLO face_count for the SOLO↔COLLAB split.
        cams: list[tuple[int, int, int, int]] = []
        gp: dict[str, float] = {}
        gameplay_present = False
        try:
            from src.vision.visual_analyzer import VisualAnalyzer

            analyzer = VisualAnalyzer()
            try:
                cams = analyzer.detect_facecams(video_path)

                # 5. Gameplay gate (same VisualAnalyzer instance)
                if self.config.video_processing.region_detection.enabled:
                    gp = analyzer.detect_gameplay_presence(video_path)
                    open_threshold = (
                        GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT
                        if gaming_hint
                        else GAMEPLAY_MIN_OPEN_AREA_FRAC
                    )
                    gameplay_present = gp["open_area_frac"] >= open_threshold and (
                        gp["non_person_motion"] >= GAMEPLAY_MIN_NONPERSON_MOTION
                        or gaming_hint
                        or has_hud
                    )
                    logger.debug(
                        f"Gameplay gate — open_area: {gp['open_area_frac']:.2f}, "
                        f"motion: {gp['non_person_motion']:.2f}, "
                        f"hud: {has_hud}, hint: {gaming_hint}, confirmed: {gameplay_present}"
                    )
            finally:
                analyzer.release()
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")

        cam_count = len(cams)

        # Build structured evidence dict (used for logging AND LLM fallback)
        evidence: dict[str, object] = {
            "webcam_count": cam_count,
            "gameplay_present": gameplay_present,
            "hud_score": round(hud_score, 4),
            "hud_detected": has_hud,
            "gaming_hint": gaming_hint,
        }
        if gp:
            evidence["open_area_frac"] = round(float(gp.get("open_area_frac", 1.0)), 3)
            evidence["non_person_motion"] = round(float(gp.get("non_person_motion", 0.0)), 2)

        # 6. Decision tree (strengthened — detect_facecams already filters out game
        #    characters, so ≥2 cams + gameplay is definitively GAMING_COLLAB).
        if gameplay_present and cam_count >= 2:
            logger.info(
                "Video type detected as gaming with 2 webcams, using gaming collaboration layout."
            )
            return ContentTypeDetectionResult(
                content_type=ContentType.GAMING_COLLAB, evidence=evidence
            )

        if gameplay_present:
            logger.info("Video type detected as gaming with 1 webcam, using gaming solo layout.")
            return ContentTypeDetectionResult(
                content_type=ContentType.GAMING_SOLO, evidence=evidence
            )

        if cam_count >= 2:
            # Two+ persistent webcams, no gameplay → talking heads / panel / podcast.
            logger.info("Video type detected as panel or podcast with 2 faces and no gameplay.")
            return ContentTypeDetectionResult(content_type=ContentType.PODCAST, evidence=evidence)

        # Donation overlay detection (colour heuristic) for single-face, no-gameplay content.
        has_donation_alerts = self._check_donation_overlays(frames)
        evidence["donation_detected"] = has_donation_alerts
        if has_donation_alerts:
            logger.info("Video type detected as live stream with donation alerts.")
            return ContentTypeDetectionResult(content_type=ContentType.JUST_CHAT, evidence=evidence)

        if cam_count == 1:
            return ContentTypeDetectionResult(content_type=ContentType.PODCAST, evidence=evidence)

        # Truly ambiguous: no gameplay, no webcams, no donation, no faces.
        logger.info("Video type undetermined, will have AI decide per clip.")
        return ContentTypeDetectionResult(content_type=None, evidence=evidence)

    def classify_from_analysis(self, analysis: dict, gaming_hint: bool) -> ContentType | None:
        """Classify ONE clip window from its ``VisualAnalyzer.analyze_window()`` result.

        Used only by the renderer for manual-mode clips (no LLM call). The per-clip signals
        are less reliable than the whole-video detector, so this is the fallback path.

        Args:
            analysis: dict from ``VisualAnalyzer.analyze_window``.
            gaming_hint: True when the saved YouTube metadata marks this video as Gaming.

        Returns:
            A ``ContentType``, or ``None`` when the window is ambiguous (defaults to PODCAST).
        """
        open_frac = float(analysis.get("open_area_frac", 1.0))
        motion = float(analysis.get("non_person_motion", 0.0))
        face_count = int(analysis.get("face_count", 0))
        mediashare = bool(analysis.get("mediashare_present", False))

        open_threshold = (
            GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT if gaming_hint else GAMEPLAY_MIN_OPEN_AREA_FRAC
        )
        open_enough = open_frac >= open_threshold
        gameplay_confirmed = open_enough and (
            motion >= GAMEPLAY_MIN_NONPERSON_MOTION or gaming_hint
        )

        if gameplay_confirmed:
            if face_count >= 2:
                logger.info(
                    "Clip type detected as gaming with 2 faces, using gaming collaboration layout."
                )
                return ContentType.GAMING_COLLAB
            return ContentType.GAMING_SOLO

        if face_count >= 2:
            return ContentType.PODCAST
        if mediashare:
            return ContentType.JUST_CHAT
        if face_count == 1:
            return ContentType.PODCAST
        return None

    def metadata_gaming_hint(self, video_path: Path) -> bool:
        """True if the saved YouTube metadata marks this video as Gaming content."""
        meta_path = DATA_DIR / f"{video_path.stem}_metadata.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        categories = " ".join(meta.get("categories") or []).lower()
        tags = " ".join(meta.get("tags") or []).lower()
        return "gaming" in categories or "game" in tags

    # ------------------------------------------------------------------
    # Private helpers (whole-video detection only)
    # ------------------------------------------------------------------

    def _sample_frames(
        self, video_path: Path, num_samples: int = 25
    ) -> list[tuple[float, np.ndarray]]:
        """Sample N frames evenly from the video, excluding first and last 10%."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        try:
            _w, _h, fps, total_frames = video_props(cap)
            if total_frames <= 0:
                return []

            start_frame = int(total_frames * SAMPLE_FRAME_START_FRAC)
            end_frame = int(total_frames * SAMPLE_FRAME_END_FRAC)
            if end_frame <= start_frame:
                start_frame = 0
                end_frame = total_frames

            indices = sample_frame_indices_evenly(start_frame, end_frame, num_samples)
            return sample_frames_timed(cap, indices, fps)
        finally:
            cap.release()

    def _compute_hud_score(self, frames: list[np.ndarray]) -> float:
        """Detect static graphic UI elements (HUDs) by temporal variance & spatial gradient."""
        import cv2

        if not frames:
            return 0.0

        small_grays = []
        for f in frames:
            small = cv2.resize(f, (128, 128))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small_grays.append(gray)

        stack = np.stack(small_grays, axis=0)
        temp_std = np.std(stack, axis=0)

        avg_frame = np.mean(stack, axis=0).astype(np.uint8)
        sobelx = cv2.Sobel(avg_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(avg_frame, cv2.CV_64F, 0, 1, ksize=3)
        spatial_grad = np.sqrt(sobelx**2 + sobely**2)

        hud_mask = (temp_std < HUD_TEMP_STD_THRESH) & (spatial_grad > HUD_SPATIAL_GRAD_THRESH)

        margin = HUD_EDGE_MARGIN
        hud_mask[:margin, :] = False
        hud_mask[-margin:, :] = False
        hud_mask[:, :margin] = False
        hud_mask[:, -margin:] = False

        return float(np.sum(hud_mask) / (HUD_ANALYSIS_SIZE * HUD_ANALYSIS_SIZE))

    def _check_donation_overlays(self, frames: list[np.ndarray]) -> bool:
        """Sample for donation alert popup signatures (bright transient overlay rectangles)."""
        import cv2

        if len(frames) < 3:
            return False

        matched_alerts = 0
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red/pink alert range: H: 170-180 (or 0-10), S: 100-255, V: 100-255
            mask_rp = cv2.inRange(hsv, DONATION_HSV_RED_LOWER1, DONATION_HSV_RED_UPPER1)
            mask_rp2 = cv2.inRange(hsv, DONATION_HSV_RED_LOWER2, DONATION_HSV_RED_UPPER2)
            mask_rp = cv2.bitwise_or(mask_rp, mask_rp2)

            # Orange alert range: H: 5-25, S: 100-255, V: 100-255
            mask_or = cv2.inRange(hsv, DONATION_HSV_ORANGE_LOWER, DONATION_HSV_ORANGE_UPPER)

            mask = cv2.bitwise_or(mask_rp, mask_or)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < DONATION_MIN_AREA or area > DONATION_MAX_AREA:
                    continue
                x, y, cw, ch = cv2.boundingRect(c)
                aspect = cw / max(1, ch)
                if DONATION_ASPECT_MIN <= aspect <= DONATION_ASPECT_MAX:
                    matched_alerts += 1
                    break

        # Donation overlays are transient: present in some frames but not all.
        # 0 < matches < DONATION_PERSISTENCE_MAX_RATIO fraction of frames = a real popup.
        ratio = matched_alerts / max(1, len(frames))
        return 0 < ratio < DONATION_PERSISTENCE_MAX_RATIO
