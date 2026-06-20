from __future__ import annotations

import json
import numpy as np

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    ContentType,
    GAMEPLAY_MIN_NONPERSON_MOTION,
    GAMEPLAY_MIN_OPEN_AREA_FRAC,
    GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT,
)
from src.core.workspace import DATA_DIR


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

        Returns:
            A ``ContentType`` enum value, or ``None`` when the video is ambiguous
            (the LLM resolves uncertain videos).
        """
        # 1. Config override
        override_val = self.config.video_processing.content_type_override
        if override_val != "auto":
            try:
                ct = ContentType(override_val.upper())
                logger.info(f"Video type set manually in config: {ct}")
                return ct
            except ValueError:
                logger.warning(
                    f"Invalid video type override '{override_val}' in config — ignoring, will auto-detect."
                )

        logger.info(f"Analysing video to detect content type: {video_path.name}")

        # 2. Sample frames from video (HUD heuristic + face count across middle 80%)
        sampled = self._sample_frames(video_path, num_samples=25)
        if not sampled:
            logger.warning("Could not read any video frames. Defaulting to PODCAST type.")
            return ContentType.PODCAST

        frames = [f[1] for f in sampled]
        h, w, _ = frames[0].shape

        # 3. Gaming hints (YouTube category + HUD score)
        gaming_hint = self.metadata_gaming_hint(video_path)
        hud_score = self._compute_hud_score(frames)
        hud_threshold = 0.0015 if gaming_hint else 0.003
        has_hud = hud_score >= hud_threshold
        logger.debug(
            f"Game HUD scan — score: {hud_score:.4f}, detected: {has_hud}, "
            f"gaming category hint: {gaming_hint}"
        )

        # 4. Reliable webcam count (filters by persistence/area/edge/separation).
        #    Much more reliable than raw YOLO face_count for the SOLO↔COLLAB split.
        cams: list[tuple[int, int, int, int]] = []
        try:
            from src.vision.visual_analyzer import VisualAnalyzer

            analyzer = VisualAnalyzer()
            try:
                cams = analyzer.detect_facecams(video_path)
            finally:
                analyzer.release()
        except Exception as e:
            logger.warning(f"Webcam detection failed: {e}")

        cam_count = len(cams)
        logger.info(f"Webcam detection: {cam_count} webcam(s) found.")

        # 5. Gameplay gate (VisualAnalyzer probes the whole video)
        gameplay_present = False
        if self.config.video_processing.region_detection.enabled:
            try:
                from src.vision.visual_analyzer import VisualAnalyzer

                analyzer = VisualAnalyzer()
                try:
                    gp = analyzer.detect_gameplay_presence(video_path)
                    # When gaming_hint is set, relax the open-area threshold — close-up cam
                    # shots in a gaming stream still have gameplay behind/around the streamer.
                    open_threshold = (
                        GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT
                        if gaming_hint
                        else GAMEPLAY_MIN_OPEN_AREA_FRAC
                    )
                    gameplay_present = (
                        gp["open_area_frac"] >= open_threshold
                        and (
                            gp["non_person_motion"] >= GAMEPLAY_MIN_NONPERSON_MOTION
                            or gaming_hint
                            or has_hud
                        )
                    )
                    logger.debug(
                        f"Gameplay gate — open_area: {gp['open_area_frac']:.2f}, "
                        f"motion: {gp['non_person_motion']:.2f}, "
                        f"hud: {has_hud}, hint: {gaming_hint}, confirmed: {gameplay_present}"
                    )
                finally:
                    analyzer.release()
            except Exception as e:
                logger.warning(f"Gameplay presence probe failed: {e}. Assuming no gameplay.")

        # 6. Decision tree
        if gameplay_present:
            if cam_count >= 2:
                # Two genuine webcams — could be SOLO or COLLAB. Defer to the LLM which also
                # sees the transcript, audio speaker count, and game/show metadata.
                logger.info("Gaming content: 2 webcams detected — uncertain SOLO vs COLLAB, deferring to LLM.")
                return None
            logger.info("Gaming content: solo stream.")
            return ContentType.GAMING_SOLO

        # No gameplay confirmed.
        if cam_count >= 2:
            # Two+ persistent faces, no gameplay → talking heads / panel / podcast.
            return ContentType.PODCAST

        # Donation overlay detection (colour heuristic) for single-face, no-gameplay content.
        has_donation_alerts = self._check_donation_overlays(frames)
        if has_donation_alerts:
            logger.info("Donation alerts detected in video.")
            return ContentType.JUST_CHAT

        if cam_count == 1:
            return ContentType.PODCAST  # single talking head

        # Ambiguous: no faces, no gameplay, no donation → uncertain.
        return None

    def classify_from_analysis(
        self, analysis: dict, gaming_hint: bool
    ) -> ContentType | None:
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
            GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT
            if gaming_hint
            else GAMEPLAY_MIN_OPEN_AREA_FRAC
        )
        open_enough = open_frac >= open_threshold
        gameplay_confirmed = open_enough and (
            motion >= GAMEPLAY_MIN_NONPERSON_MOTION or gaming_hint
        )

        if gameplay_confirmed:
            if face_count >= 2:
                return None  # ambiguous, defer to LLM
            return ContentType.GAMING_SOLO

        # No gameplay confirmed.
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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        if total_frames <= 0:
            cap.release()
            return []

        start_frame = int(total_frames * 0.10)
        end_frame = int(total_frames * 0.90)
        if end_frame <= start_frame:
            start_frame = 0
            end_frame = total_frames

        indices = [
            int(start_frame + i * (end_frame - start_frame) / num_samples)
            for i in range(num_samples)
        ]

        samples = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                samples.append((idx / fps, frame))

        cap.release()
        return samples

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

        hud_mask = (temp_std < 8.0) & (spatial_grad > 18.0)

        margin = 6
        hud_mask[:margin, :] = False
        hud_mask[-margin:, :] = False
        hud_mask[:, :margin] = False
        hud_mask[:, -margin:] = False

        return float(np.sum(hud_mask) / (128 * 128))

    def _check_donation_overlays(self, frames: list[np.ndarray]) -> bool:
        """Sample for donation alert popup signatures (bright transient overlay rectangles)."""
        import cv2

        if len(frames) < 3:
            return False

        matched_alerts = 0
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Red/pink alert range: H: 170-180 (or 0-10), S: 100-255, V: 100-255
            mask_rp = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            mask_rp2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            mask_rp = cv2.bitwise_or(mask_rp, mask_rp2)

            # Orange alert range: H: 5-25, S: 100-255, V: 100-255
            mask_or = cv2.inRange(hsv, (5, 100, 100), (25, 255, 255))

            mask = cv2.bitwise_or(mask_rp, mask_or)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < 2000 or area > 50000:
                    continue
                x, y, cw, ch = cv2.boundingRect(c)
                aspect = cw / max(1, ch)
                if 1.2 <= aspect <= 4.0:
                    matched_alerts += 1
                    break

        # Donation overlays are transient: present in some frames but not all.
        # 0 < matches < 60% of frames = a real popup (not a permanent HUD element).
        ratio = matched_alerts / max(1, len(frames))
        return 0 < ratio < 0.6
