from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    COCO_CLASS_PERSON,
    COCO_SCREEN_CLASSES,
    FACECAM_EDGE_SCORE_MIN,
    FACECAM_MAX_AREA_FRAC,
    FACECAM_MIN_AREA_FRAC,
    FACECAM_MIN_PERSISTENCE,
    FACECAM_MIN_SEP_FRAC,
    FACECAM_PICK_MIN_PERSISTENCE,
    GAMEPLAY_PAN_DEADZONE_FRAC,
    GAMEPLAY_PAN_EMA,
    GAMEPLAY_PAN_MAX_DEVIATION_FRAC,
    GAMEPLAY_PAN_MAX_VELOCITY_FRAC,
    MOTION_GRID_H,
    MOTION_GRID_W,
    MOTION_INTENSITY_HIGH,
    MOTION_INTENSITY_MODERATE,
    OPEN_AREA_GRID_COLS,
    OPEN_AREA_GRID_ROWS,
    PERSON_COUNT_CONF_MIN,
    STACK2_PANEL_ASPECT,
    YOLO_CLUSTER_DIST_FRAC,
)
from src.core.utils import SystemUtils, boxes_overlap, make_even
from src.core.workspace import MODELS_DIR


class VisualAnalyzer:
    """Detects facecam / gameplay / mediashare regions in a video time window via YOLOv8n.

    A single shared engine consumed by three callers:
      - clip selection (a text descriptor per candidate window for the LLM),
      - rendering layout (precise crop boxes per clip),
      - content-type detection (person count + screen inset + motion signals).

    The YOLO model is loaded lazily and reused across windows; call ``release()`` to free it.
    """

    def __init__(self) -> None:
        self.config = load_config()
        self.cfg = self.config.video_processing.region_detection
        self._model = None  # lazily-loaded ultralytics YOLO

    # ------------------------------------------------------------------ Model

    def _load_model(self) -> object:
        """Lazy-load the YOLOv8 model, routing weights into the portable workspace."""
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.warning(
                f"Object detection package not installed ({e}) — scene detection disabled."
            )
            return None

        model_path = MODELS_DIR / self.cfg.model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # YOLO() downloads the named asset to the given path if it does not exist yet.
        target = str(model_path) if model_path.exists() else self.cfg.model_name
        logger.info("Loading object detection model...")
        self._model = YOLO(target)
        # Keep weights inside the workspace for portability on first download.
        if not model_path.exists():
            try:
                downloaded = Path(self.cfg.model_name)
                if downloaded.exists():
                    downloaded.replace(model_path)
            except OSError as e:
                logger.debug(f"Could not move model file to workspace folder: {e}")
        return self._model

    def release(self) -> None:
        """Free the YOLO model from memory (RAM safety before loading other ML models)."""
        if self._model is not None:
            logger.debug("Releasing object detection model from memory.")
            del self._model
            self._model = None
            gc.collect()

    # --------------------------------------------------------------- Analysis

    def analyze_window(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        track_gameplay: bool = False,
        facecam_override: tuple[int, int, int, int] | None = None,
        facecam_boxes: list | None = None,
        gameplay_aspect: float = STACK2_PANEL_ASPECT,
        mediashare_cached: tuple[bool, tuple[int, int, int, int] | None] | None = None,
    ) -> dict:
        """Analyze one time window and return region metadata + an LLM text descriptor.

        Args:
            video_path: Path to the source video.
            start_time: Window start in seconds.
            end_time: Window end in seconds.
            track_gameplay: When True, also compute an animated gameplay crop track that
                follows the peak-motion region over time (render only; skipped for selection).
            mediashare_cached: Pre-computed donation scan result as (mediashare_present, mediashare_box).
                When provided, skips the expensive dense frame scan and uses this value instead.
                Supplied by the renderer when reusing the selection-phase scan results.

        Returns:
            dict with keys: facecam_box, persons, gameplay_box, gameplay_track,
            screen_inset_box, mediashare_present, mediashare_box, mediashare_events,
            face_count, motion_level, descriptor.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Cannot open video for scene analysis: {video_path}")
            return self._empty_analysis()

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            frames = self._sample_frames(cap, start_time, end_time)
        finally:
            cap.release()

        if not frames:
            return self._empty_analysis(width, height)

        # Sparse YOLO pass: facecam / persons / screen inset.
        persons: list[dict] = []
        screen_box: tuple[int, int, int, int] | None = None
        max_persons_in_frame: int = 0
        if self.cfg.enabled:
            persons, screen_box, max_persons_in_frame = self._detect_regions(frames, width, height)
        # Use the stable, video-level facecam box when provided so framing is identical
        # across every clip (the cam is positionally static; per-clip boxes drift with pose).
        # COLLAB passes the reliable detect_facecams() PAIR: [0] = primary (top), [1] = collaborator
        # (bottom). Both cams are excluded from the gameplay crop so neither bleeds into the centre.
        collab_box: tuple[int, int, int, int] | None = None
        if facecam_boxes:
            facecam_box = facecam_boxes[0]
            collab_box = facecam_boxes[1] if len(facecam_boxes) >= 2 else None
            exclude_boxes = [b for b in facecam_boxes if b is not None]
        else:
            facecam_box = facecam_override or self._pick_facecam(persons, width, height)
            exclude_boxes = [facecam_box] if facecam_box is not None else []

        # A YOLO "screen" box that overlaps the facecam is the webcam in a border — not a popup.
        if (
            screen_box is not None
            and facecam_box is not None
            and boxes_overlap(screen_box, facecam_box, 0.3)
        ):
            screen_box = None

        # Dense MediaShare/donation event scan (catches transient popups the sparse pass misses).
        # The facecam is a moving bordered inset, so it must be excluded or it becomes a false popup.
        # Three paths, in priority order:
        #   1. Caller supplied a cached result → skip the scan entirely (renderer reusing pipeline data).
        #   2. preserve_donation_overlays is off → skip; no routing will happen, scan is pure waste.
        #   3. Normal path: run the dense scan.
        if mediashare_cached is not None:
            mediashare_present, ms_box = mediashare_cached
            mediashare_box = ms_box or screen_box
            ms_events = []  # events only needed for the LLM descriptor (not used at render)
            logger.debug(
                f"Scene analysis [{start_time:.1f}-{end_time:.1f}s]: reusing cached donation scan "
                f"(mediashare_present={mediashare_present})."
            )
        elif self.config.video_processing.preserve_donation_overlays:
            ms_events, ms_box = self._scan_mediashare(video_path, start_time, end_time, facecam_box)
            mediashare_box = ms_box or screen_box
            mediashare_present = bool(ms_events)
        else:
            ms_events, ms_box = [], None
            mediashare_box = None
            mediashare_present = False

        # Gameplay region: animated pan (render) or static motion crop (selection). The panel aspect
        # is parameterised so the 3-stack collab centre uses the same engine at its wider 1.6875 shape.
        if track_gameplay:
            gameplay_track, gameplay_box, motion_level = self._gameplay_pan(
                video_path, start_time, end_time, exclude_boxes, width, height, gameplay_aspect
            )
        else:
            gameplay_track = []
            gameplay_box, motion_level = self._motion_region(
                frames, exclude_boxes, width, height, gameplay_aspect
            )

        analysis = {
            "video_width": width,
            "video_height": height,
            "facecam_box": facecam_box,
            "collab_box": collab_box,
            "persons": [p["box"] for p in persons],
            "gameplay_box": gameplay_box,
            "gameplay_track": gameplay_track,
            "screen_inset_box": screen_box,
            "mediashare_present": mediashare_present,
            "mediashare_box": mediashare_box,
            "mediashare_events": ms_events,
            "face_count": max_persons_in_frame,
            "motion_level": motion_level,
            # Classification-ready signals (consumed by ContentTypeDetector.classify_from_analysis):
            #   non_person_motion — motion_level is an accumulated sum over frame pairs; normalise to a
            #   per-pair mean so it is comparable to the GAMEPLAY_MIN_NONPERSON_MOTION gate.
            #   open_area_frac — fraction of a coarse grid NOT covered by person boxes (game screen room).
            "non_person_motion": motion_level / max(1, len(frames) - 1),
            "open_area_frac": self._open_area_frac(persons, width, height),
        }
        analysis["descriptor"] = self._build_descriptor(analysis)
        logger.info(self._build_log_summary(analysis, start_time, end_time))
        logger.debug(f"Scene descriptor (LLM input): {analysis['descriptor']}")
        return analysis

    def detect_gameplay_presence(self, video_path: Path) -> dict:
        """Measure how much animated content exists in the non-person screen area.

        Returns a dict with:
          ``non_person_motion``  — mean frame-diff intensity in non-person pixels
                                   (low ≈ static background / podcast set; high ≈ active game)
          ``open_area_frac``     — fraction of a coarse cell grid NOT covered by any person box
                                   (small ≈ people fill the frame; large ≈ a game screen is visible)
          ``person_count``       — number of persistent person clusters detected

        This is the key signal for the gameplay gate in ContentTypeDetector: a genuine gaming
        video has both substantial non-person screen space AND motion in that space.  A podcast
        with a static studio set may have a falsely-elevated HUD score but will fail on
        open_area_frac (people fill the frame) and/or non_person_motion (borders are static).
        """
        import cv2

        if not self.cfg.enabled:
            return {"non_person_motion": 0.0, "open_area_frac": 1.0, "person_count": 0}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"non_person_motion": 0.0, "open_area_frac": 1.0, "person_count": 0}

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

            # YOLO pass: detect persistent person clusters (12 frames, middle 80%)
            n = 12
            idxs = [int(total * (0.1 + 0.8 * i / max(1, n - 1))) for i in range(n)]
            yolo_frames: list[np.ndarray] = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    yolo_frames.append(frame)
        finally:
            cap.release()

        if not yolo_frames:
            return {"non_person_motion": 0.0, "open_area_frac": 1.0, "person_count": 0}

        persons, _, _max = self._detect_regions(yolo_frames, width, height)
        person_count = len(persons)
        open_area_frac = self._open_area_frac(persons, width, height)

        # Motion bursts: 3 short windows across the middle 80%.
        # Each burst = 6 consecutive frames at ~2 fps apart.  Person regions zeroed.
        burst_motions: list[float] = []
        small_w, small_h = MOTION_GRID_W, MOTION_GRID_H
        scale_x, scale_y = width / small_w, height / small_h
        mask_rects = [
            (
                max(0, int(p["box"][0] / scale_x)),
                min(small_w, int((p["box"][0] + p["box"][2]) / scale_x)),
                max(0, int(p["box"][1] / scale_y)),
                min(small_h, int((p["box"][1] + p["box"][3]) / scale_y)),
            )
            for p in persons
        ]

        cap2 = cv2.VideoCapture(str(video_path))
        if cap2.isOpened():
            try:
                burst_step_frames = max(1, int(fps * 0.5))  # ~2 fps within a burst
                burst_len = 6
                anchors = [int(total * frac) for frac in (0.25, 0.50, 0.75)]
                for anchor in anchors:
                    indices = [anchor + i * burst_step_frames for i in range(burst_len)]
                    indices = [min(max(0, idx), total - 1) for idx in indices]
                    burst_frames: list[np.ndarray] = []
                    for idx in indices:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap2.read()
                        if ret:
                            burst_frames.append(
                                cv2.cvtColor(
                                    cv2.resize(frame, (small_w, small_h)),
                                    cv2.COLOR_BGR2GRAY,
                                ).astype(np.float32)
                            )

                    if len(burst_frames) < 2:
                        continue

                    diffs: list[np.ndarray] = []
                    for i in range(1, len(burst_frames)):
                        d = np.abs(burst_frames[i] - burst_frames[i - 1])
                        for mx0, mx1, my0, my1 in mask_rects:
                            d[my0:my1, mx0:mx1] = 0.0
                        diffs.append(d)

                    if diffs:
                        burst_motions.append(float(np.mean(np.stack(diffs, axis=0))))
            finally:
                cap2.release()

        non_person_motion = float(np.median(burst_motions)) if burst_motions else 0.0
        logger.debug(
            f"Gameplay probe — open_area: {open_area_frac:.2f}, "
            f"non_person_motion: {non_person_motion:.2f}, persons: {person_count}"
        )
        return {
            "non_person_motion": non_person_motion,
            "open_area_frac": open_area_frac,
            "person_count": person_count,
        }

    def detect_stable_facecam(self, video_path: Path) -> tuple[int, int, int, int] | None:
        """Detect one stable facecam box for the whole video (sampled across its length).

        The webcam is positionally static, so a single box reused for every clip keeps the
        top-panel framing consistent instead of drifting with the streamer's pose.

        Returns:
            The facecam box (x, y, w, h), or None if no cam is found / model disabled.
        """
        import cv2

        if not self.cfg.enabled:
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            # 12 frames spanning the middle 80% of the video.
            n = 12
            idxs = [int(total * (0.1 + 0.8 * i / max(1, n - 1))) for i in range(n)]
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        finally:
            cap.release()

        if not frames:
            return None

        persons, _, _max = self._detect_regions(frames, width, height)
        box = self._pick_facecam(persons, width, height)
        if box is None:
            logger.info(
                "No fixed webcam found in video. Webcam position will be detected per clip."
            )
            return None
        position = self._where(box, width, height)
        logger.info(f"Primary webcam locked at {position}, assigned for Top Panel.")
        return box

    def detect_facecams(
        self, video_path: Path, max_cams: int = 2
    ) -> list[tuple[int, int, int, int]]:
        """Detect up to ``max_cams`` distinct streamer webcams across the video (YOLO persons).

        Far more reliable than MediaPipe for small corner cams, so it drives the GAMING_SOLO vs
        GAMING_COLLAB split. A cam is a persistent, cam-sized person near a frame edge/corner; central
        in-game characters and full-frame talking heads are filtered out, and two cams must be
        spatially separated. Returns the cam boxes (largest first), or [] when the model is disabled.
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            n = 12  # frames spanning the middle 80% of the video
            idxs = [int(total * (0.1 + 0.8 * i / max(1, n - 1))) for i in range(n)]
            frames = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        finally:
            cap.release()

        if not frames:
            return []

        persons, _, _max = self._detect_regions(frames, width, height)
        frame_area = float(width * height)
        diag = float((width**2 + height**2) ** 0.5)
        cams: list[tuple[float, float, float, float]] = []
        for p in persons:  # _cluster_boxes returns these sorted by area descending
            area_frac = p["area"] / frame_area
            if not (FACECAM_MIN_AREA_FRAC <= area_frac <= FACECAM_MAX_AREA_FRAC):
                continue
            if p["persistence"] < FACECAM_MIN_PERSISTENCE:
                continue
            if self._edge_score(p["box"], width, height) < FACECAM_EDGE_SCORE_MIN:
                continue
            b = p["box"]
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            too_close = any(
                float(np.hypot(cx - (q[0] + q[2] / 2), cy - (q[1] + q[3] / 2)))
                <= FACECAM_MIN_SEP_FRAC * diag
                for q in cams
            )
            if too_close:
                continue
            cams.append(b)
            if len(cams) >= max_cams:
                break

        boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in cams]
        if not boxes:
            logger.warning("No webcam detected in the video.")
        elif len(boxes) == 1:
            pos = self._where(boxes[0], width, height)
            logger.info(f"1 webcam detected at {pos}.")
        else:
            positions = [self._where(b, width, height) for b in boxes]
            pos_list = ", ".join(f"webcam {i + 1} {p}" for i, p in enumerate(positions))
            logger.info(f"{len(boxes)} webcams detected: {pos_list}.")
        return boxes

    @staticmethod
    def _edge_score(box: tuple[float, float, float, float], width: int, height: int) -> float:
        """1 − (centre's distance to the nearest frame edge)/(min(w,h)/2).

        ~0 for an interior game character (far from every border), high for a webcam hugging any
        edge/corner. More robust than corner distance — a bottom-centre cam still scores high.
        """
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        margin = min(cx, width - cx, cy, height - cy)
        half = max(1.0, min(width, height) / 2.0)
        return max(0.0, 1.0 - margin / half)

    def _empty_analysis(self, width: int = 1920, height: int = 1080) -> dict:
        """Neutral analysis when no frames/model are available."""
        crop_h = make_even(height)
        crop_w = make_even(min(width, int(round(height * STACK2_PANEL_ASPECT))))
        center = {"x": make_even((width - crop_w) // 2), "y": 0, "w": crop_w, "h": crop_h}
        return {
            "video_width": width,
            "video_height": height,
            "facecam_box": None,
            "collab_box": None,
            "persons": [],
            "gameplay_box": center,
            "gameplay_track": [],
            "screen_inset_box": None,
            "mediashare_present": False,
            "mediashare_box": None,
            "mediashare_events": [],
            "face_count": 0,
            "motion_level": 0.0,
            "non_person_motion": 0.0,
            "open_area_frac": 1.0,
            "descriptor": "No visual signal available.",
        }

    def _scan_mediashare(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        facecam_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[tuple[float, float]], tuple[int, int, int, int] | None]:
        """Dense scan for MediaShare/donation popups; returns absolute-time events + a box.

        Reuses OverlayDetector (colour signatures + transient motion inset) at a bounded
        ~2 fps cadence. Intervals overlapping the facecam are dropped — the webcam is itself
        a moving bordered inset and would otherwise be a false popup (the duplicate-cam bug).
        """
        try:
            from src.vision.overlay_detector import OverlayDetector

            intervals = OverlayDetector().detect_overlays(
                video_path,
                start_time,
                end_time,
                sample_step_seconds=0.5,
                facecam_box=facecam_box,
            )
        except Exception as e:
            logger.warning(f"Donation alert scan failed: {e}")
            return [], None

        if facecam_box is not None:
            kept = [iv for iv in intervals if not boxes_overlap(iv["box"], facecam_box, 0.3)]
            dropped = len(intervals) - len(kept)
            if dropped:
                logger.info(
                    f"Ignored {dropped} donation alert candidate(s) that overlapped the webcam."
                )
            intervals = kept

        if not intervals:
            return [], None

        events = [(start_time + iv["start_time"], start_time + iv["end_time"]) for iv in intervals]
        # Representative box = the longest-active interval's box.
        longest = max(intervals, key=lambda iv: iv["end_time"] - iv["start_time"])
        b = longest["box"]
        return events, (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def _gameplay_crop_geom(
        self,
        cams: list,
        width: int,
        height: int,
        aspect: float,
    ) -> tuple[int, int, int]:
        """Zoomed panel-aspect (w, h, y) for the gameplay crop, clearing the bottom cam band.

        Normally a vertically-centred zoomed crop. For collab (cams in the bottom third) the crop is
        shrunk + raised so its bottom edge sits at or above the highest bottom cam — the gameplay
        panel then shows only gameplay, never a corner webcam.
        """
        zoom = max(1.0, float(self.cfg.gameplay_zoom))
        base_w = min(width, int(round(height * aspect)))
        crop_w = make_even(int(base_w / zoom))
        crop_h = make_even(min(height, int(round(crop_w / aspect))))

        bottom_tops = [int(b[1]) for b in (cams or []) if (b[1] + b[3] / 2.0) > 2 * height / 3.0]
        if bottom_tops and min(bottom_tops) >= 0.45 * height:
            cam_top = min(bottom_tops)
            if crop_h > cam_top:  # shrink so the whole crop fits above the cam band
                crop_h = make_even(cam_top)
                crop_w = make_even(min(width, int(round(crop_h * aspect))))
            crop_y = make_even(max(0, cam_top - crop_h))
        else:
            crop_y = make_even(max(0, (height - crop_h) // 2))
        return crop_w, crop_h, crop_y

    def _gameplay_pan(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        exclude_boxes: list | None,
        width: int,
        height: int,
        aspect: float = STACK2_PANEL_ASPECT,
    ) -> tuple[list[dict], dict, float]:
        """Zoomed, static-first gameplay crop that glides slowly only when the subject moves.

        Densely samples the clip (~2 fps, bounded). The crop is zoomed in (``gameplay_zoom``) and
        centred. Its target is the motion CENTROID (diffuse particle motion averages to centre, so
        the camera stays put; a moving character pulls it). The camera holds still while the target
        is within a deadzone of the current centre, and when it must move it glides at a capped
        slow velocity — so viewers don't notice the pan. The crop is kept clear of the facecam.
        Returns the keyframe track, a representative static box, and the mean motion level.
        """
        import cv2

        # Zoomed panel-aspect crop; vertically biased above the bottom cam band for collab.
        cams = [b for b in (exclude_boxes or []) if b is not None]
        crop_w, crop_h, crop_y = self._gameplay_crop_geom(cams, width, height, aspect)
        center_x = make_even((width - crop_w) // 2)
        fallback = {"x": center_x, "y": crop_y, "w": crop_w, "h": crop_h}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], fallback, 0.0

        # Smooth, static-first tuning.
        frame_center = width / 2.0
        target_ema = (
            GAMEPLAY_PAN_EMA  # smooth the motion target so it doesn't jitter frame-to-frame
        )
        deadzone = (
            GAMEPLAY_PAN_DEADZONE_FRAC * width
        )  # hold still until the target drifts past this from the crop centre
        max_vel = (
            GAMEPLAY_PAN_MAX_VELOCITY_FRAC * width
        )  # then glide at most this many px per keyframe (slow, imperceptible)
        max_dev = (
            GAMEPLAY_PAN_MAX_DEVIATION_FRAC * width
        )  # crop centre may drift at most ±12% of width from frame centre
        # Horizontal bounds that keep the crop clear of every cam that overlaps it vertically (no
        # duplicate cam). Cams fully above/below the crop band need no horizontal exclusion.
        min_left, max_left = 0.0, float(width - crop_w)
        for cam in cams:
            cy0, cy1 = cam[1], cam[1] + cam[3]
            if cy1 <= crop_y or cy0 >= crop_y + crop_h:
                continue  # cam sits outside the crop's vertical band — no overlap possible
            fcx = cam[0] + cam[2] / 2.0
            if fcx > frame_center:  # cam on the right → keep crop fully left of it
                max_left = min(max_left, float(cam[0] - crop_w))
            else:  # cam on the left → keep crop fully right of it
                min_left = max(min_left, float(cam[0] + cam[2]))
        if max_left < min_left:  # cams too central to fully exclude → fall back to frame bounds
            min_left, max_left = 0.0, float(width - crop_w)

        small_w, small_h = MOTION_GRID_W, MOTION_GRID_H
        scale_x = width / small_w
        scale_y = height / small_h
        # Pre-compute each cam's region in the small motion grid so its movement never pulls the pan.
        mask_rects = [
            (
                max(0, int(c[0] / scale_x)),
                min(small_w, int((c[0] + c[2]) / scale_x)),
                max(0, int(c[1] / scale_y)),
                min(small_h, int((c[1] + c[3]) / scale_y)),
            )
            for c in cams
        ]

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            s_frame = max(0, min(int(start_time * fps), max(0, total - 1)))
            e_frame = max(s_frame + 1, min(int(end_time * fps), total))
            step = max(1, int(fps * 0.5))  # ~2 fps
            indices = list(range(s_frame, e_frame, step))[:120]  # bound cost

            track: list[dict] = []
            levels: list[float] = []
            prev = None
            cur_c = frame_center  # current crop centre
            smoothed_target = frame_center  # EMA-smoothed motion target
            cols = np.arange(small_w, dtype=np.float32)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(cv2.resize(frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    diff = cv2.GaussianBlur(cv2.absdiff(gray, prev), (21, 21), 0)
                    for mx0, mx1, my0, my1 in mask_rects:
                        diff[my0:my1, mx0:mx1] = 0
                    levels.append(float(diff.mean()))
                    col_motion = diff.sum(axis=0)
                    total_motion = float(col_motion.sum())
                    if total_motion > 0:
                        # Motion CENTROID (stable) rather than argmax (jumpy).
                        centroid = float((col_motion * cols).sum() / total_motion) * scale_x
                        smoothed_target += target_ema * (centroid - smoothed_target)
                    # Static-first: only move when the target leaves the deadzone, then glide slowly.
                    delta = smoothed_target - cur_c
                    if abs(delta) > deadzone:
                        cur_c += max(-max_vel, min(max_vel, delta))
                    cur_c = min(frame_center + max_dev, max(frame_center - max_dev, cur_c))
                prev = gray
                left = min(max_left, max(min_left, cur_c - crop_w / 2.0))
                track.append(
                    {
                        "timestamp": (idx - s_frame) / fps,
                        "crop_x": make_even(int(left)),
                        "crop_y": crop_y,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                    }
                )
        finally:
            cap.release()

        if not track:
            return [], fallback, 0.0
        mid = track[len(track) // 2]
        repr_box = {"x": mid["crop_x"], "y": crop_y, "w": crop_w, "h": crop_h}
        motion_level = float(np.mean(levels)) if levels else 0.0
        return track, repr_box, motion_level

    def _sample_frames(self, cap: object, start_time: float, end_time: float) -> list[np.ndarray]:
        """Sample up to ``sample_frames`` frames evenly across the window."""
        import cv2

        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        s_frame = max(0, min(int(start_time * fps), max(0, total - 1)))
        e_frame = max(s_frame + 1, min(int(end_time * fps), total))

        n = max(1, self.cfg.sample_frames)
        indices = [int(s_frame + i * (e_frame - s_frame) / n) for i in range(n)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return frames

    def _detect_regions(
        self, frames: list[np.ndarray], width: int, height: int
    ) -> tuple[list[dict], tuple[int, int, int, int] | None, int]:
        """Run YOLO across frames → persistent person clusters + a screen-inset box + max simultaneous person count.

        Returns:
            persons: Persistent YOLO person clusters (used for facecam picking / gameplay exclusion).
            screen_box: A transient screen-class inset (MediaShare candidate), or None.
            max_persons_in_frame: The maximum number of high-confidence person boxes detected
                in any single sampled frame.  This is the honest "how many people are on screen
                at once" — used for face_count / num_faces sizing.  Cross-frame clustering is
                intentionally NOT used here because adjacent people would merge into one cluster.
        """
        model = self._load_model()
        if model is None:
            return [], None, 0

        device = SystemUtils.resolve_device(self.cfg.device)
        person_dets: list[tuple[float, float, float, float]] = []
        screen_dets: list[tuple[float, float, float, float]] = []
        max_persons_in_frame: int = 0

        for frame in frames:
            results = model.predict(frame, verbose=False, device=device)
            frame_person_count = 0
            for res in results:
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    rect = (x1, y1, x2 - x1, y2 - y1)
                    if cls_id == COCO_CLASS_PERSON:
                        person_dets.append(rect)
                        if conf >= PERSON_COUNT_CONF_MIN:
                            frame_person_count += 1
                    elif cls_id in COCO_SCREEN_CLASSES:
                        screen_dets.append(rect)
            max_persons_in_frame = max(max_persons_in_frame, frame_person_count)

        persons = self._cluster_boxes(person_dets, width, height, len(frames))

        # Screen inset = a screen-class blob seen in a minority of frames (transient popup).
        # A "screen" overlapping a person box is the webcam in a border — discard it.
        screen_box = None
        if screen_dets:
            clustered = self._cluster_boxes(screen_dets, width, height, len(frames))
            for cand in clustered:
                box = cand["box"]
                if any(boxes_overlap(box, p["box"], 0.3) for p in persons):
                    continue
                screen_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                break

        return persons, screen_box, max_persons_in_frame

    def _cluster_boxes(
        self,
        boxes: list[tuple[float, float, float, float]],
        width: int,
        height: int,
        num_frames: int,
    ) -> list[dict]:
        """Cluster detections across frames into persistent regions (sorted by area desc)."""

        diag = float(np.sqrt(width**2 + height**2))
        thr = YOLO_CLUSTER_DIST_FRAC * diag
        clusters: list[list[tuple[float, float, float, float]]] = []

        def center(b: tuple[float, float, float, float]) -> tuple[float, float]:
            return (b[0] + b[2] / 2, b[1] + b[3] / 2)

        for box in boxes:
            placed = False
            for cl in clusters:
                avg = np.mean(cl, axis=0)
                c1, c2 = center(box), center(avg)
                if float(np.hypot(c1[0] - c2[0], c1[1] - c2[1])) < thr:
                    cl.append(box)
                    placed = True
                    break
            if not placed:
                clusters.append([box])

        out = []
        for cl in clusters:
            avg = np.mean(cl, axis=0)
            out.append(
                {
                    "box": (float(avg[0]), float(avg[1]), float(avg[2]), float(avg[3])),
                    "persistence": len(cl) / max(1, num_frames),
                    "area": float(avg[2] * avg[3]),
                }
            )
        out.sort(key=lambda d: d["area"], reverse=True)
        return out

    def _pick_facecam(
        self, persons: list[dict], width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Choose the streamer facecam among detected persons.

        Stream cams sit near a corner; talking-head cams are large and central. Score each
        persistent person by corner-proximity so a corner webcam wins over an in-frame
        game character, while a single central speaker is still selected.
        """

        candidates = [
            p for p in persons if p["persistence"] >= FACECAM_PICK_MIN_PERSISTENCE
        ] or persons
        if not candidates:
            return None
        if len(candidates) == 1:
            b = candidates[0]["box"]
            return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # With multiple persons, prefer cam-sized boxes (a webcam is a small fraction of the
        # frame) so a large/central game character is not mistaken for the facecam.
        frame_area = float(width * height)
        cam_sized = [p for p in candidates if p["area"] < 0.45 * frame_area]
        pool = cam_sized or candidates

        corners = [(0, 0), (width, 0), (0, height), (width, height)]
        diag = float(np.sqrt(width**2 + height**2))

        def corner_score(box: tuple[float, float, float, float]) -> float:
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            nearest = min(float(np.hypot(cx - cxn, cy - cyn)) for cxn, cyn in corners)
            return 1.0 - (nearest / diag)  # higher = closer to a corner

        best = max(pool, key=lambda p: corner_score(p["box"]) * p["persistence"])
        b = best["box"]
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def _motion_region(
        self,
        frames: list[np.ndarray],
        exclude_boxes: list | None,
        width: int,
        height: int,
        aspect: float = STACK2_PANEL_ASPECT,
    ) -> tuple[dict, float]:
        """Zoomed panel-aspect crop centred on the highest-motion (gameplay) region + motion level."""
        import cv2

        # Match _gameplay_pan's zoomed crop + bottom-cam-band bias so the static fallback frames the
        # same way as the panning path.
        cams = [b for b in (exclude_boxes or []) if b is not None]
        crop_w, crop_h, crop_y = self._gameplay_crop_geom(cams, width, height, aspect)

        small_w, small_h = MOTION_GRID_W, MOTION_GRID_H
        scale_x, scale_y = width / small_w, height / small_h
        motion = np.zeros((small_h, small_w), dtype=np.float32)
        prev = None
        for frame in frames:
            gray = cv2.cvtColor(cv2.resize(frame, (small_w, small_h)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                motion += cv2.absdiff(gray, prev).astype(np.float32)
            prev = gray

        for fx, fy, fw, fh in cams:  # zero every cam region so it never pulls the gameplay centroid
            motion[
                max(0, int(fy / scale_y)) : min(small_h, int((fy + fh) / scale_y)),
                max(0, int(fx / scale_x)) : min(small_w, int((fx + fw) / scale_x)),
            ] = 0.0

        motion_level = float(motion.mean())
        if motion.sum() <= 0.0:
            cx = width / 2.0
        else:
            cols = np.arange(small_w, dtype=np.float32)
            cx = float((motion.sum(axis=0) * cols).sum() / motion.sum()) * scale_x

        crop_x = make_even(int(max(0, min(cx - crop_w / 2.0, width - crop_w))))
        return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}, motion_level

    @staticmethod
    def _open_area_frac(persons: list[dict], width: int, height: int) -> float:
        """Fraction of a coarse 16×9 grid NOT covered by any person box.

        Small ≈ people fill the frame (talking-heads); large ≈ open screen space for a game. Feeds the
        gameplay gate in ``ContentTypeDetector.classify_from_analysis``.
        """
        grid_cols, grid_rows = OPEN_AREA_GRID_COLS, OPEN_AREA_GRID_ROWS
        cell_w = width / grid_cols
        cell_h = height / grid_rows
        covered: set[tuple[int, int]] = set()
        for p in persons:
            bx, by, bw, bh = p["box"]
            c0 = int(max(0, bx / cell_w))
            c1 = int(min(grid_cols, (bx + bw) / cell_w)) + 1
            r0 = int(max(0, by / cell_h))
            r1 = int(min(grid_rows, (by + bh) / cell_h)) + 1
            for r in range(r0, r1):
                for c in range(c0, c1):
                    covered.add((r, c))
        return 1.0 - len(covered) / (grid_cols * grid_rows)

    def _build_descriptor(self, a: dict) -> str:
        """Compact natural-language summary of the window for a (text-only) LLM."""
        w, h = a["video_width"], a["video_height"]
        parts = [f"{a['face_count']} person box(es) detected (may include on-screen characters)"]
        if a["facecam_box"]:
            parts.append(f"facecam at {self._where(a['facecam_box'], w, h)}")
        if a.get("collab_box"):
            parts.append(f"second facecam at {self._where(a['collab_box'], w, h)}")
        if a["mediashare_present"]:
            events = a.get("mediashare_events") or []
            if events:
                e0, e1 = events[0]
                parts.append(
                    f"Donation overlay reaction at ~{e0:.0f}-{e1:.0f}s (high-value moment)"
                )
            else:
                parts.append("Donation overlay (screen inset) visible")
        level = a["motion_level"]
        intensity = (
            "high"
            if level > MOTION_INTENSITY_HIGH
            else "moderate"
            if level > MOTION_INTENSITY_MODERATE
            else "low"
        )
        parts.append(f"{intensity} on-screen motion")
        return "; ".join(parts) + "."

    def _build_log_summary(self, a: dict, start_time: float, end_time: float) -> str:
        """Plain one-line sentence for the INFO log (user-readable).

        Intentionally separate from ``_build_descriptor`` which serves the LLM and keeps
        its technical phrasing (facecam, screen inset, etc.) intact.
        """
        w, h = a["video_width"], a["video_height"]
        n = a["face_count"]
        people = f"{n} {'person' if n == 1 else 'people'}"
        parts = [people]
        cam_idx = 1
        if a["facecam_box"]:
            parts.append(f"webcam {cam_idx} {self._where(a['facecam_box'], w, h)}")
            cam_idx += 1
        if a.get("collab_box"):
            parts.append(f"webcam {cam_idx} {self._where(a['collab_box'], w, h)}")
        if a["mediashare_present"]:
            events = a.get("mediashare_events") or []
            if events:
                e0, e1 = events[0]
                parts.append(f"donation alert around {e0:.0f}–{e1:.0f}s")
            else:
                parts.append("donation overlay visible")
        level = a["motion_level"]
        intensity = (
            "high"
            if level > MOTION_INTENSITY_HIGH
            else "moderate"
            if level > MOTION_INTENSITY_MODERATE
            else "low"
        )
        parts.append(f"{intensity} motion")
        return f"Scene [{start_time:.1f}–{end_time:.1f}s]: {', '.join(parts)}."

    def _where(self, box: tuple[int, int, int, int], width: int, height: int) -> str:
        """Human-readable screen position of a box (e.g. 'bottom-right')."""
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        vert = "top" if cy < height / 3 else "bottom" if cy > 2 * height / 3 else "centre"
        horiz = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "centre"
        return "centre" if vert == "centre" and horiz == "centre" else f"{vert}-{horiz}"
