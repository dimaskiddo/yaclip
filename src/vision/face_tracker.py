from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    COCO_CLASS_PERSON,
    ContentType,
    FACE_COUNT_MARGIN,
    FACE_LANDMARKER_MAX_FACES,
    SPEAKER_HOLD_SECONDS,
)
from src.core.workspace import MODELS_DIR


class FaceTracker:
    """Tracks faces and speakers in video clips to generate smooth crop coordinates."""

    def __init__(self) -> None:
        self.config = load_config()
        self.device = self.config.video_processing.device.lower()

    def track_clip(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        content_type: ContentType,
        person_count: int = 0,
    ) -> list[dict]:
        """Track face coordinates frame-by-frame and return crop boxes.

        Args:
            video_path: Path to the video file.
            start_time: Start time of the clip in seconds.
            end_time: End time of the clip in seconds.
            content_type: The content type of the video.
            person_count: Number of persons detected in the video by YOLO (from
                ``analysis["face_count"]``).  Used to size the FaceLandmarker capacity
                (PODCAST mode).  Defaults to 0 (treated as 1 after margin is applied).

        Returns:
            A list of crop dictionaries, one per frame of the clip.
            Each dict: {"timestamp": float, "crop_x": int, "crop_y": int, "crop_w": int, "crop_h": int}
        """
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video for tracking: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Clip boundaries
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        clip_frame_count = end_frame - start_frame

        logger.debug(
            f"Tracking faces in clip: {start_time:.2f}s to {end_time:.2f}s ({clip_frame_count} frames)"
        )

        # We will downsample detection to 5 fps to save CPU/GPU resources
        detection_interval = max(1, int(fps / 5))

        raw_detections: list[
            dict
        ] = []  # List of dicts: {"frame_idx": int, "faces": list}

        # PODCAST uses MediaPipe FaceLandmarker (lip-landmark speaker detection) so it can
        # identify and cut to the active speaker when 2+ faces are present.  All other types
        # use YOLO person detection (no lip landmarks needed; simpler crop).
        if content_type == ContentType.PODCAST:
            raw_detections = self._track_speaker_mesh(
                cap, start_frame, end_frame, detection_interval, width, height,
                person_count=person_count,
            )
        else:
            raw_detections = self._track_standard_detection(
                cap, start_frame, end_frame, detection_interval, width, height
            )

        cap.release()

        # Interpolate detections frame-by-frame
        crops = self._generate_smooth_crops(
            raw_detections, start_frame, end_frame, fps, width, height, content_type
        )
        return crops

    def _download_model(self, url: str, path: Path) -> None:
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading speaker-tracking model...")
        urllib.request.urlretrieve(url, str(path))

    def _track_standard_detection(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        interval: int,
        width: int,
        height: int,
    ) -> list[dict]:
        """Detect persons using YOLOv8 at interval steps.

        Detects COCO class 0 (person) boxes at each sample frame using the same YOLO
        model as VisualAnalyzer — no separate model download needed.
        """
        import cv2
        from src.core.utils import SystemUtils

        cfg = self.config.video_processing.region_detection
        model_path = MODELS_DIR / cfg.model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("Object detection unavailable — using center crops for this clip.")
            return []

        target = str(model_path) if model_path.exists() else cfg.model_name

        logger.info("Loading object detection model...")
        model = YOLO(target)
        device = SystemUtils.resolve_device(cfg.device)

        detections = []
        for idx in range(start_frame, end_frame, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False, device=device)
            faces = []
            for res in results:
                for box in res.boxes:
                    if int(box.cls[0]) != COCO_CLASS_PERSON:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    faces.append(
                        {
                            "box": (
                                max(0.0, x1),
                                max(0.0, y1),
                                x2 - x1,
                                y2 - y1,
                            ),
                            "score": float(box.conf[0]),
                        }
                    )
            detections.append({"frame_idx": idx, "faces": faces})

        return detections

    def _track_speaker_mesh(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        interval: int,
        width: int,
        height: int,
        person_count: int = 0,
    ) -> list[dict]:
        """Detect faces and capture lip landmarks using MediaPipe FaceLandmarker Tasks API at interval steps.

        Used for PODCAST content so the crop cuts to the active speaker when 2+ faces are
        present (EMA of lip distance identifies who is talking).  A single-face podcast
        simply tracks that face; the speaker-switching logic in _generate_smooth_crops
        naturally no-ops in that case.

        The FaceLandmarker ``num_faces`` capacity is sized from the YOLO person count passed
        in via ``person_count`` (from ``analysis["face_count"]``), adding ``FACE_COUNT_MARGIN``
        headroom so partially-visible or angled faces are never missed.
        """
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = MODELS_DIR / "face_landmarker.task"
        if not model_path.exists():
            self._download_model(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_path,
            )

        num_faces = max(1, min(person_count + FACE_COUNT_MARGIN, FACE_LANDMARKER_MAX_FACES))
        logger.info(f"Speaker tracking: {person_count} person(s) in video, tracking up to {num_faces} speakers.")

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
        )

        detections = []
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            for idx in range(start_frame, end_frame, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect(mp_image)
                faces = []

                if res.face_landmarks:
                    for landmarks in res.face_landmarks:
                        # Extract bounding box from all landmarks
                        xs = [lm.x for lm in landmarks]
                        ys = [lm.y for lm in landmarks]
                        xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
                        ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))

                        box = (
                            xmin * width,
                            ymin * height,
                            (xmax - xmin) * width,
                            (ymax - ymin) * height,
                        )

                        # Extract inner lip landmarks for speaker activity
                        # Landmark 13 is inner upper lip, 14 is inner lower lip
                        lip_top = landmarks[13]
                        lip_bot = landmarks[14]
                        lip_dist = np.sqrt(
                            (lip_top.x - lip_bot.x) ** 2 + (lip_top.y - lip_bot.y) ** 2
                        )

                        faces.append({"box": box, "lip_distance": float(lip_dist)})
                detections.append({"frame_idx": idx, "faces": faces})

        return detections

    def _generate_smooth_crops(
        self,
        detections: list[dict],
        start_frame: int,
        end_frame: int,
        fps: float,
        width: int,
        height: int,
        content_type: ContentType,
    ) -> list[dict]:
        """Smooth raw face bounding boxes and output dynamic crop boxes frame-by-frame."""
        import numpy as np
        if not detections:
            # Fallback to static center crops
            return self._generate_fallback_crops(
                start_frame, end_frame, fps, width, height
            )

        # Target crop dimensions depend on content type.
        # Facecam crop is pre-shaped to the destination panel aspect so the downstream
        # scale (1080x960 for 2-stack, 1080x640 for collab) adds zero distortion.
        from src.core.constants import STACK2_PANEL_ASPECT, STACK3_PANEL_ASPECT

        if content_type == ContentType.PODCAST:
            # 9:16 full-height vertical pillar
            crop_w = int(height * (9.0 / 16.0))
            crop_h = height
        elif content_type == ContentType.GAMING_COLLAB:
            # Mode C facecam panel is 1080x640 (aspect 1.6875)
            crop_h = int(height * 0.4)
            crop_w = int(crop_h * STACK3_PANEL_ASPECT)
        else:
            # Mode B 2-stack facecam panel is 1080x960 (aspect 1.125) for GAMING_SOLO/JUST_CHAT.
            # Crop ~half the source height around the tracked face → crop-fill zoom on the cam.
            crop_h = int(height * 0.5)
            crop_w = int(crop_h * STACK2_PANEL_ASPECT)

        # Ensure crop fits within source frame
        crop_w = max(100, min(crop_w, width))
        crop_h = max(100, min(crop_h, height))
        if crop_w % 2 != 0:
            crop_w -= 1
        if crop_h % 2 != 0:
            crop_h -= 1

        # We will compute the target face center for each step frame
        step_centers = []

        # For PODCAST mode, we also track speaker activity when 2+ faces are present.
        # EMA of per-face lip distance — responsive from frame 1, no warm-up window.
        active_speaker_idx = 0
        speaker_lip_ema: dict[int, float] = {}
        _LIP_EMA_ALPHA = 0.35  # higher = faster reaction to new lip movement

        # We need to map face bounding boxes to consistent speaker IDs
        face_ids: list[tuple[float, float]] = []  # list of average x,y centers

        for step in detections:
            frame_idx = step["frame_idx"]
            faces = step["faces"]

            if not faces:
                # Keep previous center if none found
                step_centers.append(
                    {
                        "frame_idx": frame_idx,
                        "target_x": width // 2,
                        "target_y": height // 2,
                        "speaker_id": -1,
                    }
                )
                continue

            # Map faces to consistent IDs
            mapped_faces = []
            for face in faces:
                box = face["box"]
                face_center = (box[0] + box[2] / 2, box[1] + box[3] / 2)

                # Find matching face ID
                face_id = -1
                for fid, pos in enumerate(face_ids):
                    dist = np.sqrt(
                        (face_center[0] - pos[0]) ** 2 + (face_center[1] - pos[1]) ** 2
                    )
                    if dist < 0.12 * width:
                        face_id = fid
                        # Update rolling position
                        face_ids[fid] = (
                            0.8 * pos[0] + 0.2 * face_center[0],
                            0.8 * pos[1] + 0.2 * face_center[1],
                        )
                        break

                if face_id == -1:
                    face_id = len(face_ids)
                    face_ids.append(face_center)

                mapped_faces.append((face_id, face))

            # Determine who is speaking in PODCAST mode (2+ faces: follow active speaker)
            if content_type == ContentType.PODCAST:
                # Update per-face EMA of lip distance
                visible_fids = set()
                for fid, face in mapped_faces:
                    visible_fids.add(fid)
                    lip_dist = face.get("lip_distance", 0.0)
                    if fid not in speaker_lip_ema:
                        speaker_lip_ema[fid] = lip_dist
                    else:
                        speaker_lip_ema[fid] = (
                            _LIP_EMA_ALPHA * lip_dist
                            + (1 - _LIP_EMA_ALPHA) * speaker_lip_ema[fid]
                        )

                # Active speaker = highest current EMA among visible faces
                max_activity = -1.0
                for fid, ema_val in speaker_lip_ema.items():
                    if fid in visible_fids and ema_val > max_activity:
                        max_activity = ema_val
                        active_speaker_idx = fid

                # Find the active speaker face in current frame
                target_face = None
                for fid, face in mapped_faces:
                    if fid == active_speaker_idx:
                        target_face = face
                        break

                # If active speaker not visible in this frame, use first face
                if not target_face and mapped_faces:
                    target_face = mapped_faces[0][1]
            else:
                # For non-interview, just track the first face (usually the only streamer/host)
                target_face = mapped_faces[0][1]

            if target_face:
                box = target_face["box"]
                fx = box[0] + box[2] / 2
                fy = box[1] + box[3] / 2
                step_centers.append(
                    {
                        "frame_idx": frame_idx,
                        "target_x": int(fx),
                        "target_y": int(fy),
                        "speaker_id": active_speaker_idx
                        if content_type == ContentType.PODCAST
                        else 0,
                    }
                )

        # ── Static-cut crop: no camera pan, no EMA glide ─────────────────────────────────
        # Build committed speaker segments with debounce:
        #   - A candidate speaker must hold for >= SPEAKER_HOLD_SECONDS before we commit.
        #   - Each committed segment owns a FIXED crop center (median of face-box centers
        #     in that segment).  The crop is constant for the whole segment; switching to
        #     a new speaker is a hard frame cut — zero viewer dizziness.
        # For non-PODCAST (single face tracked) the step_centers list all share speaker_id 0
        # → one segment → fully static center crop.
        import numpy as np

        hold_steps = max(1, int(SPEAKER_HOLD_SECONDS * fps / max(1, int(fps / 5))))

        # --- Phase 1: debounced segment boundaries ----------------------------------------
        # Each entry: {"speaker_id", "frame_idx", "target_x", "target_y"}
        segments: list[dict] = []  # {"speaker_id", "start_f", "end_f", "cx", "cy"}

        if not step_centers:
            # No detections at all → static center fallback (handled by caller).
            return self._generate_fallback_crops(start_frame, end_frame, fps, width, height)

        committed_sid = step_centers[0]["speaker_id"]
        pending_sid = committed_sid
        pending_count = 0
        seg_xs: list[float] = []
        seg_ys: list[float] = []
        seg_start = start_frame

        def _close_segment(sid: int, xs: list, ys: list, sf: int, ef: int) -> dict:
            cx = float(np.median(xs)) if xs else width / 2.0
            cy = float(np.median(ys)) if ys else height / 2.0
            # Clamp center so the crop box stays inside the frame.
            cx = max(crop_w / 2.0, min(cx, width - crop_w / 2.0))
            cy = max(crop_h / 2.0, min(cy, height - crop_h / 2.0))
            return {"speaker_id": sid, "start_f": sf, "end_f": ef, "cx": cx, "cy": cy}

        for sc in step_centers:
            sid = sc["speaker_id"]
            tx = float(sc["target_x"])
            ty = float(sc["target_y"])
            if sid == committed_sid:
                # Still the committed speaker — accumulate.
                pending_sid = sid
                pending_count = 0
                seg_xs.append(tx)
                seg_ys.append(ty)
            else:
                if sid == pending_sid:
                    pending_count += 1
                else:
                    pending_sid = sid
                    pending_count = 1
                # Commit the switch only after hold threshold is met.
                if pending_count >= hold_steps:
                    # Close current segment up to this step's frame.
                    segments.append(
                        _close_segment(committed_sid, seg_xs, seg_ys, seg_start, sc["frame_idx"])
                    )
                    committed_sid = sid
                    pending_count = 0
                    seg_start = sc["frame_idx"]
                    seg_xs = [tx]
                    seg_ys = [ty]
                # While waiting for hold, still accumulate coords for pending speaker
                # so if it commits we already have a good median.

        # Close the final open segment.
        segments.append(
            _close_segment(committed_sid, seg_xs, seg_ys, seg_start, end_frame)
        )

        # --- Phase 2: emit one static crop box per frame ----------------------------------
        # Build frame→segment index for O(n) lookup.
        seg_by_frame: list[dict] = []  # ordered list (start_f, end_f, cx, cy)
        for seg in segments:
            seg_by_frame.append(seg)

        crops = []
        seg_iter = iter(seg_by_frame)
        cur_seg = next(seg_iter)

        def _crop_from_center(cx: float, cy: float) -> tuple[int, int]:
            xm = int(cx - crop_w / 2)
            xm = max(0, min(xm, width - crop_w))
            ym = int(cy - crop_h / 2)
            ym = max(0, min(ym, height - crop_h))
            return xm, ym

        for f_idx in range(start_frame, end_frame):
            # Advance segment when this frame is past the current one's end.
            while f_idx >= cur_seg["end_f"]:
                try:
                    cur_seg = next(seg_iter)
                except StopIteration:
                    break
            x_min, y_min = _crop_from_center(cur_seg["cx"], cur_seg["cy"])
            crops.append(
                {
                    "timestamp": (f_idx - start_frame) / fps,
                    "crop_x": x_min,
                    "crop_y": y_min,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                }
            )

        return crops

    def _generate_fallback_crops(
        self, start_frame: int, end_frame: int, fps: float, width: int, height: int
    ) -> list[dict]:
        """Generate static center 9:16 crop coordinates."""
        crop_w = int(height * (9.0 / 16.0))
        crop_w = max(100, min(crop_w, width))
        if crop_w % 2 != 0:
            crop_w -= 1

        x_min = (width - crop_w) // 2
        crops = []
        for idx in range(start_frame, end_frame):
            crops.append(
                {
                    "timestamp": (idx - start_frame) / fps,
                    "crop_x": x_min,
                    "crop_y": 0,
                    "crop_w": crop_w,
                    "crop_h": height,
                }
            )
        return crops
