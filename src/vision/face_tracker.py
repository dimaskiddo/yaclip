from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.constants import ContentType
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
    ) -> list[dict]:
        """Track face coordinates frame-by-frame and return crop boxes.

        Args:
            video_path: Path to the video file.
            start_time: Start time of the clip in seconds.
            end_time: End time of the clip in seconds.
            content_type: The content type of the video.

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

        logger.info(
            f"Tracking faces in clip: {start_time:.2f}s to {end_time:.2f}s ({clip_frame_count} frames)"
        )

        # We will downsample detection to 5 fps to save CPU/GPU resources
        detection_interval = max(1, int(fps / 5))

        raw_detections: list[
            dict
        ] = []  # List of dicts: {"frame_idx": int, "faces": list}

        # Initialize MediaPipe models based on ContentType
        # We use FaceMesh for Interview (speaker detection), and FaceDetection for others (faster)
        if content_type == ContentType.INTERVIEW:
            raw_detections = self._track_interview_mesh(
                cap, start_frame, end_frame, detection_interval, width, height
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
        logger.info(f"Downloading MediaPipe model from {url} to {path}")
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
        """Detect faces using standard MediaPipe Face Detector Tasks API at interval steps."""
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = MODELS_DIR / "face_detector.tflite"
        if not model_path.exists():
            self._download_model(
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                model_path,
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options, running_mode=vision.RunningMode.IMAGE
        )

        detections = []
        with vision.FaceDetector.create_from_options(options) as detector:
            for idx in range(start_frame, end_frame, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_image)
                faces = []
                if res.detections:
                    for det in res.detections:
                        bbox = det.bounding_box
                        xmin = max(0, bbox.origin_x)
                        ymin = max(0, bbox.origin_y)
                        w_box = bbox.width
                        h_box = bbox.height

                        faces.append(
                            {
                                "box": (
                                    float(xmin),
                                    float(ymin),
                                    float(w_box),
                                    float(h_box),
                                ),
                                "score": det.categories[0].score
                                if det.categories
                                else 1.0,
                            }
                        )
                detections.append({"frame_idx": idx, "faces": faces})

        return detections

    def _track_interview_mesh(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        interval: int,
        width: int,
        height: int,
    ) -> list[dict]:
        """Detect faces and capture lip landmarks using MediaPipe FaceLandmarker Tasks API at interval steps."""
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

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=3,
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

        if content_type in (ContentType.PODCAST, ContentType.INTERVIEW):
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

        # For INTERVIEW mode, we also track speaker activity
        active_speaker_idx = 0
        speaker_lip_history: dict[int, list[float]] = {}

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

            # Determine who is speaking in INTERVIEW mode
            if content_type == ContentType.INTERVIEW:
                # Update lip distance history
                for fid, face in mapped_faces:
                    if fid not in speaker_lip_history:
                        speaker_lip_history[fid] = []
                    speaker_lip_history[fid].append(face.get("lip_distance", 0.0))
                    # Keep last 15 samples (approx 3 seconds)
                    if len(speaker_lip_history[fid]) > 15:
                        speaker_lip_history[fid].pop(0)

                # Active speaker has the highest variance of lip distance over the window
                max_variance = -1.0
                for fid, history in speaker_lip_history.items():
                    if len(history) >= 3:
                        var = float(np.var(history))
                        if var > max_variance:
                            max_variance = var
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
                        if content_type == ContentType.INTERVIEW
                        else 0,
                    }
                )

        # Now interpolate frame-by-frame center points
        crops = []

        # Initial smoothed center
        if step_centers:
            curr_x = float(step_centers[0]["target_x"])
            curr_y = float(step_centers[0]["target_y"])
        else:
            curr_x = width / 2
            curr_y = height / 2

        # Panning speed coefficient
        alpha = 0.08  # Lower means smoother, more lag. Higher means faster transition.

        # Track transitions for Interview switches
        last_speaker = -1

        for f_idx in range(start_frame, end_frame):
            # Find closest past and future step center
            past_step = None
            future_step = None
            for sc in step_centers:
                if sc["frame_idx"] <= f_idx:
                    past_step = sc
                elif sc["frame_idx"] > f_idx and not future_step:
                    future_step = sc
                    break

            # Interpolate target coordinate
            if past_step and future_step:
                # Simple linear interpolation between detection steps
                ratio = (f_idx - past_step["frame_idx"]) / (
                    future_step["frame_idx"] - past_step["frame_idx"]
                )
                target_x = past_step["target_x"] + ratio * (
                    future_step["target_x"] - past_step["target_x"]
                )
                target_y = past_step["target_y"] + ratio * (
                    future_step["target_y"] - past_step["target_y"]
                )
                speaker_id = past_step["speaker_id"]
            elif past_step:
                target_x = past_step["target_x"]
                target_y = past_step["target_y"]
                speaker_id = past_step["speaker_id"]
            else:
                target_x = width / 2
                target_y = height / 2
                speaker_id = -1

            # Adjust panning speed on speaker transition
            if (
                content_type == ContentType.INTERVIEW
                and speaker_id != last_speaker
                and last_speaker != -1
            ):
                # Faster alpha during transition (0.3s transition is approx 10 frames at 30fps)
                alpha_val = 0.22
            else:
                alpha_val = alpha

            if speaker_id != -1:
                last_speaker = speaker_id

            # Apply EMA smoothing to camera pan
            curr_x = alpha_val * target_x + (1 - alpha_val) * curr_x
            curr_y = alpha_val * target_y + (1 - alpha_val) * curr_y

            # Keep crop window inside image boundaries
            x_min = int(curr_x - crop_w / 2)
            x_min = max(0, min(x_min, width - crop_w))

            y_min = int(curr_y - crop_h / 2)
            y_min = max(0, min(y_min, height - crop_h))

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
