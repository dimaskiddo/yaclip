from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pathlib import Path
from loguru import logger

from src.core.config import load_config
from src.core.constants import ContentType
from src.core.workspace import MODELS_DIR


class ContentTypeDetector:
    """Detects the ContentType of a video based on frame sampling and vision heuristics."""

    def __init__(self) -> None:
        self.config = load_config()

    def detect_content_type(self, video_path: Path) -> ContentType:
        """Analyze the video and determine its content type.

        Args:
            video_path: Path to the downloaded video file.

        Returns:
            The detected ContentType enum value.
        """
        # 1. Config override check
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

        # 2. Sample frames from video
        logger.info(
            f"Analysing video to detect content type: {video_path.name}"
        )
        sampled = self._sample_frames(video_path, num_samples=25)
        if not sampled:
            logger.warning("Could not read any video frames. Defaulting to PODCAST type.")
            return ContentType.PODCAST

        frames = [f[1] for f in sampled]
        h, w, _ = frames[0].shape

        # 3. Detect Gaming HUD. A "Gaming" YouTube category lowers the bar (borderline HUD).
        hud_score = self._compute_hud_score(frames)
        gaming_hint = self._metadata_gaming_hint(video_path)
        hud_threshold = 0.0015 if gaming_hint else 0.003
        has_hud = hud_score >= hud_threshold
        logger.debug(
            f"Game HUD scan — score: {hud_score:.4f}, detected: {has_hud}, gaming category hint: {gaming_hint}"
        )

        # 4. Count persistent face regions
        faces_per_frame = self._detect_faces_across_frames(frames)
        num_faces = self._count_persistent_faces(faces_per_frame, w, h)
        logger.info(f"Detected {num_faces} persistent face(s) in video.")

        # 5. Check for donation overlays (colour heuristic + YOLO screen-inset signal)
        has_donation_alerts = self._check_donation_overlays(frames)
        if (
            self.config.video_processing.preserve_donation_overlays
            and self.config.video_processing.region_detection.enabled
            and sampled
        ):
            has_donation_alerts = has_donation_alerts or self._region_mediashare_signal(
                video_path, sampled[0][0], sampled[-1][0]
            )
        logger.info(f"Donation alert detection result: {has_donation_alerts}")

        # 6. Apply Decision Tree
        confidence = 0.8  # Base confidence
        detected_type = ContentType.PODCAST

        # A visible gaming HUD OR a "Gaming" category (gaming_hint) routes to the gaming branch — the
        # metadata category is a strong, reliable signal, so a sub-threshold HUD no longer drops a
        # genuine gaming stream to PODCAST. Face count then splits SOLO vs COLLAB.
        is_gaming = has_hud or gaming_hint
        if is_gaming:
            # MediaPipe misses small corner webcams, so confirm SOLO vs COLLAB with the YOLO facecam
            # detector (robust for corner cams). Take the max so either signal can establish a collab.
            cam_count = num_faces
            if self.config.video_processing.region_detection.enabled:
                try:
                    from src.vision.visual_analyzer import VisualAnalyzer

                    analyzer = VisualAnalyzer()
                    try:
                        cam_count = max(cam_count, len(analyzer.detect_facecams(video_path)))
                    finally:
                        analyzer.release()
                except Exception as e:
                    logger.warning(f"Webcam count via object detection failed: {e}")
            logger.info(f"Total webcams detected for gaming content: {cam_count}")
            detected_type = (
                ContentType.GAMING_COLLAB if cam_count >= 2 else ContentType.GAMING_SOLO
            )
        else:
            if num_faces >= 2:
                detected_type = ContentType.INTERVIEW
            elif has_donation_alerts:
                detected_type = ContentType.JUST_CHAT
            else:
                detected_type = ContentType.PODCAST

        # Check threshold
        threshold = self.config.video_processing.detection_confidence_threshold
        if confidence < threshold:
            logger.warning(
                f"Detection confidence {confidence:.2f} is below the required threshold {threshold:.2f}. "
                f"Defaulting to PODCAST type."
            )
            return ContentType.PODCAST

        logger.info(f"Video type detected: {detected_type}")
        return detected_type

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
        """Detect static graphic UI elements (HUDs) by temporal variance & average frame spatial gradient."""
        import cv2
        import numpy as np

        if not frames:
            return 0.0

        # Resize frames to 128x128 for memory and speed efficiency
        small_grays = []
        for f in frames:
            small = cv2.resize(f, (128, 128))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            small_grays.append(gray)

        stack = np.stack(small_grays, axis=0)
        temp_std = np.std(stack, axis=0)

        # Spatial gradient of average frame
        avg_frame = np.mean(stack, axis=0).astype(np.uint8)
        sobelx = cv2.Sobel(avg_frame, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(avg_frame, cv2.CV_64F, 0, 1, ksize=3)
        spatial_grad = np.sqrt(sobelx**2 + sobely**2)

        # HUD pixels have very low temporal std (constant display) and high spatial gradient (sharp bounds)
        hud_mask = (temp_std < 8.0) & (spatial_grad > 18.0)

        # Exclude edge margins (outer 5%) to ignore letterboxing artifacts
        margin = 6
        hud_mask[:margin, :] = False
        hud_mask[-margin:, :] = False
        hud_mask[:, :margin] = False
        hud_mask[:, -margin:] = False

        hud_pixel_ratio = np.sum(hud_mask) / (128 * 128)
        return float(hud_pixel_ratio)

    def _detect_faces_across_frames(
        self, frames: list[np.ndarray]
    ) -> list[list[tuple[float, float, float, float]]]:
        """Detect face rectangles per frame using MediaPipe FaceDetector Tasks API."""
        import urllib.request

        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = MODELS_DIR / "face_detector.tflite"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            logger.info(f"Downloading MediaPipe FaceDetector model: {url}")
            urllib.request.urlretrieve(url, str(model_path))

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options, running_mode=vision.RunningMode.IMAGE
        )

        all_faces = []
        with vision.FaceDetector.create_from_options(options) as detector:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = detector.detect(mp_image)
                faces = []
                if res.detections:
                    for det in res.detections:
                        bbox = det.bounding_box
                        xmin = max(0, bbox.origin_x)
                        ymin = max(0, bbox.origin_y)
                        w = bbox.width
                        h = bbox.height
                        faces.append((float(xmin), float(ymin), float(w), float(h)))
                all_faces.append(faces)

        return all_faces

    def _count_persistent_faces(
        self,
        faces_per_frame: list[list[tuple[float, float, float, float]]],
        img_w: int,
        img_h: int,
    ) -> int:
        """Count distinct persistent facecams across frames.

        A second (collab) face is only counted when it is a *genuine, persistent* cam:
        present across enough frames, comparable in size to the primary face, and spatially
        separated from it. This prevents a transient MediaShare face or a small in-game
        character face from flipping GAMING_SOLO into GAMING_COLLAB.
        """
        import numpy as np
        clusters: list[list[tuple[float, float, float, float]]] = []

        def box_distance(
            b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]
        ) -> float:
            c1 = (b1[0] + b1[2] / 2, b1[1] + b1[3] / 2)
            c2 = (b2[0] + b2[2] / 2, b2[1] + b2[3] / 2)
            return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))

        diag = float(np.sqrt(img_w**2 + img_h**2))
        match_threshold = 0.15 * diag

        for frame_faces in faces_per_frame:
            for face in frame_faces:
                matched = False
                for cluster in clusters:
                    avg_box = np.mean(cluster, axis=0)
                    if box_distance(face, avg_box) < match_threshold:
                        cluster.append(face)
                        matched = True
                        break
                if not matched:
                    clusters.append([face])

        num_sampled_frames = max(1, len(faces_per_frame))
        if not any(faces_per_frame):
            logger.info("No faces detected in any sampled video frames.")
            return 0

        # Summarise each cluster: persistence ratio, average box, area.
        summaries = []
        for cluster in clusters:
            avg_box = np.mean(cluster, axis=0)
            persistence = len(cluster) / num_sampled_frames
            area = float(avg_box[2] * avg_box[3])
            center = (avg_box[0] + avg_box[2] / 2, avg_box[1] + avg_box[3] / 2)
            summaries.append(
                {"persistence": persistence, "area": area, "center": center, "box": avg_box}
            )

        # Persistent cams must appear in >= 30% of sampled frames. A 2nd collab cam (streamer who
        # sometimes looks away) dips below a stricter floor; the size + separation guards below still
        # reject transient in-game character faces, so this stays robust against false collab.
        persistent = [s for s in summaries if s["persistence"] >= 0.30]
        persistent.sort(key=lambda s: s["area"], reverse=True)

        for i, s in enumerate(persistent):
            logger.debug(
                f"Face group {i + 1}: visible in {s['persistence']*100:.0f}% of frames, "
                f"size={s['area']:.0f}px, position=({s['center'][0]:.0f},{s['center'][1]:.0f})"
            )

        if not persistent:
            return 1  # faces seen but none persistent → treat as a single subject

        primary = persistent[0]
        count = 1
        for s in persistent[1:]:
            # Comparable size (>= 40% of primary cam area) and clearly separated → a real cam.
            size_ok = s["area"] >= 0.40 * primary["area"]
            sep = float(
                np.sqrt(
                    (s["center"][0] - primary["center"][0]) ** 2
                    + (s["center"][1] - primary["center"][1]) ** 2
                )
            )
            sep_ok = sep > 0.15 * diag
            if size_ok and sep_ok:
                count += 1

        return count

    def _metadata_gaming_hint(self, video_path: Path) -> bool:
        """True if the saved YouTube metadata marks this video as Gaming content."""
        import json

        from src.core.workspace import SUBTITLES_DIR

        meta_path = SUBTITLES_DIR / f"{video_path.stem}_metadata.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        categories = " ".join(meta.get("categories") or []).lower()
        tags = " ".join(meta.get("tags") or []).lower()
        return "gaming" in categories or "game" in tags

    def _region_mediashare_signal(
        self, video_path: Path, start_time: float, end_time: float
    ) -> bool:
        """Use the YOLO VisualAnalyzer to confirm a MediaShare/screen inset is present."""
        try:
            from src.vision.visual_analyzer import VisualAnalyzer

            analyzer = VisualAnalyzer()
            try:
                res = analyzer.analyze_window(video_path, start_time, end_time)
            finally:
                analyzer.release()
            return bool(res.get("mediashare_present"))
        except Exception as e:
            logger.warning(f"Donation alert check via object detection failed: {e}")
            return False

    def _check_donation_overlays(self, frames: list[np.ndarray]) -> bool:
        """Sample for donation alert popup signatures (bright transient overlay rectangles)."""
        import cv2
        import numpy as np

        # Check if any frames show sudden high-contrast rectangular contours in expected alert centers/corners.
        # Donation alert UIs typically use saturated colours (red/pink, bright orange) in clean card shapes.
        # We look for temporary colorful contours that are only present in a few frames.
        if len(frames) < 3:
            return False

        # Look for temporary structures: frames where local pixel differences are high compared to average
        # but structured as clean rectangles.
        # As a simplified high-performance heuristic:
        # Check standard overlay regions (lower-third center or corners) for transient bright saturation.
        # HSV threshold masks cover two common alert colour families: red/pink and bright orange.
        matched_alerts = 0
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Red/pink alert range: H: 170-180 (or 0-10), S: 100-255, V: 100-255
            # Bright orange alert range: H: 5-25, S: 100-255, V: 100-255
            mask_red1 = cv2.inRange(
                hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
            )
            mask_red2 = cv2.inRange(
                hsv, np.array([170, 100, 100]), np.array([180, 255, 255])
            )
            mask_orange = cv2.inRange(
                hsv, np.array([5, 100, 100]), np.array([25, 255, 255])
            )

            combined = mask_red1 | mask_red2 | mask_orange

            # Find contours in overlay mask
            contours, _ = cv2.findContours(
                combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2000 < area < 50000:  # Sensible size for an alert box on 1080p
                    x, y, w, h_box = cv2.boundingRect(cnt)
                    aspect = w / h_box
                    if 1.2 < aspect < 4.0:  # Typical rectangular alert box ratio
                        matched_alerts += 1
                        break

        # If we catch alert boxes in at least 1 but not all frames (they are transient), it's highly likely!
        return 0 < matched_alerts < (len(frames) * 0.6)
