from __future__ import annotations

import urllib.request
import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    import cv2

from src.core.config import load_config
from src.core.constants import (
    AV_SYNC_WINDOW_SECONDS,
    COCO_CLASS_PERSON,
    COHERENCE_MIN,
    ContentType,
    FACE_COUNT_MARGIN,
    FACE_LANDMARKER_MAX_FACES,
    GROUP_FRAMING_FIT_FACTOR,
    GROUP_MAX_GAP_FACTOR,
    GROUP_SPEAKER_ID,
    HAAR_DOWNSCALE,
    HAAR_MIN_NEIGHBORS,
    HAAR_SCALE_FACTOR,
    HEADROOM_FACTOR,
    IOU_MATCH_MIN,
    LIP_ACTIVITY_MIN,
    LIP_ACTIVITY_WINDOW_SECONDS,
    MAR_VERTICAL_PAIRS,
    MAR_WIDTH_PAIR,
    MIN_SHOT_SECONDS,
    PAN_SMOOTHING_FACTOR,
    PODCAST_DETECTION_FPS,
    SPEAKER_HOLD_SECONDS,
    SPEAKER_SWITCH_MARGIN,
    STACK2_PANEL_ASPECT,
    STACK3_PANEL_ASPECT,
    VOICE_ACTIVITY_FLOOR_FACTOR,
)
from src.core.utils import SystemUtils
from src.media.energy import AudioEnergyAnalyzer
from src.core.workspace import MODELS_DIR


class FaceTracker:
    """Tracks faces and speakers in video clips to generate smooth crop coordinates."""

    def __init__(self) -> None:
        self.config = load_config()
        self.device = self.config.video_processing.device.lower()
        # Low-spec opt-in: Haar largest-face tracking instead of MediaPipe + audio (PODCAST only).
        self.fast_mode = bool(self.config.video_processing.fast_mode)

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

        # Fast mode (opt-in, PODCAST only): skip MediaPipe + audio entirely and use a fast Haar
        # largest-face crop.  Other content types are unaffected (they never reach the PODCAST path).
        fast = self.fast_mode and content_type == ContentType.PODCAST

        # Downsample detection to save CPU/GPU.  PODCAST samples faster so active-speaker
        # detection can resolve syllable-rate (~3-5 Hz) mouth movement; other types use 5 fps.
        sample_fps = PODCAST_DETECTION_FPS if (content_type == ContentType.PODCAST and not fast) else 5
        detection_interval = max(1, int(fps / sample_fps))

        raw_detections: list[
            dict
        ] = []  # List of dicts: {"frame_idx": int, "faces": list}

        # PODCAST uses MediaPipe FaceLandmarker (lip-landmark speaker detection) so it can
        # identify and cut to the active speaker when 2+ faces are present.  All other types
        # use YOLO person detection (no lip landmarks needed; simpler crop).
        audio_rms: list[float] = []
        if fast:
            raw_detections = self._track_haar_fast(
                cap, start_frame, end_frame, detection_interval, width, height
            )
        elif content_type == ContentType.PODCAST:
            raw_detections = self._track_speaker_mesh(
                cap, start_frame, end_frame, detection_interval, width, height,
                person_count=person_count,
            )
            # Audio-visual sync: per-detection-step loudness envelope for the clip window.
            # Aligns 1:1 with the detection steps (step_seconds = detection_interval / fps).
            try:
                audio_rms = AudioEnergyAnalyzer().rms_envelope(
                    str(video_path), start_time, end_time, detection_interval / fps
                )
            except Exception as e:
                logger.warning(f"Audio envelope unavailable, using visual-only speaker detection: {e}")
                audio_rms = []
        else:
            raw_detections = self._track_standard_detection(
                cap, start_frame, end_frame, detection_interval, width, height
            )

        cap.release()

        # Interpolate detections frame-by-frame
        crops = self._generate_smooth_crops(
            raw_detections, start_frame, end_frame, fps, width, height, content_type,
            audio_rms=audio_rms, detection_interval=detection_interval, fast_mode=fast,
        )
        return crops

    def _download_model(self, url: str, path: Path) -> None:
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

    def _track_haar_fast(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        interval: int,
        width: int,
        height: int,
    ) -> list[dict]:
        """Fast CPU-only PODCAST tracking via OpenCV Haar cascade (opt-in ``fast_mode``).

        Detects frontal faces on a downscaled grayscale frame and keeps the largest one — no
        MediaPipe, no audio, no active-speaker logic.  Returns the standard detection shape so the
        single-face crop path can consume it.  Trades multi-speaker accuracy for ~10x speed.
        """
        import cv2

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        inv = 1.0 / HAAR_DOWNSCALE
        detections = []
        for idx in range(start_frame, end_frame, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (0, 0), fx=HAAR_DOWNSCALE, fy=HAAR_DOWNSCALE)
            found = cascade.detectMultiScale(
                small, scaleFactor=HAAR_SCALE_FACTOR, minNeighbors=HAAR_MIN_NEIGHBORS
            )

            faces = []
            if len(found) > 0:
                # Keep the largest face and scale its box back to full resolution.
                x, y, w, h = max(found, key=lambda b: int(b[2]) * int(b[3]))
                faces.append({"box": (x * inv, y * inv, w * inv, h * inv)})
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

                        # Mouth-Aspect-Ratio (MAR) for speaker activity: mean vertical inner-lip
                        # opening / mouth width.  A smile widens the mouth (width ↑, vertical ~flat)
                        # → low MAR; speech opens vertically → high, oscillating MAR.  The ratio is
                        # intrinsically face-size-normalised, so no extra box-height scaling needed.
                        mar = self._mouth_aspect_ratio(landmarks)

                        faces.append({"box": box, "mar": float(mar)})
                detections.append({"frame_idx": idx, "faces": faces})

        return detections

    @staticmethod
    def _mouth_aspect_ratio(landmarks: list) -> float:
        """Mouth-Aspect-Ratio = mean vertical inner-lip opening / mouth width.

        Robust speaking signal: a wide smile barely changes the ratio (width grows with the
        opening), while speech drives the vertical opening up and down.  Returns 0.0 if the
        mouth width is degenerate.  Landmark indices come from ``MAR_VERTICAL_PAIRS`` /
        ``MAR_WIDTH_PAIR`` (constants).
        """
        def _dist(a: int, b: int) -> float:
            pa, pb = landmarks[a], landmarks[b]
            return float(np.hypot(pa.x - pb.x, pa.y - pb.y))

        width = _dist(*MAR_WIDTH_PAIR)
        if width <= 1e-6:
            return 0.0
        vertical = sum(_dist(a, b) for a, b in MAR_VERTICAL_PAIRS) / len(MAR_VERTICAL_PAIRS)
        return vertical / width

    def _map_faces_across_steps(self, detections: list[dict], width: int) -> list[dict]:
        """Assign each detected face a stable integer id across steps.

        Matches primarily by IoU (box overlap — scales with face size), falling back to a
        center-distance test when no track overlaps (a face that moved between sparse steps).
        Returns one entry per step: ``{"frame_idx": int, "faces": [(fid, box, mar), ...]}``.
        """
        track_boxes: list[tuple] = []  # fid → last seen (x, y, w, h)
        steps: list[dict] = []
        for step in detections:
            mapped: list[tuple[int, tuple, float]] = []
            for face in step["faces"]:
                box = face["box"]
                fc = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                # Best IoU match first.
                fid, best_iou = -1, IOU_MATCH_MIN
                for j, prev in enumerate(track_boxes):
                    iou = self._iou(box, prev)
                    if iou >= best_iou:
                        fid, best_iou = j, iou
                # Fallback: nearest center within 12% of frame width.
                if fid == -1:
                    for j, prev in enumerate(track_boxes):
                        pcx, pcy = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
                        if np.hypot(fc[0] - pcx, fc[1] - pcy) < 0.12 * width:
                            fid = j
                            break
                if fid == -1:
                    fid = len(track_boxes)
                    track_boxes.append(box)
                else:
                    track_boxes[fid] = box  # update track to the latest box
                mapped.append((fid, box, float(face.get("mar", 0.0))))
            steps.append({"frame_idx": step["frame_idx"], "faces": mapped})
        return steps

    @staticmethod
    def _mar_motion(series: list) -> list[float]:
        """Per-step mouth motion = |ΔMAR| between consecutive *visible* samples (0 otherwise)."""
        motion = [0.0] * len(series)
        prev = None
        for i, v in enumerate(series):
            if v is not None and prev is not None:
                motion[i] = abs(v - prev)
            if v is not None:
                prev = v
        return motion

    @staticmethod
    def _voiced_mask(audio_rms: list[float], n: int) -> list[bool]:
        """Boolean per-step mask of voiced (above-threshold) audio.  All-True when no audio."""
        if not audio_rms:
            return [True] * n
        positive = [r for r in audio_rms if r > 0]
        if not positive:
            return [True] * n
        thr = float(np.median(positive)) * VOICE_ACTIVITY_FLOOR_FACTOR
        return [i < len(audio_rms) and audio_rms[i] >= thr for i in range(n)]

    @staticmethod
    def _pearson(a: list[float], b: list[float]) -> float:
        """Pearson correlation of two equal-length series, clamped to [0, 1].

        Returns 0.0 for fewer than 3 points or a flat (zero-variance) series — i.e. "no positive
        correlation" — so a mouth that does not track the audio is never credited.
        """
        if len(a) < 3 or len(b) < 3:
            return 0.0
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.std() < 1e-9 or b_arr.std() < 1e-9:
            return 0.0
        return max(0.0, float(np.corrcoef(a_arr, b_arr)[0, 1]))

    @staticmethod
    def _iou(box_a: tuple, box_b: tuple) -> float:
        """Intersection-over-Union of two ``(x, y, w, h)`` boxes."""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _audio_correlation_weight(
        motion: list[float], audio_rms: list[float], voiced: list[bool]
    ) -> float:
        """Clip-level prior: correlation of a face's mouth motion with audio over all voiced steps.

        Used only as a mild tiebreak (a chronic non-speaker / persistent smiler scores low across
        the whole clip).  Neutral (1.0) when there is no audio.
        """
        if not audio_rms:
            return 1.0
        m: list[float] = []
        a: list[float] = []
        for i in range(min(len(motion), len(audio_rms))):
            if voiced[i]:
                m.append(motion[i])
                a.append(audio_rms[i])
        if len(m) < 3:
            return 1.0
        return FaceTracker._pearson(m, a)

    @staticmethod
    def _should_group_two_shot(steps: list[dict], crop_w: int) -> bool:
        """Clip-level two-shot decision: group both faces into one static crop only when they
        are consistently adjacent.  Uses the median over multi-face steps of (a) total span and
        (b) the largest gap between adjacent faces.  The gap test prevents framing the empty
        table between two far-apart people.
        """
        spans: list[float] = []
        gaps: list[float] = []
        for s in steps:
            faces = s["faces"]
            if len(faces) < 2:
                continue
            ranges = sorted((b[0], b[0] + b[2]) for _, b, _ in faces)
            spans.append(ranges[-1][1] - ranges[0][0])
            gaps.append(max(ranges[k][0] - ranges[k - 1][1] for k in range(1, len(ranges))))
        # Require two-face shots to dominate before committing to a two-shot for the whole clip.
        if not spans or len(spans) < max(1, int(len(steps) * 0.3)):
            return False
        return (
            float(np.median(spans)) <= crop_w * GROUP_FRAMING_FIT_FACTOR
            and float(np.median(gaps)) <= crop_w * GROUP_MAX_GAP_FACTOR
        )

    def _single_face_step_centers(self, detections: list[dict], width: int) -> list[dict]:
        """Non-PODCAST: track the first (usually only) detected face per step."""
        step_centers: list[dict] = []
        for s in self._map_faces_across_steps(detections, width):
            faces = s["faces"]
            if not faces:
                if step_centers:
                    step_centers.append({**step_centers[-1], "frame_idx": s["frame_idx"]})
                continue
            _fid, box, _mar = faces[0]
            # Rule-of-thirds: lift the center so the eyes sit in the upper third of the frame.
            step_centers.append(
                {
                    "frame_idx": s["frame_idx"],
                    "target_x": int(box[0] + box[2] / 2),
                    "target_y": int(box[1] + box[3] / 2 - box[3] * HEADROOM_FACTOR),
                    "speaker_id": 0,
                }
            )
        return step_centers

    def _podcast_step_centers(
        self,
        detections: list[dict],
        width: int,
        height: int,
        crop_w: int,
        audio_rms: list[float],
        steps_per_second: float,
    ) -> list[dict]:
        """PODCAST per-step targets: stable two-shot when faces are adjacent, otherwise
        audio-visual active-speaker cuts (mouth-motion gated by audio + weighted by how well
        each mouth correlates with the loudness)."""
        steps = self._map_faces_across_steps(detections, width)
        n = len(steps)
        if n == 0:
            return []

        all_fids = sorted({fid for s in steps for fid, _, _ in s["faces"]})

        # Per-face MAR time series (None where the face is not visible) and mouth-motion series.
        mar_series: dict[int, list] = {fid: [None] * n for fid in all_fids}
        for i, s in enumerate(steps):
            for fid, _box, mar in s["faces"]:
                mar_series[fid][i] = mar
        motion = {fid: self._mar_motion(mar_series[fid]) for fid in all_fids}

        # Audio-visual sync: voiced gate + per-face correlation weight.
        voiced = self._voiced_mask(audio_rms, n)
        weights = {
            fid: self._audio_correlation_weight(motion[fid], audio_rms, voiced)
            for fid in all_fids
        }

        # Stable, clip-level two-shot decision.
        group = self._should_group_two_shot(steps, crop_w)

        have_audio = bool(audio_rms)
        activity_window = max(2, round(LIP_ACTIVITY_WINDOW_SECONDS * steps_per_second))
        sync_window = max(3, round(AV_SYNC_WINDOW_SECONDS * steps_per_second))
        step_centers: list[dict] = []
        active: int | None = None

        def _emit(frame_idx: int, box: tuple, speaker_id: int, face_h: float) -> None:
            # Rule-of-thirds: lift the center so the eyes sit in the upper third of the frame.
            cy = box[1] + box[3] / 2 - face_h * HEADROOM_FACTOR
            step_centers.append(
                {
                    "frame_idx": frame_idx,
                    "target_x": int(box[0] + box[2] / 2),
                    "target_y": int(cy),
                    "speaker_id": speaker_id,
                }
            )

        for i, s in enumerate(steps):
            frame_idx = s["frame_idx"]
            faces = s["faces"]
            if not faces:
                # Carry forward the last target so the crop never snaps to the empty center.
                if step_centers:
                    step_centers.append({**step_centers[-1], "frame_idx": frame_idx})
                continue

            if group:
                ranges = [(b[0], b[0] + b[2]) for _, b, _ in faces]
                cys = [b[1] + b[3] / 2 for _, b, _ in faces]
                mean_h = sum(b[3] for _, b, _ in faces) / len(faces)
                step_centers.append(
                    {
                        "frame_idx": frame_idx,
                        "target_x": int((min(r[0] for r in ranges) + max(r[1] for r in ranges)) / 2),
                        "target_y": int(sum(cys) / len(cys) - mean_h * HEADROOM_FACTOR),
                        "speaker_id": GROUP_SPEAKER_ID,
                    }
                )
                continue

            visible = {fid: b for fid, b, _ in faces}

            # Per-step visual activity = std-dev of MAR over the trailing window.
            activity: dict[int, float] = {}
            lo = max(0, i - activity_window + 1)
            for fid in visible:
                vals = [v for v in mar_series[fid][lo : i + 1] if v is not None]
                activity[fid] = float(np.std(vals)) if len(vals) >= 2 else 0.0

            # Local audio-visual coherence: does this mouth track the audio *right now*?
            slo = max(0, i - sync_window + 1)
            hi = min(i + 1, len(audio_rms)) if have_audio else i + 1
            coherence: dict[int, float] = {}
            for fid in visible:
                if have_audio and hi - slo >= 3:
                    coherence[fid] = self._pearson(motion[fid][slo:hi], audio_rms[slo:hi])
                else:
                    coherence[fid] = 1.0  # no audio → coherence not used as a gate

            # Eligible speaker = clearly speaking AND (when audio present) tracking the audio.
            eligible = [
                fid
                for fid in visible
                if activity[fid] >= LIP_ACTIVITY_MIN
                and (not have_audio or coherence[fid] >= COHERENCE_MIN)
            ]
            # Ranking score: activity × local coherence × mild clip-level prior.
            score = {
                fid: activity[fid] * coherence[fid] * max(weights.get(fid, 1.0), 0.2)
                for fid in visible
            }

            if active is None or active not in visible:
                # No valid current speaker on screen — pick the best eligible, else the most
                # prominent (largest) visible face as a safe default (never the empty center).
                if eligible:
                    active = max(eligible, key=lambda f: score[f])
                else:
                    active = max(visible, key=lambda f: visible[f][2] * visible[f][3])
            elif voiced[i] and eligible:
                # Switch only on voiced steps to an eligible challenger that clearly beats the
                # current speaker.  If NO face is eligible (e.g. the talker's mouth is behind a
                # mic), we fall through and HOLD the current speaker — never jump to a smiler.
                best = max(eligible, key=lambda f: score[f])
                if best != active and score[best] > score.get(active, 0.0) * SPEAKER_SWITCH_MARGIN:
                    active = best

            _emit(frame_idx, visible[active], active, visible[active][3])

        return step_centers

    def _generate_smooth_crops(
        self,
        detections: list[dict],
        start_frame: int,
        end_frame: int,
        fps: float,
        width: int,
        height: int,
        content_type: ContentType,
        audio_rms: list[float] | None = None,
        detection_interval: int = 1,
        fast_mode: bool = False,
    ) -> list[dict]:
        """Smooth raw face bounding boxes and output dynamic crop boxes frame-by-frame.

        ``audio_rms`` is an optional per-detection-step loudness envelope (PODCAST only).
        When present it gates active-speaker switching to voiced moments and weights each
        face by how well its mouth movement correlates with the audio.  ``detection_interval``
        is the frame stride between detection steps (used to convert hold/window seconds to
        a number of steps).  ``fast_mode`` (Haar PODCAST tracking) forces the single largest-face
        crop path — no audio-visual active-speaker logic.
        """
        if not detections:
            # Fallback to static center crops
            return self._generate_fallback_crops(
                start_frame, end_frame, fps, width, height
            )

        # Target crop dimensions depend on content type.
        # Facecam crop is pre-shaped to the destination panel aspect so the downstream
        # scale (1080x960 for 2-stack, 1080x640 for collab) adds zero distortion.
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

        # Detection steps occur every ``detection_interval`` frames → convert seconds to steps.
        steps_per_second = fps / max(1, detection_interval)

        # We compute the target face center for each detection step.  Fast mode (Haar) and all
        # non-PODCAST types use the single largest-face path; full PODCAST uses audio-visual
        # active-speaker selection.
        if content_type == ContentType.PODCAST and not fast_mode:
            step_centers = self._podcast_step_centers(
                detections, width, height, crop_w, audio_rms or [], steps_per_second
            )
        else:
            step_centers = self._single_face_step_centers(detections, width)

        # ── Static-cut crop: no camera pan, no EMA glide ─────────────────────────────────
        # Build committed speaker segments with debounce:
        #   - A candidate speaker must hold for >= SPEAKER_HOLD_SECONDS before we commit.
        #   - Each committed segment owns a FIXED crop center (median of face-box centers
        #     in that segment).  The crop is constant for the whole segment; switching to
        #     a new speaker is a hard frame cut — zero viewer dizziness.
        # For non-PODCAST (single face tracked) the step_centers list all share speaker_id 0
        # → one segment → fully static center crop.
        hold_steps = max(1, int(SPEAKER_HOLD_SECONDS * steps_per_second))
        # Minimum number of detection steps before a subject switch is allowed.
        # Prevents dizzy rapid alternation even when the per-step speaker signal is noisy.
        min_shot_steps = max(1, int(MIN_SHOT_SECONDS * steps_per_second))

        # --- Phase 1: debounced segment boundaries ----------------------------------------
        # Each segment: {"speaker_id", "start_f", "end_f", "cx", "cy"}
        segments: list[dict] = []

        if not step_centers:
            # No detections at all → static center fallback (handled by caller).
            return self._generate_fallback_crops(start_frame, end_frame, fps, width, height)

        committed_sid = step_centers[0]["speaker_id"]
        pending_sid = committed_sid
        pending_count = 0
        committed_steps = 0  # steps on the current committed subject (enforces MIN_SHOT_SECONDS)
        seg_xs: list[float] = []
        seg_ys: list[float] = []
        seg_start = start_frame
        # Fallback center from the previous closed segment — used when a new segment has no
        # accumulated coords (avoids snapping to frame center / empty table).
        last_seg_cx: float | None = None
        last_seg_cy: float | None = None

        def _close_segment(
            sid: int,
            xs: list,
            ys: list,
            sf: int,
            ef: int,
            fallback_cx: float | None = None,
            fallback_cy: float | None = None,
        ) -> dict:
            default_cx = fallback_cx if fallback_cx is not None else width / 2.0
            default_cy = fallback_cy if fallback_cy is not None else height / 2.0
            cx = float(np.median(xs)) if xs else default_cx
            cy = float(np.median(ys)) if ys else default_cy
            # Clamp center so the crop box stays inside the frame.
            cx = max(crop_w / 2.0, min(cx, width - crop_w / 2.0))
            cy = max(crop_h / 2.0, min(cy, height - crop_h / 2.0))
            return {"speaker_id": sid, "start_f": sf, "end_f": ef, "cx": cx, "cy": cy}

        for sc in step_centers:
            sid = sc["speaker_id"]
            tx = float(sc["target_x"])
            ty = float(sc["target_y"])
            if sid == committed_sid:
                # Still the committed subject — accumulate and reset pending.
                pending_sid = sid
                pending_count = 0
                seg_xs.append(tx)
                seg_ys.append(ty)
                committed_steps += 1
            else:
                # Different subject detected.  Only allow a switch once the minimum shot
                # length has been held — this is the main anti-dizziness gate.
                if committed_steps >= min_shot_steps:
                    if sid == pending_sid:
                        pending_count += 1
                    else:
                        pending_sid = sid
                        pending_count = 1
                    # Commit the switch only after the debounce hold threshold is met.
                    if pending_count >= hold_steps:
                        seg = _close_segment(
                            committed_sid, seg_xs, seg_ys, seg_start,
                            sc["frame_idx"], last_seg_cx, last_seg_cy
                        )
                        segments.append(seg)
                        last_seg_cx, last_seg_cy = seg["cx"], seg["cy"]
                        committed_sid = sid
                        pending_count = 0
                        committed_steps = 1
                        seg_start = sc["frame_idx"]
                        seg_xs = [tx]
                        seg_ys = [ty]
                else:
                    # Min-shot not yet satisfied — hold current subject.
                    committed_steps += 1

        # Close the final open segment.
        seg = _close_segment(
            committed_sid, seg_xs, seg_ys, seg_start, end_frame, last_seg_cx, last_seg_cy
        )
        segments.append(seg)

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

        # PODCAST: gentle EMA pan so a speaker change eases over ~1 s instead of hard-cutting.
        # Within a held shot the target is constant, so the EMA converges and stays put (no drift).
        if content_type == ContentType.PODCAST and crops:
            sx = float(crops[0]["crop_x"])
            sy = float(crops[0]["crop_y"])
            for c in crops:
                sx += (c["crop_x"] - sx) * PAN_SMOOTHING_FACTOR
                sy += (c["crop_y"] - sy) * PAN_SMOOTHING_FACTOR
                c["crop_x"] = int(sx)
                c["crop_y"] = int(sy)

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
