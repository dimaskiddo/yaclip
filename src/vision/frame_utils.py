from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import cv2


def video_props(cap: cv2.VideoCapture) -> tuple[int, int, float, int]:
    """Read (width, height, fps, total_frames) from an already-open capture.

    Args:
        cap: An opened ``cv2.VideoCapture``.

    Returns:
        (width, height, fps, total_frames). Width/height fall back to 1920/1080 and fps to
        29.97 when the container reports 0 (some streams omit these properties).
    """
    import cv2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames


def probe_video_dims(video_path: str | Path) -> tuple[int, int, float, int]:
    """Open ``video_path``, read its dimensions/fps/frame-count, and release it.

    Args:
        video_path: Path to the video file.

    Returns:
        (width, height, fps, total_frames), or the defensive fallback (1920, 1080, 29.97, 0)
        when the file cannot be opened.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1920, 1080, 29.97, 0
    try:
        return video_props(cap)
    finally:
        cap.release()


def clip_frame_range(
    fps: float, total_frames: int, start_time: float, end_time: float
) -> tuple[int, int]:
    """Convert a [start_time, end_time] window to a clamped [s_frame, e_frame) frame range.

    Args:
        fps: Frames per second of the source video.
        total_frames: Total frame count of the source video.
        start_time: Window start in seconds.
        end_time: Window end in seconds.

    Returns:
        (s_frame, e_frame) clamped to [0, total_frames] with e_frame > s_frame.
    """
    s_frame = max(0, min(int(start_time * fps), max(0, total_frames - 1)))
    e_frame = max(s_frame + 1, min(int(end_time * fps), total_frames))
    return s_frame, e_frame


def sample_frame_indices_in_range(
    total_frames: int, n: int, start_frac: float = 0.1, end_frac: float = 0.9
) -> list[int]:
    """Evenly-spaced, endpoint-inclusive frame indices across a fraction of the whole video.

    Args:
        total_frames: Total frame count of the source video.
        n: Number of indices to produce.
        start_frac: Fraction of the video to start sampling from.
        end_frac: Fraction of the video to stop sampling at (inclusive at i = n-1).

    Returns:
        ``n`` frame indices spanning ``[start_frac, end_frac]`` of ``total_frames``.
    """
    span = end_frac - start_frac
    return [
        int(total_frames * (start_frac + span * i / max(1, n - 1))) for i in range(n)
    ]


def sample_frame_indices_evenly(s_frame: int, e_frame: int, n: int) -> list[int]:
    """Evenly-spaced frame indices across a [s_frame, e_frame) window (n-divided, not n-1).

    Args:
        s_frame: First frame of the window.
        e_frame: First frame after the window.
        n: Number of indices to produce.

    Returns:
        ``n`` frame indices spanning the window (does not necessarily reach ``e_frame``).
    """
    return [int(s_frame + i * (e_frame - s_frame) / n) for i in range(n)]


def sample_frames(cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    """Seek to and read each frame index; failed reads are silently dropped.

    Args:
        cap: An opened ``cv2.VideoCapture``.
        indices: Frame indices to seek to and read.

    Returns:
        The successfully-read frames, in index order (shorter than ``indices`` on any miss).
    """
    import cv2

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def sample_frames_timed(
    cap: cv2.VideoCapture, indices: list[int], fps: float
) -> list[tuple[float, np.ndarray]]:
    """Like ``sample_frames``, but pairs each kept frame with its timestamp (idx / fps).

    Args:
        cap: An opened ``cv2.VideoCapture``.
        indices: Frame indices to seek to and read.
        fps: Frames per second, used to convert each kept index to a timestamp.

    Returns:
        (timestamp_seconds, frame) pairs for the successfully-read frames.
    """
    import cv2

    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            samples.append((idx / fps, frame))
    return samples


def load_yolo(cfg: object) -> object:
    """Load (or first-time download) the configured YOLOv8 model.

    Weights are resolved from ``workspace/models/<model_name>`` when already downloaded,
    otherwise the bare model name is passed so ultralytics fetches it itself.

    Args:
        cfg: The ``region_detection`` config section (needs ``.model_name``).

    Returns:
        The loaded ``ultralytics.YOLO`` model.

    Raises:
        ImportError: When the ``ultralytics`` package is not installed.
    """
    from ultralytics import YOLO

    from src.core.workspace import MODELS_DIR

    model_path = MODELS_DIR / cfg.model_name  # type: ignore[attr-defined]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(model_path) if model_path.exists() else cfg.model_name  # type: ignore[attr-defined]
    logger.info("Loading object detection model...")
    return YOLO(target)


def yolo_predict_boxes(
    model: object, frame: np.ndarray, device: str
) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    """Run a YOLO model on one frame and flatten its boxes to plain tuples.

    Args:
        model: A loaded ``ultralytics.YOLO`` model.
        frame: A single BGR frame.
        device: Compute device string ("cpu" or "cuda").

    Returns:
        One ``(class_id, confidence, (x, y, w, h))`` tuple per detected box.
    """
    results = model.predict(frame, verbose=False, device=device)  # type: ignore[attr-defined]
    out: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for res in results:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append((cls_id, conf, (x1, y1, x2 - x1, y2 - y1)))
    return out
