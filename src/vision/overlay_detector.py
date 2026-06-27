from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    BOTTOM_STRIP_FRAC,
    FULLWIDTH_FRAC,
    MAX_POPUP_AREA_FRAC,
    MAX_POPUP_SECONDS,
    MIN_POPUP_AREA_FRAC,
    MIN_POPUP_SECONDS,
    MIN_SCAN_FRAMES,
    NOVELTY_THRESH,
    OVERLAY_MAX_GAP_SECONDS,
    OVERLAY_MAX_SCAN_SAMPLES,
    OVERLAY_MORPH_KERNEL_SIZE,
    OVERLAY_SMALL_H,
    OVERLAY_SMALL_W,
    POPUP_ASPECT_MAX,
    POPUP_ASPECT_MIN,
    POPUP_MAX_JITTER,
    POPUP_WINDOW_COVERAGE,
    SCAN_PAD_SECONDS,
)
from src.vision.frame_utils import clip_frame_range, video_props


class OverlayDetector:
    """Detects donation/MediaShare alert popups by transient appearance against a steady baseline."""

    def __init__(self) -> None:
        self.config = load_config()

    def detect_overlays(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        sample_step_seconds: float | None = None,
        facecam_box: tuple[int, int, int, int] | None = None,
    ) -> list[dict]:
        """Detect donation / MediaShare pop-ups by appearance/disappearance and return intervals.

        Method: a popup is a region that is ABSENT, then APPEARS for a bounded interval, then is
        DISMISSED. We sample the window (padded so a clip-spanning popup is still a minority),
        build a per-pixel median baseline of the persistent background, and flag frames where a
        card-shaped region differs sharply from that baseline. Persistent furniture (chat, the
        full-width bottom ticker, the facecam) lives in the baseline and never registers.

        Args:
            video_path: Path to the video file.
            start_time: Start time of the clip in seconds.
            end_time: End time of the clip in seconds.
            sample_step_seconds: Sampling cadence (default ~0.5s ≈ 2 fps).
            facecam_box: Optional facecam box (full-res); overlapping candidates are dropped.

        Returns:
            Clip-relative alert intervals: [{"start_time", "end_time", "box": (x,y,w,h), "hits"}].
        """
        if not self.config.video_processing.preserve_donation_overlays:
            return []

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video for overlay detection: {video_path}")
            return []

        try:
            width, height, fps, total_frames = video_props(cap)

            # Pad the scan so an in-clip popup has clear before/after baseline frames.
            scan_start_t = max(0.0, start_time - SCAN_PAD_SECONDS)
            scan_end_t = min(total_frames / fps, end_time + SCAN_PAD_SECONDS)
            step_s = sample_step_seconds if sample_step_seconds is not None else 0.5
            step = max(1, int(fps * step_s))

            s_frame, e_frame = clip_frame_range(fps, total_frames, scan_start_t, scan_end_t)

            # Sample small frames; t is relative to the CLIP start (negative inside the lead pad).
            samples: list[tuple[float, np.ndarray]] = []
            for idx in range(s_frame, e_frame, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                small = cv2.resize(frame, (OVERLAY_SMALL_W, OVERLAY_SMALL_H))
                samples.append(((idx / fps) - start_time, small))
                if len(samples) >= OVERLAY_MAX_SCAN_SAMPLES:  # bound cost on long windows
                    break
        finally:
            cap.release()

        if len(samples) < MIN_SCAN_FRAMES:
            return []

        # Per-pixel median baseline = the persistent background; transient popups are excluded.
        baseline = np.median(np.stack([s for _, s in samples]), axis=0).astype(np.uint8)
        scale_x, scale_y = width / OVERLAY_SMALL_W, height / OVERLAY_SMALL_H

        active_frames: list[tuple[float, tuple[int, int, int, int]]] = []
        for t, small in samples:
            box = self._novelty_box(small, baseline, scale_x, scale_y, width, height, facecam_box)
            if box is not None:
                active_frames.append((t, box))

        intervals = self._group_intervals(
            active_frames, step / fps, end_time - start_time, width, height
        )

        # Keep only TRANSIENT, BOUNDED popups with a clear onset AND offset inside the padded scan
        # (i.e. not present in the first/last sampled frame → it genuinely appeared and dismissed).
        first_t, last_t = samples[0][0], samples[-1][0]
        window_dur = max(0.001, end_time - start_time)
        popups = []
        for iv in intervals:
            dur = iv["end_time"] - iv["start_time"]
            if iv["hits"] < 2 or dur < MIN_POPUP_SECONDS:
                continue
            if dur >= MAX_POPUP_SECONDS or dur >= POPUP_WINDOW_COVERAGE * window_dur:
                continue
            # Onset/offset: a real popup is absent at both scan ends.
            if iv["raw_start"] <= first_t + 1e-3 or iv["raw_end"] >= last_t - 1e-3:
                continue
            # Static-vs-moving: a real card holds position; drifting box = gameplay motion.
            if iv.get("jitter", 0.0) > POPUP_MAX_JITTER:
                logger.debug(
                    f"  Rejected moving novelty (jitter {iv['jitter']:.3f} > "
                    f"{POPUP_MAX_JITTER}) — gameplay, not a popup."
                )
                continue
            popups.append(iv)

        n_confirmed = len(popups)
        if n_confirmed:
            logger.info(f"Donation overlay scan: {n_confirmed} popup(s) confirmed.")
        else:
            logger.info("Donation overlay scan: no popups confirmed.")
        for idx, val in enumerate(popups):
            logger.debug(
                f"Popup {idx + 1}: {val['start_time']:.1f}s–{val['end_time']:.1f}s "
                f"({val['hits']} frames matched)"
            )
        return popups

    def _novelty_box(
        self,
        small: np.ndarray,
        baseline: np.ndarray,
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
        facecam_box: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """Largest card-shaped region of ``small`` that differs sharply from the baseline.

        Returns the box in full-resolution coordinates, or None when nothing card-like appears.
        """
        import cv2

        diff = cv2.absdiff(small, baseline)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, NOVELTY_THRESH, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (OVERLAY_MORPH_KERNEL_SIZE, OVERLAY_MORPH_KERNEL_SIZE)
        )
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(width * height)
        best: tuple[int, int, int, int] | None = None
        best_area = 0.0
        for cnt in contours:
            x, y, w, h_box = cv2.boundingRect(cnt)
            # Map to full-res coordinates.
            fx, fy = int(x * scale_x), int(y * scale_y)
            fw, fh = int(w * scale_x), int(h_box * scale_y)
            area = float(fw * fh)
            if not (MIN_POPUP_AREA_FRAC * frame_area < area < MAX_POPUP_AREA_FRAC * frame_area):
                continue
            aspect = fw / max(1, fh)
            if not (POPUP_ASPECT_MIN < aspect < POPUP_ASPECT_MAX):
                continue
            if self._is_excluded(fx, fy, fw, fh, width, height, facecam_box):
                continue
            if area > best_area:
                best, best_area = (fx, fy, fw, fh), area
        return best

    def _is_excluded(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        width: int,
        height: int,
        facecam_box: tuple[int, int, int, int] | None,
    ) -> bool:
        """True for persistent furniture that must never be a popup (ticker, full-width band, cam)."""
        cy = y + h / 2.0
        if cy > BOTTOM_STRIP_FRAC * height:  # full-width scrolling donation ticker at the bottom
            return True
        if w >= FULLWIDTH_FRAC * width:  # ticker / lower-thirds banners span the width
            return True
        if facecam_box is not None:  # the streamer cam is a moving bordered inset, not a popup
            fx, fy, fw, fh = facecam_box
            cx = x + w / 2.0
            if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                return True
        return False

    def _group_intervals(
        self,
        active_frames: list[tuple[float, tuple[int, int, int, int]]],
        step_dur: float,
        window_dur: float,
        width: int,
        height: int,
    ) -> list[dict]:
        """Group per-frame detections into contiguous presence intervals (clip-relative times)."""

        if not active_frames:
            return []

        diag = max(1.0, float((width**2 + height**2) ** 0.5))
        max_gap = OVERLAY_MAX_GAP_SECONDS
        intervals: list[dict] = []
        cur_start = cur_end = active_frames[0][0]
        cur_boxes = [active_frames[0][1]]

        def _close(raw_start: float, raw_end: float, boxes: list) -> dict:
            arr = np.array(boxes, dtype=float)
            avg = arr.mean(axis=0)
            # Box-centre jitter, normalised by the frame diagonal: ~0 for a static card, large for a
            # drifting (moving) novelty region. Distinguishes a popup from gameplay motion.
            centres = arr[:, :2] + arr[:, 2:] / 2.0
            jitter = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).mean() / diag)
            return {
                "raw_start": raw_start,
                "raw_end": raw_end,
                "start_time": max(0.0, raw_start),
                "end_time": min(window_dur, raw_end + step_dur),
                "box": (int(avg[0]), int(avg[1]), int(avg[2]), int(avg[3])),
                "hits": len(boxes),
                "jitter": jitter,
            }

        for t, box in active_frames[1:]:
            if t - cur_end <= max_gap:
                cur_end = t
                cur_boxes.append(box)
            else:
                intervals.append(_close(cur_start, cur_end, cur_boxes))
                cur_start = cur_end = t
                cur_boxes = [box]
        intervals.append(_close(cur_start, cur_end, cur_boxes))
        return intervals
