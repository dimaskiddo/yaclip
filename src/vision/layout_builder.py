from __future__ import annotations

from loguru import logger

from src.core.config import load_config
from src.core.constants import (
    FACECAM_DOMINANT_AREA_FRAC,
    FACECAM_FIT_FACTOR,
    STACK2_PANEL_ASPECT,
    STACK3_PANEL_ASPECT,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    ContentType,
    LayoutMode,
)
from src.core.utils import (
    box_center,
    boxes_overlap,
    center_crop,
    clamp_crop_x,
    expand_box_to_aspect,
    is_facecam_candidate,
)


class LayoutBuilder:
    """Turns ContentType + VisualAnalyzer regions into concrete crop/stack instructions."""

    def __init__(self) -> None:
        self.config = load_config()

    def build_layout(
        self,
        content_type: ContentType,
        analysis: dict,
        face_data: list[dict],
        overlay_data: list[dict] | None = None,
    ) -> dict:
        """Construct the layout metadata payload for one clip.

        Args:
            content_type: Locked content type for the clip.
            analysis: VisualAnalyzer.analyze_window() result for the clip window
                (facecam_box, gameplay_box, screen_inset_box, mediashare_present, persons...).
            face_data: FaceTracker smooth-pan crops — used only by Mode A (talking heads).
            overlay_data: Optional colour-based donation boxes from OverlayDetector,
                used as a MediaShare fallback when YOLO finds no screen inset.

        Returns:
            A layout_spec dict consumed by FFmpegCommandBuilder.
        """
        overlay_data = overlay_data or []
        width = int(analysis.get("video_width", 1920))
        height = int(analysis.get("video_height", 1080))

        # Determine LayoutMode. DONATION_OVERLAY is a per-clip type (promoted when a mediashare
        # popup is in the window); it reuses the 2-stack geometry with the popup forced bottom.
        if content_type == ContentType.PODCAST:
            layout_mode = LayoutMode.SINGLE_VERTICAL
        elif content_type in (
            ContentType.GAMING_SOLO,
            ContentType.JUST_CHAT,
            ContentType.DONATION_OVERLAY,
        ):
            layout_mode = LayoutMode.STACKED_SPLIT
        elif content_type == ContentType.GAMING_COLLAB:
            layout_mode = LayoutMode.MULTI_COLLAB
        else:
            layout_mode = LayoutMode.SINGLE_VERTICAL

        # Fullscreen-cam override: when the facecam fills most of the frame there is no real
        # gameplay to stack — render a single full-face vertical instead of face-over-face. Skipped
        # for DONATION_OVERLAY: a big facecam during a donation must still stack the popup.
        crops = face_data
        if (
            layout_mode == LayoutMode.STACKED_SPLIT
            and content_type != ContentType.DONATION_OVERLAY
            and self._facecam_dominant(analysis, width, height)
        ):
            layout_mode = LayoutMode.SINGLE_VERTICAL
            crops = self._facecam_single_vertical_crops(analysis, width, height)
            logger.info("Webcam fills most of the frame — using single vertical layout.")

        logger.info(f"Building {content_type.value} layout ({layout_mode.value}).")

        spec: dict = {
            "content_type": content_type.value,
            "layout_mode": layout_mode.value,
            "video_width": width,
            "video_height": height,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
            "crops": crops,  # Mode A smooth-pan facecam (or synthesized full-face crop)
        }

        if layout_mode == LayoutMode.STACKED_SPLIT:
            if content_type == ContentType.DONATION_OVERLAY:
                self._fill_donation(spec, analysis, overlay_data, width, height)
            else:
                self._fill_mode_b(spec, analysis, overlay_data, width, height)
        elif layout_mode == LayoutMode.MULTI_COLLAB:
            self._fill_mode_c(spec, analysis, overlay_data, width, height)

        return spec

    def _facecam_dominant(self, analysis: dict, width: int, height: int) -> bool:
        """True when the facecam box covers > 55% of the frame (a fullscreen-cam moment)."""
        box = analysis.get("facecam_box")
        if not box:
            return False
        return (box[2] * box[3]) > FACECAM_DOMINANT_AREA_FRAC * (width * height)

    def _facecam_single_vertical_crops(self, analysis: dict, width: int, height: int) -> list[dict]:
        """Synthesize a single static 9:16 crop centred on the facecam for Mode A rendering."""
        box = analysis.get("facecam_box")
        crop = center_crop(width, height, 9.0 / 16.0)
        cx = box_center(box)[0] if box else width / 2
        crop_x = clamp_crop_x(cx, crop["w"], width)
        return [
            {
                "timestamp": 0.0,
                "crop_x": crop_x,
                "crop_y": 0,
                "crop_w": crop["w"],
                "crop_h": crop["h"],
            }
        ]

    # ---------------------------------------------------------------- Mode B

    def _fill_mode_b(
        self, spec: dict, analysis: dict, overlay_data: list[dict], width: int, height: int
    ) -> None:
        """2-stack: facecam top (always fits), gameplay bottom.

        MediaShare/donation moments are no longer handled here — a clip with a popup is promoted to
        DONATION_OVERLAY (see _fill_donation) before reaching this path, so the bottom is always the
        gameplay region for GAMING_SOLO / JUST_CHAT.
        """
        self._set_facecam_top(spec, analysis, width, height)

        spec["bottom_mode"] = "gameplay"
        # Centred, panel-aspect gameplay crop that fills the panel (focused), excluding the corner
        # cam. The damped pan track gently follows the character (small movement).
        spec["bottom_crop"] = analysis.get("gameplay_box") or center_crop(
            width, height, STACK2_PANEL_ASPECT
        )
        spec["bottom_track"] = analysis.get("gameplay_track") or []
        kf = len(spec["bottom_track"])
        logger.info(f"Layout: webcam top, gameplay bottom ({kf} motion keyframe(s)).")
        logger.debug(f"Gameplay crop box: {spec['bottom_crop']}")

    # ------------------------------------------------------- Donation Overlay

    def _fill_donation(
        self, spec: dict, analysis: dict, overlay_data: list[dict], width: int, height: int
    ) -> None:
        """2-stack: facecam top (always fits) + the donation/mediashare popup forced as the bottom."""
        self._set_facecam_top(spec, analysis, width, height)

        mediashare_box = self._mediashare_box(analysis, overlay_data)
        if mediashare_box is not None:
            spec["bottom_mode"] = "mediashare"
            spec["bottom_crop"] = expand_box_to_aspect(
                *mediashare_box, width, height, STACK2_PANEL_ASPECT
            )
            logger.info("Layout: webcam top, donation popup bottom.")
            logger.debug(f"Donation popup crop box: {spec['bottom_crop']}")
        else:
            # Promoted to donation but no usable popup box (e.g. content_type forced via config) →
            # degrade gracefully to the gameplay bottom rather than a black panel.
            spec["bottom_mode"] = "gameplay"
            spec["bottom_crop"] = analysis.get("gameplay_box") or center_crop(
                width, height, STACK2_PANEL_ASPECT
            )
            spec["bottom_track"] = analysis.get("gameplay_track") or []
            logger.info(
                "Layout: webcam top, gameplay bottom (no popup box found — using gameplay fallback)."
            )

    # ---------------------------------------------------------------- Mode C

    def _fill_mode_c(
        self, spec: dict, analysis: dict, overlay_data: list[dict], width: int, height: int
    ) -> None:
        """3-stack: facecam top, gameplay centre, collaborator bottom (each 1080x640)."""
        spec["facecam_crop"] = self._facecam_crop(analysis, width, height, STACK3_PANEL_ASPECT)

        # Collaborator = the SECOND facecam. Prefer the reliable detected cam PAIR (collab_box, locked
        # video-wide via detect_facecams) so the bottom panel is the actual 2nd webcam. Fall back to
        # the per-clip persons heuristic (cam-sized, near an edge, not overlapping the primary) only
        # when the pair is absent.
        facecam_box = analysis.get("facecam_box")
        collab_box = analysis.get("collab_box")
        if collab_box is not None:
            spec["collab_crop"] = expand_box_to_aspect(
                int(collab_box[0]),
                int(collab_box[1]),
                int(collab_box[2]),
                int(collab_box[3]),
                width,
                height,
                STACK3_PANEL_ASPECT,
            )
            spec["gameplay_crop"] = analysis.get("gameplay_box") or center_crop(
                width, height, STACK3_PANEL_ASPECT
            )
            spec["gameplay_track"] = analysis.get("gameplay_track") or []
            return

        persons = sorted(
            analysis.get("persons", []),
            key=lambda b: float(b[2]) * float(b[3]),
            reverse=True,
        )
        frame_area = float(width * height)

        def _not_primary(box: tuple) -> bool:
            return facecam_box is None or not boxes_overlap(tuple(box), tuple(facecam_box), 0.3)

        def _edge_cam(box: tuple) -> bool:
            return is_facecam_candidate(box, frame_area, width, height)

        collab_src = next(
            (b for b in persons if _not_primary(b) and _edge_cam(b)),
            next((b for b in persons if _not_primary(b)), None),
        )

        if collab_src is not None:
            spec["collab_crop"] = expand_box_to_aspect(
                int(collab_src[0]),
                int(collab_src[1]),
                int(collab_src[2]),
                int(collab_src[3]),
                width,
                height,
                STACK3_PANEL_ASPECT,
            )
        else:
            spec["collab_crop"] = center_crop(width, height, STACK3_PANEL_ASPECT)

        # Gameplay centre panel (1080x640): same zoomed, static-first pan engine as GAMING_SOLO,
        # shaped to the 3-stack aspect (the analyzer produced gameplay_box/gameplay_track at 1.6875
        # for collab clips). Falls back to a centred panel-aspect crop when no region is available.
        spec["gameplay_crop"] = analysis.get("gameplay_box") or center_crop(
            width, height, STACK3_PANEL_ASPECT
        )
        spec["gameplay_track"] = analysis.get("gameplay_track") or []

    # --------------------------------------------------------------- Helpers

    def _facecam_crop(self, analysis: dict, width: int, height: int, aspect: float) -> dict:
        """Facecam box expanded to the destination panel aspect (or centre fallback)."""
        box = analysis.get("facecam_box")
        if not box:
            return center_crop(width, height, aspect)
        return expand_box_to_aspect(
            int(box[0]), int(box[1]), int(box[2]), int(box[3]), width, height, aspect
        )

    def _mediashare_box(
        self, analysis: dict, overlay_data: list[dict]
    ) -> tuple[int, int, int, int] | None:
        """Pick a MediaShare/donation box from the analyzer (dense scan), colour fallback.

        The streamer facecam is a moving bordered inset and must never become the bottom popup
        (the duplicate-cam bug), so any colour box overlapping the facecam is dropped here — the
        analysis pass filters this in _scan_mediashare, this closes the same gap on overlay_data.
        """
        if analysis.get("mediashare_present"):
            b = analysis.get("mediashare_box") or analysis.get("screen_inset_box")
            if b:
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        facecam_box = analysis.get("facecam_box")
        for od in overlay_data:
            ox, oy, ow, oh = od["box"]
            if facecam_box and boxes_overlap((ox, oy, ow, oh), facecam_box, 0.3):
                continue
            return (int(ox), int(oy), int(ow), int(oh))
        return None

    def _set_facecam_top(self, spec: dict, analysis: dict, width: int, height: int) -> None:
        """Compute the always-fit top-panel facecam crop on the spec."""
        crop, _ = self._facecam_panel_crop(analysis, width, height)
        spec["facecam_crop"] = crop

    def _facecam_panel_crop(self, analysis: dict, width: int, height: int) -> tuple[dict, str]:
        """Facecam top crop — prominent and panel-aspect, crop-filled sharp (no blur, no bars).

        Expand the cam box by FACECAM_FIT_FACTOR (comfortable context margin) and shape it to the
        panel aspect (1.125). The shaped crop is exact panel aspect, so the FFmpeg crop+scale fills
        the 1080×960 panel with no bars and no blur — mild upscale when the cam region is smaller
        than the panel, which keeps the cam prominent. Returns (crop_box, "fit") always.
        """
        box = analysis.get("facecam_box")
        if not box:
            # Centre crop is already panel-aspect → scales clean.
            return center_crop(width, height, STACK2_PANEL_ASPECT), "fit"

        fx, fy, fw, fh = (int(v) for v in box)
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        ew = min(float(width), fw * FACECAM_FIT_FACTOR)
        eh = min(float(height), fh * FACECAM_FIT_FACTOR)
        ex = int(max(0, min(cx - ew / 2.0, width - ew)))
        ey = int(max(0, min(cy - eh / 2.0, height - eh)))
        shaped = expand_box_to_aspect(ex, ey, int(ew), int(eh), width, height, STACK2_PANEL_ASPECT)
        return shaped, "fit"
