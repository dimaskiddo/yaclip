from __future__ import annotations

from pathlib import Path

from src.core.config import load_config
from src.core.constants import (
    STACK2_PANEL_H,
    STACK2_PANEL_W,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    LayoutMode,
)
from src.core.utils import SystemUtils
from src.core.workspace import FONTS_DIR


class FFmpegCommandBuilder:
    """Builds complex FFmpeg command lines for cropping, stacking, overlaying, and burning subtitles."""

    def __init__(self) -> None:
        self.config = load_config()
        self.ffmpeg_path = SystemUtils.get_ffmpeg_path()

    def build_render_command(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        layout_spec: dict,
        subtitle_ass_path: Path | None,
        output_path: Path,
        audio_path: Path | None = None,
        video_encoder: str | None = None,
    ) -> list[str]:
        """Generate the FFmpeg command line arguments list.

        Args:
            video_path: Path to the input video.
            start_time: Start time of the clip in seconds.
            duration: Duration of the clip in seconds.
            layout_spec: Layout specifications from LayoutBuilder.
            subtitle_ass_path: Optional path to .ass subtitle file.
            output_path: Destination path for the rendered video.
            audio_path: Optional path to the separate audio track file.
            video_encoder: Encoder override (auto|cpu|nvenc|qsv|videotoolbox).  None → read from
                config.  The renderer passes "cpu" here to rebuild after a GPU-encoder failure.

        Returns:
            A list of command strings to execute via subprocess.
        """
        layout_mode = layout_spec["layout_mode"]

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Base input options
        # We seek fast using -ss BEFORE -i, which is extremely efficient
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
        ]

        has_audio_input = False
        if audio_path and audio_path.exists():
            cmd.extend(
                [
                    "-ss",
                    f"{start_time:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(audio_path),
                ]
            )
            has_audio_input = True

        # Build filter complex graph
        filter_complex = []

        # 1. Video filtering based on layout mode
        if layout_mode == LayoutMode.SINGLE_VERTICAL.value:
            filter_complex = self._build_mode_a_filters(layout_spec, subtitle_ass_path)
        elif layout_mode == LayoutMode.STACKED_SPLIT.value:
            filter_complex = self._build_mode_b_filters(layout_spec, subtitle_ass_path)
        elif layout_mode == LayoutMode.MULTI_COLLAB.value:
            filter_complex = self._build_mode_c_filters(layout_spec, subtitle_ass_path)
        else:
            filter_complex = self._build_mode_a_filters(layout_spec, subtitle_ass_path)

        # 2. Audio from separate audio file — input already trimmed by -ss/-t at input level
        if has_audio_input:
            filter_complex.append("[1:a]asetpts=PTS-STARTPTS[a_out]")

        cmd.extend(["-filter_complex", ";".join(filter_complex)])
        cmd.extend(["-map", "[out_v]"])

        # Map audio
        if has_audio_input:
            cmd.extend(["-map", "[a_out]"])
        else:
            cmd.extend(["-map", "0:a"])

        # Platform-standard H.264/AAC encode accepted by YouTube Shorts, Reels, TikTok, Threads.
        # The video codec + rate-control flags vary per encoder (see _video_encoder_args); the
        # shared flags below are encoder-independent:
        #   high profile + level 4.1 + yuv420p   → broad decoder compatibility
        #   -r 30 -g 60                           → constant 30 fps, 2s GOP (clean keyframes)
        #   -movflags +faststart                  → moov atom up front (streaming upload/preview)
        #   -ar 48000 -ac 2                       → standard 48 kHz stereo AAC-LC
        # The filter graph already emits 1080x1920 with setsar=1 (square pixels → DAR 9:16).
        cmd.extend(self._video_encoder_args(video_encoder))
        cmd.extend(
            [
                "-profile:v",
                "high",
                "-level",
                "4.1",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-g",
                "60",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-shortest",
                str(output_path),
            ]
        )

        return cmd

    def _resolve_encoder(self, video_encoder: str | None) -> str:
        """Resolve the encoder name: explicit override → config → 'auto' probe (nvenc if CUDA)."""
        encoder = (
            video_encoder or self.config.video_processing.video_encoder or "cpu"
        ).lower()
        if encoder == "auto":
            return "nvenc" if SystemUtils.resolve_device("auto") == "cuda" else "cpu"
        return encoder

    def _video_encoder_args(self, video_encoder: str | None) -> list[str]:
        """Return the ``-c:v`` + rate-control flags for the selected encoder.

        cpu (libx264) is the default and the guaranteed fallback target.  GPU encoders use their
        own rate-control flags (nvenc has no -crf).  Unknown names degrade to libx264.
        """
        encoder = self._resolve_encoder(video_encoder)
        if encoder == "nvenc":
            # NVENC: constant-quality VBR (no -crf); -threads is ignored by the GPU encoder.
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20"]
        if encoder == "qsv":
            return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "20"]
        if encoder == "videotoolbox":
            return ["-c:v", "h264_videotoolbox", "-q:v", "60"]
        # cpu / libx264 (default) — also the fallback after a GPU-encoder failure.
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2"]

    def _get_face_crop_filter(
        self, crops: list[dict], video_w: int, video_h: int, interpolate: bool = True
    ) -> str:
        """Generate the FFmpeg crop filter for a per-frame crop track.

        Args:
            crops: Per-frame crop dicts with timestamp, crop_x, crop_y, crop_w, crop_h.
            video_w: Source video width.
            video_h: Source video height.
            interpolate: When True (default, used by opt-in gameplay pan) the crop position
                glides linearly between sampled keyframes.  When False (used by the PODCAST
                facecam / speaker-cut path) the crop position is held constant from one
                keyframe until the next and then jumps — a hard cut with zero camera motion.
        """
        if not crops:
            crop_w = int(video_h * TARGET_WIDTH / TARGET_HEIGHT)
            crop_h = video_h
            crop_x = (video_w - crop_w) // 2
            return f"crop={crop_w}:{crop_h}:{crop_x}:0"

        step = max(1, len(crops) // 60)
        keyframes = []
        for i in range(0, len(crops), step):
            c = crops[i]
            keyframes.append((c["timestamp"], c["crop_x"], c["crop_y"]))
        if crops and keyframes[-1][0] < crops[-1]["timestamp"]:
            keyframes.append(
                (crops[-1]["timestamp"], crops[-1]["crop_x"], crops[-1]["crop_y"])
            )

        crop_w = crops[0]["crop_w"]
        crop_h = crops[0]["crop_h"]

        x_expr = f"{keyframes[-1][1]}"
        y_expr = f"{keyframes[-1][2]}"

        for i in range(len(keyframes) - 2, -1, -1):
            t_curr, x_curr, y_curr = keyframes[i]
            t_next, x_next, y_next = keyframes[i + 1]
            if interpolate:
                slope_x = (x_next - x_curr) / max(0.01, t_next - t_curr)
                slope_y = (y_next - y_curr) / max(0.01, t_next - t_curr)
                x_expr = f"if(lt(t,{t_next:.3f}),{x_curr:.1f}+(t-{t_curr:.3f})*{slope_x:.3f},{x_expr})"
                y_expr = f"if(lt(t,{t_next:.3f}),{y_curr:.1f}+(t-{t_curr:.3f})*{slope_y:.3f},{y_expr})"
            else:
                # Static hold: keep the keyframe value until the next boundary, then cut.
                x_expr = f"if(lt(t,{t_next:.3f}),{x_curr:.0f},{x_expr})"
                y_expr = f"if(lt(t,{t_next:.3f}),{y_curr:.0f},{y_expr})"

        x_expr_escaped = x_expr.replace(",", "\\,")
        y_expr_escaped = y_expr.replace(",", "\\,")
        return f"crop={crop_w}:{crop_h}:{x_expr_escaped}:{y_expr_escaped}"

    def _append_subtitles_filter(
        self, v_filter: str, subtitle_ass_path: Path | None
    ) -> str:
        """Append subtitle burn-in to the final video filter chain if subtitles exist."""
        if subtitle_ass_path and subtitle_ass_path.exists():
            escaped_path = SystemUtils.escape_ffmpeg_path(subtitle_ass_path)
            escaped_fonts_dir = SystemUtils.escape_ffmpeg_path(str(FONTS_DIR))
            return (
                f"{v_filter},subtitles='{escaped_path}':fontsdir='{escaped_fonts_dir}'"
            )
        return v_filter

    def _build_mode_a_filters(
        self, spec: dict, subtitle_ass_path: Path | None
    ) -> list[str]:
        """Build filters for Mode A - Single 9:16 Vertical with gentle EMA speaker pans.

        The crop track is already EMA-smoothed in the face tracker (static while a speaker holds,
        gliding on a speaker change), so the interpolating filter renders that glide smoothly
        rather than quantizing it into ~60 stepwise jumps.
        """
        face_crop = self._get_face_crop_filter(
            spec["crops"], spec["video_width"], spec["video_height"], interpolate=True
        )
        v_filter = f"[0:v]setpts=PTS-STARTPTS,{face_crop},scale={spec['target_width']}:{spec['target_height']},format=yuv420p,setsar=1"
        v_filter = self._append_subtitles_filter(v_filter, subtitle_ass_path)
        return [v_filter + "[out_v]"]

    def _build_mode_b_filters(
        self, spec: dict, subtitle_ass_path: Path | None
    ) -> list[str]:
        """Build the Mode B 2-stack layout for GAMING_SOLO / JUST_CHAT.

        Top panel  (1080x960) = facecam, smooth-panned crop-fill (no distortion).
        Bottom panel (1080x960) = MediaShare popup when one is active in the clip, otherwise
        the motion-tracked gameplay region. The bottom crop was already pre-shaped to the
        panel aspect (1.125) by LayoutBuilder, so both panels scale in without stretching.
        Final canvas = vstack of the two panels = 1080x1920. GAMING_SOLO_BOTTOM (spec's
        "facecam_bottom" flag) swaps the vstack order — gameplay top, facecam bottom.
        """
        panel_w = STACK2_PANEL_W
        panel_h = STACK2_PANEL_H

        fc = spec["facecam_crop"]
        bc = spec["bottom_crop"]

        filters: list[str] = []

        # Split the source into top (facecam) and bottom (gameplay/mediashare) branches.
        filters.append("[0:v]setpts=PTS-STARTPTS,split=2[v_top][v_bot]")

        # Top: facecam. The crop is panel-aspect and >= panel size (LayoutBuilder enlarges a small
        # cam to fit), so a plain crop+scale downscales it sharp — no blur, no left/right bars.
        fc_crop = f"crop={fc['w']}:{fc['h']}:{fc['x']}:{fc['y']}"
        filters.append(f"[v_top]{fc_crop},scale={panel_w}:{panel_h}[top]")

        # Bottom panel:
        #   - mediashare → blurred-fill of the popup box (odd aspect → no distortion)
        #   - gameplay (default) → blurred-fill of the cam-free region (all gameplay visible)
        #   - gameplay + follow-motion → animated pan crop-fill
        bottom_track = spec.get("bottom_track") or []
        crop_box = f"crop={bc['w']}:{bc['h']}:{bc['x']}:{bc['y']}"
        if spec.get("bottom_mode") == "gameplay" and bottom_track:
            crop_expr = self._get_face_crop_filter(
                bottom_track, spec["video_width"], spec["video_height"]
            )
            filters.append(f"[v_bot]{crop_expr},scale={panel_w}:{panel_h}[bot]")
        else:
            # gameplay default and mediashare both zoom-out + blur-fill their crop region.
            filters.extend(self._blurred_fill_filters("[v_bot]", crop_box, "[bot]"))

        # Stack the two equal panels into the 1080x1920 canvas. GAMING_SOLO_BOTTOM flips the
        # order (gameplay top, facecam bottom) — same panels, mirrored stack.
        stack_order = "[bot][top]" if spec.get("facecam_bottom") else "[top][bot]"
        filters.append(f"{stack_order}vstack=inputs=2[stacked]")

        # Burn subtitles onto the stacked canvas (or just normalise pixel format).
        v_filter = "[stacked]format=yuv420p,setsar=1"
        v_filter = self._append_subtitles_filter(v_filter, subtitle_ass_path)
        filters.append(f"{v_filter}[out_v]")

        return filters

    def _blurred_fill_filters(
        self,
        src_label: str,
        crop_filter: str | None,
        out_label: str,
        prefix: str = "gp",
    ) -> list[str]:
        """Composite a source into a panel with a blurred-background vertical fill.

        The source (optionally cropped) is downscaled to fit the panel and centred over a blurred,
        panel-filling copy of itself — zoomed-out and sharp (no upscaling, no black bars), instead
        of a tight zoom. Used for the gameplay/mediashare bottom and the zoomed-out facecam top.

        Args:
            src_label: FFmpeg stream label feeding the panel, e.g. "[v_bot]".
            crop_filter: Optional "crop=w:h:x:y" applied before fill; None = use the full frame.
            out_label: Output stream label, e.g. "[bot]".
            prefix: Unique label prefix so multiple blur-fills in one graph don't collide.

        Returns:
            The list of filtergraph statements producing ``out_label`` (1080x960).
        """
        panel_w = STACK2_PANEL_W
        panel_h = STACK2_PANEL_H
        pre = f"{src_label}{crop_filter + ',' if crop_filter else ''}"
        bg, fg = f"{prefix}_bg", f"{prefix}_fg"
        bgb, fgs = f"{prefix}_bgb", f"{prefix}_fgs"
        return [
            f"{pre}split=2[{bg}][{fg}]",
            (
                f"[{bg}]scale={panel_w}:{panel_h}:force_original_aspect_ratio=increase,"
                f"crop={panel_w}:{panel_h},boxblur=20:1[{bgb}]"
            ),
            f"[{fg}]scale={panel_w}:{panel_h}:force_original_aspect_ratio=decrease[{fgs}]",
            f"[{bgb}][{fgs}]overlay=(W-w)/2:(H-h)/2{out_label}",
        ]

    def _build_mode_c_filters(
        self, spec: dict, subtitle_ass_path: Path | None
    ) -> list[str]:
        """Build Mode C multi-face collab stack (Facecam Top + Gameplay Center + Collab Grid Bottom)."""
        filters = []
        filters.append("[0:v]setpts=PTS-STARTPTS,split=3[v_f][v_g][v_c]")

        fc = spec["facecam_crop"]
        filters.append(
            f"[v_f]crop={fc['w']}:{fc['h']}:{fc['x']}:{fc['y']},scale=1080:640[facecam]"
        )

        # Gameplay centre: animated pan (same engine as Mode B) when a track exists, else static crop.
        gc = spec["gameplay_crop"]
        gameplay_track = spec.get("gameplay_track") or []
        if gameplay_track:
            crop_expr = self._get_face_crop_filter(
                gameplay_track, spec["video_width"], spec["video_height"]
            )
            filters.append(f"[v_g]{crop_expr},scale=1080:640[gameplay]")
        else:
            filters.append(
                f"[v_g]crop={gc['w']}:{gc['h']}:{gc['x']}:{gc['y']},scale=1080:640[gameplay]"
            )

        cc = spec["collab_crop"]
        filters.append(
            f"[v_c]crop={cc['w']}:{cc['h']}:{cc['x']}:{cc['y']},scale=1080:640[collab]"
        )

        v_filter = "[facecam][gameplay][collab]vstack=inputs=3,format=yuv420p,setsar=1"
        v_filter_final = self._append_subtitles_filter(v_filter, subtitle_ass_path)

        filters.append(f"{v_filter_final}[out_v]")
        return filters
