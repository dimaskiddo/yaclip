from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.core.config import load_config
from src.core.utils import load_timerange_file, parse_timerange_text
from src.core.workspace import VIDEOS_DIR, ensure_workspace_integrity, run_purge_cycle
from src.interfaces.components import (
    _CONTENT_TYPE_CHOICES,
    _ENCODER_CHOICES,
    _STRATEGY_CHOICES,
    LANGUAGE_CHOICES,
    clip_count_slider,
    clip_duration_slider,
)


def _refresh_controls() -> list[gr.update]:
    """Return Clipper-tab control updates from the current config singleton."""
    cfg = load_config()
    cs = cfg.clip_selection
    vp = cfg.video_processing
    return [
        gr.update(value=cs.auto_strategy),
        gr.update(minimum=cs.min_clips, maximum=cs.max_clips),
        gr.update(
            minimum=cs.min_clip_duration_seconds,
            maximum=cs.max_clip_duration_seconds,
        ),
        gr.update(value=cs.clip_length_margin_seconds),
        gr.update(value=cs.candidate_margin),
        gr.update(value=cs.require_review_before_render),
        gr.update(value=vp.content_type_override),
        gr.update(value=vp.fast_mode),
        gr.update(value=vp.default_resolution),
        gr.update(value=cfg.downloader.target_resolution),
        gr.update(value=vp.video_encoder),
        gr.update(value=vp.subtitles.stt_context),
    ]


async def _run_clipper_pipeline(
    url: str,
    mode: str,
    auto_strategy: str,
    clip_count: int,
    clip_duration: int,
    clip_margin: int,
    candidate_margin: int,
    require_review: bool,
    content_type_override: str,
    fast_mode: bool,
    target_resolution: str,
    download_resolution: str,
    video_encoder: str,
    stt_context: str,
    language: str,
    sys_prompt_override: str,
    force: bool,
    cookies_file_input: str | None,
    timerange_text: str,
    timerange_file: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[dict | None, str, gr.update]:
    if not url:
        raise gr.Error("YouTube URL is required.")
    manual_ranges: list[dict] | None = None
    if mode == "Manual":
        if timerange_file:
            try:
                manual_ranges = load_timerange_file(Path(timerange_file))
            except (ValueError, OSError) as e:
                raise gr.Error(f"Failed to parse timerange file: {e}") from e
        elif timerange_text.strip():
            try:
                manual_ranges = parse_timerange_text(timerange_text)
            except ValueError as e:
                raise gr.Error(str(e)) from e
        else:
            raise gr.Error("Manual mode requires timeranges (textarea or file).")
    # Patch config
    cfg = load_config()
    cs = cfg.clip_selection
    cs.default_clips = clip_count
    cs.default_clip_duration_seconds = clip_duration
    cs.auto_strategy = auto_strategy
    cs.clip_length_margin_seconds = clip_margin
    cs.candidate_margin = candidate_margin
    cs.require_review_before_render = require_review
    cs._coherent_durations()
    cfg.video_processing.subtitles.language = language
    cfg.video_processing.subtitles.stt_context = stt_context
    cfg.video_processing.content_type_override = content_type_override
    cfg.video_processing.fast_mode = fast_mode
    cfg.video_processing.default_resolution = target_resolution
    cfg.downloader.target_resolution = download_resolution
    cfg.video_processing.video_encoder = video_encoder
    cs.mode = "manual" if mode == "Manual" else "auto"
    if sys_prompt_override.strip():
        cfg.dk_clipper_sys_prompt = sys_prompt_override.strip()
    else:
        cfg.dk_clipper_sys_prompt = None

    progress(0.0, desc="Getting ready...")
    await asyncio.to_thread(ensure_workspace_integrity)
    await asyncio.to_thread(run_purge_cycle)

    try:
        progress(0.2, desc="Downloading video and audio...")
        from src.media.downloader import VideoDownloader

        cookies_path = cookies_file_input or None
        try:
            result = await asyncio.to_thread(
                VideoDownloader().download_video,
                url,
                str(VIDEOS_DIR),
                force=force,
                cookies_file=cookies_path,
            )
        except Exception as e:
            raise gr.Error(f"Download failed: {e}") from e

        audio_path = result.get("audio_path")
        video_path = result.get("video_path")
        if not audio_path or not video_path:
            raise gr.Error("Download did not produce both video and audio files.")

        progress(0.5, desc="Analyzing video type...")
        from src.vision.content_type_detector import ContentTypeDetector

        detection_result = await asyncio.to_thread(
            ContentTypeDetector().detect_content_type_full,
            Path(video_path),
        )
        content_type = detection_result.content_type

        progress(0.7, desc="AI clip selection...")
        from src.ai.pipeline import AIPipeline

        try:
            clips = await asyncio.to_thread(
                AIPipeline().process_audio,
                audio_path,
                video_path=video_path,
                force=force,
                detected_type=content_type,
                detection_evidence=(
                    detection_result.evidence
                    if not detection_result.is_confident
                    else None
                ),
                manual_ranges=manual_ranges,
                no_metadata=False,
            )
        except Exception as e:
            raise gr.Error(f"Clip selection failed: {e}") from e

        progress(0.9, desc="Finalizing results...")
        return (
            {
                "proposals": clips,
                "video_path": video_path,
                "content_type": content_type.value if content_type else None,
            },
            "✅ Clips found. Switch to **Review & Render** tab above to review.",
            gr.update(interactive=True, value="Find Clips"),
        )
    except Exception:
        raise


def build_clipper_tab(cfg) -> SimpleNamespace:
    """Build the Clipper tab. All event wiring here is self-contained (no cross-tab deps)."""
    cs = cfg.clip_selection
    with gr.Tab("Clipper") as clipper_tab:
        gr.HTML(
            "<style>"
            "#clips-css-wrap { display: none !important; }"
            ".mode-hide{display:none!important}"
            "#clipper-progress{min-height:60px}"
            "#clipper-progress>div{padding-top:8px}"
            "#render-progress{min-height:60px}"
            "#render-progress>div{padding-top:8px}"
            ".render-lock{pointer-events:none;opacity:0.55}"
            "</style>",
            elem_id="clips-css-wrap",
        )
        is_manual = cs.mode == "manual"
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        download_resolution = gr.Dropdown(
            choices=["480p", "720p", "1080p", "1440p", "4K"],
            value=cfg.downloader.target_resolution,
            label="Video Download Resolution",
        )
        force_redo = gr.Checkbox(label="Force Re-Download Video", value=False)
        cookies_file_input = gr.File(
            label="Cookies File (Optional)",
            file_types=[".txt"],
        )
        mode_radio = gr.Radio(
            choices=["Auto", "Manual"],
            value="Manual" if is_manual else "Auto",
            label="Clip Selection Mode",
        )
        with gr.Column(
            elem_classes=["auto-controls", "mode-hide"]
            if is_manual
            else ["auto-controls"]
        ):
            auto_strategy = gr.Radio(
                choices=_STRATEGY_CHOICES,
                value=cs.auto_strategy,
                label="Selection Method",
            )
            clip_count = clip_count_slider(cs)
            clip_duration = clip_duration_slider(cs)
            clip_margin = gr.Slider(
                minimum=0,
                maximum=60,
                value=cs.clip_length_margin_seconds,
                step=1,
                label="Extra Clip Length (seconds)",
            )
            candidate_margin = gr.Slider(
                minimum=0,
                maximum=15,
                value=cs.candidate_margin,
                step=1,
                label="Extra Clip Options for AI Selection",
            )
            require_review = gr.Checkbox(
                label="Review Before Rendering",
                value=cs.require_review_before_render,
            )
            content_type_override = gr.Dropdown(
                choices=_CONTENT_TYPE_CHOICES,
                value=cfg.video_processing.content_type_override,
                label="Video Type",
            )
            fast_mode = gr.Checkbox(
                label="Speed Mode (Faster But Lower Quality)",
                value=cfg.video_processing.fast_mode,
            )
            target_resolution = gr.Dropdown(
                choices=["480p", "720p", "1080p", "1440p", "4K"],
                value=cfg.video_processing.default_resolution,
                label="Video Clip Output Resolution",
            )
            video_encoder = gr.Dropdown(
                choices=_ENCODER_CHOICES,
                value=cfg.video_processing.video_encoder,
                label="Video Encoder",
            )
            stt_context = gr.Textbox(
                label="Keyword Hints",
                placeholder="names, game terms, famous people...",
                value=cfg.video_processing.subtitles.stt_context,
            )
        with gr.Column(
            elem_classes=["manual-controls", "mode-hide"]
            if not is_manual
            else ["manual-controls"]
        ):
            timerange_input = gr.TextArea(
                label="Timeranges",
                placeholder="1:30 - 2:30\n3:00 - 4:00 | JUST_CHAT",
                lines=4,
            )
            timerange_file = gr.File(
                label="Or Upload Timerange File",
                file_types=[".txt"],
            )
        language_dropdown = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value="auto",
            label="Language",
        )
        sys_prompt_override = gr.Textbox(
            label="Custom AI Instructions (Optional)",
            placeholder="Leave blank to use default settings",
            lines=3,
        )
        find_btn = gr.Button("Find Clips")
        clipper_progress = gr.Markdown("", elem_id="clipper-progress")
        pipeline_state = gr.State()
        mode_radio.change(
            None,
            inputs=[mode_radio],
            js="""
            (mode) => {
                document.querySelector('.auto-controls').classList.toggle('mode-hide', mode !== 'Auto');
                document.querySelector('.manual-controls').classList.toggle('mode-hide', mode !== 'Manual');
            }
            """,
            queue=False,
            api_name=False,
        )
        clipper_tab.select(
            fn=_refresh_controls,
            outputs=[
                auto_strategy,
                clip_count,
                clip_duration,
                clip_margin,
                candidate_margin,
                require_review,
                content_type_override,
                fast_mode,
                target_resolution,
                download_resolution,
                video_encoder,
                stt_context,
            ],
            api_name="clipper-refresh-controls",
        )
        find_btn.click(
            fn=lambda: gr.update(interactive=False, value="Finding Clips..."),
            outputs=[find_btn],
            queue=False,
            api_name=False,
        ).then(
            fn=_run_clipper_pipeline,
            inputs=[
                url_input,
                mode_radio,
                auto_strategy,
                clip_count,
                clip_duration,
                clip_margin,
                candidate_margin,
                require_review,
                content_type_override,
                fast_mode,
                target_resolution,
                download_resolution,
                video_encoder,
                stt_context,
                language_dropdown,
                sys_prompt_override,
                force_redo,
                cookies_file_input,
                timerange_input,
                timerange_file,
            ],
            outputs=[pipeline_state, clipper_progress, find_btn],
            show_progress_on=[find_btn, clipper_progress],
            api_name="clipper-find-clips",
        ).success(
            fn=lambda: gr.update(interactive=True, value="Find Clips"),
            outputs=[find_btn],
            queue=False,
            api_name=False,
        )

    return SimpleNamespace(
        tab=clipper_tab,
        url_input=url_input,
        pipeline_state=pipeline_state,
        clipper_progress=clipper_progress,
        timerange_input=timerange_input,
        timerange_file=timerange_file,
    )
