from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr

from src.core.config import load_config
from src.core.utils import load_timerange_file, parse_timerange_text
from src.core.workspace import VIDEOS_DIR, ensure_workspace_integrity, run_purge_cycle
from src.interfaces.components import clip_count_slider, clip_duration_slider


def _build_language_choices() -> list[tuple[str, str]]:
    from src.ai.prompts import LANGUAGE_MAP

    seen: set[str] = set()
    choices = [("Auto-Detect", "auto")]
    for name, code in sorted(LANGUAGE_MAP.items()):
        display = name.title()
        if display not in seen:
            seen.add(display)
            choices.append((display, code))
    return choices


_LANGUAGE_CHOICES = _build_language_choices()

_CONTENT_TYPE_CHOICES: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("Podcast", "PODCAST"),
    ("Just Chat", "JUST_CHAT"),
    ("Gaming Solo", "GAMING_SOLO"),
    ("Gaming Solo (Cam Bottom)", "GAMING_SOLO_BOTTOM"),
    ("Gaming Collab", "GAMING_COLLAB"),
    ("Donation / Alert", "DONATION_OVERLAY"),
]

# ── Human-friendly choice mappings ──

_strategy_choices: list[tuple[str, str]] = [
    ("Combined (Recommended)", "hybrid"),
    ("YouTube Most Replayed", "heatmap"),
    ("AI Analysis", "ai"),
]

_encoder_choices: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("CPU (Software)", "cpu"),
    ("NVIDIA GPU", "nvenc"),
    ("Intel GPU", "qsv"),
    ("Apple GPU", "videotoolbox"),
]

_provider_choices: list[tuple[str, str]] = [
    ("Auto (Cloud First)", "auto"),
    ("Online (Cloud)", "cloud"),
    ("On This Computer (Local)", "local"),
]

_cloud_provider_choices: list[tuple[str, str]] = [
    ("Google (Gemini)", "google"),
    ("OpenAI / Compatible", "openai"),
]

_hardware_choices: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("CPU", "cpu"),
    ("NVIDIA GPU", "cuda"),
]

_model_size_choices: list[tuple[str, str]] = [
    ("Fastest", "tiny"),
    ("Basic", "base"),
    ("Balanced", "small"),
    ("Better", "medium"),
    ("Best", "large-v3"),
]

_hallucination_choices: list[tuple[str, str]] = [
    ("Strict", "and"),
    ("Relaxed", "or"),
]

_donation_exclude_choices: list[tuple[str, str]] = [
    ("Podcast", "PODCAST"),
    ("Just Chat", "JUST_CHAT"),
    ("Gaming Solo", "GAMING_SOLO"),
    ("Gaming Collab", "GAMING_COLLAB"),
]

_alignment_choices: list[tuple[str, str]] = [
    ("Bottom Left", "bottom-left"),
    ("Bottom Center", "bottom-center"),
    ("Bottom Right", "bottom-right"),
    ("Middle Left", "middle-left"),
    ("Center", "center"),
    ("Middle Right", "middle-right"),
    ("Top Left", "top-left"),
    ("Top Center", "top-center"),
    ("Top Right", "top-right"),
]

_browser_choices: list[tuple[str, str]] = [
    ("Edge", "edge"),
    ("Chrome", "chrome"),
    ("Firefox", "firefox"),
    ("Brave", "brave"),
    ("Opera", "opera"),
]

# Populated by _build_settings_tab, consumed by _apply_settings.
_SETTINGS_PATHS: list[str] = []


def _refresh_cache_info() -> list[list]:
    from src.core.workspace import cache_usage

    rows = cache_usage()
    return [
        [
            r["name"],
            round(r["size_mb"], 2),
            r["count"],
            f"{r['oldest_days']:.1f}d" if r["oldest_days"] is not None else "-",
        ]
        for r in rows
    ]


def _run_purge(dry_run: bool) -> tuple[list[list], str]:
    from src.core.workspace import cache_usage, run_purge_cycle

    before = {r["name"]: r for r in cache_usage()}
    run_purge_cycle(force=not dry_run)
    after = {r["name"]: r for r in cache_usage()}
    lines: list[str] = []
    total_freed = 0.0
    for name in before:
        freed = before[name]["size_mb"] - after[name]["size_mb"]
        if freed > 0.01:
            lines.append(f"{name}: {freed:.2f} MB freed")
            total_freed += freed
    summary = (
        f"Dry run — would free {total_freed:.2f} MB"
        if dry_run
        else f"Freed {total_freed:.2f} MB total"
    )
    return _refresh_cache_info(), summary


def build_ui() -> gr.Blocks:
    cfg = load_config()
    cs = cfg.clip_selection

    with gr.Blocks(title="YaClip — AI Auto-Clipper") as app:
        gr.Markdown("# YaClip — AI Auto-Clipper")

        with gr.Tab("Clipper"):
            gr.HTML("<style>.mode-hide{display:none!important}</style>")

            is_manual = cs.mode == "manual"

            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            force_redo = gr.Checkbox(label="Force Re-Download Video", value=False)

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
                    choices=_strategy_choices,
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
                    label="Output Resolution",
                )
                video_encoder = gr.Dropdown(
                    choices=_encoder_choices,
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
                    label="Or upload Timerange File (.txt)",
                    file_types=[".txt"],
                )

            language_dropdown = gr.Dropdown(
                choices=_LANGUAGE_CHOICES,
                value="auto",
                label="Language",
            )
            sys_prompt_override = gr.Textbox(
                label="Custom AI Instructions (Optional)",
                placeholder="Leave blank to use default settings",
                lines=3,
            )

            find_btn = gr.Button("Find Clips")
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
            )

            find_btn.click(
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
                    video_encoder,
                    stt_context,
                    language_dropdown,
                    sys_prompt_override,
                    force_redo,
                    timerange_input,
                    timerange_file,
                ],
                outputs=[pipeline_state],
            )

        with gr.Tab("Review & Render"):
            pass

        with gr.Tab("Settings"):
            _build_settings_tab(cfg)

        with gr.Tab("Maintenance") as maintenance_tab:
            gr.Markdown("## Cache Management")

            refresh_btn = gr.Button("⟳ Refresh Cache Info")
            usage_table = gr.Dataframe(
                value=_refresh_cache_info(),
                headers=["Directory", "Size (MB)", "Files", "Oldest"],
                datatype=["str", "number", "number", "str"],
                column_count=(4, "fixed"),
                row_count=10,
            )

            with gr.Row():
                dry_run_cb = gr.Checkbox(
                    label="Dry Run (preview only, no deletion)", value=True
                )
                clear_btn = gr.Button("🗑️ Clear Cache Now", variant="stop")

            result_log = gr.Textbox(label="Result", lines=4, interactive=False)

            maintenance_tab.select(fn=_refresh_cache_info, outputs=[usage_table])
            refresh_btn.click(fn=_refresh_cache_info, outputs=[usage_table])

            clear_btn.click(
                fn=_run_purge,
                inputs=[dry_run_cb],
                outputs=[usage_table, result_log],
            )

    return app


def launch_webui(host: str = "127.0.0.1", port: int = 7860) -> None:
    cfg = load_config().web_server
    ui = build_ui()
    ui.queue().launch(
        server_name=host or cfg.host,
        server_port=port or cfg.port,
        share=cfg.share,
    )


def _build_settings_tab(cfg):
    stt = cfg.ai_pipeline.stt
    stt_c = stt.cloud
    stt_l = stt.local
    stt_a = stt_l.advanced
    llm = cfg.ai_pipeline.llm
    llm_c = llm.cloud
    llm_l = llm.local
    dl = cfg.downloader
    cs = cfg.clip_selection
    vp = cfg.video_processing
    rd = vp.region_detection
    sub = vp.subtitles

    gr.Markdown("## Settings")
    gr.Markdown(
        "Changes apply to this session only. For permanent settings, "
        "close the app and edit the `config.yaml` file."
    )

    with gr.Accordion("Speech to Text", open=True):
        s_provider = gr.Radio(
            _provider_choices,
            label="Engine",
            value=stt.provider,
        )
        with gr.Accordion("Online (Cloud)", open=False):
            s_c_provider = gr.Radio(
                _cloud_provider_choices,
                label="Online Service",
                value=stt_c.provider,
            )
            s_c_model = gr.Textbox(label="Model", value=stt_c.model)
            s_c_timeout = gr.Slider(
                30,
                600,
                step=10,
                label="Timeout (seconds)",
                value=stt_c.timeout,
            )
        with gr.Accordion("On This Computer (Local)", open=False):
            s_l_device = gr.Dropdown(
                _hardware_choices,
                label="Hardware",
                value=stt_l.device,
            )
            s_l_model_size = gr.Dropdown(
                _model_size_choices,
                label="Accuracy / Speed",
                value=stt_l.model_size,
            )
            with gr.Accordion("Advanced", open=False):
                s_l_a_beam = gr.Slider(
                    1,
                    20,
                    step=1,
                    label="Whisper Accuracy",
                    value=stt_a.beam_size,
                )
                s_l_a_vad_t = gr.Slider(
                    0.0,
                    1.0,
                    step=0.05,
                    label="Voice Detection Sensitivity",
                    value=stt_a.vad_threshold,
                )
                s_l_a_vad_ms = gr.Slider(
                    100,
                    3000,
                    step=50,
                    label="Minimum Pause Between Sentences (ms)",
                    value=stt_a.vad_min_silence_ms,
                )
                s_l_a_suppress = gr.Checkbox(
                    label="Skip Silent Sections",
                    value=stt_a.suppress_blank,
                )
                s_l_a_hall = gr.Radio(
                    _hallucination_choices,
                    label="Filter Repeating Words",
                    value=stt_a.hallucination_gate,
                )
                s_l_a_repeat = gr.Slider(
                    0.1,
                    1.0,
                    step=0.05,
                    label="Repetition Filter Level",
                    value=stt_a.repeat_token_ratio_max,
                )
                s_l_a_rep_pen = gr.Slider(
                    1.0,
                    2.0,
                    step=0.05,
                    label="Anti-Repetition Strength",
                    value=stt_a.repetition_penalty,
                )

    with gr.Accordion("AI Clip Selection", open=True):
        l_provider = gr.Radio(
            _provider_choices,
            label="Engine",
            value=llm.provider,
        )
        with gr.Accordion("Online (Cloud)", open=False):
            l_c_provider = gr.Radio(
                _cloud_provider_choices,
                label="Online Service",
                value=llm_c.provider,
            )
            l_c_base = gr.Textbox(
                label="Custom API URL (Optional)",
                value=llm_c.base_url or "",
            )
            l_c_model = gr.Textbox(label="Model", value=llm_c.model)
            l_c_timeout = gr.Slider(
                30,
                600,
                step=10,
                label="Timeout (seconds)",
                value=llm_c.timeout,
            )
        with gr.Accordion("On This Computer (Local)", open=False):
            l_l_device = gr.Dropdown(
                _hardware_choices,
                label="Hardware",
                value=llm_l.device,
            )
            l_l_n_gpu = gr.Slider(
                -1,
                128,
                step=1,
                label="GPU Offload Layers",
                value=llm_l.n_gpu_layers,
            )
            l_l_model = gr.Textbox(label="Model Name", value=llm_l.model_name)

    with gr.Accordion("Download", open=True):
        d_browser = gr.Dropdown(
            _browser_choices,
            label="Login Using Browser",
            value=dl.browser_cookies,
        )
        d_vid_fmt = gr.Dropdown(
            ["mp4", "webm", "mkv"],
            label="Video Format",
            value=dl.video_format,
        )
        d_aud_fmt = gr.Dropdown(
            ["aac", "mp3", "opus"],
            label="Audio Format",
            value=dl.audio_format,
        )
        d_aud_q = gr.Dropdown(
            ["128k", "192k", "256k", "320k"],
            label="Audio Quality",
            value=dl.audio_quality,
        )

    with gr.Accordion("Clip Limits", open=True):
        cs_min_clips = gr.Slider(
            1,
            50,
            step=1,
            label="Minimum Clips per Video",
            value=cs.min_clips,
        )
        cs_max_clips = gr.Slider(
            1,
            100,
            step=1,
            label="Maximum Clips per Video",
            value=cs.max_clips,
        )
        cs_min_dur = gr.Slider(
            10,
            300,
            step=5,
            label="Shortest Clip (seconds)",
            value=cs.min_clip_duration_seconds,
        )
        cs_max_dur = gr.Slider(
            10,
            600,
            step=5,
            label="Longest Clip (seconds)",
            value=cs.max_clip_duration_seconds,
        )
        cs_heat = gr.Slider(
            1,
            100,
            step=1,
            label="YouTube Highlights Sensitivity",
            value=cs.heatmap_threshold_percentile,
        )
        cs_spike = gr.Slider(
            10,
            200,
            step=1,
            label="Maximum Moments to Consider",
            value=cs.spike_pool_size,
        )

    with gr.Accordion("Video Processing", open=True):
        vp_device = gr.Dropdown(
            _hardware_choices,
            label="Processing Hardware",
            value=vp.device,
        )
        vp_det_conf = gr.Slider(
            0.1,
            1.0,
            step=0.05,
            label="Content Detection Certainty",
            value=vp.detection_confidence_threshold,
        )
        vp_face = gr.Checkbox(
            label="Smart Face Tracking",
            value=vp.auto_face_tracking,
        )
        vp_don = gr.Checkbox(
            label="Show Donation / Alert Popups",
            value=vp.preserve_donation_overlays,
        )
        vp_don_excl = gr.CheckboxGroup(
            _donation_exclude_choices,
            label="Don't Show Popups On",
            value=list(vp.donation_overlay_exclude_types),
        )

    with gr.Accordion("Scene Detection", open=True):
        rd_enabled = gr.Checkbox(label="Enabled", value=rd.enabled)
        rd_model = gr.Textbox(label="Detection Model File", value=rd.model_name)
        rd_frames = gr.Slider(
            1,
            20,
            step=1,
            label="Analysis Detail",
            value=rd.sample_frames,
        )
        rd_device = gr.Dropdown(
            _hardware_choices,
            label="Hardware",
            value=rd.device,
        )
        rd_motion = gr.Checkbox(
            label="Camera Follows Action",
            value=rd.gameplay_follow_motion,
        )
        rd_zoom = gr.Slider(
            1.0,
            2.0,
            step=0.05,
            label="Gameplay Zoom",
            value=rd.gameplay_zoom,
        )

    with gr.Accordion("Caption Style", open=True):
        sub_enabled = gr.Checkbox(label="Show Captions", value=sub.enabled)
        sub_collab = gr.Checkbox(
            label="Show on Multi-Player Layout", value=sub.collab_enabled
        )
        sub_upper = gr.Checkbox(label="ALL CAPS", value=sub.uppercase)
        sub_font = gr.Textbox(label="Font File", value=sub.font_file)
        sub_size = gr.Slider(20, 200, step=1, label="Font Size", value=sub.font_size)
        sub_pc = gr.ColorPicker(label="Text Color", value="#FFFFFF")
        sub_hc = gr.ColorPicker(label="Active Word Color", value="#96C8FF")
        sub_oc = gr.ColorPicker(label="Text Outline Color", value="#000000")
        sub_ot = gr.Slider(
            0, 20, step=1, label="Outline Weight", value=sub.outline_thickness
        )
        sub_bold = gr.Checkbox(label="Bold", value=sub.bold)
        sub_shadow = gr.Checkbox(label="Shadow", value=sub.shadow)
        sub_align = gr.Dropdown(
            _alignment_choices,
            label="Position",
            value="bottom-center",
        )
        sub_margin = gr.Slider(
            0, 1920, step=10, label="Distance from Bottom", value=sub.margin_v
        )

    apply_btn = gr.Button("Apply Settings")
    apply_status = gr.Markdown(visible=False)

    widget_list: list[gr.components.Component] = [
        s_provider,
        s_c_provider,
        s_c_model,
        s_c_timeout,
        s_l_device,
        s_l_model_size,
        s_l_a_beam,
        s_l_a_vad_t,
        s_l_a_vad_ms,
        s_l_a_suppress,
        s_l_a_hall,
        s_l_a_repeat,
        s_l_a_rep_pen,
        l_provider,
        l_c_provider,
        l_c_base,
        l_c_model,
        l_c_timeout,
        l_l_device,
        l_l_n_gpu,
        l_l_model,
        d_browser,
        d_vid_fmt,
        d_aud_fmt,
        d_aud_q,
        cs_min_clips,
        cs_max_clips,
        cs_min_dur,
        cs_max_dur,
        cs_heat,
        cs_spike,
        vp_device,
        vp_det_conf,
        vp_face,
        vp_don,
        vp_don_excl,
        rd_enabled,
        rd_model,
        rd_frames,
        rd_device,
        rd_motion,
        rd_zoom,
        sub_enabled,
        sub_collab,
        sub_upper,
        sub_font,
        sub_size,
        sub_pc,
        sub_hc,
        sub_oc,
        sub_ot,
        sub_bold,
        sub_shadow,
        sub_align,
        sub_margin,
    ]

    global _SETTINGS_PATHS
    _SETTINGS_PATHS.clear()
    path_list = [
        "ai_pipeline.stt.provider",
        "ai_pipeline.stt.cloud.provider",
        "ai_pipeline.stt.cloud.model",
        "ai_pipeline.stt.cloud.timeout",
        "ai_pipeline.stt.local.device",
        "ai_pipeline.stt.local.model_size",
        "ai_pipeline.stt.local.advanced.beam_size",
        "ai_pipeline.stt.local.advanced.vad_threshold",
        "ai_pipeline.stt.local.advanced.vad_min_silence_ms",
        "ai_pipeline.stt.local.advanced.suppress_blank",
        "ai_pipeline.stt.local.advanced.hallucination_gate",
        "ai_pipeline.stt.local.advanced.repeat_token_ratio_max",
        "ai_pipeline.stt.local.advanced.repetition_penalty",
        "ai_pipeline.llm.provider",
        "ai_pipeline.llm.cloud.provider",
        "ai_pipeline.llm.cloud.base_url",
        "ai_pipeline.llm.cloud.model",
        "ai_pipeline.llm.cloud.timeout",
        "ai_pipeline.llm.local.device",
        "ai_pipeline.llm.local.n_gpu_layers",
        "ai_pipeline.llm.local.model_name",
        "downloader.browser_cookies",
        "downloader.video_format",
        "downloader.audio_format",
        "downloader.audio_quality",
        "clip_selection.min_clips",
        "clip_selection.max_clips",
        "clip_selection.min_clip_duration_seconds",
        "clip_selection.max_clip_duration_seconds",
        "clip_selection.heatmap_threshold_percentile",
        "clip_selection.spike_pool_size",
        "video_processing.device",
        "video_processing.detection_confidence_threshold",
        "video_processing.auto_face_tracking",
        "video_processing.preserve_donation_overlays",
        "video_processing.donation_overlay_exclude_types",
        "video_processing.region_detection.enabled",
        "video_processing.region_detection.model_name",
        "video_processing.region_detection.sample_frames",
        "video_processing.region_detection.device",
        "video_processing.region_detection.gameplay_follow_motion",
        "video_processing.region_detection.gameplay_zoom",
        "video_processing.subtitles.enabled",
        "video_processing.subtitles.collab_enabled",
        "video_processing.subtitles.uppercase",
        "video_processing.subtitles.font_file",
        "video_processing.subtitles.font_size",
        "video_processing.subtitles.primary_color",
        "video_processing.subtitles.highlight_color",
        "video_processing.subtitles.outline_color",
        "video_processing.subtitles.outline_thickness",
        "video_processing.subtitles.bold",
        "video_processing.subtitles.shadow",
        "video_processing.subtitles.alignment",
        "video_processing.subtitles.margin_v",
    ]
    _SETTINGS_PATHS.extend(path_list)

    apply_btn.click(
        fn=_apply_settings,
        inputs=widget_list,
        outputs=[apply_status],
        queue=False,
    )


def _apply_settings(*values) -> str:
    from src.core.config import apply_session_overrides

    global _SETTINGS_PATHS
    overrides: dict[str, object] = {}
    for path, val in zip(_SETTINGS_PATHS, values, strict=True):
        if val == "" or val is None:
            val = None
        overrides[path] = val
    try:
        apply_session_overrides(overrides)
        return "✅ Settings applied for this session."
    except Exception as e:
        return f"❌ Failed: {e}"


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
    video_encoder: str,
    stt_context: str,
    language: str,
    sys_prompt_override: str,
    force: bool,
    timerange_text: str,
    timerange_file: str | None,
    progress: gr.Progress = None,
) -> dict:
    if progress is None:
        progress = gr.Progress()
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
    cfg.video_processing.video_encoder = video_encoder
    cs.mode = "manual" if mode == "Manual" else "auto"
    if sys_prompt_override.strip():
        cfg.dk_clipper_sys_prompt = sys_prompt_override.strip()
    else:
        cfg.dk_clipper_sys_prompt = None

    progress(0.0, desc="Getting ready...")
    await asyncio.to_thread(ensure_workspace_integrity)
    await asyncio.to_thread(run_purge_cycle)

    from src.media.downloader import VideoDownloader

    progress(0.1, desc="Downloading video...")
    try:
        result = await asyncio.to_thread(
            VideoDownloader().download_video,
            url,
            str(VIDEOS_DIR),
            force=force,
        )
    except Exception as e:
        raise gr.Error(f"Download failed: {e}") from e

    audio_path = result.get("audio_path")
    video_path = result.get("video_path")
    if not audio_path or not video_path:
        raise gr.Error("Download did not produce both video and audio files.")

    progress(0.4, desc="Analyzing video type...")
    from src.vision.content_type_detector import ContentTypeDetector

    detection_result = await asyncio.to_thread(
        ContentTypeDetector().detect_content_type_full,
        Path(video_path),
    )
    content_type = detection_result.content_type

    progress(0.6, desc="Finding best moments...")
    from src.ai.pipeline import AIPipeline

    try:
        clips = await asyncio.to_thread(
            AIPipeline().process_audio,
            audio_path,
            video_path=video_path,
            force=force,
            detected_type=content_type,
            detection_evidence=(
                detection_result.evidence if not detection_result.is_confident else None
            ),
            manual_ranges=manual_ranges,
            no_metadata=False,
        )
    except Exception as e:
        raise gr.Error(f"Clip selection failed: {e}") from e

    progress(1.0, desc="Done.")
    return {
        "proposals": clips,
        "video_path": video_path,
        "content_type": content_type.value if content_type else None,
    }
