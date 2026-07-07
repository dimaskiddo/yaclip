from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr

from src.core.config import (
    apply_session_overrides,
    list_config_backups,
    load_config,
    restore_config,
    save_config_to_disk,
)
from src.core.utils import load_timerange_file, mask_api_key, parse_timerange_text
from src.core.workspace import (
    VIDEOS_DIR,
    cache_usage,
    ensure_workspace_integrity,
    run_purge_cycle,
)
from src.interfaces.components import (
    _ALIGNMENT_CHOICES,
    _CLOUD_PROVIDER_CHOICES,
    _CONTENT_TYPE_CHOICES,
    _DONATION_EXCLUDE_CHOICES,
    _ENCODER_CHOICES,
    _HALLUCINATION_CHOICES,
    _HARDWARE_CHOICES,
    _MODEL_SIZE_CHOICES,
    _PROVIDER_CHOICES,
    _STRATEGY_CHOICES,
    LANGUAGE_CHOICES,
    SETTINGS_HELP,
    clip_count_slider,
    clip_duration_slider,
)

_SETTINGS_PATHS: list[str] = []

_WIDGET_LIST: list = []

_NAME_SORT = {
    "clips": 0,
    "videos": 1,
    "audios": 2,
    "subtitles": 3,
    "data": 4,
    "tmp": 5,
    "logs": 6,
}

_NAME_DISPLAY: dict[str, str] = {
    "clips": "Clips",
    "videos": "Videos",
    "audios": "Audios",
    "subtitles": "Subtitles",
    "data": "Data",
    "tmp": "Temp",
    "logs": "Logs",
}


def _refresh_cache_info() -> list[list]:
    rows = sorted(cache_usage(), key=lambda r: _NAME_SORT.get(r["name"], 99))
    return [
        [
            _NAME_DISPLAY.get(r["name"], r["name"].title()),
            round(r["size_mb"], 2),
            r["count"],
            f"{r['oldest_days']:.1f}d" if r["oldest_days"] is not None else "-",
        ]
        for r in rows
    ]


def _run_purge(targets: list[str], dry_run: bool) -> tuple[list[list], str]:
    before = {r["name"]: r for r in cache_usage()}
    run_purge_cycle(force=not dry_run, specific_target=targets or None)

    after = {r["name"]: r for r in cache_usage()}
    lines: list[str] = []
    total_freed = 0.0

    for name in before:
        freed = before[name]["size_mb"] - after[name]["size_mb"]
        if freed > 0.01:
            lines.append(f"{name}: {freed:.2f} MB freed")
            total_freed += freed

    summary = (
        f"Dry run — Would free {total_freed:.2f} MB"
        if dry_run
        else f"Freed {total_freed:.2f} MB total"
    )

    return _refresh_cache_info(), summary


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


def build_ui() -> gr.Blocks:
    cfg = load_config()
    cs = cfg.clip_selection

    with gr.Blocks(title="YaClip — AI Auto-Clipper") as app:
        gr.Markdown("# YaClip — AI Auto-Clipper")

        with gr.Tab("Clipper") as clipper_tab:
            gr.HTML("<style>.mode-hide{display:none!important}</style>")

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
                outputs=[pipeline_state],
            )

        with gr.Tab("Review & Render"):
            pass

        with gr.Tab("Settings") as settings_tab:
            _build_settings_tab(cfg, settings_tab)

        with gr.Tab("Maintenance") as maintenance_tab:
            gr.Markdown("## Cache Management")

            usage_table = gr.Dataframe(
                value=_refresh_cache_info(),
                headers=["Directory", "Size (MB)", "Files", "Oldest"],
                datatype=["str", "number", "number", "str"],
                column_count=(4, "fixed"),
                row_count=10,
            )
            refresh_btn = gr.Button("⟳ Refresh Cache Info")

            purge_targets = gr.CheckboxGroup(
                [
                    ("Clips", "clips"),
                    ("Videos", "videos"),
                    ("Audios", "audios"),
                    ("Subtitles", "subtitles"),
                    ("Data", "data"),
                    ("Temp", "tmp"),
                    ("Logs", "logs"),
                ],
                label="Select Directories to Clean",
                value=["tmp"],
            )

            with gr.Row():
                dry_run_cb = gr.Checkbox(
                    label="Dry Run (Preview Only, No Deletion)", value=True
                )
                clear_btn = gr.Button("🗑️ Clear Cache Now", variant="stop")

            result_log = gr.Textbox(label="Result", lines=4, interactive=False)

            maintenance_tab.select(fn=_refresh_cache_info, outputs=[usage_table])
            refresh_btn.click(fn=_refresh_cache_info, outputs=[usage_table])

            clear_btn.click(
                fn=_run_purge,
                inputs=[purge_targets, dry_run_cb],
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


def _build_settings_tab(cfg, settings_tab):
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
            _PROVIDER_CHOICES,
            label="Engine",
            value=stt.provider,
            info=SETTINGS_HELP["stt_provider"],
        )
        with gr.Accordion("Online (Cloud)", open=False):
            s_c_provider = gr.Radio(
                _CLOUD_PROVIDER_CHOICES,
                label="Online Service",
                value=stt_c.provider,
                info=SETTINGS_HELP["stt_cloud_provider"],
            )
            s_c_base = gr.Textbox(
                label="Custom API URL (Optional)",
                value=stt_c.base_url or "",
                info=SETTINGS_HELP["stt_cloud_base_url"],
            )
            s_c_api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder=mask_api_key(stt_c.api_key),
                info=SETTINGS_HELP["stt_cloud_api_key"],
            )
            s_c_model = gr.Textbox(
                label="Model", value=stt_c.model, info=SETTINGS_HELP["stt_cloud_model"]
            )
            s_c_timeout = gr.Slider(
                30,
                600,
                step=10,
                label="Timeout (seconds)",
                value=stt_c.timeout,
                info=SETTINGS_HELP["stt_cloud_timeout"],
            )
        with gr.Accordion("On This Computer (Local)", open=False):
            s_l_device = gr.Dropdown(
                _HARDWARE_CHOICES,
                label="Hardware",
                value=stt_l.device,
                info=SETTINGS_HELP["stt_local_device"],
            )
            s_l_model_size = gr.Dropdown(
                _MODEL_SIZE_CHOICES,
                label="Accuracy / Speed",
                value=stt_l.model_size,
                info=SETTINGS_HELP["stt_local_model_size"],
            )
            with gr.Accordion("Advanced", open=False):
                s_l_a_beam = gr.Slider(
                    1,
                    20,
                    step=1,
                    label="Whisper Accuracy",
                    value=stt_a.beam_size,
                    info=SETTINGS_HELP["stt_local_beam_size"],
                )
                s_l_a_vad_t = gr.Slider(
                    0.0,
                    1.0,
                    step=0.05,
                    label="Voice Detection Sensitivity",
                    value=stt_a.vad_threshold,
                    info=SETTINGS_HELP["stt_local_vad_threshold"],
                )
                s_l_a_vad_ms = gr.Slider(
                    100,
                    3000,
                    step=50,
                    label="Minimum Pause Between Sentences (ms)",
                    value=stt_a.vad_min_silence_ms,
                    info=SETTINGS_HELP["stt_local_vad_min_silence_ms"],
                )
                s_l_a_suppress = gr.Checkbox(
                    label="Skip Silent Sections",
                    value=stt_a.suppress_blank,
                    info=SETTINGS_HELP["stt_local_suppress_blank"],
                )
                s_l_a_hall = gr.Radio(
                    _HALLUCINATION_CHOICES,
                    label="Filter Repeating Words",
                    value=stt_a.hallucination_gate,
                    info=SETTINGS_HELP["stt_local_hallucination_gate"],
                )
                s_l_a_repeat = gr.Slider(
                    0.1,
                    1.0,
                    step=0.05,
                    label="Repetition Filter Level",
                    value=stt_a.repeat_token_ratio_max,
                    info=SETTINGS_HELP["stt_local_repeat_token_ratio_max"],
                )
                s_l_a_rep_pen = gr.Slider(
                    1.0,
                    2.0,
                    step=0.05,
                    label="Anti-Repetition Strength",
                    value=stt_a.repetition_penalty,
                    info=SETTINGS_HELP["stt_local_repetition_penalty"],
                )

    with gr.Accordion("AI Clip Selection", open=True):
        l_provider = gr.Radio(
            _PROVIDER_CHOICES,
            label="Engine",
            value=llm.provider,
            info=SETTINGS_HELP["llm_provider"],
        )
        with gr.Accordion("Online (Cloud)", open=False):
            l_c_provider = gr.Radio(
                _CLOUD_PROVIDER_CHOICES,
                label="Online Service",
                value=llm_c.provider,
                info=SETTINGS_HELP["llm_cloud_provider"],
            )
            l_c_base = gr.Textbox(
                label="Custom API URL (Optional)",
                value=llm_c.base_url or "",
                info=SETTINGS_HELP["llm_cloud_base_url"],
            )
            l_c_api_key = gr.Textbox(
                label="API Key",
                type="password",
                placeholder=mask_api_key(llm_c.api_key),
                info=SETTINGS_HELP["llm_cloud_api_key"],
            )
            l_c_model = gr.Textbox(
                label="Model", value=llm_c.model, info=SETTINGS_HELP["llm_cloud_model"]
            )
            l_c_timeout = gr.Slider(
                30,
                600,
                step=10,
                label="Timeout (seconds)",
                value=llm_c.timeout,
                info=SETTINGS_HELP["llm_cloud_timeout"],
            )
        with gr.Accordion("On This Computer (Local)", open=False):
            l_l_device = gr.Dropdown(
                _HARDWARE_CHOICES,
                label="Hardware",
                value=llm_l.device,
                info=SETTINGS_HELP["llm_local_device"],
            )
            l_l_n_gpu = gr.Slider(
                -1,
                128,
                step=1,
                label="GPU Offload Layers",
                value=llm_l.n_gpu_layers,
                info=SETTINGS_HELP["llm_local_n_gpu_layers"],
            )
            l_l_model = gr.Textbox(
                label="Model Name",
                value=llm_l.model_name,
                info=SETTINGS_HELP["llm_local_model_name"],
            )

    with gr.Accordion("Download", open=True):
        d_vid_fmt = gr.Dropdown(
            ["mp4", "webm", "mkv"],
            label="Video Format",
            value=dl.video_format,
            info=SETTINGS_HELP["downloader_video_format"],
        )
        d_aud_fmt = gr.Dropdown(
            ["aac", "mp3", "opus"],
            label="Audio Format",
            value=dl.audio_format,
            info=SETTINGS_HELP["downloader_audio_format"],
        )
        d_aud_q = gr.Dropdown(
            ["128k", "192k", "256k", "320k"],
            label="Audio Quality",
            value=dl.audio_quality,
            info=SETTINGS_HELP["downloader_audio_quality"],
        )

    with gr.Accordion("Clip Limits", open=True):
        cs_min_clips = gr.Slider(
            1,
            50,
            step=1,
            label="Minimum Clips per Video",
            value=cs.min_clips,
            info=SETTINGS_HELP["min_clips"],
        )
        cs_max_clips = gr.Slider(
            1,
            100,
            step=1,
            label="Maximum Clips per Video",
            value=cs.max_clips,
            info=SETTINGS_HELP["max_clips"],
        )
        cs_min_dur = gr.Slider(
            10,
            300,
            step=5,
            label="Shortest Clip (seconds)",
            value=cs.min_clip_duration_seconds,
            info=SETTINGS_HELP["min_clip_duration_seconds"],
        )
        cs_max_dur = gr.Slider(
            10,
            600,
            step=5,
            label="Longest Clip (seconds)",
            value=cs.max_clip_duration_seconds,
            info=SETTINGS_HELP["max_clip_duration_seconds"],
        )
        cs_heat = gr.Slider(
            1,
            100,
            step=1,
            label="YouTube Highlights Sensitivity",
            value=cs.heatmap_threshold_percentile,
            info=SETTINGS_HELP["heatmap_threshold_percentile"],
        )
        cs_spike = gr.Slider(
            10,
            200,
            step=1,
            label="Maximum Moments to Consider",
            value=cs.spike_pool_size,
            info=SETTINGS_HELP["spike_pool_size"],
        )

    with gr.Accordion("Video Processing", open=True):
        vp_device = gr.Dropdown(
            _HARDWARE_CHOICES,
            label="Processing Hardware",
            value=vp.device,
            info=SETTINGS_HELP["video_processing_device"],
        )
        vp_det_conf = gr.Slider(
            0.1,
            1.0,
            step=0.05,
            label="Content Detection Certainty",
            value=vp.detection_confidence_threshold,
            info=SETTINGS_HELP["detection_confidence_threshold"],
        )
        vp_face = gr.Checkbox(
            label="Smart Face Tracking",
            value=vp.auto_face_tracking,
            info=SETTINGS_HELP["auto_face_tracking"],
        )
        vp_don = gr.Checkbox(
            label="Show Donation / Alert Popups",
            value=vp.preserve_donation_overlays,
            info=SETTINGS_HELP["preserve_donation_overlays"],
        )
        vp_don_excl = gr.CheckboxGroup(
            _DONATION_EXCLUDE_CHOICES,
            label="Don't Show Popups On",
            value=list(vp.donation_overlay_exclude_types),
            info=SETTINGS_HELP["donation_overlay_exclude_types"],
        )

    with gr.Accordion("Scene Detection", open=True):
        rd_enabled = gr.Checkbox(
            label="Enabled",
            value=rd.enabled,
            info=SETTINGS_HELP["region_detection_enabled"],
        )
        rd_model = gr.Textbox(
            label="Detection Model File",
            value=rd.model_name,
            info=SETTINGS_HELP["region_detection_model_name"],
        )
        rd_frames = gr.Slider(
            1,
            20,
            step=1,
            label="Analysis Detail",
            value=rd.sample_frames,
            info=SETTINGS_HELP["region_detection_sample_frames"],
        )
        rd_device = gr.Dropdown(
            _HARDWARE_CHOICES,
            label="Hardware",
            value=rd.device,
            info=SETTINGS_HELP["region_detection_device"],
        )
        rd_motion = gr.Checkbox(
            label="Camera Follows Action",
            value=rd.gameplay_follow_motion,
            info=SETTINGS_HELP["gameplay_follow_motion"],
        )
        rd_zoom = gr.Slider(
            1.0,
            2.0,
            step=0.05,
            label="Gameplay Zoom",
            value=rd.gameplay_zoom,
            info=SETTINGS_HELP["gameplay_zoom"],
        )

    with gr.Accordion("Caption Style", open=True):
        sub_enabled = gr.Checkbox(
            label="Show Captions",
            value=sub.enabled,
            info=SETTINGS_HELP["subtitles_enabled"],
        )
        sub_collab = gr.Checkbox(
            label="Show on Multi-Player Layout",
            value=sub.collab_enabled,
            info=SETTINGS_HELP["subtitles_collab_enabled"],
        )
        sub_upper = gr.Checkbox(
            label="ALL CAPS",
            value=sub.uppercase,
            info=SETTINGS_HELP["subtitles_uppercase"],
        )
        sub_font = gr.Textbox(
            label="Font File",
            value=sub.font_file,
            info=SETTINGS_HELP["subtitles_font_file"],
        )
        sub_size = gr.Slider(
            20,
            200,
            step=1,
            label="Font Size",
            value=sub.font_size,
            info=SETTINGS_HELP["subtitles_font_size"],
        )
        sub_pc = gr.ColorPicker(
            label="Text Color",
            value="#FFFFFF",
            info=SETTINGS_HELP["subtitles_primary_color"],
        )
        sub_hc = gr.ColorPicker(
            label="Active Word Color",
            value="#96C8FF",
            info=SETTINGS_HELP["subtitles_highlight_color"],
        )
        sub_oc = gr.ColorPicker(
            label="Text Outline Color",
            value="#000000",
            info=SETTINGS_HELP["subtitles_outline_color"],
        )
        sub_ot = gr.Slider(
            0,
            20,
            step=1,
            label="Outline Weight",
            value=sub.outline_thickness,
            info=SETTINGS_HELP["subtitles_outline_thickness"],
        )
        sub_bold = gr.Checkbox(
            label="Bold", value=sub.bold, info=SETTINGS_HELP["subtitles_bold"]
        )
        sub_shadow = gr.Checkbox(
            label="Shadow", value=sub.shadow, info=SETTINGS_HELP["subtitles_shadow"]
        )
        sub_align = gr.Dropdown(
            _ALIGNMENT_CHOICES,
            label="Position",
            value="bottom-center",
            info=SETTINGS_HELP["subtitles_alignment"],
        )
        sub_margin = gr.Slider(
            0,
            1920,
            step=10,
            label="Distance from Bottom",
            value=sub.margin_v,
            info=SETTINGS_HELP["subtitles_margin_v"],
        )

    with gr.Accordion("Save Settings", open=True):
        persist_cb = gr.Checkbox(
            label="Also Save to config.yaml file",
            value=False,
        )
        apply_btn = gr.Button("Apply Settings")

    _init_backups = list_config_backups()
    with gr.Accordion("Restore Settings from Backup", open=False):
        restore_dd = gr.Dropdown(
            label="Configuration Settings Backups",
            choices=_init_backups or [],
            value=None,
        )
        restore_btn = gr.Button(
            "Restore Settings",
            variant="secondary",
            interactive=bool(_init_backups),
        )

    widget_list: list[gr.components.Component] = [
        s_provider,
        s_c_provider,
        s_c_api_key,
        s_c_base,
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
        l_c_api_key,
        l_c_base,
        l_c_model,
        l_c_timeout,
        l_l_device,
        l_l_n_gpu,
        l_l_model,
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

    global _SETTINGS_PATHS, _WIDGET_LIST

    _WIDGET_LIST.clear()
    _WIDGET_LIST.extend(widget_list)

    _SETTINGS_PATHS.clear()
    path_list = [
        "ai_pipeline.stt.provider",
        "ai_pipeline.stt.cloud.provider",
        "ai_pipeline.stt.cloud.api_key",
        "ai_pipeline.stt.cloud.base_url",
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
        "ai_pipeline.llm.cloud.api_key",
        "ai_pipeline.llm.cloud.base_url",
        "ai_pipeline.llm.cloud.model",
        "ai_pipeline.llm.cloud.timeout",
        "ai_pipeline.llm.local.device",
        "ai_pipeline.llm.local.n_gpu_layers",
        "ai_pipeline.llm.local.model_name",
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
        inputs=widget_list + [persist_cb],
        outputs=[s_c_api_key, l_c_api_key, restore_dd, restore_btn],
    )

    restore_btn.click(
        fn=_restore_settings,
        inputs=[restore_dd],
        outputs=widget_list + [restore_dd],
    )

    settings_tab.select(
        fn=_refresh_backup_list,
        outputs=[restore_dd, restore_btn],
    )


def _refresh_backup_list() -> tuple[gr.update, gr.update]:
    _bk = list_config_backups()
    return (
        gr.update(choices=_bk or [], interactive=bool(_bk)),
        gr.update(interactive=bool(_bk)),
    )


def _apply_settings(*values):
    *setting_values, persist = values
    overrides: dict[str, object] = {}
    for path, val in zip(_SETTINGS_PATHS, setting_values, strict=True):
        if path.endswith("api_key"):
            if val:  # blank → keep existing key, never overwrite with empty
                overrides[path] = val
            continue
        overrides[path] = None if val == "" else val

    try:
        cfg = apply_session_overrides(overrides)
        if persist:
            save_config_to_disk()
            gr.Success("Settings applied and saved to config.yaml.")
        else:
            gr.Success("Settings applied for this session.")
    except Exception as e:
        cfg = load_config()
        gr.Warning(f"Failed to apply settings: {e}")

    _bk = list_config_backups()
    return (
        gr.update(
            value="", placeholder=mask_api_key(cfg.ai_pipeline.stt.cloud.api_key)
        ),
        gr.update(
            value="", placeholder=mask_api_key(cfg.ai_pipeline.llm.cloud.api_key)
        ),
        gr.update(choices=_bk or [], interactive=bool(_bk)),
        gr.update(interactive=bool(_bk)),
    )


def _restore_settings(filename: str) -> list[gr.update]:
    """Restore a chosen backup, reload config, refresh the whole form."""
    if not filename:
        gr.Warning("Select a backup first.")
        _bk = list_config_backups()
        return [gr.update() for _ in _WIDGET_LIST] + [
            gr.update(choices=_bk or [], interactive=bool(_bk)),
        ]

    try:
        cfg = restore_config(filename)
        gr.Success("Settings restored from backup. All settings reloaded.")
    except Exception as e:
        cfg = load_config()
        gr.Warning(f"Failed to restore: {e}")

    data = cfg.model_dump()
    updates: list[gr.update] = []
    for path in _SETTINGS_PATHS:
        node = data
        for key in path.split("."):
            node = node.get(key, {})  # type: ignore[assignment]

        val = node if not isinstance(node, dict) else None
        if path.endswith("api_key"):
            updates.append(gr.update(value="", placeholder=mask_api_key(val)))
        else:
            updates.append(gr.update(value=("" if val is None else val)))

    _bk = list_config_backups()
    updates.append(gr.update(choices=_bk or [], interactive=bool(_bk)))
    return updates


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

    from src.media.downloader import VideoDownloader

    progress(0.1, desc="Downloading video...")
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
