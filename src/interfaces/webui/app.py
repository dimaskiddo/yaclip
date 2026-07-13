from __future__ import annotations

import gradio as gr

from src.core.config import load_config
from src.interfaces.webui.tabs.about import build_about_tab
from src.interfaces.webui.tabs.clipper import build_clipper_tab
from src.interfaces.webui.tabs.clipsmanager import build_clipsmanager_tab
from src.interfaces.webui.tabs.maintenance import build_maintenance_tab
from src.interfaces.webui.tabs.review import (
    _load_proposals,
    _reject_and_start_new,
    _run_render,
    _start_new_clip,
    build_review_tab,
)
from src.interfaces.webui.tabs.settings import build_settings_tab

# Loads Trakteer's overlay-button library ourselves
_TRAKTEER_LOAD_JS = """
() => {
  const slot = document.getElementById('trakteer-btn-slot');
  if (!slot || slot.dataset.done) return;
  slot.dataset.done = '1';

  const marker = document.createElement('script');
  marker.className = 'troverlay';
  slot.appendChild(marker);

  const lib = document.createElement('script');
  lib.src = 'https://edge-cdn.trakteer.id/js/trbtn-overlay.min.js?v=14-05-2025';
  lib.onload = () => {
    const id = trbtnOverlay.init(
      'Support This Project on Trakteer.ID', '#be1e2d',
      'https://trakteer.id/v1/itsdrh/tip/embed/modal',
      'https://edge-cdn.trakteer.id/images/embed/trbtn-icon.png?v=14-05-2025',
      '40', 'inline');
    trbtnOverlay.draw(id);
  };
  document.head.appendChild(lib);
}
"""


def build_ui() -> gr.Blocks:
    cfg = load_config()
    with gr.Blocks(title="Yet Another AI Auto-Clipper (YaClip)") as app:
        gr.HTML(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:4px;">'
            '<h1 style="margin:0;font-size:1.75rem;">Yet Another AI Auto-Clipper (YaClip)</h1>'
            '<span id="trakteer-btn-slot" style="margin-left:auto;"></span>'
            "</div>"
        )
        clipper = build_clipper_tab(cfg)
        clipper_tab = clipper.tab
        url_input = clipper.url_input
        pipeline_state = clipper.pipeline_state
        clipper_progress = clipper.clipper_progress
        timerange_input = clipper.timerange_input
        timerange_file = clipper.timerange_file
        review = build_review_tab()
        review_tab = review.tab
        job_type_state = review.job_type_state
        rendered_state = review.rendered_state
        proposals_state = review.proposals_state
        render_progress_md = review.render_progress_md
        review_col = review.review_col
        render_btn = review.render_btn
        rendered_actions = review.rendered_actions
        new_clip_btn = review.new_clip_btn
        reject_confirm_btn = review.reject_confirm_btn

        review_tab.select(
            fn=_load_proposals,
            inputs=[pipeline_state, rendered_state],
            outputs=[
                proposals_state,
                job_type_state,
                render_btn,
                review_col,
                rendered_actions,
            ],
        )
        clipsmanager = build_clipsmanager_tab()
        clipsmanager_tab = clipsmanager.tab

        settings = build_settings_tab(cfg)
        settings_tab = settings.tab
        maintenance = build_maintenance_tab()
        maintenance_tab = maintenance.tab
        build_about_tab()

        # ---- Deferred event wiring: render/reset/reject (after all tabs exist). ----
        _OTHER_TABS = [clipper_tab, clipsmanager_tab, settings_tab, maintenance_tab]
        _RESET_OUTPUTS = [
            proposals_state,
            rendered_state,
            pipeline_state,
            job_type_state,
            review_col,
            rendered_actions,
            render_progress_md,
            render_btn,
            clipper_progress,
            url_input,
            timerange_input,
            timerange_file,
        ]

        render_event = render_btn.click(
            fn=lambda: (
                [gr.update(interactive=False, value="Rendering Clips...")]
                + [gr.update(elem_classes=["render-lock"])]
                + [gr.update(interactive=False)] * len(_OTHER_TABS)
            ),
            outputs=[render_btn, review_col, *_OTHER_TABS],
            queue=False,
        ).then(
            fn=_run_render,
            inputs=[proposals_state, pipeline_state, job_type_state],
            outputs=[
                rendered_state,
                render_progress_md,
                review_col,
                rendered_actions,
                render_btn,
            ],
            show_progress_on=[render_btn, render_progress_md],
        )
        render_event.success(
            fn=lambda: [gr.update(interactive=True)] * len(_OTHER_TABS),
            outputs=[*_OTHER_TABS],
            queue=False,
        )
        render_event.failure(
            fn=lambda: (
                [gr.update(interactive=True, value="Render Clips")]
                + [gr.update(elem_classes=[])]
                + [gr.update(interactive=True)] * len(_OTHER_TABS)
            ),
            outputs=[render_btn, review_col, *_OTHER_TABS],
            queue=False,
        )

        new_clip_btn.click(fn=_start_new_clip, outputs=_RESET_OUTPUTS)

        reject_confirm_btn.click(
            fn=_reject_and_start_new,
            inputs=[rendered_state],
            outputs=_RESET_OUTPUTS,
        )

        app.load(None, js=_TRAKTEER_LOAD_JS)
    return app


def launch_webui(host: str = "127.0.0.1", port: int = 7860) -> None:
    cfg = load_config().web_server
    ui = build_ui()
    ui.queue().launch(
        server_name=host or cfg.host,
        server_port=port or cfg.port,
        share=cfg.share,
    )
