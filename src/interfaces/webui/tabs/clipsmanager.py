from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.core.workspace import (
    cleanup_gradio_temp,
    list_clip_subdirs,
    list_clips_in_dir,
    prewarm_gradio_cache,
)

_LOADING_SENTINEL = "LOADING"


def _on_tab_select() -> tuple[gr.update, list]:
    """Nuke stale Gradio copies, reset dropdown + clips state.

    Returns an **empty** clips state so the old clip panels with stale Gradio-
    cached paths are torn down immediately.  The user must re-pick a directory.
    """
    cleanup_gradio_temp()

    return (gr.update(choices=list_clip_subdirs(), value=None), [])


def _on_dir_change(_dir_name: str | None) -> str:
    """Nuke stale Gradio copies, then signal loading.

    Returns the loading sentinel so ``@gr.render`` shows a progress message
    while the heavy file listing runs in the ``.then()`` step.
    """
    cleanup_gradio_temp()
    return _LOADING_SENTINEL


def _load_clips(dir_name: str | None) -> list[dict[str, str]]:
    """Enumerate clips in selected directory.

    Runs as the ``.then()`` step after the loading sentinel is displayed so
    the UI shows a progress message instead of stale panels while the disk
    scan executes.

    Pre-warms each clip into Gradio's file cache before returning so the heavy
    ``gr.Video`` copy happens here (under the tab-disable lock) rather than in
    the ``@gr.render`` response — this is what makes the other-tabs lock hold
    through the real panel-populate window instead of blinking off instantly.
    """
    if not dir_name:
        return []

    clips = list_clips_in_dir(dir_name)
    prewarm_gradio_cache([c["path"] for c in clips])
    return clips


def build_clipsmanager_tab() -> SimpleNamespace:
    """Build the Clips Manager tab UI.

    Returns a ``SimpleNamespace`` exposing ``tab`` (the ``gr.Tab`` object) and
    ``clips_state`` (the ``gr.State`` driving the clip panels). ``clips_state``
    is surfaced so the orchestrator can key an other-tabs disable/enable lock
    off the panel's actual data state — its value is the loading sentinel while
    the panel shows "Loading clips…", and the clips list once panels render.
    """
    with gr.Tab("Clips Manager") as clipsmanager_tab:
        clips_state = gr.State([])

        gr.HTML(
            "<style>\n"
            "#clips-css-wrap { display: none !important; }\n"
            ".clips-dir-dropdown ul {\n"
            "  max-height: 200px !important;\n"
            "  overflow-y: auto !important;\n"
            "}\n"
            "</style>",
            elem_id="clips-css-wrap",
        )

        gr.Markdown("## Clips Manager")
        gr.Markdown(
            "Browse clips from previous renders. "
            "Select a video directory below to preview its clips."
        )

        video_dropdown = gr.Dropdown(
            elem_classes=["clips-dir-dropdown"],
            choices=[],
            label="Select Video (Clips Directory)",
            interactive=True,
        )

        @gr.render(inputs=clips_state)
        def _render_clips(clips: list[dict[str, str]]):
            if isinstance(clips, str) and clips == _LOADING_SENTINEL:
                gr.Markdown("_Loading clips…_")
                return
            if not clips:
                gr.Markdown("_No clips found. Select a video directory above._")
                return
            for idx, clip in enumerate(clips):
                with gr.Group():
                    gr.Markdown(f"**Clip {idx + 1} — {Path(clip['path']).stem}**")
                    with gr.Row(equal_height=True):
                        gr.Video(
                            value=clip["path"],
                            interactive=False,
                            height=480,
                            scale=1,
                        )
                        with gr.Column(scale=2):
                            gr.Textbox(
                                value=clip["title"],
                                label="Title",
                                interactive=False,
                            )
                            gr.Textbox(
                                value=clip["caption"],
                                label="Caption",
                                interactive=False,
                            )
                            gr.Textbox(
                                value=clip["description"],
                                label="Description",
                                interactive=False,
                                lines=3,
                            )
                            gr.Textbox(
                                value=clip["hashtags"],
                                label="Hashtags",
                                interactive=False,
                            )

        # Wire events: clean stale copies + refresh choices on tab activation;
        # clean again on directory change so only the current dir's clips stay cached.
        clipsmanager_tab.select(
            fn=_on_tab_select, outputs=[video_dropdown, clips_state],
            api_name="clips-refresh",
        )
        video_dropdown.change(
            fn=_on_dir_change, inputs=[video_dropdown], outputs=[clips_state],
            api_name="clips-select-video",
        ).then(
            fn=_load_clips, inputs=[video_dropdown], outputs=[clips_state],
            api_name="clips-load",
        )

    return SimpleNamespace(tab=clipsmanager_tab, clips_state=clips_state)
