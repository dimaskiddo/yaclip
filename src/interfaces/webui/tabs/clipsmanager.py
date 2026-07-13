from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.core.workspace import (
    cleanup_gradio_temp,
    list_clip_subdirs,
    list_clips_in_dir,
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
    """
    if not dir_name:
        return []

    return list_clips_in_dir(dir_name)


def build_clipsmanager_tab() -> SimpleNamespace:
    """Build the Clips Manager tab UI.

    Returns a ``SimpleNamespace`` with a single public attribute ``tab``
    (the ``gr.Tab`` context-manager object) — no cross-tab wiring is needed
    because this tab is entirely self-contained.
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
            fn=_on_tab_select, outputs=[video_dropdown, clips_state]
        )
        video_dropdown.change(
            fn=_on_dir_change, inputs=[video_dropdown], outputs=[clips_state]
        ).then(fn=_load_clips, inputs=[video_dropdown], outputs=[clips_state])

    return SimpleNamespace(tab=clipsmanager_tab)
