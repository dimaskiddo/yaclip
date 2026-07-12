from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.core.workspace import TMP_DIR, list_clip_subdirs, list_clips_in_dir


def _cleanup_gradio_stale() -> None:
    """Remove Gradio's file-copy directory so stale video copies never accumulate.

    Runs before every tab activation and directory change so that only the
    currently-viewed clip set occupies disk space in ``workspace/tmp/gradio/``.
    Gradio re-creates the directory automatically on the next file serve.
    """
    cache_dir = TMP_DIR / "gradio"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


_LOADING_SENTINEL = "LOADING"


def _refresh_choices() -> gr.update:
    """Re-read the clips directory tree and return an updated dropdown (choices only).

    Never sets ``value`` so Gradio does **not** fire a spurious ``.change``
    event — the user must explicitly pick a directory to trigger clip loading.
    """
    subdirs = list_clip_subdirs()
    return gr.update(choices=subdirs)


def _load_clips(dir_name: str | None) -> list[dict[str, str]]:
    """Return the clip list for the selected video directory, or empty."""
    if not dir_name:
        return []
    return list_clips_in_dir(dir_name)


def _on_tab_select() -> tuple[gr.update, list]:
    """Nuke stale Gradio copies, reset dropdown + clips state.

    Returns an **empty** clips state so the old clip panels with stale Gradio-
    cached paths are torn down immediately.  The user must re-pick a directory.
    """
    _cleanup_gradio_stale()
    from src.core.workspace import list_clip_subdirs  # avoid circular on module import

    return (gr.update(choices=list_clip_subdirs(), value=None), [])


def _on_dir_change(_dir_name: str | None) -> str:
    """Nuke stale Gradio copies, then signal loading.

    Returns the loading sentinel so ``@gr.render`` shows a progress message
    while the heavy file listing runs in the ``.then()`` step.
    """
    _cleanup_gradio_stale()
    return _LOADING_SENTINEL


def build_clipsmanager_tab() -> SimpleNamespace:
    """Build the Clips Manager tab UI.

    Returns a ``SimpleNamespace`` with a single public attribute ``tab``
    (the ``gr.Tab`` context-manager object) — no cross-tab wiring is needed
    because this tab is entirely self-contained.
    """
    with gr.Tab("Clips Manager") as clipsmanager_tab:
        clips_state = gr.State([])

        gr.Markdown("## Clips Manager")
        gr.Markdown(
            "Browse clips from previous renders. "
            "Select a video directory below to preview its clips."
        )

        video_dropdown = gr.Dropdown(
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
        clipsmanager_tab.select(fn=_on_tab_select, outputs=[video_dropdown, clips_state])
        video_dropdown.change(
            fn=_on_dir_change, inputs=[video_dropdown], outputs=[clips_state]
        ).then(fn=_load_clips, inputs=[video_dropdown], outputs=[clips_state])

    return SimpleNamespace(tab=clipsmanager_tab)
