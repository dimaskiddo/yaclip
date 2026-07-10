from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from src.core.workspace import run_purge_cycle
from src.interfaces.components import _CONTENT_TYPE_CHOICES
from src.interfaces.utils import parse_clip_sidecar


def _load_proposals(
    pipeline: dict | None,
    rendered_state: list[str],
) -> tuple[list[dict], str | None, gr.update, gr.update, gr.update]:
    """Copy proposals into an editable list[dict] (never alias the source).

    Also returns visibility updates. When render has already completed
    (``rendered_state`` is non-empty) the columns stay on the rendered view;
    otherwise the review column is shown and actions column is hidden.
    """
    has_rendered = bool(rendered_state)

    if not pipeline and not has_rendered:
        return (
            [],
            None,
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    proposals = [
        {
            "title": c.get("title", ""),
            "start_time": float(c.get("start_time", 0.0)),
            "end_time": float(c.get("end_time", 0.0)),
            "content_type": c.get("content_type", "auto"),
            "reasoning": c.get("reasoning", ""),
            "caption": c.get("caption", ""),
            "description": c.get("description", ""),
            "hashtags": c.get("hashtags", ""),
        }
        for c in pipeline.get("proposals", [])
    ]

    if has_rendered:
        return (
            proposals,
            pipeline.get("content_type") if pipeline else None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    return (
        proposals,
        pipeline.get("content_type"),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def _edit_field(idx: int, key: str):
    """Return a handler writing one field of clip `idx` back into the proposals state in place."""

    def _apply(value: object, proposals: list[dict]) -> None:
        if 0 <= idx < len(proposals):
            proposals[idx][key] = value

    return _apply


def _delete_clip(idx: int):
    """Return a handler that drops clip `idx` and returns a new list (triggers re-render)."""

    def _apply(proposals: list[dict]) -> list[dict]:
        return [c for i, c in enumerate(proposals) if i != idx]

    return _apply


def _delete_rendered_files(paths: list[str]) -> None:
    """Delete each rendered clip .mp4 and its .txt sidecar from the workspace."""
    for p in paths:
        clip = Path(p)
        with contextlib.suppress(OSError):
            clip.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            clip.with_suffix(".txt").unlink(missing_ok=True)


def _reset_review() -> tuple:
    """State + visibility resets that return the tab to its initial (empty) state.

    Return order matches _RESET_OUTPUTS:
    proposals_state, rendered_state, pipeline_state,
    job_type_state, review_col, rendered_actions, render_progress_md,
    render_btn, clipper_progress, url_input, timerange_input, timerange_file.
    """
    return (
        [],
        [],
        None,
        None,
        gr.update(visible=True, elem_classes=[]),
        gr.update(visible=False),
        "",
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=None),
    )


def _start_new_clip() -> tuple:
    """Reset the tab UI; leave rendered files on disk."""
    return _reset_review()


def _reject_and_start_new(rendered: list[str]) -> tuple:
    """Delete the rendered clip files, then reset the tab UI."""
    _delete_rendered_files(rendered)
    gr.Info(f"Rejected and deleted {len(rendered)} rendered clip(s).")
    return _reset_review()


async def _run_render(
    proposals: list[dict],
    pipeline: dict | None,
    job_content_type: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[list[str], str, gr.update, gr.update, gr.update]:
    """Render clips then return state + column visibility swaps."""
    if not proposals:
        raise gr.Error("No clips to render. Run Find Clips first.")
    if not pipeline or not pipeline.get("video_path"):
        raise gr.Error("Missing source video. Run Find Clips first.")
    video_path = pipeline["video_path"]

    from src.core.constants import ContentType

    formatted = []
    for idx, clip in enumerate(proposals):
        title = str(clip.get("title") or f"Clip_{idx + 1}")
        start = float(clip.get("start_time") or 0.0)
        end = float(clip.get("end_time") or 0.0)
        ctype = str(clip.get("content_type") or "auto")
        d: dict[str, object] = {
            "start_time": start,
            "end_time": end,
            "title": title,
            "caption": clip.get("caption", ""),
            "description": clip.get("description", ""),
            "hashtags": clip.get("hashtags", ""),
        }
        if ctype != "auto":
            d["content_type"] = ctype
        formatted.append(d)
    content_type = ContentType(job_content_type) if job_content_type else None

    progress(0.0, desc="Preparing render...")
    from src.media.renderer import ClipRenderer

    def _cb(frac: float, desc: str) -> None:
        progress(frac, desc=desc)

    try:
        rendered = await asyncio.to_thread(
            ClipRenderer().render_clips,
            Path(video_path),
            formatted,
            content_type=content_type,
            progress_callback=_cb,
        )
    except Exception as e:
        raise gr.Error(f"Render failed: {e}") from e

    progress(0.9, desc="Cleaning up...")
    await asyncio.to_thread(run_purge_cycle, force=True, specific_target="tmp")

    progress(1.0, desc="Done.")
    return (
        [str(p) for p in rendered],
        f"Rendered {len(rendered)} clip(s).",
        gr.update(visible=False, elem_classes=[]),
        gr.update(visible=True),
        gr.update(visible=False, value="Render Clips"),
    )


def build_review_tab() -> SimpleNamespace:
    """Build the Review & Render tab UI and its internal (non-cross-tab) wiring.

    Cross-tab event chains (render click, tab select, reset/reject) are wired by the
    orchestrator in app.py since they also touch Clipper-tab and other-tab components.
    """
    with gr.Tab("Review & Render") as review_tab:
        job_type_state = gr.State(None)
        rendered_state = gr.State([])
        proposals_state = gr.State([])
        gr.Markdown("## Review & Render")

        review_col = gr.Column(visible=True)
        with review_col:
            gr.Markdown("### Review Proposed Clips")

            @gr.render(inputs=proposals_state)
            def _render_clip_panels(proposals: list[dict]):
                if not proposals:
                    gr.Markdown("_No clips yet. Run Find Clips on the Clipper tab._")
                    return
                for idx, clip in enumerate(proposals):
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            title_box = gr.Textbox(
                                value=clip.get("title", ""),
                                label=f"Clip {idx + 1} — Title",
                                scale=5,
                            )
                            del_btn = gr.Button("🗑️ Delete", scale=1, variant="stop")
                        with gr.Row():
                            start_box = gr.Number(
                                value=float(clip.get("start_time", 0.0)),
                                label="Start (s)",
                            )
                            end_box = gr.Number(
                                value=float(clip.get("end_time", 0.0)),
                                label="End (s)",
                            )
                            type_dd = gr.Dropdown(
                                choices=_CONTENT_TYPE_CHOICES,
                                value=clip.get("content_type", "auto"),
                                label="Video Type",
                            )
                        gr.Textbox(
                            value=clip.get("reasoning", ""),
                            label="Reasoning",
                            interactive=False,
                            lines=2,
                        )

                    # In-place edits mutate the live state; no reassignment avoids re-render.
                    title_box.change(
                        _edit_field(idx, "title"),
                        inputs=[title_box, proposals_state],
                        outputs=[],
                    )
                    start_box.change(
                        _edit_field(idx, "start_time"),
                        inputs=[start_box, proposals_state],
                        outputs=[],
                    )
                    end_box.change(
                        _edit_field(idx, "end_time"),
                        inputs=[end_box, proposals_state],
                        outputs=[],
                    )
                    type_dd.change(
                        _edit_field(idx, "content_type"),
                        inputs=[type_dd, proposals_state],
                        outputs=[],
                    )
                    # Delete swaps the list, triggering a re-render.
                    del_btn.click(
                        _delete_clip(idx),
                        inputs=[proposals_state],
                        outputs=[proposals_state],
                    )

        render_btn = gr.Button("Render Clips", variant="primary", visible=False)
        render_progress_md = gr.Markdown(elem_id="render-progress")

        rendered_col = gr.Column(visible=True)
        with rendered_col:

            @gr.render(inputs=rendered_state)
            def _show_rendered_clips(paths: list[str]):
                if not paths:
                    return
                gr.Markdown("### Rendered Clips")
                for idx, path in enumerate(paths):
                    meta = parse_clip_sidecar(path)
                    with gr.Group():
                        gr.Markdown(f"**Clip {idx + 1} — {Path(path).stem}**")
                        with gr.Row(equal_height=True):
                            gr.Video(value=path, interactive=False, scale=1)
                            with gr.Column(scale=1):
                                gr.Textbox(
                                    value=meta.get("title", ""),
                                    label="Title",
                                    interactive=False,
                                )
                                gr.Textbox(
                                    value=meta.get("caption", ""),
                                    label="Caption",
                                    interactive=False,
                                )
                                gr.Textbox(
                                    value=meta.get("description", ""),
                                    label="Description",
                                    interactive=False,
                                    lines=3,
                                )
                                gr.Textbox(
                                    value=meta.get("hashtags", ""),
                                    label="Hashtags",
                                    interactive=False,
                                )

            rendered_actions = gr.Column(visible=False)
            with rendered_actions:
                with gr.Row():
                    new_clip_btn = gr.Button("Start New Clip")
                    reject_btn = gr.Button("🗑️ Reject & Start New Clip", variant="stop")

                reject_confirm_row = gr.Row(visible=False)
                with reject_confirm_row:
                    gr.Markdown(
                        "⚠️ This will permanently delete the rendered clip files. Continue?"
                    )
                    reject_confirm_btn = gr.Button("Yes, Delete Them", variant="stop")
                    reject_cancel_btn = gr.Button("Cancel")

        reject_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[reject_confirm_row],
            queue=False,
        )
        reject_cancel_btn.click(
            fn=lambda: gr.update(visible=False),
            outputs=[reject_confirm_row],
            queue=False,
        )

    return SimpleNamespace(
        tab=review_tab,
        job_type_state=job_type_state,
        rendered_state=rendered_state,
        proposals_state=proposals_state,
        render_progress_md=render_progress_md,
        review_col=review_col,
        render_btn=render_btn,
        rendered_col=rendered_col,
        rendered_actions=rendered_actions,
        new_clip_btn=new_clip_btn,
        reject_btn=reject_btn,
        reject_confirm_row=reject_confirm_row,
        reject_confirm_btn=reject_confirm_btn,
        reject_cancel_btn=reject_cancel_btn,
    )
