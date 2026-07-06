"""Reusable Gradio component factories for the YaClip WebUI (AGENTS.md §11.6).

Kept deliberately small — factories are added here as tabs need them, so shared UI patterns
live in one place instead of being copy-pasted across ``webui.py``. ``gradio`` is imported
lazily inside each factory (per AGENTS.md §11.3) and because these run only while a
``gr.Blocks`` context is open.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gradio as gr

    from src.core.config import ClipSelectionConfig


def section_header(title: str, subtitle: str | None = None) -> gr.Markdown:
    """A bold section title with an optional muted subtitle line."""
    import gradio as gr

    body = f"### {title}"
    if subtitle:
        body += f"\n<span style='opacity:0.7'>{subtitle}</span>"
    return gr.Markdown(body)


def clip_count_slider(cs: ClipSelectionConfig) -> gr.Slider:
    """Number-of-clips slider bounded by ``min_clips``/``max_clips`` from config."""
    import gradio as gr

    return gr.Slider(
        minimum=cs.min_clips,
        maximum=cs.max_clips,
        value=cs.default_clips,
        step=1,
        label="Number of Clips",
    )


def clip_duration_slider(cs: ClipSelectionConfig) -> gr.Slider:
    """Target clip-duration slider bounded by the config duration-slider range (seconds)."""
    import gradio as gr

    return gr.Slider(
        minimum=cs.min_clip_duration_seconds,
        maximum=cs.max_clip_duration_seconds,
        value=cs.default_clip_duration_seconds,
        step=1,
        label="Target Clip Duration (seconds)",
    )
