from __future__ import annotations

from typing import TYPE_CHECKING

from src.ai.prompts import LANGUAGE_MAP

if TYPE_CHECKING:
    import gradio as gr

    from src.core.config import ClipSelectionConfig


_CONTENT_TYPE_CHOICES: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("Podcast", "PODCAST"),
    ("Just Chat", "JUST_CHAT"),
    ("Gaming Solo (FaceCam at Top)", "GAMING_SOLO"),
    ("Gaming Solo (FaceCam at Bottom)", "GAMING_SOLO_BOTTOM"),
    ("Gaming Collab", "GAMING_COLLAB"),
    ("Donation / Alert", "DONATION_OVERLAY"),
]

_STRATEGY_CHOICES: list[tuple[str, str]] = [
    ("Combined (Recommended)", "hybrid"),
    ("YouTube Most Replayed", "heatmap"),
    ("AI Analysis", "ai"),
]

_ENCODER_CHOICES: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("CPU (Software)", "cpu"),
    ("NVIDIA GPU", "nvenc"),
    ("Intel GPU", "qsv"),
    ("Apple GPU", "videotoolbox"),
]

_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("Auto (Cloud First)", "auto"),
    ("Online (Cloud)", "cloud"),
    ("On This Computer (Local)", "local"),
]

_CLOUD_PROVIDER_CHOICES: list[tuple[str, str]] = [
    ("Google (Gemini)", "google"),
    ("OpenAI / Compatible", "openai"),
]

_HARDWARE_CHOICES: list[tuple[str, str]] = [
    ("Auto-Detect", "auto"),
    ("CPU", "cpu"),
    ("NVIDIA GPU", "cuda"),
]

_MODEL_SIZE_CHOICES: list[tuple[str, str]] = [
    ("Fastest", "tiny"),
    ("Basic", "base"),
    ("Balanced", "small"),
    ("Better", "medium"),
    ("Best", "large-v3"),
]

_HALLUCINATION_CHOICES: list[tuple[str, str]] = [
    ("Strict", "and"),
    ("Relaxed", "or"),
]

_DONATION_EXCLUDE_CHOICES: list[tuple[str, str]] = [
    ("Podcast", "PODCAST"),
    ("Just Chat", "JUST_CHAT"),
    ("Gaming Solo", "GAMING_SOLO"),
    ("Gaming Collab", "GAMING_COLLAB"),
]

_ALIGNMENT_CHOICES: list[tuple[str, str]] = [
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


def _build_language_choices() -> list[tuple[str, str]]:
    """Build language choices from LANGUAGE_MAP, deduped, with Auto-Detect first."""
    seen: set[str] = set()
    choices: list[tuple[str, str]] = [("Auto-Detect", "auto")]
    for name, code in sorted(LANGUAGE_MAP.items()):
        display = name.title()
        if display not in seen:
            seen.add(display)
            choices.append((display, code))
    return choices


LANGUAGE_CHOICES = _build_language_choices()


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
