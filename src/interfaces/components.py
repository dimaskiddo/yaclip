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

SETTINGS_HELP: dict[str, str] = {
    "stt_provider": "Where the app turns speech into text (transcription / speech-to-text). Online "
    "(cloud) sends the audio to a paid provider — needs internet and an API key and charges per "
    "use; On this computer (local) runs free and private on your own hardware; Auto tries the cloud "
    "first, then falls back to local.",
    "stt_cloud_provider": "Which cloud transcription provider to use (e.g. Google, OpenAI). These "
    "are paid and bill by how much audio you send.",
    "stt_cloud_api_key": "Your API key for the chosen transcription provider, from that provider's "
    "website. Using the service costs money. Leave empty to keep the key already saved.",
    "stt_cloud_base_url": "Custom API endpoint URL, for advanced users running their own or a "
    "compatible transcription service. Leave empty for normal use.",
    "stt_cloud_model": "Which provider model to use for transcription (e.g. gemini-2.5-flash). "
    "Leave as-is if unsure.",
    "stt_cloud_timeout": "How many seconds to wait for the provider before giving up (timeout). "
    "Raise it on a slow connection.",
    "stt_local_device": "The hardware used for local transcription. GPU (graphics card) is much "
    "faster; CPU (processor) works everywhere but is slower.",
    "stt_local_model_size": "The local speech-to-text model size. Bigger is more accurate but "
    "slower and uses more memory; smaller is faster with more mistakes.",
    "stt_local_beam_size": "How many options the model weighs per word (beam size). Higher is a bit "
    "more accurate but slower.",
    "stt_local_vad_threshold": "How loud audio must be to count as speech (voice detection "
    "sensitivity). Higher ignores quiet talking; lower catches whispers but also background noise.",
    "stt_local_vad_min_silence_ms": "How long a pause must be, in milliseconds, to split a sentence. "
    "Higher makes longer captions; lower breaks them up more often.",
    "stt_local_suppress_blank": "Skip silent parts to save time. Turn on if there is lots of quiet.",
    "stt_local_hallucination_gate": "How hard to filter out made-up words (hallucinations) from "
    "laughter or music. Strict removes more; Relaxed keeps more.",
    "stt_local_repeat_token_ratio_max": "How much repeated text to allow before treating it as a "
    "mistake. Lower cleans up more; higher keeps more.",
    "stt_local_repetition_penalty": "How strongly to stop the model repeating a word (repetition "
    "penalty). Higher repeats less; too high may drop words.",
    "llm_provider": "Where the app picks the best moments and writes titles (AI analysis). Online "
    "(cloud) uses a paid provider — needs internet and an API key and charges per use, and is "
    "usually smarter; On this computer (local) runs free and private; Auto tries the cloud first, "
    "then falls back to local.",
    "llm_cloud_provider": "Which cloud AI provider picks moments and writes titles (e.g. Google, "
    "OpenAI). These are paid and bill by how much text you send.",
    "llm_cloud_api_key": "Your API key for the chosen AI provider, from that provider's website. "
    "Using the service costs money. Leave empty to keep the key already saved.",
    "llm_cloud_base_url": "Custom API endpoint URL, for advanced users running their own or a "
    "compatible AI service (e.g. OpenRouter). Leave empty for normal use.",
    "llm_cloud_model": "Which provider model to use for analysis (e.g. gemini-2.5-flash). Leave "
    "as-is if unsure.",
    "llm_cloud_timeout": "How many seconds to wait for the provider before giving up (timeout). "
    "Raise it on a slow connection.",
    "llm_local_device": "The hardware used for the local AI model. GPU (graphics card) is much "
    "faster; CPU (processor) works everywhere but is slower.",
    "llm_local_n_gpu_layers": "How many model layers to run on your graphics card (GPU offload). "
    "Higher is faster but needs a stronger card; 0 uses the processor only.",
    "llm_local_model_name": "Which local AI model file to run (a downloaded GGUF model). Leave "
    "as-is if unsure.",
    "downloader_video_format": "The video file format (container) for the download. mp4 works "
    "almost everywhere.",
    "downloader_audio_format": "The audio format for the sound extracted from the video (e.g. aac, "
    "mp3). Leave as-is if unsure.",
    "downloader_audio_quality": "The audio bitrate pulled from the video. Higher is clearer but "
    "makes bigger files.",
    "min_clips": "The fewest clips the app will make from one video.",
    "max_clips": "The most clips the app will make from one video.",
    "min_clip_duration_seconds": "The shortest clip length (seconds) you can pick on the Clipper "
    "tab's duration slider.",
    "max_clip_duration_seconds": "The longest clip length (seconds) you can pick on the Clipper "
    "tab's duration slider.",
    "heatmap_threshold_percentile": "How popular a moment must be on YouTube's most-replayed "
    "heatmap to be picked. Higher keeps only the biggest highlights; lower includes more.",
    "spike_pool_size": "How many candidate moments (spikes) to consider before choosing. Higher "
    "looks wider but is slower; lower is faster but may miss some.",
    "video_processing_device": "The hardware used to find faces and action while rendering. GPU "
    "(graphics card) is faster; CPU (processor) works everywhere but is slower.",
    "detection_confidence_threshold": "How confident the app must be about the video's content type "
    "before it trusts the guess. Higher is safer (falls back to a plain layout); lower trusts "
    "guesses more.",
    "auto_face_tracking": "Keeps the crop centered on whoever is talking (face tracking). On "
    "follows the speaker; off stays fixed in the middle.",
    "preserve_donation_overlays": "Show donation and alert pop-ups (e.g. Trakteer, MediaShare) in "
    "the clips. Off hides them even when they appear.",
    "donation_overlay_exclude_types": "Content types that should never show donation pop-ups (e.g. "
    "podcasts, which have no live alerts).",
    "region_detection_enabled": "Let the app analyze each scene to crop it smartly (region "
    "detection). Off uses a simpler motion/face method that is less accurate.",
    "region_detection_model_name": "The detection model file used to analyze scenes (e.g. "
    "yolov8n.pt). Leave as-is if unsure.",
    "region_detection_sample_frames": "How many frames the app samples per clip to analyze. Higher "
    "is more accurate but slower; lower is faster.",
    "region_detection_device": "The hardware used to analyze scenes. GPU (graphics card) is faster; "
    "CPU (processor) works everywhere but is slower.",
    "gameplay_follow_motion": "How the gameplay panel moves. On gently pans to follow the action; "
    "off keeps a steady static crop (recommended).",
    "gameplay_zoom": "How close-up the gameplay crop is (zoom). Higher zooms in more and hides the "
    "edges; 1.0 shows the full view.",
    "camera_pan_speed": "How fast the camera glides toward its target. Higher = snappier "
    "response to scene changes; lower = smoother, slower ease.",
    "subtitles_enabled": "Burn captions (subtitles) into the clips. Off makes clips with no "
    "captions.",
    "subtitles_collab_enabled": "Also show captions on the crowded multi-player (collab) layout. "
    "Off leaves more room for the three panels.",
    "subtitles_uppercase": "Show captions in CAPITAL LETTERS (uppercase). Off uses normal case.",
    "subtitles_font_file": "The font file used for captions (from the fonts folder).",
    "subtitles_font_size": "The caption text size in pixels. Higher is easier to read; lower is "
    "smaller.",
    "subtitles_primary_color": "The color of the normal (not-yet-spoken) caption words.",
    "subtitles_highlight_color": "The highlight color of the word being spoken right now.",
    "subtitles_outline_color": "The color of the outline/border around the caption text.",
    "subtitles_outline_thickness": "The outline thickness around captions, in pixels. Higher stands "
    "out more; 0 removes it.",
    "subtitles_bold": "Show caption text in bold (thicker letters).",
    "subtitles_shadow": "Add a drop shadow behind captions so they are easier to read.",
    "subtitles_alignment": "Where the captions sit on the screen (alignment).",
    "subtitles_margin_v": "The caption distance from the screen edge, in pixels (vertical margin). "
    "Higher moves them toward the middle.",
}


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
