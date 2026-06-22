from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.core.exceptions import ConfigValidationError
from src.core.workspace import CLIPS_DIR, LOGS_DIR

CONFIG_PATH = Path("config.yaml")

# Friendly subtitle-alignment names → ASS numpad code (1-9). Raw ints are also accepted.
ALIGNMENT_NAMES = {
    "bottom-left": 1,
    "bottom-center": 2,
    "bottom-centre": 2,
    "bottom-right": 3,
    "middle-left": 4,
    "center": 5,
    "centre": 5,
    "middle-center": 5,
    "middle-right": 6,
    "top-left": 7,
    "top-center": 8,
    "top-centre": 8,
    "top-right": 9,
}


def hex_to_ass_color(value: str) -> str:
    """Convert a human ``#RRGGBB`` / ``#RRGGBBAA`` colour to ASS ``&HAABBGGRR`` (R/B swapped).

    ASS alpha is inverted (``00`` = opaque, ``FF`` = transparent). Values already in ``&H…`` form (or
    anything not starting with ``#``) pass through unchanged, so existing configs keep working.
    """
    if not isinstance(value, str) or not value.startswith("#"):
        return value
    h = value[1:].strip()
    if len(h) == 6:
        rr, gg, bb, aa = h[0:2], h[2:4], h[4:6], "00"
    elif len(h) == 8:  # #RRGGBBAA — web alpha FF=opaque → ASS alpha 00=opaque
        rr, gg, bb = h[0:2], h[2:4], h[4:6]
        aa = f"{255 - int(h[6:8], 16):02X}"
    else:
        return value   # malformed → let it through (renders as-is / surfaces the typo)
    return f"&H{aa}{bb}{gg}{rr}".upper()


def alignment_to_int(value: object) -> object:
    """Map a friendly alignment name to its ASS int; pass ints (or unknowns) through."""
    if isinstance(value, str):
        s = value.strip().lower()
        if s.isdigit():
            return int(s)
        return ALIGNMENT_NAMES.get(s, value)
    return value


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    file_path: str = Field(default=str(LOGS_DIR / "app.log"))
    rotation: str = Field(default="50 MB")
    retention: str = Field(default="7 days")


class WebServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7860)
    share: bool = Field(default=False)


class STTCloudConfig(BaseModel):
    provider: str = Field(default="google")
    api_key: str = Field(default="your-api-key-here")
    model: str = Field(default="gemini-2.5-flash")
    timeout: int = Field(default=300, ge=30, le=600)


class STTLocalConfig(BaseModel):
    device: str = Field(default="auto")
    model_size: str = Field(default="large-v3")


class AIPipelineSTTConfig(BaseModel):
    provider: str = Field(default="local")
    cloud: STTCloudConfig = Field(default_factory=STTCloudConfig)
    local: STTLocalConfig = Field(default_factory=STTLocalConfig)


class LLMCloudConfig(BaseModel):
    provider: str = Field(default="google")
    base_url: str | None = Field(default=None)
    api_key: str = Field(default="your-api-key-here")
    model: str = Field(default="gemini-2.5-flash")
    timeout: int = Field(default=300, ge=30, le=600)


class LLMLocalConfig(BaseModel):
    device: str = Field(default="auto")
    n_gpu_layers: int = Field(default=0)
    model_name: str = Field(default="microsoft/Phi-3-mini-4k-instruct-gguf:q4")


class AIPipelineLLMConfig(BaseModel):
    provider: str = Field(default="cloud")
    cloud: LLMCloudConfig = Field(default_factory=LLMCloudConfig)
    local: LLMLocalConfig = Field(default_factory=LLMLocalConfig)


class AIPipelineConfig(BaseModel):
    stt: AIPipelineSTTConfig = Field(default_factory=AIPipelineSTTConfig)
    llm: AIPipelineLLMConfig = Field(default_factory=AIPipelineLLMConfig)


class DownloaderConfig(BaseModel):
    browser_cookies: str = Field(default="edge")
    target_resolution: str = Field(default="1080p")
    video_format: str = Field(default="mp4")
    audio_format: str = Field(default="aac")
    audio_quality: str = Field(default="192K")


class ClipSelectionConfig(BaseModel):
    mode: str = Field(default="auto")
    auto_strategy: str = Field(default="hybrid")
    candidate_margin: int = Field(default=2, ge=0, le=15)
    require_review_before_render: bool = Field(default=True)
    heatmap_threshold_percentile: int = Field(default=85)
    min_clips: int = Field(default=1)
    max_clips: int = Field(default=25)
    default_clips: int = Field(default=5)
    # Clip length = [default, default + margin]. `default` is the target/floor; `margin` is how much
    # longer a clip may run than the target. min/max are NOT per-clip bounds — they are the allowed
    # RANGE of the duration slider in the (future) Gradio WebUI.
    min_clip_duration_seconds: int = Field(default=30)  # WebUI duration-slider lower bound
    max_clip_duration_seconds: int = Field(default=180)  # WebUI duration-slider upper bound
    default_clip_duration_seconds: int = Field(default=60)  # target clip length (and floor)
    clip_length_margin_seconds: int = Field(default=15, ge=0)  # a clip may run up to default+margin

    @model_validator(mode="after")
    def _coherent_durations(self) -> ClipSelectionConfig:
        """Keep min ≤ default ≤ max (slider bounds) and margin ≥ 0."""
        if self.min_clip_duration_seconds > self.max_clip_duration_seconds:
            self.min_clip_duration_seconds, self.max_clip_duration_seconds = (
                self.max_clip_duration_seconds,
                self.min_clip_duration_seconds,
            )
        self.default_clip_duration_seconds = max(
            self.min_clip_duration_seconds,
            min(self.default_clip_duration_seconds, self.max_clip_duration_seconds),
        )
        self.clip_length_margin_seconds = max(0, self.clip_length_margin_seconds)
        return self


class SubtitleConfig(BaseModel):
    enabled: bool = Field(default=True)
    collab_enabled: bool = Field(default=False)  # subtitles on the cramped 3-stack collab layout
    uppercase: bool = Field(default=True)  # render caption text in CAPITALS for readability
    language: str = Field(default="auto")
    stt_context: str = Field(default="")  # vocab hint (names/game/terms) → STT initial_prompt
    font_file: str = Field(default="Anton.ttf")
    font_size: int = Field(default=80)
    # Colours accept human #RRGGBB(/AA) and are stored as ASS &HAABBGGRR (see hex_to_ass_color).
    primary_color: str = Field(default="&H00FFFFFF")
    highlight_color: str = Field(default="&H00FFC896")  # active-word colour
    outline_color: str = Field(default="&H00000000")
    outline_thickness: int = Field(default=6)
    bold: bool = Field(default=True)
    shadow: bool = Field(default=True)
    alignment: int = Field(default=2)  # accepts names like "bottom-center" (→ ASS numpad int)
    margin_v: int = Field(
        default=760
    )  # bottom margin in px; for bottom-center alignment: 760px up from canvas bottom = ~60% down from top

    @field_validator("primary_color", "highlight_color", "outline_color", mode="before")
    @classmethod
    def _normalize_color(cls, v: object) -> object:
        return hex_to_ass_color(v) if isinstance(v, str) else v

    @field_validator("alignment", mode="before")
    @classmethod
    def _normalize_alignment(cls, v: object) -> object:
        return alignment_to_int(v)


class RegionDetectionConfig(BaseModel):
    enabled: bool = Field(default=True)
    model_name: str = Field(default="yolov8n.pt")
    sample_frames: int = Field(default=4, ge=1, le=20)
    device: str = Field(default="auto")
    gameplay_follow_motion: bool = Field(
        default=False
    )  # false = static centred crop; true = animated motion-following pan
    gameplay_zoom: float = Field(default=1.25, ge=1.0, le=2.0)  # >1 zooms the gameplay crop tighter


class VideoProcessingConfig(BaseModel):
    output_dir: str = Field(default=str(CLIPS_DIR))
    device: str = Field(default="auto")
    # FFmpeg video encoder: auto | cpu | nvenc | qsv | videotoolbox.  "auto" uses nvenc when CUDA is
    # present, else cpu (libx264).  GPU encoders that fail at runtime automatically fall back to
    # libx264 (see renderer._run_render_with_fallback), so "auto" is safe as the default.
    video_encoder: str = Field(default="auto")
    # Low-spec opt-in: when true, PODCAST clips are tracked with a fast OpenCV Haar-cascade
    # largest-face crop instead of the heavier MediaPipe + audio active-speaker pipeline.  Trades
    # multi-speaker accuracy for speed.  Does NOT affect content-type detection or Mode B/C.
    fast_mode: bool = Field(default=False)
    content_type_override: str = Field(default="auto")
    detection_confidence_threshold: float = Field(default=0.6)
    auto_face_tracking: bool = Field(default=True)
    preserve_donation_overlays: bool = Field(default=False)
    # Content types excluded from per-clip donation-overlay promotion.
    # PODCAST is excluded because it is pre-recorded (no live donation widgets).
    # GAMING_COLLAB is excluded because the popup must not replace one of the three collab panels.
    # Valid values: PODCAST | JUST_CHAT | GAMING_SOLO | GAMING_COLLAB | DONATION_OVERLAY
    donation_overlay_exclude_types: list[str] = Field(
        default_factory=lambda: ["PODCAST", "GAMING_COLLAB"]
    )
    default_resolution: str = Field(default="1080p")
    region_detection: RegionDetectionConfig = Field(default_factory=RegionDetectionConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)


class RetentionDaysConfig(BaseModel):
    videos: int = Field(default=3)
    audios: int = Field(default=3)
    subtitles: int = Field(default=3)
    data: int = Field(default=3)
    tmp: int = Field(default=1)


class WorkspaceCleanupConfig(BaseModel):
    enabled: bool = Field(default=True)
    run_on_startup: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    retention_days: RetentionDaysConfig = Field(default_factory=RetentionDaysConfig)
    protected_dirs: list[str] = Field(default_factory=lambda: ["bin", "fonts", "models"])


class AppConfig(BaseModel):
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    web_server: WebServerConfig = Field(default_factory=WebServerConfig)
    ai_pipeline: AIPipelineConfig = Field(default_factory=AIPipelineConfig)
    downloader: DownloaderConfig = Field(default_factory=DownloaderConfig)
    clip_selection: ClipSelectionConfig = Field(default_factory=ClipSelectionConfig)
    video_processing: VideoProcessingConfig = Field(default_factory=VideoProcessingConfig)
    workspace_cleanup: WorkspaceCleanupConfig = Field(default_factory=WorkspaceCleanupConfig)
    dk_clipper_sys_prompt: str | None = Field(default=None)


_config_cache: AppConfig | None = None


def load_config(force_reload: bool = False) -> AppConfig:
    """Parse, validate YAML config file against Pydantic models and return AppConfig singleton."""
    global _config_cache
    if _config_cache is not None and not force_reload:
        return _config_cache

    if not CONFIG_PATH.exists():
        raise ConfigValidationError(
            f"Missing config file: {CONFIG_PATH}. Please copy config.yaml.example to config.yaml."
        )

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigValidationError(f"Failed to read/parse {CONFIG_PATH} as YAML: {e}") from e

    try:
        _config_cache = AppConfig.model_validate(data)
        return _config_cache
    except ValidationError as e:
        raise ConfigValidationError(f"Configuration validation failed:\n{e}") from e
