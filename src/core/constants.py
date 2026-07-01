from __future__ import annotations

from enum import Enum


class ContentType(str, Enum):
    PODCAST = "PODCAST"
    JUST_CHAT = "JUST_CHAT"
    GAMING_SOLO = "GAMING_SOLO"
    GAMING_COLLAB = "GAMING_COLLAB"
    # Per-clip type (not a video-level detection): any clip whose window contains a
    # mediashare/donation overlay is promoted to this, routing to the facecam+popup 2-stack.
    DONATION_OVERLAY = "DONATION_OVERLAY"


# The 4 base (video-level-detectable) content types, excluding the render-time-only
# DONATION_OVERLAY promotion. Used to validate LLM-returned content_type values.
BASE_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        ContentType.PODCAST.value,
        ContentType.JUST_CHAT.value,
        ContentType.GAMING_SOLO.value,
        ContentType.GAMING_COLLAB.value,
    }
)

# config.yaml.example template value for unset API keys — never a real credential.
PLACEHOLDER_API_KEY: str = "your-api-key-here"


class ClipMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class LayoutMode(str, Enum):
    SINGLE_VERTICAL = "single_vertical"
    STACKED_SPLIT = "stacked_split"
    MULTI_COLLAB = "multi_collab"


# Layout and rendering constants

# Final vertical canvas (9:16)
TARGET_WIDTH: int = 1080
TARGET_HEIGHT: int = 1920

# Minimum valid clip length. Clips shorter than this (or inverted, end <= start) are degenerate —
# they come from a bad LLM timestamp mapping and would crash the slicer (-to < -ss), so they are
# dropped before slicing/rendering.
MIN_CLIP_SECONDS: float = 1.0

# Hybrid candidate windows are widened to (max_clip_duration + 2× this buffer) around each spike
# centre so the LLM has room to pick any clip length in [min, max] and the post-map duration
# enforcement can always extend a short clip up to the minimum within the transcribed window.
CANDIDATE_WINDOW_BUFFER: float = 5.0

# STT hallucination filter (faster-whisper segment metrics). Whisper transcribes laughter / music /
# non-speech as repeated filler tokens (e.g. "hehehe" x128). A segment is dropped when it is clearly
# repetitive or non-speech. Conservative so real quiet speech survives. See stt_local._filter_hallucinations.
STT_COMPRESSION_MAX: float = (
    2.0  # gzip ratio above this = looped/repeated text (hallucination)
)
STT_NO_SPEECH_MAX: float = 0.6  # no_speech_prob above this = likely silence/noise
STT_LOGPROB_MIN: float = -1.0  # avg_logprob below this = low-confidence decode
STT_REPEAT_TOKEN_MAX: int = (
    3  # a single token repeated more than this = laughter/filler segment
)
STT_NON_SPEECH_TOKENS: frozenset[str] = frozenset(
    {
        "♪",
        "♫",
        "[music]",
        "(music)",
        "[laughter]",
        "(laughter)",
        "[applause]",
        "(applause)",
        "[noise]",
        "(noise)",
        "...",
        "…",
    }
)

# Mode B 2-stack panels — top (facecam) and bottom (gameplay/mediashare) split 50/50.
# Both panels share aspect 1080/960 = 1.125, so crops pre-shaped to this aspect scale
# into the panel with zero distortion. Single source of truth for tracker + ffmpeg builder.
STACK2_PANEL_W: int = 1080
STACK2_PANEL_H: int = 960
STACK2_PANEL_ASPECT: float = STACK2_PANEL_W / STACK2_PANEL_H  # 1.125

# Facecam top panel context margin. The cam box is expanded by this factor (comfortable surround),
# shaped to the panel aspect, and crop-filled into the 1080×960 panel — prominent, no floor, no blur
# (mild upscale when the cam is smaller than the panel). Cam fills ~1/factor of panel height:
# 1.45 → cam ≈69%.
FACECAM_FIT_FACTOR: float = 1.45

# Gameplay-gate thresholds (ContentTypeDetector): gaming classification requires BOTH conditions.
# Raise GAMEPLAY_MIN_NONPERSON_MOTION if static gaming menus trigger a false positive;
# lower GAMEPLAY_MIN_OPEN_AREA_FRAC if close-up gaming streams with little visible screen
# are missed.
GAMEPLAY_MIN_NONPERSON_MOTION: float = (
    4.0  # mean frame-diff in non-person cells ("moderate" boundary)
)
GAMEPLAY_MIN_OPEN_AREA_FRAC: float = (
    0.45  # fraction of coarse grid not covered by persons
)
# Relaxed open-area threshold when the YouTube metadata says "Gaming" (gaming_hint=True).  A
# close-up cam shot in a gaming stream may show <45% free grid but the game is still present
# behind/around the streamer; 30% is enough to confirm gameplay when the category corroborates.
GAMEPLAY_MIN_OPEN_AREA_FRAC_GAMING_HINT: float = 0.30

# Streamer-facecam detection (YOLO persons) for GAMING_SOLO vs GAMING_COLLAB and the Mode C collab
# pick. A real webcam inset is cam-sized (a fraction of the frame, not a tiny in-game character nor a
# full-frame talking head), persistent across sampled frames, and sits near a frame edge/corner — an
# in-frame game character is central and scores low. Tunable.
FACECAM_MIN_AREA_FRAC: float = 0.004  # below this = tiny game character / noise
FACECAM_MAX_AREA_FRAC: float = (
    0.45  # above this = full-frame talking head, not a corner cam
)
FACECAM_MIN_PERSISTENCE: float = 0.40  # must appear in >= 40% of sampled frames
# Edge score = 1 − (centre's distance to the nearest frame edge)/(min(w,h)/2): ~0 for an interior
# game character, high for a webcam hugging any border/corner. Above this = a real cam position.
FACECAM_EDGE_SCORE_MIN: float = 0.45
FACECAM_MIN_SEP_FRAC: float = (
    0.15  # two cams must be separated by > this × frame diagonal
)

# Face-landmarker capacity for PODCAST speaker tracking.
# num_faces is set from the YOLO person count (analysis["face_count"]) + FACE_COUNT_MARGIN,
# clamped to FACE_LANDMARKER_MAX_FACES.  The margin absorbs YOLO under-counts on partially
# visible or angled faces.  Raise FACE_LANDMARKER_MAX_FACES if you routinely clip panels
# with more than 8 speakers.
FACE_LANDMARKER_MAX_FACES: int = 8
# Safety margin added to the YOLO person count before sizing the FaceLandmarker.
# If YOLO sees 4 people, capacity is set to 6 — headroom for partially-visible faces.
FACE_COUNT_MARGIN: int = 2

# Minimum consecutive detection steps a new speaker must hold before the crop commits to
# them (debounce).  Prevents a single mis-detected frame from triggering a cut.
SPEAKER_HOLD_SECONDS: float = 2.0
# Minimum YOLO box confidence for counting simultaneous persons in a single frame.
# Boxes below this threshold are excluded from the face_count used to size num_faces.
# Does NOT affect facecam-picking or gameplay exclusion (those use persistent clusters).
PERSON_COUNT_CONF_MIN: float = 0.5

# PODCAST detection sampling rate (frames per second of video sampled for face/lip tracking).
# Higher than the default 5 fps because active-speaker detection needs to resolve syllable-rate
# (~3-5 Hz) mouth oscillation — 10 fps clears the Nyquist limit.  Non-PODCAST stays at 5 fps.
PODCAST_DETECTION_FPS: int = 10

# PODCAST two-shot group framing (decided ONCE per clip from the typical two-face geometry).
# Group both faces into one static crop only when BOTH hold:
#   - total face span ≤ GROUP_FRAMING_FIT_FACTOR × crop_w (both faces physically fit), AND
#   - inter-face gap (right edge of left face → left edge of right face) ≤ GROUP_MAX_GAP_FACTOR
#     × crop_w (faces are adjacent, so the crop center lands ON the faces — never the empty
#     table between far-apart people).
GROUP_FRAMING_FIT_FACTOR: float = 0.9
GROUP_MAX_GAP_FACTOR: float = 0.25
# Stable pseudo speaker-id for a committed two-shot (group) segment.  The segment builder
# treats this id as a single stable subject so it does not cut within the two-shot.
GROUP_SPEAKER_ID: int = -2

# PODCAST active-speaker detection via Mouth-Aspect-Ratio (MAR) movement + audio-visual sync.
# MAR = mean vertical inner-lip opening / mouth width (a smile widens but does not open → low MAR;
# speech opens vertically → high, oscillating MAR).  Activity = std-dev of MAR over this window.
LIP_ACTIVITY_WINDOW_SECONDS: float = 0.8
# Absolute floor on MAR activity below which a face is treated as silent (not a switch candidate).
LIP_ACTIVITY_MIN: float = 0.003
# Hysteresis ratio: a challenger face must have activity > current speaker's × this factor
# before the crop switches to them.  Suppresses jitter when two faces have similar activity.
SPEAKER_SWITCH_MARGIN: float = 1.5
# Minimum time (seconds) the crop must stay on a subject (speaker or group) before any
# switch is allowed.  The primary anti-dizziness lever for the single-speaker cut path.
MIN_SHOT_SECONDS: float = 2.0
# Audio-visual sync: a detection step counts as "voiced" when its audio RMS ≥ this factor ×
# the clip's median RMS.  During unvoiced steps (silence/pauses) the active speaker is held —
# no switching on a laugh or a pause.  Below the median scales the gate to the clip's loudness.
VOICE_ACTIVITY_FLOOR_FACTOR: float = 0.5

# Local (windowed) audio-visual coherence — the moment-to-moment speaker test.  At each voiced
# step the recent mouth-motion window is Pearson-correlated with the recent audio-RMS window over
# this many seconds.  A face is an eligible speaker only when its local coherence ≥ COHERENCE_MIN
# (and its activity ≥ LIP_ACTIVITY_MIN).  If NO visible face is eligible (e.g. the talker's mouth
# is occluded by a mic), the crop HOLDS the current speaker instead of jumping to a visible
# non-speaker / smiler.  Larger window = steadier but slower to react.
AV_SYNC_WINDOW_SECONDS: float = 1.0
COHERENCE_MIN: float = 0.25

# Rule-of-thirds headroom: shift the crop center up by this fraction of the face-box height so the
# subject's eyes sit in the upper third rather than dead center.  Kept modest because the PODCAST
# crop is full source height (a large offset would push the chin out of frame).
HEADROOM_FACTOR: float = 0.18

# Face-identity matching across detection steps: a face matches an existing track when their boxes
# overlap by at least this IoU.  Scales with face size (unlike a fixed pixel distance); the legacy
# center-distance test is kept only as a fallback when no box overlaps.
IOU_MATCH_MIN: float = 0.3

# Gentle EMA panning (PODCAST): the final per-frame crop center glides toward its target by this
# fraction each frame.  Within a held shot the target is constant so the crop stays put (no drift);
# on a speaker change it eases over ~1 s instead of hard-cutting.  Higher = snappier.
PAN_SMOOTHING_FACTOR: float = 0.03

# Haar-cascade fast tracking (opt-in `video_processing.fast_mode`): a lightweight CPU-only PODCAST
# tracker that follows the largest frontal face — no MediaPipe, no audio.  Frames are downscaled for
# speed, then detected with OpenCV's default frontal-face cascade.
HAAR_DOWNSCALE: float = 0.5  # grayscale resize factor before detection (speed)
HAAR_SCALE_FACTOR: float = 1.1  # cascade image-pyramid step
HAAR_MIN_NEIGHBORS: int = 5  # higher = fewer false positives, more missed faces

# Mode C 3-stack panels (GAMING_COLLAB) — facecam / gameplay / collab, each 1080x640.
STACK3_PANEL_W: int = 1080
STACK3_PANEL_H: int = 640
STACK3_PANEL_ASPECT: float = STACK3_PANEL_W / STACK3_PANEL_H  # 1.6875

# YOLOv8 COCO class ids used by the VisualAnalyzer (no magic numbers, AGENTS §11.5).
COCO_CLASS_PERSON: int = 0  # → facecam / collab face regions
COCO_CLASS_CELL_PHONE: int = 67  # → screen / MediaShare inset
COCO_CLASS_BOOK: int = 73  # → screen-like rectangular inset (fallback)
COCO_CLASS_LAPTOP: int = 63  # → screen / MediaShare inset
COCO_CLASS_TV: int = 62  # → screen / MediaShare inset

# Classes that look like a rectangular video/screen popup (MediaShare candidates).
COCO_SCREEN_CLASSES: frozenset[int] = frozenset(
    {COCO_CLASS_TV, COCO_CLASS_LAPTOP, COCO_CLASS_CELL_PHONE, COCO_CLASS_BOOK}
)

# Audio RMS analysis (energy heatmap + face-tracker audio-visual sync).  Mono 8 kHz PCM is plenty
# for loudness measurement and keeps the FFmpeg decode cheap.
RMS_SAMPLE_RATE: int = 8000

# MediaPipe FaceMesh inner-lip landmark indices used to compute the Mouth-Aspect-Ratio.
# Vertical pairs (upper, lower) sample the opening at three points; the width pair are the mouth
# corners.  MAR = mean vertical opening / mouth width.
MAR_VERTICAL_PAIRS: tuple[tuple[int, int], ...] = ((13, 14), (81, 178), (311, 402))
MAR_WIDTH_PAIR: tuple[int, int] = (78, 308)

# FFmpeg stderr substrings (lower-cased match) that indicate a hardware-encoder / driver failure
# rather than a syntax error.  Any hit triggers the automatic libx264 fallback in ClipRenderer.
GPU_ENCODER_FAILURE_SIGNS: tuple[str, ...] = (
    "nvenc",
    "qsv",
    "cuvid",
    "vaapi",
    "videotoolbox",
    "cannot load",
    "device not found",
    "initialization failed for codec",
    "driver",
    "opencl",
    "no capable devices",
    "function not implemented",
)

# Native GL libraries MediaPipe needs present on headless Linux/WSL (environment preflight).
MEDIAPIPE_GL_LIBS: tuple[str, ...] = ("libEGL.so.1", "libGLESv2.so.2")

# ── Audio speaker-count heuristic (AudioEnergyAnalyzer.estimate_speaker_count) ──
# Lightweight, numpy-only pitch clustering over the same 8 kHz mono PCM used for RMS — a coarse
# "~1 vs ~2 speakers" hint for the LLM clip-classifier, NOT exact diarization.
SPEAKER_F0_MIN_HZ: float = 70.0  # lowest fundamental searched (deep male voice)
SPEAKER_F0_MAX_HZ: float = (
    300.0  # highest fundamental searched (high female/child voice)
)
SPEAKER_PITCH_FRAME_SAMPLES: int = 1024  # ~128 ms analysis frame at 8 kHz
SPEAKER_VOICE_RMS_FLOOR_FRAC: float = (
    0.5  # voiced frame = RMS ≥ this × the median frame RMS
)
SPEAKER_MIN_VOICED_FRAMES: int = (
    12  # fewer voiced/pitched frames than this → assume 1 speaker
)
SPEAKER_PITCH_CLARITY_MIN: float = (
    0.3  # autocorr peak ≥ this × zero-lag energy → a clear pitch
)
SPEAKER_MIN_SEPARATION_HZ: float = (
    45.0  # two pitch clusters this far apart → distinct speakers
)
SPEAKER_MIN_CLUSTER_FRAC: float = (
    0.25  # each cluster must hold ≥ this share of pitched frames
)
SPEAKER_KMEANS_ITERS: int = 10  # iterations for the tiny 1-D 2-means split

# ── Overlay/MediaShare popup detector thresholds (src/vision/overlay_detector.py) ──
# A genuine donation/MediaShare popup is transient and bounded — it appears then disappears
# against an otherwise steady background. These gates keep only bounded, transient intervals
# and reject persistent furniture (chat list, bottom ticker, the facecam itself).
MIN_POPUP_SECONDS: float = 1.0  # Below this a novelty blip is noise, not an alert card.
MAX_POPUP_SECONDS: float = (
    15.0  # Donation alerts are short; longer = persistent background.
)
POPUP_WINDOW_COVERAGE: float = (
    0.7  # Reject intervals covering >=70% of the window (scene change).
)
SCAN_PAD_SECONDS: float = (
    3.0  # Pad the scan window so a clip-spanning popup is still a minority.
)
OVERLAY_SMALL_W: int = 480  # Downscaled analysis width for fast median + diff.
OVERLAY_SMALL_H: int = 270  # Downscaled analysis height.
MIN_SCAN_FRAMES: int = 5  # Need enough frames to build a meaningful median baseline.
NOVELTY_THRESH: int = 30  # Per-pixel abs-diff vs baseline that counts as "changed".
MIN_POPUP_AREA_FRAC: float = 0.025  # Donation card is sizeable, not a moving character.
MAX_POPUP_AREA_FRAC: float = (
    0.40  # Larger than this is a scene change, not an alert card.
)
POPUP_ASPECT_MIN: float = 1.5  # Alert cards are clearly wider than tall.
POPUP_ASPECT_MAX: float = 6.0
# Static-vs-moving discriminator: a real donation card holds the SAME on-screen position while shown,
# so its per-frame novelty box barely drifts. Gameplay motion drifts frame-to-frame. Reject intervals
# whose box centre jitter (normalised by frame diagonal) exceeds this threshold.
POPUP_MAX_JITTER: float = 0.035
BOTTOM_STRIP_FRAC: float = (
    0.88  # Boxes centred below this (full-width ticker) are dropped.
)
FULLWIDTH_FRAC: float = (
    0.70  # Boxes spanning >=70% width (ticker / lower-thirds) are dropped.
)
OVERLAY_DEFAULT_SAMPLE_STEP: float = 0.5  # ~2 fps sampling cadence for overlay scan.
OVERLAY_MAX_SCAN_SAMPLES: int = (
    200  # Upper bound on frames sampled to limit cost on long windows.
)
OVERLAY_MORPH_KERNEL_SIZE: int = (
    9  # Kernel size for morphological close (bridges small gaps).
)
OVERLAY_MAX_GAP_SECONDS: float = (
    1.5  # Max gap between detections still considered the same popup.
)

# ── VisualAnalyzer motion-grid constants (src/vision/visual_analyzer.py) ──
# The gameplay motion analysis downscales the frame to this grid for fast frame-diff calculations.
# Both the gameplay-presence probe and the gameplay-pan / motion-region methods use the same grid.
MOTION_GRID_W: int = 320
MOTION_GRID_H: int = 180
# Coarse grid for the open-area fraction calculation (how much of the frame is NOT a person).
OPEN_AREA_GRID_COLS: int = 16
OPEN_AREA_GRID_ROWS: int = 9
GAMEPLAY_BURST_FPS: float = 0.5  # Burst sample rate for gameplay motion probe (~2 fps).
# Gameplay pan smoothing params (opt-in legacy mode, disabled by default).
GAMEPLAY_PAN_EMA: float = 0.2  # EMA smoothing factor for motion-follow pan.
GAMEPLAY_PAN_DEADZONE_FRAC: float = (
    0.06  # Fraction of width — centre deadzone (no pan below this).
)
GAMEPLAY_PAN_MAX_VELOCITY_FRAC: float = (
    0.012  # Max pan speed as fraction of width per frame.
)
GAMEPLAY_PAN_MAX_DEVIATION_FRAC: float = (
    0.12  # Max dev from frame centre as fraction of width.
)
MOTION_INTENSITY_HIGH: int = 12  # Mean frame-diff above this = "high motion".
MOTION_INTENSITY_MODERATE: int = 4  # Mean frame-diff above this = "moderate motion".
YOLO_CLUSTER_DIST_FRAC: float = (
    0.15  # YOLO box cluster distance threshold × frame diagonal.
)
FACECAM_PICK_MIN_PERSISTENCE: float = (
    0.5  # Min fraction of sampled frames a facecam candidate must appear in.
)

# ── LayoutBuilder constants (src/vision/layout_builder.py) ──
FACECAM_DOMINANT_AREA_FRAC: float = (
    0.55  # Facecam covers >55% of frame → render as full-face
)
# single vertical (Mode A) instead of 2-stack.

# ── ContentTypeDetector HUD-score & donation thresholds ──
# HUD (Heads-Up Display) score thresholds for detecting gaming UI elements like health bars,
# minimaps, and kill feeds. Used to corroborate gameplay presence.
HUD_SCORE_THRESHOLD_GAMING_HINT: float = (
    0.0015  # Relaxed when YouTube category says "Gaming".
)
HUD_SCORE_THRESHOLD_DEFAULT: float = 0.003
HUD_ANALYSIS_SIZE: int = 128  # Downscale resolution for HUD temporal-variance analysis.
HUD_TEMP_STD_THRESH: float = 8.0  # Min temporal std-dev across frames to count as HUD.
HUD_SPATIAL_GRAD_THRESH: float = 18.0  # Min spatial gradient magnitude to count as HUD.
HUD_EDGE_MARGIN: int = 6  # Border margin (px) for HUD mask edge exclusion.
SAMPLE_FRAME_START_FRAC: float = 0.10  # Skip first 10% of video when sampling frames.
SAMPLE_FRAME_END_FRAC: float = 0.90  # Skip last 10% of video when sampling frames.
# Donation alert HSV colour ranges (for the colour-heuristic donation check on single-face
# content where no mediashare novelty scan is available).
DONATION_HSV_RED_LOWER1: tuple[int, int, int] = (0, 100, 100)  # Red hue band 1 lower.
DONATION_HSV_RED_UPPER1: tuple[int, int, int] = (10, 255, 255)  # Red hue band 1 upper.
DONATION_HSV_RED_LOWER2: tuple[int, int, int] = (
    170,
    100,
    100,
)  # Red hue band 2 lower (wraps around).
DONATION_HSV_RED_UPPER2: tuple[int, int, int] = (180, 255, 255)  # Red hue band 2 upper.
DONATION_HSV_ORANGE_LOWER: tuple[int, int, int] = (
    5,
    100,
    100,
)  # Orange/alert hue lower.
DONATION_HSV_ORANGE_UPPER: tuple[int, int, int] = (
    25,
    255,
    255,
)  # Orange/alert hue upper.
DONATION_MIN_AREA: int = 2000  # Min contour area (px) for donation-alert detection.
DONATION_MAX_AREA: int = (
    50000  # Max contour area (px); larger = scene change, not alert.
)
DONATION_ASPECT_MIN: float = 1.2  # Min aspect ratio for donation alert bounding box.
DONATION_ASPECT_MAX: float = 4.0  # Max aspect ratio for donation alert bounding box.
DONATION_PERSISTENCE_MAX_RATIO: float = (
    0.6  # Max ratio of frames a donation must appear in (too many
)
# frames = persistent UI, not transient popup).

# ── FaceTracker crop-dimension constants (src/vision/face_tracker.py) ──
PODCAST_CROP_ASPECT: float = (
    9.0 / 16.0
)  # Full-height crop for single-speaker PODCAST (9:16).
COLLAB_CROP_HEIGHT_FRAC: float = (
    0.4  # Mode C bottom panel (collab face) is ~40% of frame height.
)
SOLO_CROP_HEIGHT_FRAC: float = (
    0.5  # Mode B panels (facecam/gameplay) are ~50% of frame height.
)
MIN_CROP_DIMENSION: int = (
    100  # Minimum crop width/height in pixels (avoids degenerate FFmpeg crops).
)

# ── Human-readable content-type names for logging ──
# Maps ContentType enum values to plain-language labels for INFO-level logging.
CONTENT_TYPE_HUMAN_NAMES: dict[str, str] = {
    "PODCAST": "podcast or panel",
    "JUST_CHAT": "live stream",
    "GAMING_SOLO": "gaming solo",
    "GAMING_COLLAB": "gaming collaboration",
    "DONATION_OVERLAY": "donation overlay",
}

# ── LLM inference constants ──
# Single-transcript: full video transcript → clips with titles/captions/descriptions/hashtags.
# Batch: candidate transcripts + visual descriptors → detailed comparison + rich per-clip metadata.

# Token limits for local LLM responses (single transcript vs. batch candidate selection).
LLM_LOCAL_MAX_TOKENS_SINGLE: int = 2048
LLM_LOCAL_MAX_TOKENS_BATCH: int = 4096

# Cloud LLM response token limits (OpenAI max_completion_tokens / Gemini max_output_tokens).
LLM_CLOUD_MAX_TOKENS_SINGLE: int = 4096
LLM_CLOUD_MAX_TOKENS_BATCH: int = 8192

# Default context window for llama.cpp models (n_ctx parameter).
LLAMA_N_CTX: int = 4096

# ── Pipeline thread-pool sizes ──
# STT transcription: number of parallel workers for candidate audio slice decoding.
STT_THREAD_POOL: int = 4
# Region/visual analysis: number of parallel workers for candidate video window scanning.
VISUAL_THREAD_POOL: int = 5

# ── Bytes-to-MB conversion factor ──
BYTES_PER_MB: int = 1024 * 1024

# ── Gameplay scan upper bound ──
# Maximum number of frames scanned during gameplay pan / motion-region analysis.
MAX_GAMEPLAY_SCAN_FRAMES: int = 120
