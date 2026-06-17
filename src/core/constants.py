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


class ClipMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"


class LayoutMode(str, Enum):
    SINGLE_VERTICAL = "single_vertical"
    STACKED_SPLIT = "stacked_split"
    MULTI_COLLAB = "multi_collab"


class AIProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    LOCAL = "local"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Layout and rendering constants
ASPECT_RATIO_VERTICAL: float = 9.0 / 16.0
DEFAULT_FONT_NAME: str = "Anton.ttf"

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
STT_COMPRESSION_MAX: float = 2.4    # gzip ratio above this = looped/repeated text (hallucination)
STT_NO_SPEECH_MAX: float = 0.6      # no_speech_prob above this = likely silence/noise
STT_LOGPROB_MIN: float = -1.0       # avg_logprob below this = low-confidence decode
STT_REPEAT_TOKEN_MAX: int = 3       # a single token repeated more than this = laughter/filler segment

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
GAMEPLAY_MIN_NONPERSON_MOTION: float = 4.0  # mean frame-diff in non-person cells ("moderate" boundary)
GAMEPLAY_MIN_OPEN_AREA_FRAC: float = 0.45   # fraction of coarse grid not covered by persons

# Streamer-facecam detection (YOLO persons) for GAMING_SOLO vs GAMING_COLLAB and the Mode C collab
# pick. A real webcam inset is cam-sized (a fraction of the frame, not a tiny in-game character nor a
# full-frame talking head), persistent across sampled frames, and sits near a frame edge/corner — an
# in-frame game character is central and scores low. Tunable.
FACECAM_MIN_AREA_FRAC: float = 0.004   # below this = tiny game character / noise
FACECAM_MAX_AREA_FRAC: float = 0.45    # above this = full-frame talking head, not a corner cam
FACECAM_MIN_PERSISTENCE: float = 0.40  # must appear in >= 40% of sampled frames
# Edge score = 1 − (centre's distance to the nearest frame edge)/(min(w,h)/2): ~0 for an interior
# game character, high for a webcam hugging any border/corner. Above this = a real cam position.
FACECAM_EDGE_SCORE_MIN: float = 0.45
FACECAM_MIN_SEP_FRAC: float = 0.15     # two cams must be separated by > this × frame diagonal

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
PAN_SMOOTHING_FACTOR: float = 0.12

# Haar-cascade fast tracking (opt-in `video_processing.fast_mode`): a lightweight CPU-only PODCAST
# tracker that follows the largest frontal face — no MediaPipe, no audio.  Frames are downscaled for
# speed, then detected with OpenCV's default frontal-face cascade.
HAAR_DOWNSCALE: float = 0.5       # grayscale resize factor before detection (speed)
HAAR_SCALE_FACTOR: float = 1.1    # cascade image-pyramid step
HAAR_MIN_NEIGHBORS: int = 5       # higher = fewer false positives, more missed faces

# Mode C 3-stack panels (GAMING_COLLAB) — facecam / gameplay / collab, each 1080x640.
STACK3_PANEL_W: int = 1080
STACK3_PANEL_H: int = 640
STACK3_PANEL_ASPECT: float = STACK3_PANEL_W / STACK3_PANEL_H  # 1.6875

# YOLOv8 COCO class ids used by the VisualAnalyzer (no magic numbers, AGENTS §11.5).
COCO_CLASS_PERSON: int = 0          # → facecam / collab face regions
COCO_CLASS_CELL_PHONE: int = 67     # → screen / MediaShare inset
COCO_CLASS_BOOK: int = 73           # → screen-like rectangular inset (fallback)
COCO_CLASS_LAPTOP: int = 63         # → screen / MediaShare inset
COCO_CLASS_TV: int = 62             # → screen / MediaShare inset

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
    "nvenc", "qsv", "cuvid", "vaapi", "videotoolbox", "cannot load",
    "device not found", "initialization failed for codec", "driver", "opencl",
    "no capable devices", "function not implemented",
)

# Native GL libraries MediaPipe needs present on headless Linux/WSL (environment preflight).
MEDIAPIPE_GL_LIBS: tuple[str, ...] = ("libEGL.so.1", "libGLESv2.so.2")
