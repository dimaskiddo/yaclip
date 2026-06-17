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
SPEAKER_HOLD_SECONDS: float = 1.0
# Minimum YOLO box confidence for counting simultaneous persons in a single frame.
# Boxes below this threshold are excluded from the face_count used to size num_faces.
# Does NOT affect facecam-picking or gameplay exclusion (those use persistent clusters).
PERSON_COUNT_CONF_MIN: float = 0.5

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
