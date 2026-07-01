# Yet Another AI Auto-Clipper (YaClip) — Agent Instructions

## 🎯 Role & Objective

You are an expert Python software architect, AI integration specialist, and multimedia processing engineer. Your task is to build a modern, cross-platform **YaClip** — an application that downloads YouTube videos, automatically detects the most engaging moments, and renders them as vertical short-form clips ready for **YouTube Shorts, Instagram Reels, and TikTok**.

The source video can be any of these content types: **Podcast / Panel Discussion, Gaming Stream, Just Chat / Just Chill, or Solo Commentary**. Each type has its own detection heuristics, face-tracking behavior, and vertical layout strategy. The application must handle all of them correctly without manual configuration, unless the user explicitly wants to override detection.

The application features both a **CLI** and a **WebUI** (Gradio). It must be cross-platform (Windows, macOS, Linux, WSL), optimized for **low-to-medium spec hardware**, implement a **100% Portable Asset Cache**, and follow a **Cloud-First with Local-Fallback** AI architecture.

---

## 📁 Directory Context

- **Current Directory (`./`)**: The root of the YaClip Python project.

---

## 📋 Workflow & Task Tracking (CRITICAL)

This is a complex ground-up project. The agent MUST NOT attempt to write the entire codebase in a single response. Check TASKS.md before each stage.

1. **Project Initialization & Planning:** Create `TASKS.md` and the `./docs/` directory first. Map the build into a checklist in `TASKS.md`. Simultaneously generate `./docs/ARCHITECTURE.md` and `./docs/WORKFLOWS.md`.
2. **State Sync:** Always read `TASKS.md` before starting any new session to orient to the current project state.
3. **Immutability:** NEVER rework or touch items marked done (`[x]`) in `TASKS.md` unless explicitly instructed.
4. **Auto-Update:** Update `TASKS.md` immediately after successfully testing a feature.

---

## 🛠️ Skills & Utilization (CRITICAL)

**🔥 GLOBAL OVERRIDE DIRECTIVE:** Every prompt, interaction, and task execution MUST be processed as if `"Use caveman mode full"` has been explicitly injected at all times.

1. **Boot Sequence:** Before ANY coding task, invoke and read the `using-superpowers`, `karpathy-guidelines`, and `caveman` skills first.
2. **Skill Routing:** Use the `using-superpowers` framework to identify and load other relevant skills per task.
3. **Plan-Validate-Execute:** State which skills were loaded, validate against all constraints, then execute.

---

## 🏗️ Strict Architectural Constraints

### 1. Product Scope & Content Type Definitions

This section is the source of truth for what this application is and how it thinks about content. Every module in the codebase MUST be built around these definitions.

#### 1.1 Supported Content Types

The application recognizes four content types. Detection is automatic unless overridden in `config.yaml`.

| Content Type | Key Characteristics |
|---|---|
| `PODCAST` | Talking-head content with no confirmed gameplay: single speaker, panel discussion, interview, or Q&A. When two or more persistent face regions are detected, the camera follows the active speaker (lip-movement tracking). When a single face is dominant, the camera centers and tracks that speaker only. |
| `JUST_CHAT` | Single streamer, no gaming HUD, donation overlays (Trakteer.ID, MediaShare) are common and must be preserved |
| `GAMING_SOLO` | Confirmed gameplay detected (animated non-person screen region), single facecam in corner, donation overlays must be preserved |
| `GAMING_COLLAB` | Confirmed gameplay detected, multiple persistent webcam faces (Discord grid, dual-screen collab layout), donation overlays must be preserved |
| `DONATION_OVERLAY` | **Per-clip, not a video-level type.** Any clip whose window contains a transient mediashare/donation overlay is promoted to this type (from whatever base type the video has) and routed to a dedicated facecam + popup layout. This is the single home for donation handling — the other layouts no longer embed donations. |

#### 1.2 Target Output Formats

All rendered clips MUST be 9:16 vertical aspect ratio, suitable for direct upload to:
- YouTube Shorts (max 60s recommended, 3 min absolute max)
- Instagram Reels (max 90s recommended)
- TikTok (max 60s default, up to 3 min)

Clip length is driven by the **target plus a small margin**, not by min/max. Every clip runs **`[default_clip_duration_seconds, default_clip_duration_seconds + clip_length_margin_seconds]`** — e.g. **60–75s** at default 60 / margin 15. `default_clip_duration_seconds` is the LLM's target and the guaranteed floor (a shorter pick is extended up to it within the transcribed window); `default + clip_length_margin_seconds` is the hard cap (a longer pick is trimmed to it). `min_clip_duration_seconds` / `max_clip_duration_seconds` are **NOT** per-clip bounds — they are repurposed as the allowed **range of the Gradio WebUI duration slider** (how low/high a user may set the target). Hybrid candidate windows are sized to `default + margin` (not the slider max) so the runtime stays fast; CLI `--duration` overrides the target per run.

#### 1.3 Clip Selection Modes

Two modes exist and MUST be fully supported:

- **Auto Mode:** The application analyzes the video using AI transcript analysis, YouTube heatmap spike detection, or both ("hybrid"), then proposes the most engaging timestamps. The user reviews proposals before rendering begins.
- **Manual Mode:** The user provides explicit start/end timestamp ranges (supports bulk input, one range per line). Layout detection, face tracking, and rendering run identically to Auto Mode — only the timestamp sourcing differs.

---

### 2. Pure Python 3 Ecosystem & Portable Asset Cache (`./workspace/`)

- **Strict Virtual Environment (CRITICAL):** Always ensure `.venv` is active before executing any Python script or installing packages (`source .venv/bin/activate` on Linux/macOS, `.\.venv\Scripts\activate` on Windows, or `uv run`). NEVER touch the global system environment.
- **Package Management:** Use `uv` (preferred) or `pip3`. All install commands MUST use no-cache flags (`--no-cache-dir` / `--no-cache`) to prevent disk bloat on low-spec hardware.
- **PyTorch — CPU build by default (CRITICAL):** `requirements.txt` and `pyproject.toml` pin `torch==2.12.0+cpu` / `torchvision==0.27.0+cpu` from the PyTorch CPU index. The CUDA torch build ships `triton`, which segfaults (`SIGSEGV`) when imported after MediaPipe runs its native inference on CPU-only / WSL environments (MediaPipe × triton native-lib ABI collision). The CPU build carries no triton and eliminates this crash class entirely. A `guard_triton_segfault()` function in `src/core/environment.py` provides a second safety net: if a CUDA torch is somehow installed on a box with no GPU it masks `triton` via `sys.modules["triton"] = None` so torch's own `has_triton_package()` returns `False` (via `ImportError`) instead of crashing. GPU users who install a CUDA torch must set `YACLIP_FORCE_TRITON=1` to bypass the guard.
- **Portable Workspace (`./workspace/`):** All runtime assets live inside `./workspace/` within the project root. NEVER use the OS system temp directory (`/tmp/`, `%TEMP%`).
- **Startup Integrity Check:** On every app boot, verify and auto-repair the workspace environment:
  - `./workspace/bin/` — check for FFmpeg and yt-dlp. Auto-download the correct OS-specific binary if missing.
  - `./workspace/fonts/` — check for the primary subtitle font. Auto-download an open-source high-impact font (e.g., Anton or Oswald) if missing.
  - `./workspace/videos/`, `./workspace/audios/`, `./workspace/subtitles/`, `./workspace/data/`, `./workspace/tmp/` — create if missing.
- Advanced users may manually place custom binaries or `.ttf` fonts directly into the relevant `./workspace/` subdirectory.

---

### 3. Hybrid AI Pipeline — Independent STT & LLM Provider Selection

STT (transcription) and LLM (clip analysis) are configured **independently** under `ai_pipeline.stt` and `ai_pipeline.llm`. This makes the most useful mixed mode explicit: **local STT + cloud LLM** — faster-whisper transcribes for free at good accuracy, while a cloud LLM provides superior clip selection and title generation.

Each component has its own `provider` setting:
- `"cloud"` — use the configured cloud provider
- `"local"` — use the configured local model
- `"auto"` — try cloud first; fall back to local on API failure or missing credentials

**Provider combination behaviour:**

| `stt.provider` | `llm.provider` | Pipeline behaviour |
|---|---|---|
| `cloud` (google) | `cloud` (google) | **Single unified Gemini call** — audio uploaded once, STT + analysis in one request. Most cost-efficient. |
| `cloud` (openai) | `cloud` (any) | OpenAI Whisper API for STT, separate GPT/Gemini call for LLM. |
| `local` | `cloud` | faster-whisper transcribes locally, transcript passed to cloud LLM. **Recommended default.** |
| `cloud` | `local` | Cloud STT for language accuracy, local llama-cpp for analysis. |
| `local` | `local` | Fully offline — no API calls, no credentials required. |

**Memory safety (when both are local):** STT and LLM models MUST never coexist in RAM. Strict sequential load/unload: `Load STT → Transcribe → del model → gc.collect() → Load LLM → Analyze → del model → gc.collect()`.

---

### 4. Dual Interface Mode (CLI + Gradio WebUI)

- **CLI:** Implemented with `typer`. Supports all clip operations, workspace management, and configuration overrides via flags.
- **WebUI:** Implemented with `gradio`. Must be visually clean and logically organized into tabs: **Clipper**, **Review & Render**, **Settings**, and **Maintenance**.
- **Gradio Task Queuing (CRITICAL):** Always initialize with `.queue().launch()`. Never use `.launch()` alone — this causes socket timeouts on long video jobs.
- **WSL Compatibility:** Gradio MUST bind to `127.0.0.1:7860` (configurable) so headless Linux/WSL environments expose the UI to the host Windows browser cleanly.
- **Execution Routing:** `app.py` is entry/routing only (load config → environment → logging) and delegates to the Typer app in `src/interfaces/cli.py`. If `sys.argv` has CLI arguments → route to Typer; if run bare → launch Gradio. CLI commands: `clip <URL>` (the pipeline, with `--clips/--duration/--language/--output-dir/--force/--debug` overrides, plus `--manual/--timerange-file/--no-metadata` for manual mode — see §5.2), `config`, `cache status`, `cache purge [--dry-run]`, `serve` . `clean-workspace` remains as a hidden back-compat alias of `cache purge`.
- **WebUI parity

---

### 5. Clip Selection — Auto Mode & Manual Mode

#### 5.1 Auto Mode

Clips are proposed by the AI pipeline before rendering. Three strategies are supported, configured via `clip_selection.auto_strategy`:

| Strategy | Behaviour |
|---|---|
| `"ai"` | Full transcript analysis by the LLM. Best for Podcast, Interview, and Just Chat content where heatmap data is sparse or absent. |
| `"heatmap"` | Uses YouTube's built-in heatmap replay data (most-replayed timestamps). Best for Gaming content with clear engagement spikes. |
| `"hybrid"` | Both signals combined. Heatmap identifies candidate windows; AI ranks and titles them. Recommended default. |

**Candidate Pre-Ranking & Batched LLM Analysis (CRITICAL):**

RMS Energy and Heatmap both produce more candidate windows than the user wants. Sending all of them to the LLM wastes compute and STT cost. The pipeline MUST apply a **candidate margin** to pre-filter before any AI call.

The selection flow for auto mode:
1. RMS or Heatmap produces N ranked spike windows (e.g., 25), already sorted by score (energy level or heatmap value) — highest first.
2. Compute the candidate pool size: `target_clips + candidate_margin` (e.g., 5 clips + 2 = 7 candidates).
3. Take only the **top candidates by score** from step 1. Discard the rest entirely — no STT, no LLM call on them.
4. STT-transcribe those top candidates.
5. Send **all transcripts in a single batched LLM call**, asking the model to return the best `target_clips` from the pool. The LLM compares candidates against each other and picks the most engaging and diverse set.
6. LLM returns exactly `target_clips` results with titles and reasoning. **Post-processing enforces the cap:** if the LLM returns more than `target_clips`, the list is clamped to `target_clips` before deduplication — cosmetic noise never over-produces.

This means: 25 RMS spikes + user wants 5 clips + margin 2 → **7 STT calls, 1 LLM call**, not 25 of each.

`candidate_margin` is configurable in `config.yaml`. It is **additive** (`pool = target_clips + margin`), not a multiplier, so STT cost stays bounded as the requested clip count grows (15 clips + 2 = 17 candidates, not 30). A value of `2` is the recommended default — enough headroom for the LLM to make a meaningful selection without excessive STT cost. Raising to `3`–`5` gives the LLM a broader pool at higher cost. Minimum valid value is `0` (LLM receives exactly the target count — no selection, just titling and boundary refinement).

**Review Gate:** After clip proposal, all candidates (Start Time, End Time, Title, Reasoning, Content Type detected) MUST be displayed in the WebUI review panel before any rendering begins. The user can approve, edit, or discard individual proposals. This gate is skippable via `clip_selection.require_review_before_render: false`.

#### 5.2 Manual Mode

- The user provides start and end timestamps directly. The format MUST accept both `MM:SS - MM:SS` and `HH:MM:SS - HH:MM:SS` on the same input.
- **Bulk Input:** Multiple ranges entered one per line in a textarea (WebUI) or via a `.txt` file upload or `--timerange-file` CLI flag (`src.core.utils.parse_timerange_line` / `load_timerange_file`). Each line is parsed into a separate clip job.
- **CLI flags (implemented):** `--manual` (bool, requires `--timerange-file`) and `--timerange-file <path>` (requires `--manual`) — passing one without the other logs an error and exits non-zero. `--no-metadata` (manual-mode only; errors if passed without `--manual`) skips STT+LLM titling entirely.
- **Selection-config bypass:** Manual mode ignores `default_clips`, `min_clip_duration_seconds`/`max_clip_duration_seconds`, `candidate_margin`, dedup, and duration-enforcement — clips render at exactly the user's boundaries (only a `start < end` safety check applies, not `MIN_CLIP_SECONDS`).
- **LLM titling still runs by default:** unlike AI clip *selection*, manual mode by default still transcribes each fixed range and sends it through the batched LLM so it gets a real title/caption/description/hashtags — only the timerange boundaries are user-fixed, not the metadata. Pass `--no-metadata` to skip STT+LLM entirely; clips then get a default `Manual_<start>_<end>` title (e.g. `Manual_1-30_2-30`) and no `.txt` sidecar is written.
- **Rendering is identical to Auto Mode:** Manual mode bypasses AI timestamp *selection* only. Layout detection, face tracking, subtitle generation, and all rendering logic run exactly the same as in Auto Mode.
- Manual clips MUST still go through the Review Gate (displayed for confirmation) unless the review gate is globally disabled in config.

#### 5.3 Shared UI Controls (WebUI)

The Gradio Clipper tab MUST provide:
- A YouTube URL input field.
- A mode toggle: **Auto** / **Manual**.
- In Manual mode: a multiline text area for bulk timestamp entry and a `.txt` file upload component.
- In Auto mode: sliders for Target Clip Duration and Number of Clips to Generate, with min/max/default values loaded from `config.yaml`.
- A language selector for subtitles: dropdown with common languages plus an "Auto-Detect" option.

#### 5.4 WSL Cookie Resolution (CRITICAL)

When `yt-dlp` uses browser cookies for bot-bypass, the downloader MUST detect if it is running inside WSL. If so, it must dynamically resolve the Windows host username, map the cookie database path via `/mnt/c/Users/<WinUser>/AppData/...`, and copy the SQLite file to `./workspace/tmp/` before extraction to sidestep cross-OS file lock issues.

---

### 6. Video Type Detection Engine

This is the core intelligence of the application. Content type is detected **once per video**, before any clip selection or rendering. Its output — a `ContentType` enum value — drives all downstream decisions: layout mode, face-switching behavior, donation overlay handling, and the LLM's system-prompt context. The detector aggregates evidence across the **entire video** (25 sampled frames + YOLO facecam detection + gameplay presence probe + HUD heuristic), making it far more reliable than per-clip classification, which suffers from noisy signals in short 60s windows.

**Visual Analysis (`src/vision/visual_analyzer.py`):** A shared YOLOv8n (Ultralytics COCO) engine analyses sampled frames of a time window and returns region metadata — `facecam_box`, `gameplay_box`/`gameplay_track`, `screen_inset_box`, persistent person count, motion level, `mediashare_events` — plus a compact **text descriptor**. This one engine serves three consumers: (a) **clip selection** — the descriptor is attached to each candidate window so a (possibly non-multimodal) LLM weighs on-screen events; (b) **rendering layout** — precise per-clip crop boxes replace generic face-tracking for Mode B/C; (c) **content-type detection** — person/screen/motion signals supplement the HUD heuristic. Configured under `video_processing.region_detection`; the YOLO region pass is sparse for low-spec hardware.

Two refinements matter for correctness:
- **Dense MediaShare event scan.** A transient MediaShare/donation popup is easily missed by sparse sampling, so a separate **~2 fps bounded scan** (appearance/disappearance against a median-frame baseline, via `overlay_detector`) runs over each window and reports `mediashare_events` with timing. The descriptor states e.g. *"Donation overlay reaction at ~412–418s (high-value moment)"*, and the LLM prompt treats such candidates as **high-priority** — this is what lets a mediashare moment actually be selected.
  - **Static-vs-moving gate (CRITICAL, anti-false-positive):** a real donation card holds the **same on-screen position** while shown; gameplay novelty (camera pan, moving character) **drifts** frame-to-frame and would otherwise be misread as a popup. The detector computes per-interval **box-centre jitter** (normalised by the frame diagonal) and **rejects** intervals above `POPUP_MAX_JITTER` — only a positionally-stable card survives. The aspect gate also requires cards clearly wider than tall (`POPUP_ASPECT_MIN`).
- **Static gameplay crop.** At render, the gameplay bottom panel uses a **static centred crop** (motion-centroid–centred, zoomed by `gameplay_zoom`, cam-band-aware) — no panning, no animation. The crop is computed once per clip window by `_motion_region` and emitted as a constant `crop=` FFmpeg filter. MediaShare bottom is also a static crop. The legacy motion-following pan (`_gameplay_pan`, `gameplay_follow_motion: true`) is preserved as an opt-in escape hatch but disabled by default.

Audio (heatmap/energy/STT) and visual analysis run over the **same time windows** so selection and layout share aligned evidence. **Game/show context:** the downloader saves `{video_id}_metadata.json` (title/category/tags/description); the batched LLM prompt is prefixed with this so it can infer the game, and a "Gaming" category sets `gaming_hint`, which corroborates the gameplay gate during classification.

#### 6.1 Detection Pipeline (video-level, with LLM fallback for uncertain)

Detection runs **once** before clip selection or rendering, aggregating evidence across the whole video:

1. **Config Override** — If `video_processing.content_type_override` is set to anything other than `"auto"`, that value is returned immediately; no detection runs.

2. **Frame Sampling** — 25 frames are sampled evenly from the middle 80% of the video. These are used for the HUD score heuristic, donation overlay detection, and (via the VisualAnalyzer) the gameplay presence probe.

3. **Gaming Detection** — Three corroboration signals are combined:
   - `VisualAnalyzer.detect_gameplay_presence(video_path)` measures `open_area_frac` (fraction of frame NOT occupied by person boxes) and `non_person_motion` (mean frame-diff in non-person regions across 3 short motion bursts). Gaming requires `open_area_frac ≥ 0.45` (or `≥ 0.30` when `gaming_hint=True` for close-up cam shots).
   - `gaming_hint` (YouTube "Gaming" category in `{video_id}_metadata.json`) bypasses the motion check — a close-up cam shot in a gaming stream still has gameplay behind/around the streamer.
   - **HUD score** detects static graphic UI elements (health bars, minimaps, kill feeds). A high HUD score corroborates gameplay even when the streamer's cam obscures most of the screen.

4. **Webcam Count** — `VisualAnalyzer.detect_facecams(video_path)` runs once (reliable filtered cams, not raw YOLO person boxes). Returns webcam boxes filtered by persistence (≥40% of frames), area (cam-sized, not game characters), edge proximity (webcams sit near frame edges), and spatial separation. This replaces the old raw `face_count` which over-counted game characters and cutscene NPCs — the primary cause of false `GAMING_COLLAB` detection.

5. **Decision Tree:**
   - Gameplay confirmed + `< 2` webcams → **`GAMING_SOLO`** (confident).
   - Gameplay confirmed + `≥ 2` webcams → **`GAMING_COLLAB`** (confident — `detect_facecams` already filters out game characters, so ≥2 genuine webcams + gameplay is definitively collab).
   - No gameplay + `≥ 2` persistent faces → **`PODCAST`**.
   - No gameplay + 1 face + donation alerts → **`JUST_CHAT`**.
   - No gameplay + 1 face → **`PODCAST`**.
   - Ambiguous (no faces, no gameplay, no donation) → **`None`** (uncertain — defer to LLM with structured detection evidence injected into prompt).

6. **LLM Fallback (uncertain only)** — When the detector returns `None`, the batched selection LLM classifies each clip. The prompt includes: **structured detection evidence block** with raw numbers (`gameplay_present`, `webcam_count`, `hud_score`, `gaming_hint`, `open_area_frac`, `non_person_motion`, `donation_detected`), the video-level hint, audio speaker count, visual context descriptors, and game/show metadata. The LLM's `content_type` output is **post-validated** — audio ~1 speaker can never be COLLAB (forced to SOLO/PODCAST), and audio ≥2 speakers + gameplay + ≥2 webcams forces COLLAB. When the detector returns a confident type, it is forced on all clips and no evidence is sent to the LLM.

**Key difference from per-clip classification:** whole-video evidence is far more reliable — 25 frames across the full video length vs 6 frames in a 60s clip. `detect_facecams` replaces raw `face_count`, eliminating the cutscene-NPC false positive. `gaming_hint` relaxes the open-area threshold for close-up gaming streams. HUD score adds a third corroboration signal.

#### 6.2 Detection Output & Downstream Routing

The detected `ContentType` is threaded from `ContentTypeDetector.detect_content_type_full()` → `cli.py` → `AIPipeline.process_audio(detected_type=..., detection_evidence=...)` → `ClipRenderer.render_clips(content_type=...)`. When the detector returns a confident type (`is_confident=True`), evidence is not passed to the LLM. When uncertain (`None`), the evidence dict goes to the LLM prompt to prevent hallucination.

| Detected Type | Layout Mode | Face Switching | Donation Handling |
|---|---|---|---|
| `PODCAST` | Mode A — Single Vertical | Yes — active speaker when 2+ faces | **Excluded by default** (pre-recorded; no live donation widgets) |
| `JUST_CHAT` | Mode B — Stacked Split-Screen | No | **Disabled by default** (opt-in via `preserve_donation_overlays`) |
| `GAMING_SOLO` | Mode B — Stacked Split-Screen | No | **Disabled by default** (opt-in via `preserve_donation_overlays`) |
| `GAMING_COLLAB` | Mode C — Multi-Face Collab Stack | No | **Excluded by default** (popup must not displace a collab panel) |
| `DONATION_OVERLAY` | Mode B geometry — Facecam top + popup bottom | No | This **is** the donation layout |

**Per-clip donation promotion:** on top of the video-level base type, any clip whose window contains a transient mediashare/donation popup (`mediashare_present`) is promoted to `DONATION_OVERLAY` and routed to the facecam + popup 2-stack. **Disabled by default** (`preserve_donation_overlays: false`). When enabled, gated by `video_processing.preserve_donation_overlays` (off → donations are shown nowhere). An explicit per-clip `content_type` (review-gate edit) is never overridden. **Configurable exclusion:** `video_processing.donation_overlay_exclude_types` lists which base types are never promoted (default: `["PODCAST", "GAMING_COLLAB"]`). `PODCAST` is excluded because it is pre-recorded content with no live donation widgets; `GAMING_COLLAB` is excluded because the popup must not replace one of the three collab panels. Remove a type from this list to enable donation routing for it.

#### 6.3 Detection Confidence & Fallback

- When the video-level detector returns `None` (uncertain), the LLM decides per-clip content types in auto mode, and in manual mode too unless `--no-metadata` was passed (no LLM call) — those clips fall back to `PODCAST` (safest default) via the renderer's visual classification. `video_processing.detection_confidence_threshold` remains the configured floor.
- The detected content type MUST be displayed in the WebUI Review panel so the user can override it before rendering if the detection was wrong.

---

### 7. Smart Vertical Cropping, Face Tracking & Layout Modes

Use `opencv-python-headless` and `mediapipe` or `yolov8-face`. The headless OpenCV variant prevents display-server crashes in WSL environments. All layout logic is driven by the `ContentType` produced by the Detection Engine in Section 6.

---

#### Layout Mode A — Single Vertical (Podcast)

**Applies to:** `PODCAST`

- Render as a **single 9:16 vertical crop**. No stacking, no split panels.
- Face tracking applies: the crop window centers on the active speaker. It is **static while a speaker holds** and **glides gently** to the next speaker on a change (EMA pan, `PAN_SMOOTHING_FACTOR`=0.03, τ≈1.1s) — no drift within a held shot. The crop center is lifted by `HEADROOM_FACTOR × face_height` (rule-of-thirds: eyes in the upper third).
- If no face is detected (e.g., screen-share segment), default to a center-biased static crop with no movement.

**Multi-speaker mode (2+ faces detected):**
- The FaceLandmarker capacity (`num_faces`) is sized from the **YOLO person count** (`analysis["face_count"]`, computed during the visual analysis pass) plus `FACE_COUNT_MARGIN` (default `+2`) as a safety margin, clamped to `FACE_LANDMARKER_MAX_FACES` (default 8). `face_count` is the **maximum number of high-confidence person boxes visible in any single sampled frame** (not a cross-frame cluster count) — so 4 people appearing simultaneously register as 4, not 2 even if they are seated close together. The margin provides headroom for partially-visible or angled faces.
- **Detection sampling:** PODCAST samples faces/lips at `PODCAST_DETECTION_FPS` (default 10 fps, vs 5 fps for other types) so speaker detection resolves syllable-rate (~3–5 Hz) mouth movement (Nyquist).
- **Two-shot grouping (primary path), decided ONCE per clip:** the generator measures, across all multi-face steps, the median total face span **and** the median largest gap between adjacent faces. Both faces are framed together (crop centered on the cluster midpoint, pseudo-subject `GROUP_SPEAKER_ID = -2`, zero cuts) only when **both** `span ≤ GROUP_FRAMING_FIT_FACTOR × crop_w` (default 0.9×) **and** `gap ≤ GROUP_MAX_GAP_FACTOR × crop_w` (default 0.25×). The gap test prevents centering on the empty table between two far-apart people; the clip-level decision (not per-step) removes group↔single flicker.
- **Speaking signal = Mouth-Aspect-Ratio (MAR):** `mean vertical inner-lip opening ÷ mouth width` (FaceMesh verticals 13/14, 81/178, 311/402; width 78/308). A wide smile barely changes MAR; speech drives it up and down. Per-step activity = std-dev of MAR over `LIP_ACTIVITY_WINDOW_SECONDS` (default 0.8 s). MAR is intrinsically face-size-normalised (no separate box-height scaling).
- **Audio-visual sync (the accuracy core):** a per-clip RMS loudness envelope (`AudioEnergyAnalyzer.rms_envelope`, aligned 1:1 to detection steps) gates and weights speaker selection. (a) **Voice gate** — switching is only considered on *voiced* steps (RMS ≥ `VOICE_ACTIVITY_FLOOR_FACTOR × median`, default 0.5×); during silence/pauses the current speaker is held. (b) **Local coherence (moment-to-moment)** — at each voiced step, each face's recent mouth-motion window is Pearson-correlated with the recent audio window (`AV_SYNC_WINDOW_SECONDS`, default 1.0 s). A face is an **eligible speaker** only when its local coherence ≥ `COHERENCE_MIN` (default 0.25) **and** its activity ≥ `LIP_ACTIVITY_MIN`. Eligible faces are ranked by `activity × local_coherence × clip-level prior`. (c) **Occlusion-aware hold** — if **no** visible face is eligible (e.g. the talker's lips are behind a mic, or only a non-speaker is smiling), the crop **holds** the current speaker rather than jumping to the visible non-speaker. The clip-level global correlation (`_audio_correlation_weight`) is now only a mild tiebreak prior (penalizes chronic non-speakers), not a floor. If audio decode fails the system falls back to visual-only MAR activity (never crashes a clip).
- **Hysteresis:** a challenger must exceed the current speaker's score × `SPEAKER_SWITCH_MARGIN` (default 1.5×) before the per-step pick switches.
- **Minimum shot length (`MIN_SHOT_SECONDS`, default 2.0 s):** once committed to a subject (speaker or group), no switch is allowed until this many seconds have elapsed (the primary anti-dizziness lever); `SPEAKER_HOLD_SECONDS` (default 2.0 s) is the confirm window on top. Step→seconds conversion uses the actual detection interval.
- When the active speaker commits, the crop holds a fixed center on their face box (median of detected face centers in that segment); on a speaker change the EMA pan glides to the new center. No-face steps carry forward the last known target (never snap to frame center / empty table).
- **Face identity** is tracked across detection steps by IoU box-overlap (`IOU_MATCH_MIN`), falling back to center-distance when no box overlaps.
- Single-speaker clips use the lighter FaceDetector (Mode B/C) and track that face only.

---

#### Layout Mode B — Stacked Split-Screen (Gaming Solo / Just Chat)

**Applies to:** `GAMING_SOLO`, `JUST_CHAT`

- Construct a **two-region vertical stack** (each panel 1080×960, total 1080×1920):
  - **Top panel = Facecam (always fits).** A **single stable cam box** is detected once for the whole video (`detect_stable_facecam`, sampled across its length) and reused for every clip, so the framing does not drift with the streamer's pose. The cam box is expanded by `FACECAM_FIT_FACTOR` (≈1.45× — comfortable context margin, cam ≈69% of panel height) and shaped to the panel aspect (1.125), then **crop-filled into the 1080×960 panel — sharp, prominent, no blur and no left/right bars** (mild upscale when the cam region is smaller than the panel).
  - **Bottom panel = Gameplay.** A **centered, panel-aspect gameplay crop** zoomed by `region_detection.gameplay_zoom` (default 1.25× — crops the bottom ticker from the panel). The crop is **fully static** (motion-centroid–centered at analysis time by `_motion_region`; no panning at render). This keeps the viewer's eye steady on the gameplay. The legacy motion-following pan (`gameplay_follow_motion: true` in config) is available but disabled by default. (Donation/MediaShare popups are **not** shown here — a clip with a popup is promoted to `DONATION_OVERLAY`; see below.)
  - **Fullscreen-cam override:** when the facecam covers >55% of the frame (a just-chatting / reaction moment with no real gameplay), the clip renders as a **single full-face 9:16 vertical** (Mode A) instead of a 2-stack — avoids face-over-face. (Skipped for `DONATION_OVERLAY`.)
- **CRITICAL LAYOUT ORDER:** Facecam is always the top panel, gameplay the bottom. No third (black/subtitle-pad) zone — the stack is exactly two equal panels.
- Detect static streaming HUD elements (corner overlays, alert boxes) and exclude them from the gameplay motion crop so the crop locks onto gameplay, not the HUD.

---

#### Layout Mode — Donation Overlay (per-clip)

**Applies to:** `DONATION_OVERLAY` (any clip whose window contains a transient mediashare/donation popup, promoted from its base type)

- Reuses the Mode B 2-stack geometry (two 1080×960 panels):
  - **Top panel = Facecam** — same always-fits rule as Mode B (the streamer's reaction).
  - **Bottom panel = the donation/MediaShare popup**, forced. The popup box (from the appearance/disappearance detector, with the colour-overlay detector as fallback) is expanded to the panel aspect and blurred-background filled. The webcam is a moving bordered inset and is excluded from popup detection so the face is never duplicated top+bottom.
  - If a clip is forced to this type but no usable popup box is found (e.g. `content_type_override` set to `DONATION_OVERLAY`), the bottom degrades gracefully to the gameplay crop rather than a black panel.
- **Donation Preservation (CRITICAL):** This dedicated layout is the **only** place donations are composited — never a small PiP, and never embedded in Mode A/B/C. Gated by `preserve_donation_overlays`.

---

#### Layout Mode C — Multi-Face Collab Stack (Gaming Collab)

**Applies to:** `GAMING_COLLAB`

- Construct a **three-region vertical stack**: Primary Facecam at top, Gameplay in the absolute center, Collaborator face(s) at the bottom.
- **Both cams come from the reliable `VisualAnalyzer.detect_facecams` PAIR**, locked once video-wide (the same detector that drives the SOLO-vs-COLLAB split), not from the per-clip `persons` list. `analyze_window(facecam_boxes=[primary, collaborator])` sets `facecam_box` (top panel) and `collab_box` (bottom panel); the layout uses `collab_box` directly and only falls back to the per-clip edge-cam `persons` heuristic when the pair is absent. The descriptor names **both** cams ("facecam at bottom-right; second facecam at bottom-left").
- **CRITICAL LAYOUT ORDER:** Gameplay canvas is always center. Both face regions are secondary. Each panel is a panel-aspect (1080×640) crop-fill (no distortion). The centre gameplay panel uses the **same static centred crop as Mode B** (`gameplay_zoom` + `_motion_region`, shaped to the 3-stack aspect 1.6875) — no panning. **Both cam boxes are excluded from the gameplay crop** — masked out of the motion search and, when they sit in the bottom third, the crop is shrunk + biased upward so its bottom edge sits at/above the highest bottom cam — so neither corner cam ever bleeds into the centre (no "double facecam"). A `GAMING_COLLAB` clip stays a 3-stack even when a donation is present (donation promotion is skipped for collab).
- **Donation Handling:** Mode C does not composite donations. A collab clip whose window contains a donation popup is promoted to `DONATION_OVERLAY` (facecam + popup 2-stack) for that clip, trading the collab face for proper popup display.

---

### 8. Subtitle Engine & Language Handling

Subtitles are a first-class feature of the application, not an afterthought. The subtitle pipeline has three distinct behaviors controlled by user configuration and UI input.

#### 8.1 Subtitle Modes

| Mode | Config / UI | Behaviour |
|---|---|---|
| **Auto** | `subtitles.enabled: true`, `language: "auto"` | Detect spoken language from audio, transcribe with the detected language, render captions |
| **Manual Language** | `subtitles.enabled: true`, `language: "id"` (or any ISO 639-1 code) | Skip language detection, transcribe directly in the specified language |
| **Disabled** | `subtitles.enabled: false` | Skip all STT and caption steps; render clips with no subtitles |

The WebUI MUST expose a language dropdown on the Clipper tab with common presets (Auto-Detect, Indonesian, English, Japanese, Korean, etc.) and a free-text input for any ISO 639-1 code. This selection overrides `config.yaml` for that session.

#### 8.2 Transcription & Caption Rendering

- Word-level timestamps from STT output are processed into an `.ass` subtitle schema. The effect is **word-by-word focus**: the whole phrase is shown with the currently-spoken word **bold + highlight-coloured**, the rest normal (one Dialogue event per word). Text is upper-cased when `subtitles.uppercase` is set.
- Captions are burned permanently into video frames using FFmpeg's `vf="subtitles=..."` filter, referencing the `.ttf` font from `./workspace/fonts/`.
- Place captions in the **lower-center** margin with thick contrasting outlines for legibility across all layouts and backgrounds.
- Caption style (font size, color, outline, bold) is fully configurable under `video_processing.subtitles` in `config.yaml`. Colours are written as human **`#RRGGBB`** hex (auto-converted to ASS `&HAABBGGRR`), `bold`/`shadow` as `true`/`false`, and `alignment` as a name (`bottom-center`…) — the legacy ASS/int forms still load. An optional `stt_context` string (names/game/terms) is fed to STT as the `initial_prompt` to sharpen word accuracy.
- For cloud STT (Google Gemini): language is passed as a parameter in the API call.
- For local STT (faster-whisper): language is passed to `model.transcribe(language=...)`. When `"auto"`, pass `language=None` to let faster-whisper detect it automatically.
- **Language-locking primer:** when an explicit language is set, a native-language "accurate transcription" sentence (`prompts.LANGUAGE_PROMPTS`, ~34 ISO 639-1 languages, name- or code-keyed) is passed as the `initial_prompt` (local/OpenAI) or appended to the Gemini instruction — all three providers share the same hint.
- **Hallucination filter (CRITICAL):** whisper transcribes laughter / music / reaction noise as repeated filler tokens (e.g. "hehehe" ×128). STT segments are confidence-filtered (`stt_local._is_hallucinated_segment`) before use — a segment is dropped when its `compression_ratio`, `no_speech_prob`+`avg_logprob`, or single-token repetition exceeds the thresholds in `constants.py`. This runs on **all** STT outputs (LLM selection transcript, the word cache, and the render subtitle pass), so a laughter clip renders captionless instead of garbage. Complements the word-level de-dup in §8.2.

#### 8.3 Language Detection (Auto Mode)

- Cloud path: Gemini Flash handles language detection natively.
- Local path: faster-whisper detects language automatically on the first segment when `language=None` is passed. The detected language MUST be logged and displayed in the WebUI so the user is aware of what was detected.

---

### 9. Unified YAML Configuration

All application configuration is parsed from a single `config.yaml` in the project root. A hidden override key `dk_clipper_sys_prompt` can replace the default system prompt but MUST NOT appear in the base config file. All values are validated against Pydantic models at startup.

```yaml
logging:
  level: "INFO"                        # DEBUG | INFO | WARNING | ERROR
  file_path: "./workspace/logs/app.log"
  rotation: "50 MB"
  retention: "7 days"

web_server:
  host: "127.0.0.1"
  port: 7860
  share: false

ai_pipeline:
  stt:
    provider: "local"                  # cloud | local | auto (auto = try cloud, fall back to local)
    cloud:
      provider: "google"               # google | openai
      api_key: "your-api-key-here"
      model: "gemini-2.5-flash"        # Used for google; openai always uses whisper-1
      timeout: 300                     # Request timeout in seconds for cloud STT calls (30-600)
    local:
      device: "auto"                   # auto | cpu | cuda
      model_size: "large-v3"           # tiny | base | small | medium | large-v3

  llm:
    provider: "cloud"                  # cloud | local | auto (auto = try cloud, fall back to local)
    cloud:
      provider: "google"               # google | openai (openai covers OpenRouter and compatible APIs)
      base_url: null                   # null = provider default; set for OpenRouter or self-hosted endpoints
      api_key: "your-api-key-here"
      model: "gemini-2.5-flash"
      timeout: 300                     # Request timeout in seconds for cloud LLM calls (30-600)
    local:
      device: "auto"                   # auto | cpu | cuda
      n_gpu_layers: 0                  # 0 = CPU only; -1 = all layers to GPU
      model_name: "microsoft/Phi-3-mini-4k-instruct-gguf:q4"

downloader:
  browser_cookies: "edge"              # Browser for yt-dlp cookie extraction: edge | chrome | firefox
  target_resolution: "1080p"
  video_format: "mp4"
  audio_format: "aac"
  audio_quality: "192k"

clip_selection:
  mode: "auto"                         # auto = AI detects clips | manual = user provides timestamps
  auto_strategy: "hybrid"              # For auto mode only: ai | heatmap | hybrid
  candidate_margin: 2                  # RMS/heatmap produces N spikes; LLM receives top (target_clips + margin)
                                       # e.g. 5 clips + 2 = top 7 candidates sent to LLM, rest discarded
                                       # Additive (not ×): 15 clips + 2 = 17, not 30 — keeps STT cost bounded
                                       # Min: 0 (exact count, no selection) | Recommended: 2 | Max: 15
  require_review_before_render: true   # Show review panel before rendering starts
  heatmap_threshold_percentile: 85     # Minimum heatmap spike percentile to qualify as a clip candidate
  min_clips: 1
  max_clips: 25
  default_clips: 5
  # Clip length = [default, default + margin] (e.g. 60-75s). min/max are NOT per-clip bounds.
  default_clip_duration_seconds: 60    # target clip length (and floor every clip is extended up to)
  clip_length_margin_seconds: 15       # a clip may run up to default+margin (caps overly-long picks)
  min_clip_duration_seconds: 30        # WebUI duration-slider lower bound (not a per-clip floor)
  max_clip_duration_seconds: 180       # WebUI duration-slider upper bound (not a per-clip cap)

video_processing:
  output_dir: "./workspace/clips"
  video_encoder: "cpu"                 # cpu | auto | nvenc | qsv | videotoolbox (FFmpeg H.264 encoder)
  fast_mode: false                     # true = low-spec Haar largest-face PODCAST tracking (no MediaPipe/audio)
  content_type_override: "auto"        # auto | PODCAST | JUST_CHAT | GAMING_SOLO | GAMING_COLLAB | DONATION_OVERLAY
  detection_confidence_threshold: 0.6  # Below this, detection falls back to PODCAST and logs a warning
  auto_face_tracking: true             # Smooth face-centered crop for Mode A (talking heads)
  preserve_donation_overlays: false    # Detect and composite Trakteer/MediaShare alerts (default: disabled)
  donation_overlay_exclude_types:      # Types excluded from per-clip donation routing (configurable)
    - "PODCAST"                        #   pre-recorded content — no live donation widgets
    - "GAMING_COLLAB"                  #   popup must not displace a collab panel
  default_resolution: "1080p"
  region_detection:                    # YOLOv8n visual analysis: facecam / gameplay / mediashare regions
    enabled: true                      # false = fall back to motion/face heuristics only
    model_name: "yolov8n.pt"           # Ultralytics COCO nano model (auto-downloaded to workspace/models/)
    sample_frames: 4                   # Frames sampled per candidate/clip window (sparse = fast)
    device: "auto"                     # auto | cpu | cuda
    gameplay_follow_motion: false      # false = static centred crop (default, no pan) | true = legacy gentle pan
    gameplay_zoom: 1.25                 # >1 zooms the gameplay panel tighter (1.25 = 25% closer, crops the ticker out)
  subtitles:
    enabled: true
    collab_enabled: false              # subtitles on the cramped 3-stack GAMING_COLLAB layout
    uppercase: true                    # render caption text in CAPITALS for readability
    language: "auto"                   # auto = detect from audio | ISO 639-1 code or name (34 languages)
    stt_context: ""                    # optional STT vocab hint: names/game/terms (improves word accuracy)
    font_file: "Anton.ttf"
    font_size: 80
    primary_color: "#FFFFFF"           # human #RRGGBB (legacy ASS &HAABBGGRR still accepted)
    highlight_color: "#96C8FF"         # active-word colour — soft blue, eye-friendly
    outline_color: "#000000"
    outline_thickness: 6
    bold: true                         # true/false (legacy 0/1 still accepted)
    shadow: true
    alignment: bottom-center           # name (bottom-center/center/top-center/...) or ASS int
    margin_v: 760                      # bottom margin in px (bottom-center alignment) — 760px up from canvas bottom = ~60% down from top

workspace_cleanup:
  enabled: true
  run_on_startup: true
  dry_run: false                       # true = log candidates without deleting; use to validate retention settings
  retention_days:
    videos: 3                          # Raw yt-dlp downloads — set to -1 to disable purge for this dir
    audios: 3                          # Extracted audio tracks
    subtitles: 3                       # STT transcriptions and AI JSON results — set to -1 to disable
    tmp: 1                             # FFmpeg scratch files — purge aggressively; 0 = purge every run
  protected_dirs:                      # These directories are NEVER touched by auto-purge
    - "bin"
    - "fonts"
    - "models"
```

---

### 10. System Prompt Engineering

The application MUST use a hardcoded system prompt injected into every LLM analysis call. The prompt is dynamically enriched with the detected `ContentType` so the LLM can apply content-aware reasoning. Override this prompt at runtime only if `dk_clipper_sys_prompt` is present in `config.yaml`.

**Base Prompt Template (populated at runtime with `{content_type}` and `{target_duration}`):**

> "You are an expert social media content curator specializing in {content_type} videos. Analyze the provided transcript and timestamps. Identify the most engaging, viral, or emotionally resonant segments that would perform exceptionally well as {target_duration}-second YouTube Shorts, Instagram Reels, or TikTok clips.
>
> For PODCAST content: prioritize complete, punchy thoughts, strong opinions, surprising facts, or emotional peaks. Avoid segments that feel incomplete without visual context.
> For JUST_CHAT content: prioritize high-energy reactions, funny moments, and segments where donation interactions produce strong streamer responses.
> For GAMING_SOLO and GAMING_COLLAB content: prioritize intense gameplay moments, clutch plays, funny failures, and strong streamer reactions. Donation-triggered reactions are high-value clip targets.
>
> Return ONLY a valid JSON array of objects. Each object must contain:
> `candidate_index` (int, 1-based), `start_time` (float, seconds), `end_time` (float, seconds),
> `title` (<=50 chars, hook/bait style), `caption` (<=150 chars, short hook caption),
> `description` (<=300 chars, hook + context + CTA), `hashtags` (string, 5-8 space-separated hashtags like `#gaming #mlbb #shorts`),
> `content_type` (PODCAST|JUST_CHAT|GAMING_SOLO|GAMING_COLLAB), and `reasoning` (one sentence).
> Never cut a clip mid-sentence. Ensure each clip is a complete, standalone moment."

**Language & tone instruction:** The prompt is enriched with a `{language_instruction}` block that tells the LLM the video's spoken language (or to detect it from the transcript), and instructs it to write all titles, captions, and descriptions in a relaxed, informal, hook/bait human tone — avoiding robotic or corporate AI wording. The LLM is also instructed to base all generated text on the actual transcript content per candidate and to avoid hashtags inside the description text.

---

### 11. Python 3 Best Practices & Domain-Driven Project Layout

The application MUST avoid monolithic architecture. All sub-rules below are CRITICAL and non-negotiable.

#### 11.1 Strict Type Hinting

- Every function signature MUST be fully annotated — no bare unannotated `def` anywhere.
- Use `TypedDict` for structured dict shapes, `dataclasses` or `pydantic.BaseModel` for inter-module data, and `Protocol` for shared interfaces (e.g., cloud vs. local AI provider).
- Prefer `X | None` over `Optional[X]` (Python 3.10+). Add `from __future__ import annotations` at the top of every file.
- Never use `Any` unless at an unavoidable untyped third-party boundary — annotate the reason with `# type: ignore`.

#### 11.2 Data Models & Config Validation (Pydantic)

- All config sections, clip results, pipeline state, and API responses MUST be Pydantic `BaseModel` instances. Use `Field()` for value constraints.
- `config.yaml` is loaded and validated in `src/core/config.py` at startup. Type mismatches and missing required fields raise a clear `ValidationError` — never fail silently.
- Raw dict key access to config values (`config["key"]`) is forbidden outside `src/core/config.py`.

#### 11.3 Import Management & Lazy Loading (Performance Critical)

- Standard library → lightweight third-party → internal modules at the top of each file, per PEP 8.
- Heavy ML/CV modules (`faster-whisper`, `llama-cpp-python`, `torch`, `mediapipe`, `cv2`) MUST use lazy imports inside their execution functions only — not at file top-level.

```python
# ✅ DO — lazy import inside the function that needs it
def transcribe_audio(audio_path: Path) -> list[WordTimestamp]:
    from faster_whisper import WhisperModel
    ...

# ❌ DON'T — top-level import of a heavy ML module
from faster_whisper import WhisperModel
```

#### 11.4 Error Handling & Custom Exceptions

- Define a custom exception hierarchy in `src/core/exceptions.py`. All errors inherit from `YaClipError`, with subclasses: `DownloadError`, `DetectionError`, `RenderError`, `AIProviderError`, `CacheInitError`.
- Never use bare `except:` or `except Exception: pass`. Every caught exception must be re-raised, logged, or wrapped with context using `raise X from Y`.

```python
# ✅ DO — specific exception, preserved chain
raise RenderError(f"FFmpeg failed: {e.stderr.decode()}") from e

# ❌ DON'T — silent swallow
except: pass
```

#### 11.5 Constants & Enums (No Magic Values)

- Zero magic strings or numbers in the codebase. All fixed values in `src/core/constants.py` as `Enum` classes or typed constants.
- Key enums: `ContentType` (`PODCAST`, `JUST_CHAT`, `GAMING_SOLO`, `GAMING_COLLAB`, `DONATION_OVERLAY`), `LayoutMode` (`SINGLE_VERTICAL`, `STACKED_SPLIT`, `MULTI_COLLAB`), `AIProvider` (`GOOGLE`, `OPENAI`, `LOCAL`), `ClipMode` (`AUTO`, `MANUAL`).
- Key numeric constants (selection): `GAMEPLAY_MIN_NONPERSON_MOTION`, `GAMEPLAY_MIN_OPEN_AREA_FRAC` — gameplay gate thresholds; `FACE_LANDMARKER_MAX_FACES` — ceiling for FaceLandmarker capacity (default 8); `FACE_COUNT_MARGIN` — safety headroom added to the YOLO person count before sizing `num_faces` (default 2, so 4 YOLO persons → capacity 6); `PERSON_COUNT_CONF_MIN` — minimum YOLO box confidence for counting simultaneous persons per frame (default 0.5); `SPEAKER_HOLD_SECONDS` — debounce confirm window before committing a speaker cut (default 2.0 s); `PODCAST_DETECTION_FPS` — face/lip sampling rate for PODCAST (default 10, vs 5 for other types); `GROUP_FRAMING_FIT_FACTOR` — two-shot allowed when all-faces span ≤ this × crop_w (default 0.9); `GROUP_MAX_GAP_FACTOR` — two-shot also requires the largest inter-face gap ≤ this × crop_w (default 0.25, prevents empty-table centering); `GROUP_SPEAKER_ID` — pseudo-id for a group/two-shot segment (−2); `LIP_ACTIVITY_WINDOW_SECONDS` — rolling window for MAR-movement std-dev (default 0.8 s); `LIP_ACTIVITY_MIN` — floor below which a face is treated as silent (default 0.003); `SPEAKER_SWITCH_MARGIN` — challenger score ratio to trigger hysteresis switch (default 1.5); `MIN_SHOT_SECONDS` — minimum hold time before any subject switch is allowed (default 2.0 s); `VOICE_ACTIVITY_FLOOR_FACTOR` — a step is "voiced" when its audio RMS ≥ this × the clip median (default 0.5; gates speaker switching to voiced moments); `AV_SYNC_WINDOW_SECONDS` — window for moment-to-moment audio-visual coherence (default 1.0 s); `COHERENCE_MIN` — minimum local mouth↔audio correlation for a face to be an eligible speaker (default 0.25; below → occlusion-aware hold); `HEADROOM_FACTOR` — rule-of-thirds vertical lift as a fraction of face height (default 0.18); `IOU_MATCH_MIN` — minimum box IoU to match a face to an existing track (default 0.3); `PAN_SMOOTHING_FACTOR` — gentle EMA pan rate for PODCAST crop centers (default 0.03; τ ≈ 1.1 s, 95% settle ~3 s — cinematic glide); `HAAR_DOWNSCALE` / `HAAR_SCALE_FACTOR` / `HAAR_MIN_NEIGHBORS` — OpenCV Haar params for the opt-in `fast_mode` PODCAST tracker (0.5 / 1.1 / 5).

**Render encoding & GPU fallback:** the FFmpeg video encoder is selected by `video_processing.video_encoder` (`auto` default = nvenc when CUDA present else libx264; `cpu`/`nvenc`/`qsv`/`videotoolbox` force a specific encoder). `FFmpegCommandBuilder._video_encoder_args` maps each to its `-c:v` + rate-control flags (shared flags stay common). If a GPU encoder fails at runtime, `ClipRenderer._run_render_with_fallback` detects the hardware-failure signature in FFmpeg stderr, rebuilds the command with `video_encoder="cpu"`, and retries once — a genuine (non-hardware) error is re-raised. **`fast_mode`** (opt-in, default off) swaps the MediaPipe+audio PODCAST tracker for an OpenCV Haar largest-face crop (`FaceTracker._track_haar_fast` → single-face path); content-type detection and Mode B/C are untouched.

#### 11.6 DRY Principle & Code Reuse

- Centralize: FFmpeg command construction → `src/media/ffmpeg_builder.py`; API request/retry → `src/ai/api_client.py`; Gradio component factories → `src/interfaces/components.py`; temp file lifecycle → `src/core/workspace.py`.
- Any logic copy-pasted with minor variations is a refactoring target. Parameterize the variation.

#### 11.7 Function Design & Guard Clauses

- One function = one thing. Name every function with a verb-noun phrase (`download_video()`, `detect_content_type()`, `build_ffmpeg_filter()`).
- Maximum 40 lines per function. Extract sub-steps into named helpers beyond that.
- Guard clauses and early returns reduce nesting. Never nest more than 3 levels deep.
- No boolean flag parameters that change core behavior — split into two named functions instead.

```python
# ✅ DO — guard early, flat structure
def render_clip(clip: ClipMetadata, layout: LayoutMode) -> Path:
    if not clip["start_time"] < clip["end_time"]:
        raise RenderError("Invalid clip boundaries.")
    ...

# ❌ DON'T — boolean flag, deep nesting
def render_clip(clip, stacked=False):
    if clip:
        if clip["start"] < clip["end"]:
            if stacked: ...
```

#### 11.8 Context Managers & Resource Safety

- All resources with an open/close lifecycle (file handles, OpenCV captures, model instances) MUST use `with` blocks or `contextlib.contextmanager` generators.
- Local AI model loading MUST be wrapped in a context manager guaranteeing `del model` and `gc.collect()` on exit, even on exceptions.

#### 11.9 Async & Concurrency Model

- Use `asyncio` for I/O-bound operations (API calls, downloads).
- Use `concurrent.futures.ThreadPoolExecutor` for CPU-bound subprocess calls (FFmpeg, yt-dlp).
- Never call blocking functions directly inside Gradio event handlers — always offload with `asyncio.to_thread()` or an executor.

#### 11.10 Subprocess Safety

- Always use `subprocess` in **list form**. Never `shell=True` with dynamic input.
- Always capture `stderr`, set an explicit `timeout`, and pass `pathlib.Path` objects as `str(path)`.

```python
# ✅ DO
subprocess.run(["ffmpeg", "-i", str(input_path), str(output_path)], capture_output=True, timeout=300, check=True)

# ❌ DON'T
os.system(f"ffmpeg -i {input_path} {output_path}")
```

#### 11.11 Cross-Platform Path Handling (CRITICAL)

- All paths via `pathlib.Path`. Never string concatenation or `os.path.join`.
- Use `path.resolve()` for absolute paths, `path.relative_to(base)` for display paths, `str(path)` when passing to subprocess.

#### 11.12 Toolchain: Linting, Formatting & Type Checking

Configured in `pyproject.toml`. Not optional.

- **`ruff`** — replaces `flake8`, `isort`, `black`. Configure: `line-length=100`, `target-version="py310"`, rule sets `E`, `F`, `I`, `UP`, `B`, `SIM`.
- **`mypy`** — `--strict` mode. Zero errors permitted in production code. `ignore_missing_imports=true` for untyped third-party libs.
- **Pre-commit** (recommended) — wire both tools as hooks.

#### 11.13 Testing Standards

- `pytest` with a `tests/` directory mirroring `src/`.
- Unit test all pure functions: timestamp parsing, config validation, clip boundary logic, FFmpeg command construction, content type detection heuristics.
- Mock all external boundaries (yt-dlp, subprocess, API calls) with `pytest-mock`.
- Integration tests (full download → detect → clip → render) gated behind `@pytest.mark.integration`.
- Minimum 80% coverage on `src/core/` and `src/media/`.

#### 11.14 Docstrings & Inline Comments

- All public functions, classes, and modules: Google-style docstring with `Args:`, `Returns:`, and `Raises:` sections.
- Inline comments explain **why**, not **what**.
- Every non-trivial FFmpeg `filter_complex` string MUST have a plain-English comment block above it explaining each step.

#### 11.15 Naming Conventions

| Construct | Convention | Example |
|---|---|---|
| Module / package | `snake_case` | `ffmpeg_builder.py` |
| Class | `PascalCase` | `ClipMetadata`, `ContentTypeDetector` |
| Function / method | `snake_case` verb-noun | `classify_from_analysis()`, `build_filter()` |
| Variable | `snake_case` descriptive | `output_clip_path`, `face_bbox` |
| Constant | `SCREAMING_SNAKE_CASE` | `MAX_CLIP_DURATION`, `CACHE_DIR` |
| Private helper | `_snake_case` prefix | `_resolve_wsl_path()` |
| Enum member | `SCREAMING_SNAKE_CASE` | `ContentType.GAMING_SOLO` |

#### 11.16 Modular Directory Tree

```
yaclip/
├── app.py                       # Entry point: routes to Typer CLI or Gradio WebUI
├── config.yaml                  # User configuration (gitignored — copy from config.yaml.example)
├── config.yaml.example          # Distributable config template
├── TASKS.md                     # Task tracking backlog
├── requirements.txt             # pip3 compatibility
├── requirements-dev.txt         # Dev/test dependencies
├── pyproject.toml               # uv/pip deps + ruff/mypy config
├── README.md                    # Setup and usage guide
├── Dockerfile                   # Container build
├── docs/
│   ├── ARCHITECTURE.md          # System design and component overview
│   └── WORKFLOWS.md             # Pipeline diagrams and operational flows
├── tests/                       # Mirrors src/ structure
│   ├── conftest.py              # Shared pytest fixtures
│   ├── core/
│   ├── media/
│   ├── ai/
│   ├── vision/
│   └── interfaces/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py            # Pydantic config loading and validation
│   │   ├── constants.py         # ContentType, LayoutMode, AIProvider, ClipMode enums; typed constants
│   │   ├── environment.py       # setup_environment(), guard_triton_segfault(), ensure_vision_runtime()
│   │   ├── exceptions.py        # YaClipError hierarchy
│   │   ├── logger.py            # Loguru setup; InterceptHandler for stdlib logging
│   │   ├── utils.py             # SystemUtils, AIUtils, cross-platform helpers
│   │   └── workspace.py         # Workspace init, sequential purge engine
│   ├── media/
│   │   ├── audio.py             # FFmpeg audio extraction
│   │   ├── downloader.py        # yt-dlp download logic, WSL cookie resolution
│   │   ├── energy.py            # FFmpeg RMS pseudo-heatmap generator
│   │   ├── ffmpeg_builder.py    # FFmpeg filter graph builder (Mode A/B/C/Donation)
│   │   ├── renderer.py          # Clip export orchestrator (3-pass: regions → subs → encode)
│   │   ├── slicer.py            # FFmpeg -c copy audio slicer
│   │   └── subtitles.py         # .ass subtitle generator with word-by-word focus effect
│   ├── ai/
│   │   ├── api_client.py        # Retry decorator for cloud API calls
│   │   ├── heatmap.py           # YouTube heatmap JSON parser + spike ranker
│   │   ├── llm_cloud.py         # Cloud LLM (Google Gemini / OpenAI-compatible) highlight extraction
│   │   ├── llm_local.py         # llama-cpp-python local highlight detection
│   │   ├── pipeline.py          # AI orchestrator: hybrid/manual strategy, pre-ranking, dedup
│   │   ├── prompts.py           # Content-aware system prompts, LANGUAGE_PROMPTS, strip_json_markdown
│   │   ├── stt_cloud.py         # Cloud STT (Google Gemini / OpenAI Whisper API)
│   │   └── stt_local.py         # faster-whisper local STT with VAD, word timestamps, hallucination filter
│   ├── vision/
│   │   ├── visual_analyzer.py        # YOLOv8n region engine: facecam/gameplay/mediashare + LLM text descriptor
│   │   ├── content_type_detector.py  # detect_content_type: whole-video detection → ContentType | None; classify_from_analysis for manual fallback
│   │   ├── face_tracker.py           # OpenCV/MediaPipe face detection, crop box, speaker-switch logic (Mode A)
│   │   ├── layout_builder.py         # ContentType + VisualAnalyzer regions → FFmpeg layout spec
│   │   └── overlay_detector.py       # Appearance/disappearance novelty detection (median baseline diff; cam-exclusion guards)
│   └── interfaces/
│       ├── cli.py               # Typer CLI commands: clip, config, cache status/purge, serve, clean-workspace (alias)
│       ├── webui.py             # Gradio layout: Clipper, Review & Render, Settings, Maintenance tabs
│       └── components.py        # Reusable Gradio component factories
└── workspace/
    ├── bin/                     # FFmpeg, Bun JS runtime (auto-downloaded on first boot)
    ├── fonts/                   # .ttf subtitle fonts (Anton.ttf auto-downloaded)
    ├── logs/                    # Loguru rotating log files (app.log)
    ├── models/                  # Local GGUF LLM files; models/hf/ = HuggingFace Hub cache
    ├── clips/                   # Final rendered vertical clips (never auto-purged)
    ├── videos/                  # Raw yt-dlp downloads (3-day retention)
    ├── audios/                  # Extracted audio tracks (3-day retention)
    ├── subtitles/               # STT transcripts, AI JSON, word cache, heatmap data (3-day retention)
    └── tmp/                     # FFmpeg scratch, audio slices, ASS files (1-day retention)
```

---

### 12. Logging, Observability & Strict Memory Optimization

- **Loguru (CRITICAL):** Completely abandon `print()`. All output goes through `loguru`. Configuration (level, rotation, retention, file path) loaded from `config.yaml` at init. Rotating `.log` files written to `./workspace/logs/` capture silent FFmpeg failures.
- **Log level standard:** `INFO` messages must be **short, single-line, plain-language sentences readable by a standard user** — no internal variable dumps, enum reprs (`ContentType.PODCAST`), raw crop coordinates, or boolean flags. Technical detail (crop boxes, per-frame data, debug booleans) belongs in `DEBUG`. Example: `"Content type detected: PODCAST."` not `"Video type detected: ContentType.PODCAST"`. A `_build_log_summary` helper in `VisualAnalyzer` produces the user-facing scene summary; the separate `_build_descriptor` remains unchanged as the LLM input.
- **Sequential Local AI Execution:** Never load `faster-whisper` and `llama-cpp` simultaneously. Strict order: `Load STT → Transcribe → del model → gc.collect() → Load LLM → Analyze → del model → gc.collect()`.
- **CV Resource Releasing:** OpenCV capture handles closed in `finally` blocks. Frame arrays (`del frame`) deleted immediately after use.
- **Gradio State Hygiene:** Never pass multi-GB structures into persistent `gr.State()`. Reset, overwrite, or garbage collect UI state on every new submission.
- **Disk Cleanup:** Temp files in `./workspace/tmp/` MUST be scrubbed on pipeline completion or crash — never left behind.

---

### 13. Cache Lifecycle & Scheduled Purge System

The sequential workspace purge system prevents unbounded disk growth. It is not optional.

#### 13.1 Purgeable vs. Protected Directories

| Directory | Contents | Default Retention |
|---|---|---|
| `./workspace/videos/` | Raw downloads | 3 days |
| `./workspace/audios/` | Audio tracks | 3 days |
| `./workspace/subtitles/` | `.ass` subtitle files | 3 days |
| `./workspace/data/` | STT transcripts + AI/cache JSON (clip proposals, word timings, mediashare cache, heatmap, metadata) | 3 days |
| `./workspace/tmp/` | FFmpeg scratch files (fire-and-forget: audio slices, cookie copies) | 1 day |

Protected (NEVER purged): `./workspace/bin/`, `./workspace/fonts/`, `./workspace/models/`. Driven by `workspace_cleanup.protected_dirs` in config.

#### 13.2 Scheduler

- Sequential cleanup execution. Triggered once from `src/core/workspace.py` on every startup (`run_purge_cycle` called after workspace integrity check).
- If `workspace_cleanup.run_on_startup: false` → skip. If `workspace_cleanup.enabled: false` → log one INFO notice and skip entirely.
- No background daemon / APScheduler — cleanup is synchronous at boot, not periodic.

#### 13.3 Purge Logic

- `./workspace/tmp/` walked recursively. All other dirs walked non-recursively (top-level only).
- Compare `stat().st_mtime` against `now - retention_days`. If older → `unlink()` and log filename, size, age.
- Wrap each `unlink()` in try/except. Log `PermissionError`/`OSError` at WARNING; never crash the scheduler over a single file.
- Emit a summary log after each cycle: total files deleted, total MB freed.
- `retention_days = 0` → purge all files every run. `retention_days = -1` → skip that directory.
- `dry_run: true` → log `[DRY RUN] Would delete: <file> (<MB>, <age>d)` without deleting.

#### 13.4 Active Pipeline Guard

Since cleanup is sequential at startup, there are no race conditions with active jobs.
- Set on job start. Clear in `finally` block on completion or error.
- If flag is set when purge cycle fires → skip cycle, log at DEBUG, wait for next interval.

#### 13.5 CLI & WebUI Integration

**CLI commands:**

| Command | Behaviour |
|---|---|
| `app.py clip <URL> [--clips N] [--duration S] [--language L] [--output-dir D] [--force] [--debug]` | Full pipeline: download → AI selection → render, with config overrides |
| `app.py config` | Print the validated configuration (API keys masked) |
| `app.py cache status` | Per-directory workspace disk usage (size, count, oldest) |
| `app.py cache purge [target] [--dry-run]` | Immediate manual purge (all dirs or a specific target) |
| `app.py clean-workspace [target]` | Hidden back-compat alias of `cache purge` |

**WebUI — Maintenance tab:**
- "🗑️ Clear Cache Now" button → triggers `run_purge_cycle()`, shows summary in `gr.Textbox`.
- Read-only disk usage panel: size (MB) and oldest file per purgeable directory, refreshed on tab open.
- Dry-run toggle (`gr.Checkbox`) to preview without committing.

---

### 14. Complete Code Generation (No Stubbing)

NEVER use placeholders like `# ... rest of the code`, `pass`, or `# TODO` in delivered code. Every implementation MUST be complete, production-ready, and fully typed.

---

## 🛑 Interactive Clarification Protocol (CRITICAL)

**DO NOT GUESS** on: FFmpeg multi-stream filter graph combinations, speaker diarization implementation choices, content type detection model selection, or Gradio callback architecture for complex state flows. If any ambiguity exists in these areas — pause, describe the roadblock clearly with the specific options being considered, and await a decision from the human engineer before proceeding.

---

### 15. Pipeline Testing Protocol (CRITICAL — NEVER AUTOMATE)

End-to-end pipeline testing (download → AI → render) is **always user-run** and must **NEVER be executed automatically by an agent**.

**Reasons:**
- Pipeline requires network access to YouTube (rate-limited, bot-protection)
- AI steps require valid API keys (cost-incurring, per-request billed)
- Local AI steps require GPU or significant CPU time
- Full pipeline may take minutes to hours depending on video length and hardware

**Agent rule:** After implementing or fixing pipeline code, the agent MUST:
1. Provide the exact verification command(s) for the user to run manually.
2. Explain what a successful output looks like.
3. Wait for the user to report results before declaring the fix verified.

**Agent must NEVER:**
- Execute `python app.py` or any pipeline command autonomously.
- Claim the pipeline is "working" or "fixed" without user-reported confirmation.
- Use subprocess, browser tools, or any other mechanism to invoke the pipeline.

**Example verification handoff (correct):**
```
To verify fix, run manually in your .venv:
  python app.py clip <youtube-url>
Success: Should print clip timestamps with no ImportError or AttributeError.
```