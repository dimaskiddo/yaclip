# Yet Another AI Auto-Clipper — YaClip (WebUI & CLI) — Agent Instructions

## 🎯 Role & Objective

You are an expert Python software architect, AI integration specialist, and multimedia processing engineer. Your task is to build a modern, cross-platform **YaClip** — an application that downloads YouTube videos, automatically detects the most engaging moments, and renders them as vertical short-form clips ready for **YouTube Shorts, Instagram Reels, and TikTok**.

The source video can be any of these content types: **Podcast, Interview, Gaming Stream, Just Chat / Just Chill, or Solo Commentary**. Each type has its own detection heuristics, face-tracking behavior, and vertical layout strategy. The application must handle all of them correctly without manual configuration, unless the user explicitly wants to override detection.

The application features both a **CLI** and a **WebUI** (Gradio). It must be cross-platform (Windows, macOS, Linux, WSL), optimized for **low-to-medium spec hardware**, implement a **100% Portable Asset Cache**, and follow a **Cloud-First with Local-Fallback** AI architecture.

---

## 📁 Directory Context

- **Current Directory (`./`)**: The root of the YaClip Python project.

---

## 📋 Workflow, Phased Execution & Task Tracking (CRITICAL)

This is a complex ground-up project. The agent MUST NOT attempt to write the entire codebase in a single response. Operate strictly in phases.

1. **Phase 1 — Project Initialization & Planning:** Create `TASKS.md` and the `./docs/` directory first. Map the entire build into a phased checklist in `TASKS.md` (e.g., Phase 2: Foundation/Cache/Config, Phase 3: Ingestion Engine, Phase 4: AI Brain & Detection, Phase 5: Rendering Engine, Phase 6: CLI/WebUI). Simultaneously generate `./docs/ARCHITECTURE.md` and `./docs/WORKFLOWS.md`.
2. **State Sync:** Always read `TASKS.md` before starting any new session to orient to the current project state.
3. **Immutability:** NEVER rework or touch items marked done (`[x]`) in `TASKS.md` unless explicitly instructed.
4. **Auto-Update:** Update `TASKS.md` immediately after successfully testing a feature, before moving to the next phase.

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

The application recognizes five content types. Detection is automatic unless overridden in `config.yaml`.

| Content Type | Key Characteristics |
|---|---|
| `PODCAST` | Single dominant speaker, static camera or minimal movement, no gaming HUD, no persistent second face, monologue or Q&A with off-screen host |
| `INTERVIEW` | Two or more clearly separated face regions, alternating speech patterns, face-switching required when speaker changes |
| `JUST_CHAT` | Single streamer, no gaming HUD, donation overlays (Trakteer.ID, Saweria) are common and must be preserved |
| `GAMING_SOLO` | Persistent gaming HUD/interface visible, single facecam in corner, donation overlays must be preserved |
| `GAMING_COLLAB` | Persistent gaming HUD, multiple persistent faces (Discord grid, dual-screen collab layout), donation overlays must be preserved |
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
  - `./workspace/videos/`, `./workspace/audios/`, `./workspace/subtitles/`, `./workspace/tmp/` — create if missing.
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
- **Execution Routing:** `app.py` is entry/routing only (load config → environment → logging) and delegates to the Typer app in `src/interfaces/cli.py`. If `sys.argv` has CLI arguments → route to Typer; if run bare → launch Gradio. CLI commands: `clip <URL>` (the pipeline, with `--clips/--duration/--language/--output-dir/--force/--debug` overrides), `config`, `cache status`, `cache purge [--dry-run]`, `serve` (WebUI stub). `test-pipeline` and `clean-workspace` remain as hidden back-compat aliases.
- **WebUI parity (next phase, REQUIREMENT):** every `config.yaml` field MUST be editable from the Gradio WebUI, with each control's **default loaded from `config.yaml`** — `config.yaml` stays the single source of truth, and the UI is an editor over it (a session override or a save-back), never a competing config store.

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

RMS Energy and Heatmap both produce more candidate windows than the user wants. Sending all of them to the LLM wastes compute and STT cost. The pipeline MUST apply a **candidate multiplier** to pre-filter before any AI call.

The selection flow for auto mode:
1. RMS or Heatmap produces N ranked spike windows (e.g., 25), already sorted by score (energy level or heatmap value) — highest first.
2. Compute the candidate pool size: `target_clips × candidate_multiplier` (e.g., 5 clips × 2 = 10 candidates).
3. Take only the **top candidates by score** from step 1. Discard the rest entirely — no STT, no LLM call on them.
4. STT-transcribe those top candidates.
5. Send **all transcripts in a single batched LLM call**, asking the model to return the best `target_clips` from the pool. The LLM compares candidates against each other and picks the most engaging and diverse set.
6. LLM returns exactly `target_clips` results with titles and reasoning.

This means: 25 RMS spikes + user wants 5 clips + multiplier 2 → **10 STT calls, 1 LLM call**, not 25 of each.

`candidate_multiplier` is configurable in `config.yaml`. A value of `2` is the recommended default — wide enough for the LLM to make a meaningful selection without excessive STT cost. Raising to `3` gives the LLM a broader pool at higher cost. Minimum valid value is `1` (LLM receives exactly the target count — no selection, just titling and boundary refinement).

**Review Gate:** After clip proposal, all candidates (Start Time, End Time, Title, Reasoning, Content Type detected) MUST be displayed in the WebUI review panel before any rendering begins. The user can approve, edit, or discard individual proposals. This gate is skippable via `clip_selection.require_review_before_render: false`.

#### 5.2 Manual Mode

- The user provides start and end timestamps directly. The format MUST accept both `MM:SS - MM:SS` and `HH:MM:SS - HH:MM:SS` on the same input.
- **Bulk Input:** Multiple ranges entered one per line in a textarea (WebUI) or via a `.txt` file upload or `--timestamps-file` CLI flag. Each line is parsed into a separate clip job.
- **Rendering is identical to Auto Mode:** Manual mode bypasses AI timestamp detection only. Layout detection, face tracking, subtitle generation, and all rendering logic run exactly the same as in Auto Mode.
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

This is the core intelligence of the application. The detection engine runs once per video, immediately after download and before any clip selection or rendering. Its output — a `ContentType` enum value — drives all downstream decisions: layout mode, face-switching behavior, donation overlay handling, and system prompt context.

**Visual Analysis (`src/vision/visual_analyzer.py`):** A shared YOLOv8n (Ultralytics COCO) engine analyses sampled frames of a time window and returns region metadata — `facecam_box`, `gameplay_box`/`gameplay_track`, `screen_inset_box`, persistent person count, motion level, `mediashare_events` — plus a compact **text descriptor**. This one engine serves three consumers: (a) **clip selection** — the descriptor is attached to each candidate window so a (possibly non-multimodal) LLM weighs on-screen events; (b) **rendering layout** — precise per-clip crop boxes replace generic face-tracking for Mode B/C; (c) **content-type detection** — person/screen/motion signals supplement the HUD heuristic. Configured under `video_processing.region_detection`; the YOLO region pass is sparse for low-spec hardware.

Two refinements matter for correctness:
- **Dense MediaShare event scan.** A transient MediaShare/donation popup is easily missed by sparse sampling, so a separate **~2 fps bounded scan** (appearance/disappearance against a median-frame baseline, via `overlay_detector`) runs over each window and reports `mediashare_events` with timing. The descriptor states e.g. *"Donation overlay reaction at ~412–418s (high-value moment)"*, and the LLM prompt treats such candidates as **high-priority** — this is what lets a mediashare moment actually be selected.
  - **Static-vs-moving gate (CRITICAL, anti-false-positive):** a real donation card holds the **same on-screen position** while shown; gameplay novelty (camera pan, moving character) **drifts** frame-to-frame and would otherwise be misread as a popup. The detector computes per-interval **box-centre jitter** (normalised by the frame diagonal) and **rejects** intervals above `POPUP_MAX_JITTER` — only a positionally-stable card survives. The aspect gate also requires cards clearly wider than tall (`POPUP_ASPECT_MIN`).
- **Animated gameplay pan.** At render, the gameplay bottom panel follows the action: a `gameplay_track` of peak-motion crop boxes (argmax, not centroid — better for slow/tiny subjects) drives the same smooth-pan FFmpeg crop expression used by the facecam. MediaShare bottom stays a static crop.

Audio (heatmap/energy/STT) and visual analysis run over the **same time windows** so selection and layout share aligned evidence. **Game/show context:** the downloader saves `{video_id}_metadata.json` (title/category/tags/description); the batched LLM prompt is prefixed with this so it can infer the game, and a "Gaming" category lowers the HUD threshold in detection.

#### 6.1 Detection Pipeline

Detection MUST be performed in this order, stopping at the first confident match:

1. **Gaming detection (HUD OR Gaming category)** — Sample frames and detect persistent HUD elements (health bars, minimaps, ammo counters, score displays). A positive HUD match classifies the video as gaming. **The YouTube "Gaming" category (`gaming_hint` from saved metadata) ALSO routes to gaming** even when the HUD is small/sub-threshold — the category is a strong, reliable signal (a small corner HUD must not drop a genuine gaming stream to PODCAST). Face count then splits the type:
   - If two or more persistent face regions are found → `GAMING_COLLAB`
   - If a single persistent face region is found → `GAMING_SOLO`
   - Face counting for SOLO-vs-COLLAB uses the **YOLO facecam detector** (`VisualAnalyzer.detect_facecams`): persistent, cam-sized person boxes near a frame **edge** (small corner webcams that MediaPipe misses are caught; interior game characters are rejected). Two separated cams ⇒ `GAMING_COLLAB`.

2. **Multi-Face Region Analysis** — If not gaming, count persistent face regions across sampled frames.
   - Two or more distinct, spatially-separated persistent face regions with alternating speech patterns → `INTERVIEW`
   - Single persistent face region → proceed to step 3.

3. **Donation Overlay Sampling** — For single-face, no-HUD content, sample for known donation overlay signatures (bright pop-up alerts in corners, Trakteer/Saweria visual patterns).
   - Overlay detected → `JUST_CHAT`
   - No overlay detected → `PODCAST`

4. **Config Override** — If `video_processing.content_type_override` is set to anything other than `"auto"` in `config.yaml`, skip all detection steps and use the configured value directly.

#### 6.2 Detection Output & Downstream Routing

The detected `ContentType` is stored in the pipeline state and passed to every downstream module. The routing table:

| Detected Type | Layout Mode | Face Switching | Donation Handling |
|---|---|---|---|
| `PODCAST` | Mode A — Single Vertical | No | — (promoted per-clip if a popup appears) |
| `INTERVIEW` | Mode A — Single Vertical + Speaker Switch | Yes | — (promoted per-clip if a popup appears) |
| `JUST_CHAT` | Mode B — Stacked Split-Screen | No | — (promoted per-clip if a popup appears) |
| `GAMING_SOLO` | Mode B — Stacked Split-Screen | No | — (promoted per-clip if a popup appears) |
| `GAMING_COLLAB` | Mode C — Multi-Face Collab Stack | No | — (promoted per-clip if a popup appears) |
| `DONATION_OVERLAY` | Mode B geometry — Facecam top + popup bottom | No | This **is** the donation layout |

**Per-clip donation promotion:** content type is detected once per video, but after per-clip visual analysis any clip whose window contains a transient mediashare/donation popup (`mediashare_present`) is promoted to `DONATION_OVERLAY` and routed to the facecam + popup 2-stack — regardless of the video's base type. Gated by `video_processing.preserve_donation_overlays` (off → donations are shown nowhere). An explicit per-clip `content_type` (review-gate edit) is never overridden. **Exception — `GAMING_COLLAB` always keeps its 3-stack:** the donation promotion is skipped for collab videos (the popup does not replace a panel), so both collaborators and the gameplay stay on screen.

#### 6.3 Detection Confidence & Fallback

- Each detection step should produce a confidence score. If all scores fall below a configurable threshold (`video_processing.detection_confidence_threshold`), the engine MUST fall back to `PODCAST` (safest default) and log a warning with the frame samples that caused uncertainty.
- The detected content type MUST be displayed in the WebUI Review panel so the user can override it before rendering if the detection was wrong.

---

### 7. Smart Vertical Cropping, Face Tracking & Layout Modes

Use `opencv-python-headless` and `mediapipe` or `yolov8-face`. The headless OpenCV variant prevents display-server crashes in WSL environments. All layout logic is driven by the `ContentType` produced by the Detection Engine in Section 6.

---

#### Layout Mode A — Single Vertical (Podcast / Interview)

**Applies to:** `PODCAST`, `INTERVIEW`

- Render as a **single 9:16 vertical crop**. No stacking, no split panels.
- Face tracking applies: the crop window smoothly follows the active speaker, keeping their face in the upper-center region.
- If no face is detected (e.g., screen-share segment), default to a center-biased static crop with no movement.

**Interview sub-mode (INTERVIEW only):**
- The face tracker MUST identify and register two or more distinct face regions at the start of each clip.
- During the clip, monitor audio activity per face region using speaker diarization or lip-movement detection to determine who is speaking at any given moment.
- When the active speaker changes, **smoothly transition the crop window** to center the new speaker. Do not cut abruptly — use a short eased pan (0.3–0.5s transition).
- The primary interview subject (the person being interviewed) is given crop priority during equal-activity frames.

---

#### Layout Mode B — Stacked Split-Screen (Gaming Solo / Just Chat)

**Applies to:** `GAMING_SOLO`, `JUST_CHAT`

- Construct a **two-region vertical stack** (each panel 1080×960, total 1080×1920):
  - **Top panel = Facecam (always fits).** A **single stable cam box** is detected once for the whole video (`detect_stable_facecam`, sampled across its length) and reused for every clip, so the framing does not drift with the streamer's pose. The cam box is expanded by `FACECAM_FIT_FACTOR` (≈1.45× — comfortable context margin, cam ≈69% of panel height) and shaped to the panel aspect (1.125), then **crop-filled into the 1080×960 panel — sharp, prominent, no blur and no left/right bars** (mild upscale when the cam region is smaller than the panel).
  - **Bottom panel = Gameplay.** A **centered, panel-aspect gameplay crop** zoomed by `region_detection.gameplay_zoom` (default 1.25× — crops the bottom ticker from the panel). When `region_detection.gameplay_follow_motion: true`, the crop **holds still by default** (static-first) and only glides slowly to follow character motion when the weighted-centroid target drifts beyond a deadzone (≈6% of panel width), capped at ≈1.2% of panel width per keyframe so the movement is imperceptible to viewers. (Donation/MediaShare popups are **not** shown here — a clip with a popup is promoted to `DONATION_OVERLAY`; see below.)
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
- **CRITICAL LAYOUT ORDER:** Gameplay canvas is always center. Both face regions are secondary. Each panel is a panel-aspect (1080×640) crop-fill (no distortion). The centre gameplay panel uses the **same zoomed, static-first slow-pan engine as Mode B** (`gameplay_zoom` + velocity-capped glide), shaped to the 3-stack aspect (1.6875). **Both cam boxes are excluded from the gameplay crop** — masked out of the motion search and, when they sit in the bottom third, the crop is shrunk + biased upward so its bottom edge sits at/above the highest bottom cam — so neither corner cam ever bleeds into the centre (no "double facecam"). A `GAMING_COLLAB` clip stays a 3-stack even when a donation is present (donation promotion is skipped for collab).
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
  candidate_multiplier: 2              # RMS/heatmap produces N spikes; LLM receives top (target_clips × multiplier)
                                       # e.g. 5 clips × 2 = top 10 candidates sent to LLM, rest discarded
                                       # Min: 1 (exact count, no selection) | Recommended: 2 | Max: 5
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
  content_type_override: "auto"        # auto | PODCAST | INTERVIEW | JUST_CHAT | GAMING_SOLO | GAMING_COLLAB | DONATION_OVERLAY
  detection_confidence_threshold: 0.6  # Below this, detection falls back to PODCAST and logs a warning
  auto_face_tracking: true             # Smooth face-centered crop for Mode A (talking heads)
  preserve_donation_overlays: true     # Detect and composite Trakteer/Saweria/MediaShare alerts
  default_resolution: "1080p"
  region_detection:                    # YOLOv8n visual analysis: facecam / gameplay / mediashare regions
    enabled: true                      # false = fall back to motion/face heuristics only
    model_name: "yolov8n.pt"           # Ultralytics COCO nano model (auto-downloaded to workspace/models/)
    sample_frames: 4                   # Frames sampled per candidate/clip window (sparse = fast)
    device: "auto"                     # auto | cpu | cuda
    gameplay_follow_motion: true       # true = gentle centred pan follows char (small move) | false = static centred crop
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
    margin_v: 760                      # pixels from the top of the 1920 canvas (~60% down)

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
> For PODCAST and INTERVIEW content: prioritize complete, punchy thoughts, strong opinions, surprising facts, or emotional peaks. Avoid segments that feel incomplete without visual context.
> For JUST_CHAT content: prioritize high-energy reactions, funny moments, and segments where donation interactions produce strong streamer responses.
> For GAMING_SOLO and GAMING_COLLAB content: prioritize intense gameplay moments, clutch plays, funny failures, and strong streamer reactions. Donation-triggered reactions are high-value clip targets.
>
> Return ONLY a valid JSON array of objects. Each object must contain: `start_time` (float, seconds), `end_time` (float, seconds), `title` (catchy, max 50 characters), and `reasoning` (one sentence). Never cut a clip mid-sentence. Ensure each clip is a complete, standalone moment."

**Timestamp anchoring & scoring rubric:** transcript lines reach the LLM prefixed with `[start - end]` seconds (relative to the candidate). The live prompt instructs the model to pick a **contiguous run of whole lines** — `start_time` = first line's start, `end_time` = last line's end (no mid-sentence cuts) — spanning ≈`{target_duration}`s, and to rank candidates by a rubric (**HOOK** in the first ~3s, clear **PAYOFF**, **STANDALONE** comprehension, **ENERGY**). The JSON output contract is unchanged.

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
- Key enums: `ContentType` (`PODCAST`, `INTERVIEW`, `JUST_CHAT`, `GAMING_SOLO`, `GAMING_COLLAB`, `DONATION_OVERLAY`), `LayoutMode` (`SINGLE_VERTICAL`, `STACKED_SPLIT`, `MULTI_COLLAB`), `AIProvider` (`GOOGLE`, `OPENAI`, `LOCAL`), `ClipMode` (`AUTO`, `MANUAL`).

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

Configured in `pyproject.toml` from Phase 1. Not optional.

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
| Function / method | `snake_case` verb-noun | `detect_content_type()`, `build_filter()` |
| Variable | `snake_case` descriptive | `output_clip_path`, `face_bbox` |
| Constant | `SCREAMING_SNAKE_CASE` | `MAX_CLIP_DURATION`, `CACHE_DIR` |
| Private helper | `_snake_case` prefix | `_resolve_wsl_path()` |
| Enum member | `SCREAMING_SNAKE_CASE` | `ContentType.GAMING_SOLO` |

#### 11.16 Modular Directory Tree

```
yaclip/
├── app.py                       # Entry point: routes to CLI or Gradio WebUI
├── config.yaml                  # User configuration
├── TASKS.md                     # Phase execution backlog and task tracking
├── requirements.txt             # pip3 compatibility
├── pyproject.toml               # uv/pip deps + ruff/mypy config
├── README.md                    # Setup and usage guide
├── docs/
│   ├── ARCHITECTURE.md          # System design and component overview
│   └── WORKFLOWS.md             # Pipeline diagrams and operational flows
├── logs/
│   └── app.log                  # Loguru rotating log file
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
│   │   ├── exceptions.py        # YaClipError hierarchy
│   │   ├── workspace.py         # Workspace init, sequential purge engine
│   │   └── logger.py            # Loguru setup
│   ├── media/
│   │   ├── downloader.py        # yt-dlp download logic, WSL cookie resolution
│   │   └── ffmpeg_builder.py    # FFmpeg command construction and filter graph helpers
│   ├── ai/
│   │   ├── api_client.py        # Cloud API wrappers (Google, OpenAI-compatible) with retry logic
│   │   ├── stt_local.py         # faster-whisper local STT with language detection
│   │   └── llm_local.py        # llama-cpp-python local highlight detection
│   ├── vision/
│   │   ├── visual_analyzer.py        # YOLOv8n region engine: facecam/gameplay/mediashare + LLM text descriptor
│   │   ├── content_type_detector.py  # Gaming HUD, face count, donation overlay signals → ContentType
│   │   ├── face_tracker.py           # OpenCV/MediaPipe face detection, crop box, speaker-switch logic (Mode A)
│   │   ├── layout_builder.py         # ContentType + VisualAnalyzer regions → FFmpeg layout assembly
│   │   └── overlay_detector.py       # Appearance/disappearance novelty detection (median baseline diff; cam-exclusion guards)
│   └── interfaces/
│       ├── cli.py               # Typer CLI commands (clip, workspace, config)
│       ├── webui.py             # Gradio layout: Clipper, Review, Settings, Maintenance tabs
│       └── components.py        # Reusable Gradio component factories
└── workspace/
    ├── bin/                     # FFmpeg, yt-dlp binaries (auto-downloaded)
    ├── fonts/                   # .ttf subtitle fonts
    ├── models/                  # Local GGUF LLM and Whisper model files
    ├── videos/                  # Raw yt-dlp downloads
    ├── audios/                  # Extracted audio tracks
    ├── subtitles/               # STT transcriptions and AI JSON clip proposals
    └── tmp/                     # FFmpeg scratch workspace
```

---

### 12. Logging, Observability & Strict Memory Optimization

- **Loguru (CRITICAL):** Completely abandon `print()`. All output goes through `loguru`. Configuration (level, rotation, retention, file path) loaded from `config.yaml` at init. Rotating `.log` files written to `./workspace/logs/` capture silent FFmpeg failures.
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
| `./workspace/subtitles/` | Transcriptions & AI JSON | 3 days |
| `./workspace/tmp/` | FFmpeg scratch files | 1 day |

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