# YaClip — Agent Instructions

Build a cross-platform YouTube auto-clipper: download → detect engaging moments → render 9:16 vertical clips for Shorts/Reels/TikTok. CLI (typer) + WebUI (gradio). Low-spec optimized, portable workspace, cloud-first with local-fallback AI.

---

## Workflow Rules

1. Read `TASKS.md` before every session to orient to current state.
2. Never rework items marked `[x]` in `TASKS.md` unless explicitly instructed.
3. Update `TASKS.md` immediately after testing a feature.
4. Never attempt to write the entire codebase in a single response.
5. Create `TASKS.md`, `docs/ARCHITECTURE.md`, `docs/WORKFLOWS.md` during project init.

## Skills & Caveman Mode

- **GLOBAL:** All prompts processed as if `"Use caveman mode full"` is injected.
- Before ANY coding task, invoke and read: `using-superpowers`, `karpathy-guidelines`, `caveman`.
- Use `using-superpowers` to route to other relevant skills per task.

---

## Content Types

| Type | Key Traits |
|---|---|
| `PODCAST` | Talking-head, no gameplay. Speaker tracking via lip-movement (MAR). Two-shot grouping when faces fit crop. |
| `JUST_CHAT` | Single streamer, no HUD. Donation overlays common. |
| `GAMING_SOLO` | Gameplay + single facecam. |
| `GAMING_COLLAB` | Gameplay + ≥2 persistent webcams. |
| `DONATION_OVERLAY` | **Per-clip promotion.** Any clip with a mediashare/donation popup → facecam + popup 2-stack. Disabled by default. |
| `GAMING_SOLO_BOTTOM` | **Pin/override only.** Mirrored GAMING_SOLO (gameplay top, facecam bottom). Never auto-detected. |

Detection: whole-video (25 frames + YOLO + gameplay probe + HUD score). See `docs/ARCHITECTURE.md §2`.

## Output Format

All clips: 9:16 vertical. Clip length = `[default_clip_duration_seconds, default + clip_length_margin_seconds]` (e.g. 60–75s). `min/max_clip_duration_seconds` = WebUI slider bounds, NOT per-clip caps. CLI `--duration` overrides target.

## Clip Selection

- **Auto:** strategies `ai` | `heatmap` | `hybrid` (recommended). Pre-ranking with `candidate_margin`: pool = target + margin → top candidates only get STT + single batched LLM call. Review gate before render.
- **Manual:** user provides `START - END` timestamps (bulk, per-line). Optional `| CONTENT_TYPE` suffix pins layout. `--manual` + `--timerange-file` + `--no-metadata` flags. LLM titling runs by default (skip with `--no-metadata`).

---

## Critical Constraints

### Python Ecosystem
- `.venv` mandatory. `source .venv/bin/activate` (Linux/macOS) or `.\.venv\Scripts\activate` (Windows) or `uv run`. Never touch global Python.
- Package install: `uv` preferred or `pip3`. Always `--no-cache-dir`/`--no-cache`.
- **CPU torch by default.** `torch==2.11.0+cpu` / `torchvision==0.26.0+cpu` pinned. CUDA torch ships `triton` which segfaults with MediaPipe on CPU/WSL. `guard_triton_segfault()` in `src/core/environment.py` masks triton if no GPU. GPU users: set `YACLIP_FORCE_TRITON=1`.
- All runtime assets in `./workspace/`. Never `/tmp/` or `%TEMP%`.
- Startup boot: verify `workspace/bin/` (FFmpeg, yt-dlp), `workspace/fonts/` (Anton.ttf), create missing dirs.

### AI Pipeline
- Independent STT + LLM provider selection (`ai_pipeline.stt.provider`, `ai_pipeline.llm.provider`): `cloud` | `local` | `auto`.
- Cloud Google STT + Cloud Google LLM = single unified Gemini call (cheapest).
- **Memory safety:** when both local, strict sequential load/unload with `del model` + `gc.collect()`. Never coexist in RAM.

### Interfaces
- **CLI:** `typer`. Commands: `clip <URL>`, `config`, `cache status/purge/clean`, `serve`.
- **WebUI:** `gradio`. Tabs: Clipper, Review & Render, Settings, Maintenance. Always `.queue().launch()`.
- `app.py` = entry/router only. CLI args → Typer; bare → Gradio.
- Gradio binds `127.0.0.1:7860` (configurable) for WSL host browser access.

### Content Detection
- Detected once per video, before clip selection. Drives all layout/face/donation decisions.
- Signals: YOLOv8n gameplay presence (`open_area_frac ≥ 0.45`), `gaming_hint` (YouTube category), HUD score, `detect_facecams()` (filtered by persistence/area/edge).
- Decision tree: gameplay+<2 cams → SOLO, gameplay+≥2 cams → COLLAB, no gameplay+≥2 faces → PODCAST, no gameplay+1 face+donation → JUST_CHAT, no gameplay+1 face → PODCAST, ambiguous → `None`.
- `None` → LLM classifies with structured evidence block. Post-validated (audio 1 speaker ≠ COLLAB; rejection falls back to PODCAST).
- `detection_confidence_threshold` (0.6): below this, detection falls back to PODCAST with a warning log.
- `content_type_override` in config bypasses detection entirely.

### Layout Modes
- **Mode A (PODCAST):** Single 9:16 vertical crop. Active speaker tracking via MAR + audio-visual sync (RMS envelope, Pearson coherence, voice gate). EMA pan (factor 0.03, τ≈1.1s). Two-shot grouping when `span ≤ 0.9×crop_w` AND `gap ≤ 0.25×crop_w`. Min shot: 2.0s. Hysteresis: 1.5× margin. `PODCAST_DETECTION_FPS` = 10. Occlusion-aware hold. `fast_mode` opt-in swaps MediaPipe+audio for OpenCV Haar largest-face (single-speaker only). See `docs/ARCHITECTURE.md §5`.
- **Mode B (SOLO/CHAT):** 2-stack (2× 1080×960). Top=facecam (stable box, FACECAM_FIT_FACTOR≈1.45×, crop-fill), bottom=static gameplay crop (gameplay_zoom 1.25×). Fullscreen-cam override at >55% frame. `GAMING_SOLO_BOTTOM` = mirrored order.
- **Mode B — Donation:** Top=facecam, bottom=donation popup (appearance/disappearance detector). Gated by `preserve_donation_overlays`.
- **Mode C (COLLAB):** 3-stack (3× 1080×640). Primary cam top, gameplay center, collaborator bottom. Both cams from `detect_facecams` pair. Cam boxes excluded from gameplay crop. No donation in 3-stack → promoted to DONATION_OVERLAY.
- Gameplay crops use motion-centroid centering. `gameplay_follow_motion: true` (default) = gentle motion-following pan; `false` = fully static centered crop.

### Subtitles
- Word-by-word `.ass` focus effect (active word bold+highlighted). Burned via FFmpeg `vf=subtitles`.
- **Hallucination filter:** segments with high `compression_ratio`, `no_speech_prob`+`avg_logprob`, or token repetition → dropped. Runs on all STT outputs.
- Language-locking: native-language primer (`LANGUAGE_PROMPTS`, ~34 langs) as `initial_prompt`.
- Auto-detect: cloud=native, local=faster-whisper `language=None`. Detected language logged + shown in WebUI.

### Configuration
- Single `config.yaml` in project root. Pydantic validated at startup (`src/core/config.py`). Full reference: `config.yaml.example`.
- `dk_clipper_sys_prompt` = hidden override for system prompt (must not appear in base config).

### Cache & Purge
- Startup purge cycle: `videos/` 3d, `audios/` 3d, `subtitles/` 3d, `data/` 3d, `tmp/` 1d. Protected: `bin/`, `fonts/`, `models/`.
- `tmp/` walked recursively; others non-recursively. `retention_days=-1` skips dir, `0`=purge all.
- Dry-run mode available. Active pipeline guard skips purge.

### System Prompt
- Hardcoded template with `{content_type}`, `{target_duration}`, `{language_instruction}`. Informal hook/bait tone. JSON array output with `candidate_index`, `start_time`, `end_time`, `title`, `caption`, `description`, `hashtags`, `content_type`, `reasoning`.

---

## Python Standards

- **Constants (CRITICAL):** Before implementing any module, read `src/core/constants.py` for exact thresholds, defaults, and enum values. Never guess numeric constants — the file is the single source of truth. It contains 100+ typed constants covering STT hallucination thresholds, face-tracking, gameplay detection, overlay detection, audio analysis, LLM limits, and all module-specific defaults.
- **Type hints:** Every function fully annotated. `X | None` (not Optional). `from __future__ import annotations`. TypedDict for dict shapes, Pydantic BaseModel for data, Protocol for interfaces.
- **Pydantic:** All config, clip results, pipeline state, API responses = BaseModel. No raw dict access outside `config.py`.
- **Lazy imports:** Heavy ML/CV modules (`faster_whisper`, `llama_cpp`, `torch`, `mediapipe`, `cv2`) imported inside execution functions only.
- **Exceptions:** Custom hierarchy in `src/core/exceptions.py`: `YaClipError` → `ConfigValidationError`, `DownloadError`, `DetectionError`, `RenderError`, `AIProviderError`, `CacheInitError`. No bare `except:`. Chain with `raise X from Y`.
- **Constants:** All in `src/core/constants.py` as Enum/typed constants. Zero magic values. Key enums: `ContentType`, `LayoutMode` (`SINGLE_VERTICAL`, `STACKED_SPLIT`, `MULTI_COLLAB`), `AIProvider`, `ClipMode`.
- **Logging:** Loguru only. No `print()`. INFO = user-readable single line. DEBUG = technical detail. Config from `config.yaml`.
- **Paths:** `pathlib.Path` only. No `os.path.join` or string concat.
- **Subprocess:** List form only. No `shell=True`. Capture stderr, set timeout, pass `str(path)`.
- **Function design:** Max 40 lines. One function = one thing. Guard clauses, max 3 nesting levels. No boolean flags changing core behavior.
- **Resources:** `with` blocks / `contextlib.contextmanager` for all open/close lifecycles. Local models: guaranteed `del` + `gc.collect()` on exit.
- **Async:** `asyncio` for I/O. `ThreadPoolExecutor` for CPU subprocesses. Offload blocking from Gradio handlers.
- **Toolchain:** `ruff` (line-length=100, py310), `mypy --strict`. Configured in `pyproject.toml`.
- **Testing:** `pytest`, `tests/` mirrors `src/`. Mock external boundaries. Integration tests behind `@pytest.mark.integration`. ≥80% coverage on `src/core/` and `src/media/`.
- **Docstrings:** Google-style (`Args:`, `Returns:`, `Raises:`). Comments explain *why*, not *what*. FFmpeg `filter_complex` blocks get plain-English comments.
- **Naming:** modules `snake_case`, classes `PascalCase`, functions `verb_noun`, constants `SCREAMING_SNAKE`, private `_prefix`. Clip filenames: `{VIDEO_ID}/{NN}_{Title-Case}` (zero-padded).

---

## Non-Negotiable Rules

1. **No stubs.** Never use `# ... rest of the code`, bare `pass`, or `# TODO`. Every file must be complete, production-ready, fully typed.
2. **Never auto-run pipeline.** After implementing/fixing, provide the exact `python app.py clip <url>` command + expected output. Wait for user to report results.
3. **No guessing** on FFmpeg filter graphs, speaker diarization, content type detection model, or Gradio callback architecture. Describe the ambiguity + options, await decision.
4. **Never use system temp dirs.** All files in `./workspace/`.
5. **WSL cookie resolution:** Detect WSL, resolve Windows host username, map cookie DB via `/mnt/c/Users/<WinUser>/AppData/...`, copy to `workspace/tmp/`.

---

## Directory Tree

```
yaclip/
├── app.py                    # Entry: routes CLI or WebUI
├── config.yaml               # User config (gitignored, copy from .example)
├── config.yaml.example       # Distributable config template
├── TASKS.md                  # Task tracking backlog
├── requirements.txt          # pip deps
├── pyproject.toml            # uv/pip deps + ruff/mypy config
├── Dockerfile                # CPU-only build
├── Dockerfile.CUDA           # GPU build (nvidia-container-toolkit)
├── docs/
│   ├── ARCHITECTURE.md       # Module map, detection, layouts
│   └── WORKFLOWS.md          # Pipeline diagrams
├── src/
│   ├── core/                 # config, constants, exceptions, logger, utils, workspace
│   ├── media/                # audio, downloader, energy, ffmpeg_builder, renderer, slicer, subtitles
│   ├── ai/                   # api_client, heatmap, llm_cloud/local, pipeline, prompts, stt_cloud/local
│   ├── vision/               # visual_analyzer, content_type_detector, face_tracker, layout_builder, overlay_detector
│   └── interfaces/
│       ├── cli/              # app.py + commands/{clip,config,cache,serve}.py
│       ├── webui/            # app.py + tabs/{clipper,review,settings,maintenance}.py
│       ├── components.py     # Gradio component factories
│       └── utils.py          # Shared helpers
└── workspace/
    ├── bin/                  # FFmpeg, yt-dlp (auto-downloaded)
    ├── fonts/                # .ttf fonts (Anton auto-downloaded)
    ├── models/               # GGUF LLM files, HuggingFace cache
    ├── logs/                 # Rotating loguru logs
    ├── clips/                # Final rendered clips (never purged)
    ├── videos/               # Raw downloads (3d retention)
    ├── audios/               # Extracted audio (3d retention)
    ├── subtitles/            # STT/AI data (3d retention)
    └── tmp/                  # FFmpeg scratch (1d retention)
```

---

## References

| File | Purpose |
|---|---|
| `config.yaml.example` | Full annotated config reference |
| `docs/ARCHITECTURE.md` | Module map, detection pipeline details, layout mode specs, face tracking algorithms |
| `docs/WORKFLOWS.md` | Pipeline flow diagrams, operational sequences |
| `TASKS.md` | Current project state — read before every session |
| `Makefile` | Build targets (setup, dev, build, test, lint, clean) |
| `pyproject.toml` | Dependencies and ruff/mypy configuration |
| `app.py` | Entry point — routes CLI or WebUI |
| `src/core/constants.py` | All enums (`ContentType`, `LayoutMode`, etc.) and numeric constants with defaults |
| `src/core/config.py` | Pydantic config, YAML validation, runtime config |
| `src/core/exceptions.py` | Custom error hierarchy (`YaClipError` → domain errors) |
| `src/core/workspace.py` | Workspace directory management and startup boot |
| `src/ai/pipeline.py` | Orchestrates STT → LLM → candidate ranking |
| `src/vision/content_type_detector.py` | Content type detection logic (YOLO + heuristic + LLM fallback) |
| `src/vision/face_tracker.py` | MAR-based speaker tracking |
| `src/vision/layout_builder.py` | Layout mode rendering (Mode A/B/C) |
| `src/vision/overlay_detector.py` | Donation/mediashare popup detection |
| `src/media/renderer.py` | Final clip rendering pipeline |
| `src/interfaces/cli/` | Typer CLI entry point and command modules |
| `src/interfaces/webui/` | Gradio WebUI entry point and tab modules |
