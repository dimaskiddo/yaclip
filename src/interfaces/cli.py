"""Typer command-line interface for Yet Another AI Auto-Clipper (YaClip).

`app.py` routes here when CLI arguments are present. Commands:
  clip <URL>        — full pipeline (download → AI selection → render), with override flags
  config            — print the validated configuration (secrets masked)
  cache status      — per-directory workspace disk usage
  cache purge       — retention-respecting manual cleanup (opt-in --concern)
  cache clean       — force-clean specific workspace directories
  serve             — launch the Gradio WebUI (stub; next phase)
  clean-workspace   — hidden alias of `cache clean`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from loguru import logger

from src.core.config import load_config
from src.core.constants import BYTES_PER_MB
from src.core.environment import ensure_vision_runtime
from src.core.exceptions import DetectionError
from src.core.utils import SystemUtils
from src.core.workspace import (
    AUDIOS_DIR,
    CLIPS_DIR,
    DATA_DIR,
    SUBTITLES_DIR,
    TMP_DIR,
    VIDEOS_DIR,
    ensure_workspace_integrity,
    run_purge_cycle,
)

cli = typer.Typer(help="Yet Another AI Auto-Clipper (YaClip) — CLI", add_completion=False)
cache_app = typer.Typer(help="Inspect or purge the ./workspace cache.")
cli.add_typer(cache_app, name="cache")


def _apply_overrides(
    clips: int | None,
    duration: int | None,
    min_duration: int | None,
    max_duration: int | None,
    language: str | None,
    output_dir: str | None,
) -> None:
    """Patch the loaded config singleton with CLI flag overrides (Pydantic models are mutable)."""
    cfg = load_config()
    cs = cfg.clip_selection
    if clips is not None:
        cs.default_clips = clips
    if duration is not None:
        cs.default_clip_duration_seconds = duration
    if min_duration is not None:
        cs.min_clip_duration_seconds = min_duration
    if max_duration is not None:
        cs.max_clip_duration_seconds = max_duration
    # Re-run the coherence validator so min ≤ default ≤ max holds after the overrides.
    cs._coherent_durations()
    if language is not None:
        cfg.video_processing.subtitles.language = language
    if output_dir is not None:
        cfg.video_processing.output_dir = output_dir


def _run_pipeline(url: str, force: bool, debug: bool) -> None:
    """Download → AI clip selection → render. Shared by `clip` and the `test-pipeline` alias."""
    ensure_workspace_integrity()
    run_purge_cycle()

    from src.ai.pipeline import AIPipeline
    from src.media.downloader import VideoDownloader
    from src.media.renderer import ClipRenderer

    try:
        # Fail fast (before any download) if MediaPipe's GL/EGL system libs are missing.
        ensure_vision_runtime()

        logger.info("--- STEP 1: DOWNLOAD AND EXTRACT ---")
        result = VideoDownloader().download_video(url, str(VIDEOS_DIR))
        logger.info(f"Download complete: {result.get('title', 'Unknown')}")

        audio_path = result.get("audio_path")
        video_path_str = result.get("video_path")
        if not audio_path or not video_path_str:
            logger.error("Download did not produce both video and audio files. Cannot continue.")
            return

        # Detect content type ONCE for the whole video (aggregated evidence: gameplay gate,
        # webcam count, HUD, donation overlay). Returns None when uncertain → LLM decides.
        from src.vision.content_type_detector import ContentTypeDetector

        detection_result = ContentTypeDetector().detect_content_type_full(Path(video_path_str))
        content_type = detection_result.content_type

        logger.info("--- STEP 2: AI CLIP SELECTION ---")
        clips = AIPipeline().process_audio(
            audio_path,
            video_path=video_path_str,
            force=force,
            detected_type=content_type,
            detection_evidence=(
                detection_result.evidence if not detection_result.is_confident else None
            ),
        )

        video_id = Path(audio_path).stem
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_json = DATA_DIR / f"{video_id}.json"
        out_json.write_text(json.dumps(clips, indent=2), encoding="utf-8")
        logger.info(f"Saved AI highlight results to {SystemUtils.display_path(out_json)}")

        logger.info("--- FINAL EXTRACTED CLIPS ---")
        for idx, c in enumerate(clips):
            logger.info(
                f"[{idx + 1}] {c.get('start_time', 0.0):.1f}s - "
                f"{c.get('end_time', 0.0):.1f}s | {c.get('title', 'Unknown')}"
            )

        logger.info("--- STEP 3: RENDERING CLIPS ---")
        formatted = [
            {
                "start_time": float(c.get("start_time") or c.get("start", 0.0)),
                "end_time": float(c.get("end_time") or c.get("end", 0.0)),
                "title": c.get("title", f"Clip_{idx + 1}"),
                "caption": c.get("caption", ""),
                "description": c.get("description", ""),
                "hashtags": c.get("hashtags", ""),
            }
            for idx, c in enumerate(clips)
        ]
        rendered = ClipRenderer().render_clips(
            Path(video_path_str), formatted, content_type=content_type
        )

        logger.info(f"Pipeline complete, {len(rendered)} clip(s) rendered.")
        for f in rendered:
            logger.info(f" -> {SystemUtils.display_path(f)}")
            txt_candidate = Path(str(f).replace(".mp4", ".txt"))
            if txt_candidate.exists():
                logger.info(f" -> {SystemUtils.display_path(txt_candidate)}")
    except DetectionError as e:
        logger.error(str(e))
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
    finally:
        if debug:
            logger.info("Debug mode: keeping workspace/tmp scratch files.")
        else:
            run_purge_cycle(force=True, specific_target="tmp")


@cli.command("clip")
def clip(
    url: str = typer.Argument(..., help="YouTube URL to clip"),
    clips: int | None = typer.Option(
        None, "--clips", "-n", help="Number of clips (overrides config)"
    ),
    duration: int | None = typer.Option(
        None, "--duration", "-t", help="Target clip duration in seconds (overrides config)"
    ),
    min_duration: int | None = typer.Option(
        None, "--min-duration", help="Minimum clip duration in seconds (guaranteed floor)"
    ),
    max_duration: int | None = typer.Option(
        None, "--max-duration", help="Maximum clip duration in seconds (hard cap)"
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Subtitle language: auto | ISO 639-1 code or name"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for rendered clips"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-download / re-transcribe, ignoring caches"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Keep workspace/tmp scratch files"),
) -> None:
    """Download a video, auto-select highlights, and render vertical 9:16 clips."""
    _apply_overrides(clips, duration, min_duration, max_duration, language, output_dir)
    _run_pipeline(url, force=force, debug=debug)


@cli.command("config")
def show_config() -> None:
    """Print the validated configuration (API keys masked)."""

    data = load_config().model_dump()

    def _mask(node: object) -> object:
        if isinstance(node, dict):
            return {k: ("***" if k == "api_key" and v else _mask(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [_mask(v) for v in node]
        return node

    typer.echo(yaml.safe_dump(_mask(data), sort_keys=False, allow_unicode=True))


@cache_app.command("status")
def cache_status() -> None:
    """Show per-directory size, file count, and oldest file for the workspace cache."""

    targets = {
        "videos": VIDEOS_DIR,
        "audios": AUDIOS_DIR,
        "subtitles": SUBTITLES_DIR,
        "data": DATA_DIR,
        "clips": CLIPS_DIR,
        "tmp": TMP_DIR,
    }
    typer.echo(f"{'dir':<12}{'size (MB)':>12}{'files':>8}{'oldest':>12}")
    now = time.time()
    for name, path in targets.items():
        files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
        size_mb = sum(p.stat().st_size for p in files) / BYTES_PER_MB
        oldest = f"{(now - min(p.stat().st_mtime for p in files)) / 86400:.1f}d" if files else "-"
        typer.echo(f"{name:<12}{size_mb:>12.1f}{len(files):>8}{oldest:>12}")


@cache_app.command("purge")
def cache_purge(
    target: list[str] | None = typer.Argument(  # noqa: B008
        None,
        help="Space-separated dirs to purge (videos|audios|subtitles|data|tmp|clips|logs); all if omitted",
    ),
    concern: bool = typer.Option(
        False, "--concern", help="Will run the purging with user concern confirmed"
    ),
) -> None:
    """Manually purge the workspace cache (bypasses retention)."""
    if not concern:
        load_config().workspace_cleanup.dry_run = True
    logger.info(f"Manual cache purge (target: {target or 'all'}, concern: {concern})...")
    run_purge_cycle(force=concern, specific_target=target)


@cache_app.command("clean")
def cache_clean(
    target: list[str] = typer.Argument(  # noqa: B008
        None, help="Space-separated workspace directories to force-clean"
    ),
) -> None:
    """Force-clean specific workspace directories (bypasses retention & dry-run)."""
    logger.info(f"Cache clean: {target or 'all'}...")
    run_purge_cycle(force=True, specific_target=target)


@cli.command("clean-workspace", hidden=True)
def clean_workspace(
    target: list[str] = typer.Argument(  # noqa: B008
        None, help="Space-separated workspace directories to clean"
    ),
) -> None:
    """Deprecated alias of `cache purge`."""
    run_purge_cycle(force=True, specific_target=target)


@cli.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host IP to bind Gradio to"),
    port: int = typer.Option(7860, help="Port to bind Gradio to"),
) -> None:
    """Launch the Gradio WebUI (placeholder — next phase)."""
    ensure_workspace_integrity()
    run_purge_cycle()
    logger.info(f"WebUI not yet implemented. Starting on http://{host}:{port}...")
