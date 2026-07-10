from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

from src.core.config import load_config
from src.core.environment import ensure_vision_runtime
from src.core.exceptions import DetectionError
from src.core.utils import SystemUtils, load_timerange_file
from src.core.workspace import (
    DATA_DIR,
    VIDEOS_DIR,
    ensure_workspace_integrity,
    run_purge_cycle,
)


def _apply_overrides(
    clips: int | None,
    duration: int | None,
    min_duration: int | None,
    max_duration: int | None,
    language: str | None,
    output_dir: str | None,
    manual: bool = False,
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
    if manual:
        cs.mode = "manual"


def _run_pipeline(
    url: str,
    force: bool,
    debug: bool,
    manual_ranges: list[dict] | None = None,
    no_metadata: bool = False,
    cookies_file: str | None = None,
) -> None:
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
        result = VideoDownloader().download_video(
            url, str(VIDEOS_DIR), force=force, cookies_file=cookies_file
        )
        logger.info(f"Download complete: {result.get('title', 'Unknown')}")

        audio_path = result.get("audio_path")
        video_path_str = result.get("video_path")
        if not audio_path or not video_path_str:
            logger.error(
                "Download did not produce both video and audio files. Cannot continue."
            )
            return

        # Detect content type ONCE for the whole video (aggregated evidence: gameplay gate,
        # webcam count, HUD, donation overlay). Returns None when uncertain → LLM decides.
        from src.vision.content_type_detector import ContentTypeDetector

        detection_result = ContentTypeDetector().detect_content_type_full(
            Path(video_path_str)
        )
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
            manual_ranges=manual_ranges,
            no_metadata=no_metadata,
        )

        video_id = Path(audio_path).stem
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_json = DATA_DIR / f"{video_id}.json"
        out_json.write_text(json.dumps(clips, indent=2), encoding="utf-8")
        logger.info(
            f"Saved AI highlight results to {SystemUtils.display_path(out_json)}"
        )

        logger.info("--- SELECTED CLIPS ---")
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


def register(cli: typer.Typer) -> None:
    @cli.command("clip")
    def clip(
        url: str = typer.Argument(..., help="YouTube URL to clip"),
        clips: int | None = typer.Option(
            None, "--clips", "-n", help="Number of clips (overrides config)"
        ),
        duration: int | None = typer.Option(
            None,
            "--duration",
            "-t",
            help="Target clip duration in seconds (overrides config)",
        ),
        min_duration: int | None = typer.Option(
            None,
            "--min-duration",
            help="Minimum clip duration in seconds (guaranteed floor)",
        ),
        max_duration: int | None = typer.Option(
            None, "--max-duration", help="Maximum clip duration in seconds (hard cap)"
        ),
        language: str | None = typer.Option(
            None,
            "--language",
            "-l",
            help="Subtitle language: auto | ISO 639-1 code or name",
        ),
        output_dir: str | None = typer.Option(
            None, "--output-dir", "-o", help="Output directory for rendered clips"
        ),
        force: bool = typer.Option(
            False, "--force", "-f", help="Re-download / re-transcribe, ignoring caches"
        ),
        debug: bool = typer.Option(
            False, "--debug", "-d", help="Keep workspace/tmp scratch files"
        ),
        manual: bool = typer.Option(
            False,
            "--manual",
            help="Manual mode: use fixed timeranges from --timerange-file, bypassing AI "
            "selection (clip count, min-duration, candidate margin all ignored)",
        ),
        timerange_file: Path | None = typer.Option(
            None,
            "--timerange-file",
            help="Path to a manual timerange file (one 'START - END' per line, "
            "MM:SS or HH:MM:SS, optional '| CONTENT_TYPE' to pin the layout); requires --manual",
        ),
        no_metadata: bool = typer.Option(
            False,
            "--no-metadata",
            help="Skip LLM titling/metadata entirely (manual mode only); clips get "
            "Manual_<start>_<end> titles and no .txt sidecar",
        ),
        cookies_file: Path | None = typer.Option(
            None,
            "--cookies-file",
            "-c",
            help="Path to a cookies.txt file for YouTube authentication",
        ),
    ) -> None:
        """Download a video, auto-detect the best moments, and render vertical 9:16 clips.

        AUTO mode (default): AI analyzes the video to find the most engaging segments,
        transcribes audio, and renders each highlight as a Shorts/Reels/TikTok clip.

        MANUAL mode (--manual): Skip AI selection. Provide a timerange file with
        explicit start-end timestamps for each clip you want to render.

        Output is written to ``./workspace/clips/<video_id>_<clip_number>.mp4``
        (or your custom --output-dir).
        """
        if manual and not timerange_file:
            logger.error(
                "--manual requires --timerange-file (path to your timerange file)."
            )
            raise typer.Exit(1)
        if timerange_file and not manual:
            logger.error("--timerange-file requires --manual to be passed.")
            raise typer.Exit(1)
        if no_metadata and not manual:
            logger.error(
                "--no-metadata requires --manual (auto/hybrid mode needs the LLM for clip selection)."
            )
            raise typer.Exit(1)

        manual_ranges: list[dict] | None = None
        if manual:
            try:
                manual_ranges = load_timerange_file(timerange_file)  # type: ignore[arg-type]
            except (ValueError, OSError) as e:
                logger.error(f"Failed to load timerange file: {e}")
                raise typer.Exit(1) from e
            logger.info(
                f"Manual mode: loaded {len(manual_ranges)} timerange(s) from {timerange_file}."
            )

        _apply_overrides(
            clips,
            duration,
            min_duration,
            max_duration,
            language,
            output_dir,
            manual=manual,
        )
        _run_pipeline(
            url,
            force=force,
            debug=debug,
            manual_ranges=manual_ranges,
            no_metadata=no_metadata,
            cookies_file=str(cookies_file) if cookies_file else None,
        )
