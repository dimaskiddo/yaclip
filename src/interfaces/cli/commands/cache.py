from __future__ import annotations

import typer
from loguru import logger

from src.core.config import load_config
from src.core.workspace import cache_usage, run_purge_cycle
from src.interfaces.utils import format_cache_rows


def register(cache_app: typer.Typer) -> None:
    @cache_app.command("status")
    def cache_status() -> None:
        """Show per-directory size, file count, and oldest file in the workspace cache.

        Use this to see how much disk space each directory (videos, audios, clips, ...)
        is using before deciding what to purge.
        """

        typer.echo(f"{'dir':<12}{'size (MB)':>12}{'files':>8}{'oldest':>12}")
        for row in format_cache_rows(cache_usage()):
            typer.echo(f"{row[0]:<12}{row[1]:>12.1f}{row[2]:>8}{row[3]:>12}")

    @cache_app.command("purge")
    def cache_purge(
        target: list[str] | None = typer.Argument(
            None,
            help="Space-separated dirs to purge (videos|audios|subtitles|data|tmp|clips|logs); all if omitted",
        ),
        concern: bool = typer.Option(
            False,
            "--concern",
            help="Confirm you want to purge. Without this flag the command runs in dry-run mode (no files deleted).",
        ),
    ) -> None:
        """Remove expired workspace files respecting retention policies.

        By default this is a **dry run** — nothing is deleted. Pass ``--concern``
        to actually delete files older than their configured retention period
        (see ``workspace_cleanup.retention_days`` in ``config.yaml``).
        """
        if not concern:
            load_config().workspace_cleanup.dry_run = True
        logger.info(
            f"Manual cache purge (target: {target or 'all'}, concern: {concern})..."
        )
        run_purge_cycle(force=concern, specific_target=target)

    @cache_app.command("clean")
    def cache_clean(
        target: list[str] = typer.Argument(
            None,
            help="Space-separated workspace directories to force-clean (videos|audios|subtitles|data|tmp|clips|logs); all if omitted",
        ),
    ) -> None:
        """Force-delete ALL files in selected directories (bypasses retention & dry-run).

        Unlike ``cache purge``, this deletes **everything** in the selected
        directories regardless of age. Use with care — there is no undo.
        """
        logger.info(f"Cache clean: {target or 'all'}...")
        run_purge_cycle(force=True, specific_target=target)
