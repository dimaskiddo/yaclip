from __future__ import annotations

import typer

from src.core.workspace import ensure_workspace_integrity, run_purge_cycle


def serve(
    host: str = typer.Option(None, help="Host IP to bind Gradio to (default: from config.yaml)"),
    port: int = typer.Option(None, help="Port to bind Gradio to (default: from config.yaml)"),
) -> None:
    """Launch the YaClip WebUI in your browser.

    Opens a YaClip WebUI interface at http://<host>:<port> where you can
    download videos, select clips, render layouts, and manage the cache
    — all through a visual UI instead of the command line.
    """
    ensure_workspace_integrity()
    run_purge_cycle()

    from src.interfaces.webui import launch_webui

    launch_webui(host, port)


def register(cli: typer.Typer) -> None:
    cli.command("serve")(serve)
