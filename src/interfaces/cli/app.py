from __future__ import annotations

import typer

from src.interfaces.cli.commands import cache as cache_cmd
from src.interfaces.cli.commands import clip as clip_cmd
from src.interfaces.cli.commands import config as config_cmd
from src.interfaces.cli.commands import serve as serve_cmd

cli = typer.Typer(
    help="Yet Another AI Auto-Clipper (YaClip)",
    add_completion=False,
)
cache_app = typer.Typer(
    help="Check workspace cache usage, purge expired files, or force-clean specific directories."
)
cli.add_typer(cache_app, name="cache")

clip_cmd.register(cli)
config_cmd.register(cli)
cache_cmd.register(cache_app)
serve_cmd.register(cli)

serve = serve_cmd.serve
