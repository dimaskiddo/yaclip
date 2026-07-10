from __future__ import annotations

import typer
import yaml

from src.core.config import load_config
from src.interfaces.utils import mask_config_keys


def register(cli: typer.Typer) -> None:
    @cli.command("config")
    def show_config() -> None:
        """Show the current configuration (API keys masked).

        Displays all settings from ``config.yaml`` with secrets hidden
        (``***``) so you can inspect or share your setup safely.
        """
        data = load_config().model_dump()
        typer.echo(
            yaml.safe_dump(mask_config_keys(data), sort_keys=False, allow_unicode=True)
        )
