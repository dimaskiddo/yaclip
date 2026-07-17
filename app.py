#!/usr/bin/env python3

import sys

from src.core.config import load_config
from src.core.environment import setup_environment
from src.core.logger import setup_logger
from src.interfaces.cli import cli
from src.interfaces.cli.commands.serve import _launch_webui


def main() -> None:
    """Entry point. Loads config, then routes: CLI args → Typer, bare invocation → WebUI."""
    try:
        load_config()
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)

    setup_environment()
    setup_logger()

    if len(sys.argv) > 1:
        cli()
    else:
        _launch_webui()


if __name__ == "__main__":
    main()
