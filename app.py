#!/usr/bin/env python3

import sys

from src.core.config import load_config
from src.core.environment import setup_environment
from src.core.logger import setup_logger
from src.interfaces.cli import cli, serve


def main() -> None:
    """Entry point. Loads config, then routes: CLI args → Typer, bare invocation → WebUI."""
    # 0. Load and validate configuration (fail fast on a bad config.yaml).
    try:
        load_config()
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)

    # 1. Environment (PATH, HF_HOME) + 2. logging.
    setup_environment()
    setup_logger()

    # 3. Dual-interface routing: arguments → Typer CLI; no arguments → Gradio WebUI.
    if len(sys.argv) > 1:
        cli()
    else:
        serve(host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
