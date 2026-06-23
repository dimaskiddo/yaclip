from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

from loguru import logger

from src.core.config import load_config
from src.core.workspace import LOGS_DIR


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and route them to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger() -> None:
    """Configure loguru dynamically from config."""
    logger.remove()

    # Route standard python logging to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

    # Silence noisy third-party loggers
    for name in ["urllib3", "httpx", "httpcore", "huggingface_hub", "faster_whisper"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Route standard warnings into loguru
    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object = None,
        line: str | None = None,
    ) -> None:
        logger.warning(f"{category.__name__}: {message}")

    warnings.showwarning = showwarning

    try:
        config = load_config()
        log_config = config.logging
        level = log_config.level
        file_path = log_config.file_path
        rotation = log_config.rotation
        retention = log_config.retention
    except Exception:
        # Hardcoded as last-resort fallback when config.yaml is missing.
        # Must match LoggingConfig Field defaults in config.py.
        level = "INFO"
        file_path = str(LOGS_DIR / "app.log")
        rotation = "50 MB"
        retention = "7 days"

    log_path = Path(file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _pad_source(record: "logger.Record") -> None:
        """Pad source location to x chars for aligned block display."""
        source = f"{record['name']}:{record['function']}:{record['line']}"
        record["extra"]["source_padded"] = source.ljust(65)

    logger.configure(patcher=_pad_source)

    fmt_stdout = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[source_padded]}</cyan> | <level>{message}</level>"
    fmt_file = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[source_padded]} | {message}"

    logger.add(sys.stdout, format=fmt_stdout, level=level)
    logger.add(
        log_path,
        format=fmt_file,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )
