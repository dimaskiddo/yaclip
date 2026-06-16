import sys
import logging
import warnings

from pathlib import Path
from loguru import logger

from src.core.config import load_config


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages and route them to loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger() -> None:
    """Configure loguru dynamically from config."""
    logger.remove()

    # Route standard python logging to loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

    # Silence noisy third-party loggers
    for name in ["urllib3", "httpx", "httpcore", "huggingface_hub", "faster_whisper"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Route standard warnings into loguru
    def showwarning(message, category, filename, lineno, file=None, line=None):
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
        from src.core.workspace import LOGS_DIR
        level = "INFO"
        file_path = str(LOGS_DIR / "app.log")
        rotation = "50 MB"
        retention = "7 days"

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )

    log_path = Path(file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
    )
