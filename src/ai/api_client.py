from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


def retry_api_call(
    max_retries: int = 3, initial_delay: float = 2.0, backoff_factor: float = 2.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry API calls on transient errors with exponential backoff.
    Protects cloud network requests from rate limits and temporary outages.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    err_name = type(e).__name__
                    if attempt == max_retries:
                        break

                    logger.warning(
                        f"Cloud service request failed (attempt {attempt}/{max_retries}): {err_name}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

            logger.error(f"Cloud service request failed after {max_retries} attempts. Giving up.")
            raise last_err

        return wrapper

    return decorator
