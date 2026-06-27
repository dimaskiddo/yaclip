from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")


def require_package(import_name: str, pip_name: str) -> None:
    """Raise a clear ImportError when an optional dependency is missing.

    Call from inside an ``except ImportError`` block:
        try:
            import google.generativeai as genai
        except ImportError as e:
            require_package("google.generativeai", "google-generativeai")
            raise ImportError("unreachable") from e
    Kept simple (logs + raises) rather than returning the module — the caller still owns its
    own ``import ... as alias`` line so static type checkers can resolve the symbol.
    """
    logger.error(f"{import_name} package is not installed.")
    raise ImportError(f"{pip_name} package missing.")


def make_openai_client(api_key: str | None, base_url: str | None, timeout: float) -> Any:
    """Build an OpenAI SDK client with the shared per-channel timeout + retry settings.

    Args:
        api_key: API key for the configured provider.
        base_url: Optional custom endpoint (OpenRouter / self-hosted proxies).
        timeout: Seconds applied to connect/read/write/pool.

    Returns:
        A configured ``openai.OpenAI`` client.

    Raises:
        ImportError: When the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        logger.error("openai package is not installed.")
        raise ImportError("openai package missing.") from e

    from httpx import Timeout as HTTPXTimeout

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=HTTPXTimeout(connect=timeout, read=timeout, write=timeout, pool=timeout),
        max_retries=3,
    )


def gemini_generate(
    model_name: str,
    contents: Any,
    timeout: float,
    system_instruction: str | None = None,
) -> str:
    """Stream a Gemini ``generate_content`` call and join the text chunks.

    Retries on transient failure via ``retry_api_call``. Shared by the cloud LLM/STT providers
    and the unified-Gemini pipeline path — only ``contents``/``system_instruction`` vary.

    Args:
        model_name: Gemini model name (e.g. "gemini-2.5-flash").
        contents: Prompt string, or a list mixing an uploaded file + prompt string.
        timeout: Seconds for both the request timeout and the google-api-core retry deadline.
        system_instruction: Optional system prompt (unified pipeline path only).

    Returns:
        The concatenated streamed response text.

    Raises:
        ImportError: When ``google-generativeai`` is not installed.
    """
    try:
        import google.generativeai as genai
    except ImportError as e:
        logger.error("google-generativeai package is not installed.")
        raise ImportError("google-generativeai package missing.") from e

    @retry_api_call(max_retries=3)
    def _generate() -> str:
        from google.api_core import retry as google_retry

        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
        stream = model.generate_content(
            contents,
            stream=True,
            request_options={
                "timeout": timeout,
                "retry": google_retry.Retry(deadline=timeout),
            },
        )
        chunks: list[str] = []
        for chunk in stream:
            if chunk.text:
                chunks.append(chunk.text)
        return "".join(chunks)

    return _generate()


def gemini_upload_and_wait(audio_path: str) -> Any:
    """Upload a file to Gemini and block until it leaves the PROCESSING state.

    Args:
        audio_path: Path to the local file to upload.

    Returns:
        The Gemini file handle once processing succeeds.

    Raises:
        ImportError: When ``google-generativeai`` is not installed.
        RuntimeError: When Gemini reports the upload as FAILED.
    """
    import google.generativeai as genai

    @retry_api_call(max_retries=3)
    def _upload() -> Any:
        return genai.upload_file(path=audio_path)

    uploaded_file = _upload()
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded file.")
    return uploaded_file


def gemini_delete_quiet(uploaded_file: Any) -> None:
    """Delete an uploaded Gemini file, logging (not raising) on failure.

    Intended for a ``finally`` block — cleanup should never mask the caller's real error.
    """
    import google.generativeai as genai

    try:
        genai.delete_file(uploaded_file.name)
    except Exception as e:
        logger.warning(f"Failed to delete file {uploaded_file.name}: {e}")


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
