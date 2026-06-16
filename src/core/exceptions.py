from __future__ import annotations


class YaClipError(Exception):
    """Base exception for all YaClip errors."""

    pass


class ConfigValidationError(YaClipError):
    """Raised when config.yaml is invalid or missing required fields."""

    pass


class DownloadError(YaClipError):
    """Raised when video downloading or cookie resolution fails."""

    pass


class DetectionError(YaClipError):
    """Raised when content/vision detection cannot run (e.g. missing system GL libraries)."""

    pass


class RenderError(YaClipError):
    """Raised when FFmpeg rendering or processing fails."""

    pass


class AIProviderError(YaClipError):
    """Raised when AI (cloud or local) fails or returns invalid format."""

    pass


class CacheInitError(YaClipError):
    """Raised when cache directories or binary resources cannot be initialized."""

    pass
