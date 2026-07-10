"""Shared pure helpers used by both CLI and WebUI interface modules.

No Gradio, Typer, or Gradio-component dependencies — these are pure functions
and constants that can be imported from either interface without framework coupling.
"""

from __future__ import annotations

from pathlib import Path

# ── Cache-management display constants ──────────────────────────────────────

CACHE_DIR_SORT_ORDER: dict[str, int] = {
    "clips": 0,
    "videos": 1,
    "audios": 2,
    "subtitles": 3,
    "data": 4,
    "tmp": 5,
    "logs": 6,
}

CACHE_DIR_LABEL: dict[str, str] = {
    "clips": "Clips",
    "videos": "Videos",
    "audios": "Audios",
    "subtitles": "Subtitles",
    "data": "Data",
    "tmp": "Temp",
    "logs": "Logs",
}


def format_oldest_days(days: float | None) -> str:
    """Format a cache ``oldest_days`` value for human display.

    Used by both the CLI ``cache status`` table and the WebUI Maintenance
    tab Dataframe.
    """
    return f"{days:.1f}d" if days is not None else "-"


def format_cache_rows(rows: list[dict]) -> list[list]:
    """Sort raw ``cache_usage()`` rows and convert to display rows.

    Each output row is ``[label, size_mb, count, oldest_str]`` — the same
    shape consumed by both ``typer.echo()`` and ``gr.Dataframe``.
    """
    sorted_rows = sorted(rows, key=lambda r: CACHE_DIR_SORT_ORDER.get(r["name"], 99))
    return [
        [
            CACHE_DIR_LABEL.get(r["name"], r["name"].title()),
            round(r["size_mb"], 2),
            r["count"],
            format_oldest_days(r["oldest_days"]),
        ]
        for r in sorted_rows
    ]


# ── Config-key masking ──────────────────────────────────────────────────────


def mask_config_keys(node: object, key_name: str = "api_key") -> object:
    """Recursively mask all values whose dictionary key is ``key_name``.

    Handles nested dicts and lists. Replaces the CLI ``config`` command's
    local ``_mask()`` helper and can be reused in any interface that needs
    to render a configuration tree with secrets masked.
    """
    if isinstance(node, dict):
        return {
            k: ("***" if k == key_name and v else mask_config_keys(v, key_name))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [mask_config_keys(v, key_name) for v in node]
    return node


# ── Rendered-clip metadata sidecar helpers ─────────────────────────────────


def read_clip_sidecar(clip_path: str) -> str:
    """Read the ``.txt`` metadata sidecar next to a rendered clip.

    The sidecar contains ``Title:/Caption:/Description:/Hashtags:`` lines
    written by ``ClipRenderer._write_clip_metadata()``.
    """
    sidecar = Path(str(clip_path).replace(".mp4", ".txt"))
    if not sidecar.exists():
        return "(no metadata sidecar)"
    return sidecar.read_text(encoding="utf-8")


def parse_clip_sidecar(clip_path: str) -> dict[str, str]:
    """Return key→value pairs from a rendered clip's ``.txt`` sidecar."""
    raw = read_clip_sidecar(clip_path)
    if raw == "(no metadata sidecar)":
        return {}
    result: dict[str, str] = {}
    for line in raw.strip().split("\n"):
        if ": " in line:
            key, _, value = line.partition(": ")
            result[key.lower()] = value.strip()
    return result
