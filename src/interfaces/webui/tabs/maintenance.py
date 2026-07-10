from __future__ import annotations

import contextlib
from types import SimpleNamespace

import gradio as gr

from src.core.constants import BYTES_PER_MB
from src.core.workspace import CACHE_DIR_PATHS, cache_usage, run_purge_cycle
from src.interfaces.utils import format_cache_rows


def _refresh_cache_info() -> list[list]:
    return format_cache_rows(cache_usage())


def _estimate_purge_size(targets: list[str]) -> float:
    """Return total MB in selected cache directories (force-clear estimate, no retention filter)."""
    total_bytes = 0
    for dir_name in targets or CACHE_DIR_PATHS:
        dir_path = CACHE_DIR_PATHS.get(dir_name)
        if not dir_path or not dir_path.exists():
            continue
        for path in dir_path.rglob("*"):
            if path.is_file():
                with contextlib.suppress(OSError):
                    total_bytes += path.stat().st_size
    return total_bytes / BYTES_PER_MB


def _run_purge(targets: list[str], dry_run: bool) -> tuple[list[list], str]:
    if dry_run:
        freed = _estimate_purge_size(targets)
        summary = (
            f"Would free approximately {freed:.2f} MB (estimated)."
            if freed > 0.01
            else "No files to clean."
        )
        return _refresh_cache_info(), summary
    estimated = _estimate_purge_size(targets)
    run_purge_cycle(force=True, specific_target=targets or None)
    summary = f"Freed approximately {estimated:.2f} MB."
    return _refresh_cache_info(), summary


def build_maintenance_tab() -> SimpleNamespace:
    """Build the Maintenance tab UI and its internal (self-contained) wiring."""
    with gr.Tab("Maintenance") as maintenance_tab:
        gr.Markdown("## Cache Management")
        usage_table = gr.Dataframe(
            value=_refresh_cache_info(),
            headers=["Directory", "Size (MB)", "Files", "Oldest"],
            datatype=["str", "number", "number", "str"],
            column_count=(4, "fixed"),
            row_count=10,
        )
        refresh_btn = gr.Button("⟳ Refresh Cache Info")
        purge_targets = gr.CheckboxGroup(
            [
                ("Clips", "clips"),
                ("Videos", "videos"),
                ("Audios", "audios"),
                ("Subtitles", "subtitles"),
                ("Data", "data"),
                ("Temp", "tmp"),
                ("Logs", "logs"),
            ],
            label="Select Directories to Clean",
            value=["tmp"],
        )
        with gr.Row():
            dry_run_cb = gr.Checkbox(
                label="Dry-Run Clean (Preview Only, No Deletion)", value=True
            )
            clear_btn = gr.Button("🗑️ Cache Clean", variant="stop")
        result_log = gr.Textbox(label="Result", lines=4, interactive=False)
        maintenance_tab.select(fn=_refresh_cache_info, outputs=[usage_table])
        refresh_btn.click(fn=_refresh_cache_info, outputs=[usage_table])
        clear_btn.click(
            fn=_run_purge,
            inputs=[purge_targets, dry_run_cb],
            outputs=[usage_table, result_log],
        )

    return SimpleNamespace(
        tab=maintenance_tab,
        usage_table=usage_table,
        result_log=result_log,
    )
