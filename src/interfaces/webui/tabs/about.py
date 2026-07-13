from __future__ import annotations

from types import SimpleNamespace

import gradio as gr


def build_about_tab() -> SimpleNamespace:
    """Build the About tab UI.

    Returns a ``SimpleNamespace`` with a ``tab`` attribute (the ``gr.Tab``
    context-manager object).
    """
    with gr.Tab("About") as about_tab:
        gr.Markdown(
            "## What's YaClip?\n\n"
            "**YouTube ➜ Your Shorts, Reels & TikToks — one shot.**\n\n"
            "YaClip is your AI video editor that sniffs out the best moments "
            "from any YouTube video — podcasts, gaming streams, or just-chillin' "
            "rants — and serves them up as polished 9:16 shorts, ready for "
            "YouTube Shorts, Instagram Reels, and TikTok.\n\n"
            "Think of it as having a clip-hunting, caption-writing, face-tracking "
            "robot sidekick. You bring the URL, it brings the highlight reel.\n\n"
            "---\n\n"
            "### ✦ Quick Links\n\n"
            "- **\U0001f3e0 Homepage** → [dimaskiddo.my.id](https://dimaskiddo.my.id)\n"
            "- **☕ Support Me** → [gift.trakteer.id/itsdrh](https://gift.trakteer.id/itsdrh)\n\n"
            "---\n\n"
            "### ⚡ Powered By\n\n"
            "**[Trakteer.ID](https://trakteer.id)** — *Where Creator and Supporter Met Together "
            "in One Place!*"
        )
        gr.Image(
            value="public/trakteer-logo.png",
            show_label=False,
            container=False,
            interactive=False,
            buttons=[],
            height=28,
            width=136,
        )

    return SimpleNamespace(tab=about_tab)
