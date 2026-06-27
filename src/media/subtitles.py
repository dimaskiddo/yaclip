from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.core.config import load_config
from src.core.utils import SystemUtils


class SubtitleGenerator:
    """Processes word-level transcription metadata into styled .ass karaoke subtitles."""

    def __init__(self) -> None:
        self.config = load_config()

    def generate_ass(
        self,
        transcript_segments: list[dict],
        output_ass_path: Path,
        clip_start: float = 0.0,
    ) -> bool:
        """Translate segments with word-level timestamps into an .ass file.

        Args:
            transcript_segments: Word-level segment metadata from STT engine.
            output_ass_path: Destination path for the .ass subtitle file.
            clip_start: Start time of the clip relative to the source video.

        Returns:
            True if generated successfully, False otherwise.
        """
        sub_cfg = self.config.video_processing.subtitles
        if not sub_cfg.enabled:
            logger.info("Subtitles are disabled in config. Skipping subtitle file generation.")
            return False

        logger.info(f"Generating subtitle file at: {SystemUtils.display_path(output_ass_path)}")

        force_upper = sub_cfg.uppercase
        words = self._extract_flat_words(transcript_segments, force_upper)
        if not words:
            return False

        timing = sub_cfg.timing
        clip_words = self._filter_and_shift_words(
            words,
            clip_start,
            timing.word_duration_min_ms / 1000.0,
            timing.word_duration_max_ms / 1000.0,
        )
        if not clip_words:
            logger.warning("No words found within clip boundaries for subtitles.")
            return False

        clip_words = self._collapse_repeats(clip_words)
        lines = self._group_into_lines(clip_words, timing.line_max_words, timing.line_gap_seconds)

        ass_lines = self._build_ass_header()
        secondary_color = sub_cfg.highlight_color
        gap_smooth = timing.gap_smooth_ms / 1000.0
        ass_lines.extend(self._build_karaoke_dialogues(lines, gap_smooth, secondary_color))

        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_lines) + "\n")

        logger.info(f"Subtitle file generated successfully ({len(lines)} subtitle lines).")
        return True

    def _extract_flat_words(self, transcript_segments: list[dict], force_upper: bool) -> list[dict]:
        """Extract all word-level entries from segments, uppercasing when configured."""
        words = []
        for segment in transcript_segments:
            seg_words = segment.get("words", [])
            text = segment.get("text", "").strip()
            if not seg_words:
                words.append(
                    {
                        "word": text.upper() if force_upper else text,
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                    }
                )
            else:
                for w in seg_words:
                    wt = w.get("word", "").strip()
                    words.append(
                        {
                            "word": wt.upper() if force_upper else wt,
                            "start": w.get("start", 0.0),
                            "end": w.get("end", 0.0),
                        }
                    )
        return words

    def _filter_and_shift_words(
        self,
        words: list[dict],
        clip_start: float,
        word_duration_min: float,
        word_duration_max: float,
    ) -> list[dict]:
        """Shift timestamps relative to clip start and drop degenerate words."""
        clip_words = []
        for w in words:
            rel_start = w["start"] - clip_start
            rel_end = w["end"] - clip_start
            dur = rel_end - rel_start
            if dur <= 0 or not (word_duration_min <= dur <= word_duration_max):
                continue
            if rel_start >= -0.5:
                clip_words.append(
                    {
                        "word": w["word"],
                        "start": max(0.0, rel_start),
                        "end": max(0.0, rel_end),
                    }
                )
        return clip_words

    def _group_into_lines(
        self, clip_words: list[dict], line_max_words: int, line_gap_seconds: float
    ) -> list[list[dict]]:
        """Group words into subtitle lines, splitting by count or gap."""
        lines = []
        current: list[dict] = []
        for w in clip_words:
            if not current:
                current.append(w)
            else:
                gap = w["start"] - current[-1]["end"]
                if len(current) >= line_max_words or gap > line_gap_seconds:
                    lines.append(current)
                    current = [w]
                else:
                    current.append(w)
        if current:
            lines.append(current)
        return lines

    def _build_ass_header(self) -> list[str]:
        """ASS script + style header lines from config."""
        sub_cfg = self.config.video_processing.subtitles
        font_name = sub_cfg.font_file
        font_size = sub_cfg.font_size
        primary_color = sub_cfg.primary_color
        outline_color = sub_cfg.outline_color
        outline_thick = sub_cfg.outline_thickness
        bold = int(sub_cfg.bold)
        shadow = int(sub_cfg.shadow)
        alignment = sub_cfg.alignment
        margin_v = sub_cfg.margin_v
        secondary_color = sub_cfg.highlight_color

        return [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_name},{font_size},{primary_color},{secondary_color},{outline_color},&H00000000,{bold},0,0,0,100,100,0,0,1,{outline_thick},{shadow},{alignment},10,10,{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

    def _build_karaoke_dialogues(
        self, lines: list[list[dict]], gap_smooth: float, secondary_color: str
    ) -> list[str]:
        """Word-by-word karaoke Dialogue events — one event per word advances focus."""
        events = []
        for line_words in lines:
            line_end = line_words[-1]["end"]
            for i, active in enumerate(line_words):
                seg_start = active["start"]
                next_start = line_words[i + 1]["start"] if i + 1 < len(line_words) else line_end
                seg_end = next_start if next_start - active["end"] <= gap_smooth else active["end"]
                if seg_end <= seg_start:
                    seg_end = seg_start + 0.05

                parts = []
                for j, w in enumerate(line_words):
                    if j == i:
                        parts.append(
                            f"{{\\b1\\fscx112\\fscy112\\c{secondary_color}}}{w['word']}{{\\r}}"
                        )
                    else:
                        parts.append(w["word"])
                line_text = " ".join(parts)
                events.append(
                    f"Dialogue: 0,{self._format_ass_time(seg_start)},"
                    f"{self._format_ass_time(seg_end)},Default,,0,0,0,,{line_text}"
                )
                if seg_end < next_start:
                    gap_text = " ".join(w["word"] for w in line_words)
                    events.append(
                        f"Dialogue: 0,{self._format_ass_time(seg_end)},"
                        f"{self._format_ass_time(next_start)},Default,,0,0,0,,{gap_text}"
                    )
        return events

    @staticmethod
    def _norm(word: str) -> str:
        """Normalise a word for repeat comparison: lowercase, alphanumerics only."""
        return "".join(ch for ch in word.lower() if ch.isalnum())

    def _collapse_repeats(self, words: list[dict]) -> list[dict]:
        """Merge consecutive identical words into one spanning the whole run."""
        timing = self.config.video_processing.subtitles.timing
        min_repeats = timing.collapse_min_repeats
        collapsed: list[dict] = []
        i = 0
        while i < len(words):
            w = words[i]
            key = self._norm(w["word"])
            if not key:
                collapsed.append(dict(w))
                i += 1
                continue
            # Scan forward for consecutive identical words
            run_end = i
            while run_end + 1 < len(words) and self._norm(words[run_end + 1]["word"]) == key:
                run_end += 1
            run_len = run_end - i + 1
            if run_len >= min_repeats:
                entry = dict(w)
                entry["end"] = words[run_end]["end"]  # span the whole run
                collapsed.append(entry)
            else:
                for j in range(i, run_end + 1):
                    collapsed.append(dict(words[j]))
            i = run_end + 1
        return collapsed

    def _format_ass_time(self, seconds: float) -> str:
        """Convert float seconds to ASS timestamp format (H:MM:SS.cs)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        if cs >= 100:
            s += 1
            cs = 0
            if s >= 60:
                m += 1
                s = 0
                if m >= 60:
                    h += 1
                    m = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
