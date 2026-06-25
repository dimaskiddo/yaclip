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
        # Exclude subtitles if disabled in config
        if not self.config.video_processing.subtitles.enabled:
            logger.info("Subtitles are disabled in config. Skipping subtitle file generation.")
            return False

        logger.info(f"Generating subtitle file at: {SystemUtils.display_path(output_ass_path)}")

        # Uppercase improves caption readability at a glance (configurable).
        force_upper = self.config.video_processing.subtitles.uppercase

        def _case(text: str) -> str:
            return text.upper() if force_upper else text

        # Extract all words across all segments
        words = []
        for segment in transcript_segments:
            seg_words = segment.get("words", [])
            if not seg_words:
                # Fallback if no word-level timestamps are present: treat segment as a single word
                words.append(
                    {
                        "word": _case(segment.get("text", "").strip()),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                    }
                )
            else:
                for w in seg_words:
                    words.append(
                        {
                            "word": _case(w.get("word", "").strip()),
                            "start": w.get("start", 0.0),
                            "end": w.get("end", 0.0),
                        }
                    )

        # Filter words to only include those in the clip boundary
        # Wait, if we pass transcript segments already trimmed to the clip start,
        # then we just subtract clip_start.
        timing = self.config.video_processing.subtitles.timing
        min_dur = timing.word_duration_min_ms / 1000.0
        max_dur = timing.word_duration_max_ms / 1000.0

        clip_words = []
        for w in words:
            # Shift timestamps relative to clip start
            rel_start = w["start"] - clip_start
            rel_end = w["end"] - clip_start

            # Defensive word duration/impossible timing check
            dur = rel_end - rel_start
            if dur <= 0 or not (min_dur <= dur <= max_dur):
                continue

            # Keep words that fall within the clip (rel_start >= 0)
            if rel_start >= -0.5:  # Allow slight overlap at start
                clip_words.append(
                    {
                        "word": w["word"],
                        "start": max(0.0, rel_start),
                        "end": max(0.0, rel_end),
                    }
                )

        if not clip_words:
            logger.warning("No words found within clip boundaries for subtitles.")
            return False

        # Collapse runs of consecutive identical words (case/punctuation-insensitive). Whisper
        # transcribes laughter/filler as repeated tokens ("HEHEHE HEHEHE HEHEHE"); show it once,
        # extending the first occurrence to cover the whole run. Normal speech is untouched.
        clip_words = self._collapse_repeats(clip_words)

        # Group words into lines (max timing.line_max_words per line, or split if gap > timing.line_gap_seconds)
        lines = []
        current_line: list[dict] = []
        for w in clip_words:
            if not current_line:
                current_line.append(w)
            else:
                gap = w["start"] - current_line[-1]["end"]
                if len(current_line) >= timing.line_max_words or gap > timing.line_gap_seconds:
                    lines.append(current_line)
                    current_line = [w]
                else:
                    current_line.append(w)
        if current_line:
            lines.append(current_line)

        # Build ASS contents
        sub_cfg = self.config.video_processing.subtitles
        font_name = sub_cfg.font_file
        font_size = sub_cfg.font_size
        primary_color = sub_cfg.primary_color
        outline_color = sub_cfg.outline_color
        outline_thick = sub_cfg.outline_thickness
        bold = int(sub_cfg.bold)  # config exposes true/false; ASS needs 0/1
        shadow = int(sub_cfg.shadow)
        alignment = sub_cfg.alignment
        margin_v = sub_cfg.margin_v

        # Active-word highlight colour (configurable; default soft blue, eye-friendly).
        secondary_color = sub_cfg.highlight_color

        ass_lines = [
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

        # Word-by-word focus: render the whole phrase, with the currently-spoken word bold +
        # highlight-coloured, the rest normal. One Dialogue event per word advances the focus.
        gap_smooth = timing.gap_smooth_ms / 1000.0
        for line_words in lines:
            line_end = line_words[-1]["end"]
            for i, active in enumerate(line_words):
                seg_start = active["start"]
                next_start = line_words[i + 1]["start"] if i + 1 < len(line_words) else line_end

                # Bridge pause if <= gap_smooth, else end highlight at the actual word end
                seg_end = next_start if next_start - active["end"] <= gap_smooth else active["end"]

                if seg_end <= seg_start:
                    seg_end = seg_start + 0.05

                parts = []
                for j, w in enumerate(line_words):
                    if j == i:
                        # Active word pops: bold + 12% larger + highlight colour, then revert.
                        parts.append(
                            f"{{\\b1\\fscx112\\fscy112\\c{secondary_color}}}{w['word']}{{\\r}}"
                        )
                    else:
                        parts.append(w["word"])
                line_text = " ".join(parts)

                ass_lines.append(
                    f"Dialogue: 0,{self._format_ass_time(seg_start)},"
                    f"{self._format_ass_time(seg_end)},Default,,0,0,0,,{line_text}"
                )

                # If there's a significant gap after this word, add a non-highlighted frame
                # for the duration of the silence so the highlight doesn't linger on a silent word.
                if seg_end < next_start:
                    gap_text = " ".join(w["word"] for w in line_words)
                    ass_lines.append(
                        f"Dialogue: 0,{self._format_ass_time(seg_end)},"
                        f"{self._format_ass_time(next_start)},Default,,0,0,0,,{gap_text}"
                    )

        # Ensure directory exists and write
        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_lines) + "\n")

        logger.info(f"Subtitle file generated successfully ({len(lines)} subtitle lines).")
        return True

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
