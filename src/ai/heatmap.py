import json

from typing import Dict, List
from loguru import logger

from src.core.config import load_config
from src.core.workspace import SUBTITLES_DIR


class HeatmapAnalyzer:
    """Parses the yt-dlp heatmap JSON file and extracts the most viral (replayed) segments."""

    def __init__(self) -> None:
        self.config = load_config()

    def analyze_heatmap(self, video_id: str) -> List[Dict]:
        """
        Parses the yt-dlp heatmap JSON file and extracts the most viral (replayed) segments.
        """
        heatmap_file = SUBTITLES_DIR / f"{video_id}_heatmap_youtube.json"
        if not heatmap_file.exists():
            logger.info(
                f"No Most Replayed data found for {video_id}. Skipping Most Replayed clip extraction."
            )
            return []

        try:
            heatmap_data = json.loads(heatmap_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to read Most Replayed data: {e}")
            return []

        if not heatmap_data:
            return []

        clip_cfg = self.config.clip_selection
        threshold_percentile = clip_cfg.heatmap_threshold_percentile

        # Calculate percentile without external dependencies like numpy
        values = sorted([item.get("value", 0.0) for item in heatmap_data])
        if not values:
            return []

        idx = int((threshold_percentile / 100.0) * (len(values) - 1))
        threshold_value = values[idx]
        logger.debug(
            f"Heatmap threshold (P{threshold_percentile}) calculated as: {threshold_value:.4f}"
        )

        # Sort by start_time to ensure correct adjacent-segment merging
        heatmap_data = sorted(heatmap_data, key=lambda x: x.get("start_time", 0.0))

        clips = []
        current_clip = None

        for item in heatmap_data:
            start = item.get("start_time", 0.0)
            end = item.get("end_time", 0.0)
            val = item.get("value", 0.0)

            if val >= threshold_value:
                if current_clip is None:
                    current_clip = {
                        "start_time": start,
                        "end_time": end,
                        "title": "Most Replayed Moment",
                        "reasoning": "Detected extremely high viewer replay activity (YouTube Most Replayed).",
                        "score": float(val),
                    }
                else:
                    current_clip["end_time"] = end
                    current_clip["score"] = max(current_clip["score"], float(val))
            else:
                if current_clip is not None:
                    clips.append(current_clip)
                    current_clip = None

        if current_clip is not None:
            clips.append(current_clip)

        min_duration = clip_cfg.min_clip_duration_seconds
        max_duration = clip_cfg.max_clip_duration_seconds

        valid_clips = []
        for clip in clips:
            duration = clip["end_time"] - clip["start_time"]

            if duration < min_duration:
                # Expand the clip equally on both sides to reach the minimum duration
                diff = min_duration - duration
                clip["start_time"] = max(0.0, clip["start_time"] - (diff / 2.0))
                clip["end_time"] = clip["end_time"] + (diff / 2.0)

            if (clip["end_time"] - clip["start_time"]) > max_duration:
                # Truncate if the spike is too long
                clip["end_time"] = clip["start_time"] + max_duration

            valid_clips.append(clip)

        if valid_clips:
            logger.info(f"Extracted {len(valid_clips)} clip(s) from YouTube Most Replayed data.")

        return valid_clips
