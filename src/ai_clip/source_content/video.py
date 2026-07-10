from __future__ import annotations

from pathlib import Path
import re

from ai_clip.core.config import WhisperConfig
from ai_clip.extract.remote import (
    VideoScript,
    VideoScriptResult,
    download_video_audio,
    fetch_video_script,
    fetch_video_script_report,
    fetch_video_subtitles,
    transcribe_video_audio,
)
from ai_clip.radar.models import ChannelSpec, RadarVideo

__all__ = [
    "VideoScript",
    "VideoScriptResult",
    "add_source_content",
    "download_video_audio",
    "fetch_video_script",
    "fetch_video_script_report",
    "fetch_video_subtitles",
    "transcribe_video_audio",
]


def add_source_content(
    videos: list[RadarVideo],
    channels: list[ChannelSpec],
    out_dir: Path,
    whisper: WhisperConfig | None = None,
    transcribe_missing: bool = True,
) -> list[RadarVideo]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cookies_by_channel = {channel.url: channel.cookies for channel in channels if channel.cookies}
    enriched = []
    for video in videos:
        result = fetch_video_script_report(
            video.url,
            out_dir / _safe_name(video.video_id),
            cookies_by_channel.get(video.channel_url, ""),
            whisper=whisper,
            transcribe_missing=transcribe_missing,
        )
        update = {
            "content_status": result.status,
            "content_error": result.error,
            "content_cache_path": result.cache_path,
            "content_attempts": list(result.attempts),
        }
        if result.script is not None:
            update.update({
                "transcript_text": result.script.text,
                "transcript_language": result.script.language,
                "transcript_source": result.script.source,
                "transcript_segments": result.script.segments,
            })
        enriched.append(video.model_copy(update=update))
    return enriched


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
