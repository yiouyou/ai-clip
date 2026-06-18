"""Download stage: wrap yt-dlp into a Clip artifact.

yt-dlp is imported lazily so the package (and tests) load without it installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ai_clip.core.models import Clip, Platform


def detect_platform(url: str) -> Platform:
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return Platform.youtube
    if "bilibili.com" in u or "b23.tv" in u:
        return Platform.bilibili
    if "douyin.com" in u:
        return Platform.douyin
    if "kuaishou.com" in u or "kuaishou.cn" in u:
        return Platform.kuaishou
    if "tiktok.com" in u:
        return Platform.tiktok
    return Platform.unknown


def make_clip_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def download(url: str, dest_dir: str | Path, clip_id: str | None = None) -> Clip:
    """Download `url` into dest_dir and return a Clip. Picks best mp4-friendly
    streams and merges via ffmpeg (yt-dlp handles that)."""
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "yt-dlp not installed. Install with: pip install 'ai-clip[download]'"
        ) from exc

    clip_id = clip_id or make_clip_id(url)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest / f"{clip_id}.%(ext)s")

    opts = {
        "outtmpl": outtmpl,
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_path = dest / f"{clip_id}.mp4"
    if not video_path.exists():
        # Fall back to whatever extension yt-dlp produced.
        matches = list(dest.glob(f"{clip_id}.*"))
        if not matches:
            raise RuntimeError(f"download produced no file for {url}")
        video_path = matches[0]

    return Clip(
        clip_id=clip_id,
        source_url=url,
        platform=detect_platform(url),
        video_path=str(video_path),
        title=info.get("title", ""),
        duration_sec=float(info.get("duration") or 0.0),
        meta={
            "uploader": info.get("uploader"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
        },
    )
