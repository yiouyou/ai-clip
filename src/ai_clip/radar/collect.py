from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from ai_clip.core.config import RadarConfig
from ai_clip.core.models import Platform
from ai_clip.discover.base import age_days_from
from ai_clip.radar.models import (
    ChannelCollectResult,
    ChannelSpec,
    RadarCollectReport,
    RadarSnapshot,
    RadarVideo,
)


class RadarCollectError(RuntimeError):
    pass


def load_channels(path: str | Path) -> list[ChannelSpec]:
    p = Path(path)
    if not p.exists():
        raise RadarCollectError(
            f"channel config not found: {p}. Create it from config/channels.example.yaml"
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [ChannelSpec.model_validate(item) for item in raw.get("channels", [])]


def collect_channels(
    channels: list[ChannelSpec],
    radar_cfg: RadarConfig,
    collected_at: datetime | None = None,
) -> list[RadarSnapshot]:
    return collect_channels_with_diagnostics(channels, radar_cfg, collected_at).snapshots


def collect_channels_with_diagnostics(
    channels: list[ChannelSpec],
    radar_cfg: RadarConfig,
    collected_at: datetime | None = None,
) -> RadarCollectReport:
    collected_at = collected_at or datetime.now(timezone.utc)
    snapshots: list[RadarSnapshot] = []
    results: list[ChannelCollectResult] = []
    seen: set[str] = set()
    for channel in channels:
        videos, result = collect_channel_with_diagnostics(channel, radar_cfg)
        results.append(result)
        for video in videos:
            if video.video_id in seen:
                continue
            seen.add(video.video_id)
            snapshots.append(RadarSnapshot(collected_at=collected_at.isoformat(), video=video))
    return RadarCollectReport(
        collected_at=collected_at.isoformat(),
        snapshots=snapshots,
        channels=results,
    )


def collect_channel_with_diagnostics(
    channel: ChannelSpec,
    radar_cfg: RadarConfig,
) -> tuple[list[RadarVideo], ChannelCollectResult]:
    start = time.monotonic()
    try:
        videos = _collect_channel(channel, radar_cfg)
    except Exception as exc:
        return [], ChannelCollectResult(
            platform=channel.platform,
            url=channel.url,
            name=channel.name,
            status="failed",
            duration_sec=_elapsed(start),
            error=_sanitize_error(exc),
        )
    return videos, ChannelCollectResult(
        platform=channel.platform,
        url=channel.url,
        name=channel.name,
        status="succeeded",
        count=len(videos),
        duration_sec=_elapsed(start),
        video_ids=[video.video_id for video in videos],
    )


def _collect_channel(channel: ChannelSpec, radar_cfg: RadarConfig) -> list[RadarVideo]:
    if channel.platform not in (Platform.youtube, Platform.bilibili):
        raise RadarCollectError(f"unsupported radar platform: {channel.platform}")

    import yt_dlp  # noqa: PLC0415

    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "playlistend": radar_cfg.channel_limit,
        "extract_flat": False,
        "ignoreerrors": True,
        "socket_timeout": 10,
        "retries": 1,
        "extractor_retries": 1,
    }
    if channel.cookies:
        opts["cookiefile"] = channel.cookies

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(_listing_url(channel), download=False)

    entries = info.get("entries", [info]) if info else []
    videos = []
    for entry in entries:
        if not entry:
            continue
        video = entry_to_video(entry, channel)
        max_duration_sec = (
            channel.max_duration_sec
            if channel.max_duration_sec is not None
            else radar_cfg.max_duration_sec
        )
        if max_duration_sec and video.duration_sec > max_duration_sec:
            continue
        if radar_cfg.since_days and video.age_days > radar_cfg.since_days:
            continue
        videos.append(video)
    return videos


def entry_to_video(entry: dict, channel: ChannelSpec) -> RadarVideo:
    url = str(entry.get("webpage_url") or entry.get("url") or entry.get("id") or "")
    if url and not url.startswith("http"):
        if channel.platform == Platform.youtube:
            url = f"https://www.youtube.com/watch?v={url}"
        elif channel.platform == Platform.bilibili:
            url = f"https://www.bilibili.com/video/{url}"
    video_id = str(entry.get("id") or "") or _stable_id(url)
    return RadarVideo(
        video_id=f"{channel.platform}:{video_id}",
        url=url,
        platform=channel.platform,
        channel_url=channel.url,
        channel_name=channel.name,
        pool=channel.pool,
        role=channel.role,
        title=str(entry.get("title") or ""),
        uploader=str(entry.get("uploader") or entry.get("channel") or channel.name),
        tags=list(channel.tags),
        priority=float(channel.priority or 1.0),
        lens_fit=float(channel.lens_fit or 1.0),
        duration_sec=float(entry.get("duration") or 0.0),
        published_date=published_date_from(entry.get("timestamp"), entry.get("upload_date")),
        age_days=age_days_from(entry.get("timestamp"), entry.get("upload_date")),
        view_count=_int(entry.get("view_count")),
        like_count=_int(entry.get("like_count")),
        comment_count=_int(entry.get("comment_count")),
        share_count=_optional_int(entry.get("repost_count") or entry.get("share_count")),
        favorite_count=_optional_int(entry.get("favorite_count") or entry.get("favorites")),
        coin_count=_optional_int(entry.get("coin_count")),
        danmaku_count=_optional_int(entry.get("danmaku_count") or entry.get("bullet_comments")),
    )


def _listing_url(channel: ChannelSpec) -> str:
    url = channel.url.rstrip("/")
    if channel.platform == Platform.youtube and "/watch?" not in url and not url.endswith("/videos"):
        return f"{url}/videos"
    if channel.platform == Platform.bilibili and "space.bilibili.com" in url and not url.endswith("/video"):
        return f"{url}/video"
    return channel.url


def _stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def published_date_from(timestamp: int | None, upload_date: str | None) -> str:
    if timestamp:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
    if upload_date and len(upload_date) == 8:
        try:
            return datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
        except ValueError:
            return ""
    age_days = age_days_from(timestamp, upload_date)
    if age_days:
        return (datetime.now(timezone.utc).date() - timedelta(days=round(age_days))).isoformat()
    return ""


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _int(value)


def _elapsed(start: float) -> float:
    return round(max(time.monotonic() - start, 0.0), 3)


def _sanitize_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(r"(?i)(api[_-]?key|authorization|cookie|token)=\S+", r"\1=<redacted>", text)
    return text[:500]


