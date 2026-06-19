"""Bilibili discover provider via yt-dlp keyword search (`bilisearch`) / space listing.

Mirrors the YouTube provider: the info->Candidate mapping is a pure function so it
can be unit-tested without the network; search() does the yt-dlp I/O.
"""

from __future__ import annotations

from ai_clip.core.models import Candidate, Platform
from ai_clip.discover.base import age_days_from


def entry_to_candidate(entry: dict) -> Candidate:
    url = entry.get("webpage_url") or entry.get("url") or ""
    if url and not url.startswith("http"):
        url = f"https://www.bilibili.com/video/{url}"
    return Candidate(
        url=url,
        platform=Platform.bilibili,
        title=entry.get("title", ""),
        uploader=entry.get("uploader") or entry.get("channel") or "",
        view_count=int(entry.get("view_count") or 0),
        like_count=int(entry.get("like_count") or 0),
        comment_count=int(entry.get("comment_count") or 0),
        duration_sec=float(entry.get("duration") or 0.0),
        age_days=age_days_from(entry.get("timestamp"), entry.get("upload_date")),
    )


class BilibiliProvider:
    platform = "bilibili"

    def __init__(self, max_duration_sec: float = 90.0):
        self.max_duration_sec = max_duration_sec

    def _query(self, topic: str, channel: str | None, limit: int) -> str:
        if channel:
            return channel  # a space/user URL; yt-dlp lists their uploads
        return f"bilisearch{limit}:{topic}"

    def search(
        self, topic: str, channel: str | None, since_days: int, limit: int
    ) -> list[Candidate]:
        import yt_dlp  # noqa: PLC0415

        opts = {"quiet": True, "no_warnings": True, "noprogress": True,
                "playlistend": limit}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(self._query(topic, channel, limit), download=False)

        entries = info.get("entries", [info]) if info else []
        out: list[Candidate] = []
        for entry in entries:
            if not entry:
                continue
            cand = entry_to_candidate(entry)
            if self.max_duration_sec and cand.duration_sec > self.max_duration_sec:
                continue
            if since_days and cand.age_days > since_days:
                continue
            out.append(cand)
        return out
