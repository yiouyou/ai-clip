"""Best-effort discover for douyin / kuaishou.

Reality: yt-dlp can list a *user/channel* page and fetch single URLs for these
platforms, but there is no reliable keyword search API. So:
  - with --channel <user page URL>: list and rank that creator's recent clips
  - keyword only: raise with guidance (use --channel, or pass a direct URL to
    `ai-clip download`).
These extractors are brittle and may break; failures surface clearly rather than
silently returning nothing.
"""

from __future__ import annotations

from ai_clip.core.models import Candidate, Platform
from ai_clip.discover.base import age_days_from


def _entry_to_candidate(entry: dict, platform: Platform) -> Candidate:
    url = entry.get("webpage_url") or entry.get("url") or ""
    return Candidate(
        url=url,
        platform=platform,
        title=entry.get("title", ""),
        uploader=entry.get("uploader") or entry.get("channel") or "",
        view_count=int(entry.get("view_count") or 0),
        like_count=int(entry.get("like_count") or 0),
        comment_count=int(entry.get("comment_count") or 0),
        duration_sec=float(entry.get("duration") or 0.0),
        age_days=age_days_from(entry.get("timestamp"), entry.get("upload_date")),
    )


class _SocialProvider:
    platform: Platform = Platform.unknown

    def __init__(self, max_duration_sec: float = 90.0):
        self.max_duration_sec = max_duration_sec

    def search(
        self, topic: str, channel: str | None, since_days: int, limit: int
    ) -> list[Candidate]:
        if not channel:
            raise NotImplementedError(
                f"{self.platform} has no keyword search; pass --channel <user page "
                f"URL> to rank a creator's recent clips, or give a direct video URL "
                f"to `ai-clip download`. (topic={topic!r})"
            )
        import yt_dlp  # noqa: PLC0415

        opts = {"quiet": True, "no_warnings": True, "noprogress": True,
                "playlistend": limit}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel, download=False)
        except yt_dlp.utils.DownloadError as exc:
            # yt-dlp does not support douyin/kuaishou user-page listing (verified:
            # the /user/ URL returns "Unsupported URL"). Fail with clear guidance.
            raise NotImplementedError(
                f"{self.platform} channel listing is not supported by yt-dlp "
                f"({channel}). Pass a direct video URL to `ai-clip download` instead."
            ) from exc

        entries = info.get("entries", [info]) if info else []
        out: list[Candidate] = []
        for entry in entries:
            if not entry:
                continue
            cand = _entry_to_candidate(entry, self.platform)
            if self.max_duration_sec and cand.duration_sec > self.max_duration_sec:
                continue
            if since_days and cand.age_days > since_days:
                continue
            out.append(cand)
        return out


class DouyinProvider(_SocialProvider):
    platform = Platform.douyin


class KuaishouProvider(_SocialProvider):
    platform = Platform.kuaishou
