"""Discover provider protocol + shared age helper."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from ai_clip.core.models import Candidate


class DiscoverProvider(Protocol):
    platform: str

    def search(
        self, topic: str, channel: str | None, since_days: int, limit: int
    ) -> list[Candidate]:
        """Return candidates matching the topic (optionally within a channel),
        published within since_days. virality is filled by the caller."""
        ...


def age_days_from(timestamp: int | None, upload_date: str | None) -> float:
    """Compute age in days from a unix timestamp or a YYYYMMDD upload_date."""
    now = datetime.now(timezone.utc)
    dt: datetime | None = None
    if timestamp:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    elif upload_date and len(upload_date) == 8:
        dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
    if dt is None:
        return 0.0
    return max((now - dt).total_seconds() / 86400.0, 0.0)
