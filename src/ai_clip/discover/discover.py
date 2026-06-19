"""Discover orchestrator: search a platform, score, and rank candidates."""

from __future__ import annotations

from ai_clip.core.models import CandidateList, Platform
from ai_clip.discover.ranking import rank, virality_score
from ai_clip.discover.social import DouyinProvider, KuaishouProvider
from ai_clip.discover.youtube import YouTubeProvider

_PROVIDERS = {
    Platform.youtube: YouTubeProvider,
    Platform.douyin: DouyinProvider,
    Platform.kuaishou: KuaishouProvider,
}


def _provider(platform: Platform):
    if platform not in _PROVIDERS:
        raise NotImplementedError(
            f"discover provider for {platform} not implemented yet; "
            f"available: {[p.value for p in _PROVIDERS]}. "
            "For other platforms, pass a direct URL to `ai-clip download`."
        )
    return _PROVIDERS[platform]()


def discover(
    topic: str,
    platform: Platform = Platform.youtube,
    channel: str | None = None,
    since_days: int = 7,
    limit: int = 15,
    top_n: int = 5,
) -> CandidateList:
    provider = _provider(platform)
    candidates = provider.search(topic, channel, since_days, limit)
    for c in candidates:
        c.virality = virality_score(c.view_count, c.like_count, c.comment_count, c.age_days)
    return CandidateList(
        topic=topic, platform=platform, candidates=rank(candidates, top_n)
    )
