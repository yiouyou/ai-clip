"""Virality scoring: surface what is *recently* spreading, not historical hits.

score = velocity * (1 + engagement_rate) * recency_decay

- velocity        = views / age (a brand-new video with many views ranks high)
- engagement_rate = (likes + comments) / views (resonance, not just reach)
- recency_decay   = exp(-age / tau) (older content fades even if it has velocity)
"""

from __future__ import annotations

import math

_TAU_DAYS = 14.0


def virality_score(
    view_count: int, like_count: int, comment_count: int, age_days: float
) -> float:
    views = max(view_count, 0)
    age = max(age_days, 0.5)  # avoid div-by-zero; <12h treated as 0.5d
    velocity = views / age
    engagement_rate = (like_count + comment_count) / views if views else 0.0
    recency_decay = math.exp(-age / _TAU_DAYS)
    return velocity * (1.0 + engagement_rate) * recency_decay


def rank(candidates: list, top_n: int | None = None) -> list:
    """Sort candidates by virality (desc). Each must have .virality set."""
    ordered = sorted(candidates, key=lambda c: c.virality, reverse=True)
    return ordered[:top_n] if top_n else ordered
