from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median

import yaml
from pydantic import BaseModel, Field

from ai_clip.radar.models import RadarSnapshot, RadarVideo

_WEATHER_TOPIC_KEYWORDS = {
    "台风",
    "颱風",
    "气象",
    "氣象",
    "天气",
    "天氣",
    "暴雨",
    "洪水",
    "山洪",
    "风暴",
    "風暴",
    "飓风",
    "颶風",
    "寒潮",
    "高温",
    "高溫",
    "降雨",
    "降水",
    "灾害",
    "災害",
    "typhoon",
    "storm",
    "hurricane",
    "weather",
    "meteorology",
    "rainstorm",
    "flood",
}


class RankingFeedback(BaseModel):
    upweight_keywords: list[str] = Field(default_factory=list)
    downrank_keywords: list[str] = Field(default_factory=list)
    avoid_video_ids: list[str] = Field(default_factory=list)
    preferred_pools: list[str] = Field(default_factory=list)
    downrank_pools: list[str] = Field(default_factory=list)
    upweight_multiplier: float = 1.15
    downrank_multiplier: float = 0.7
    learned_pool_multipliers: dict[str, float] = Field(default_factory=dict)
    learned_tag_multipliers: dict[str, float] = Field(default_factory=dict)


@dataclass(frozen=True)
class RankingBaselines:
    channel_views: dict[str, float]
    platform_velocity: dict[str, float]
    platform_engagement: dict[str, float]


def rank_videos(
    snapshots: list[RadarSnapshot],
    previous: dict[str, RadarVideo],
    top_n: int,
    feedback: RankingFeedback | None = None,
) -> list[RadarVideo]:
    baselines = _build_baselines([snapshot.video for snapshot in snapshots])
    scored = [
        score_video(
            snapshot.video,
            previous.get(snapshot.video.video_id),
            feedback,
            baselines=baselines,
        )
        for snapshot in snapshots
    ]
    ordered = sorted(scored, key=lambda video: video.score, reverse=True)
    return _diverse_top(ordered, top_n)


def score_video(
    video: RadarVideo,
    previous: RadarVideo | None = None,
    feedback: RankingFeedback | None = None,
    *,
    baselines: RankingBaselines | None = None,
) -> RadarVideo:
    view_delta = _delta(video.view_count, previous.view_count if previous else None)
    like_delta = _delta(video.like_count, previous.like_count if previous else None)
    comment_delta = _delta(video.comment_count, previous.comment_count if previous else None)
    share_delta = _delta(video.share_count, previous.share_count if previous else None)
    favorite_delta = _delta(video.favorite_count, previous.favorite_count if previous else None)

    age = max(video.age_days, 0.25)
    velocity = max(view_delta, video.view_count) / age
    engagement_rate = _engagement_rate(video)
    if baselines is None:
        reach = math.log1p(max(view_delta, video.view_count * 0.25)) * 8.0
        velocity_score = math.log1p(velocity) * 12.0
        engagement_score = engagement_rate * 200.0
        channel_ratio = 1.0
        platform_velocity_ratio = 1.0
        platform_engagement_ratio = 1.0
    else:
        channel_base = baselines.channel_views.get(_channel_key(video), max(video.view_count, 1))
        platform_key = video.platform.value
        velocity_base = baselines.platform_velocity.get(platform_key, max(velocity, 1.0))
        engagement_base = baselines.platform_engagement.get(
            platform_key,
            max(engagement_rate, 0.001),
        )
        channel_ratio = max(video.view_count, 0) / max(channel_base, 1.0)
        platform_velocity_ratio = velocity / max(velocity_base, 1.0)
        platform_engagement_ratio = engagement_rate / max(engagement_base, 0.001)
        reach = math.log1p(channel_ratio) * 28.0 + math.log1p(max(video.view_count, 0)) * 2.0
        velocity_score = (
            math.log1p(platform_velocity_ratio) * 28.0 + math.log1p(max(velocity, 0)) * 2.0
        )
        engagement_score = min(platform_engagement_ratio, 5.0) * 12.0
    delta_score = math.log1p(max(like_delta, 0) + max(comment_delta, 0) * 2) * 10.0
    if share_delta is not None:
        delta_score += math.log1p(max(share_delta, 0)) * 8.0
    if favorite_delta is not None:
        delta_score += math.log1p(max(favorite_delta, 0)) * 6.0
    recency_score = math.exp(-age / 3.0) * 30.0
    priority_multiplier = max(video.priority, 0.1)
    role_multiplier = _role_multiplier(video.role)
    lens_multiplier = 0.75 + max(video.lens_fit, 0.0) * 0.25
    topic_multiplier = _topic_multiplier(video)
    feedback_multiplier = _feedback_multiplier(video, feedback)

    score = (
        reach + velocity_score + engagement_score + delta_score + recency_score
    ) * (
        priority_multiplier
        * role_multiplier
        * lens_multiplier
        * topic_multiplier
        * feedback_multiplier
    )
    components = {
        "reach": round(reach, 3),
        "velocity": round(velocity_score, 3),
        "engagement": round(engagement_score, 3),
        "delta": round(delta_score, 3),
        "recency": round(recency_score, 3),
        "priority_multiplier": round(priority_multiplier, 3),
        "role_multiplier": round(role_multiplier, 3),
        "lens_multiplier": round(lens_multiplier, 3),
        "topic_multiplier": round(topic_multiplier, 3),
        "feedback_multiplier": round(feedback_multiplier, 3),
        "channel_reach_ratio": round(channel_ratio, 3),
        "platform_velocity_ratio": round(platform_velocity_ratio, 3),
        "platform_engagement_ratio": round(platform_engagement_ratio, 3),
    }
    reasons = [
        f"pool={video.pool}",
        f"role={video.role}",
        f"lens_fit={video.lens_fit:g}",
        f"views={video.view_count:,}",
        f"likes={video.like_count:,}",
        f"comments={video.comment_count:,}",
        f"age={age:.1f}d",
    ]
    if previous:
        reasons.append(f"view_delta={view_delta:,}")
        reasons.append(f"like_delta={like_delta:,}")
    if video.share_count is not None:
        reasons.append(f"shares={video.share_count:,}")
    if video.favorite_count is not None:
        reasons.append(f"favorites={video.favorite_count:,}")
    if topic_multiplier < 1.0:
        reasons.append(f"topic_downrank=weather:{topic_multiplier:g}")
    if feedback_multiplier != 1.0:
        reasons.append(f"feedback_multiplier={feedback_multiplier:g}")

    return video.model_copy(update={
        "score": round(score, 3),
        "score_reasons": reasons,
        "score_components": components,
    })


def rerank_by_content(videos: list[RadarVideo], top_n: int) -> list[RadarVideo]:
    """Rerank a bounded metadata shortlist after scripts have been acquired."""
    if not videos or top_n <= 0:
        return []
    metadata_order = {
        video.video_id: index
        for index, video in enumerate(sorted(videos, key=lambda item: item.score, reverse=True))
    }
    denominator = max(len(videos) - 1, 1)
    reranked = []
    for video in videos:
        metadata_percentile = 1.0 - metadata_order[video.video_id] / denominator
        content_fit = _content_fit(video)
        rerank_score = metadata_percentile * 70.0 + content_fit * 30.0
        components = {
            **video.score_components,
            "metadata_percentile": round(metadata_percentile, 3),
            "content_fit": round(content_fit, 3),
            "rerank_score": round(rerank_score, 3),
        }
        reasons = [*video.score_reasons, f"content_fit={content_fit:.3f}"]
        reranked.append(video.model_copy(update={
            "score_components": components,
            "score_reasons": reasons,
        }))
    ordered = sorted(
        reranked,
        key=lambda item: item.score_components["rerank_score"],
        reverse=True,
    )
    return _diverse_top(ordered, top_n)


def load_ranking_feedback(path: str | Path) -> RankingFeedback:
    p = Path(path)
    if not p.exists():
        return RankingFeedback()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return RankingFeedback.model_validate(raw)


def _delta(current: int | None, old: int | None) -> int:
    if current is None:
        return 0
    if old is None:
        return 0
    return max(current - old, 0)


def _role_multiplier(role: str) -> float:
    if role == "signal":
        return 1.0
    if role == "reference":
        return 0.75
    if role == "style":
        return 0.65
    return 0.85


def _topic_multiplier(video: RadarVideo) -> float:
    text = " ".join([video.title, video.pool, " ".join(video.tags)]).lower()
    if any(keyword.lower() in text for keyword in _WEATHER_TOPIC_KEYWORDS):
        return 0.72
    return 1.0


def _feedback_multiplier(video: RadarVideo, feedback: RankingFeedback | None) -> float:
    if feedback is None:
        return 1.0
    if video.video_id in set(feedback.avoid_video_ids):
        return 0.01
    text = " ".join([video.title, video.pool, " ".join(video.tags)]).lower()
    multiplier = 1.0
    if video.pool in set(feedback.preferred_pools):
        multiplier *= feedback.upweight_multiplier
    if video.pool in set(feedback.downrank_pools):
        multiplier *= feedback.downrank_multiplier
    if any(keyword.lower() in text for keyword in feedback.upweight_keywords):
        multiplier *= feedback.upweight_multiplier
    if any(keyword.lower() in text for keyword in feedback.downrank_keywords):
        multiplier *= feedback.downrank_multiplier
    multiplier *= feedback.learned_pool_multipliers.get(video.pool, 1.0)
    for tag in set(video.tags):
        multiplier *= feedback.learned_tag_multipliers.get(tag, 1.0)
    return min(max(multiplier, 0.4), 1.8)


def _build_baselines(videos: list[RadarVideo]) -> RankingBaselines:
    channel_values: dict[str, list[float]] = {}
    velocity_values: dict[str, list[float]] = {}
    engagement_values: dict[str, list[float]] = {}
    for video in videos:
        channel_values.setdefault(_channel_key(video), []).append(float(max(video.view_count, 0)))
        platform = video.platform.value
        age = max(video.age_days, 0.25)
        velocity_values.setdefault(platform, []).append(max(video.view_count, 0) / age)
        engagement_values.setdefault(platform, []).append(_engagement_rate(video))
    return RankingBaselines(
        channel_views={key: _positive_median(values, 1.0) for key, values in channel_values.items()},
        platform_velocity={
            key: _positive_median(values, 1.0) for key, values in velocity_values.items()
        },
        platform_engagement={
            key: _positive_median(values, 0.001) for key, values in engagement_values.items()
        },
    )


def _positive_median(values: list[float], fallback: float) -> float:
    positive = [value for value in values if value > 0]
    return median(positive) if positive else fallback


def _channel_key(video: RadarVideo) -> str:
    return video.channel_url or video.channel_name or video.uploader or video.video_id


def _engagement_rate(video: RadarVideo) -> float:
    weighted = video.like_count + video.comment_count * 2 + (video.share_count or 0) * 3
    if video.platform.value == "bilibili":
        weighted += (
            (video.favorite_count or 0) * 2
            + (video.coin_count or 0) * 2
            + (video.danmaku_count or 0)
        )
    return weighted / max(video.view_count, 1)


_CONTENT_LENS_KEYWORDS = {
    "机制", "系统", "反馈", "循环", "演化", "进化", "生态", "适应", "博弈", "激励",
    "网络", "涌现", "免疫", "群体", "边界", "路径依赖", "mechanism", "system", "feedback",
    "evolution", "ecology", "adaptive", "incentive", "network", "emergence",
}


def _content_fit(video: RadarVideo) -> float:
    text = f"{video.title}\n{video.transcript_text}".lower()
    keyword_hits = sum(1 for keyword in _CONTENT_LENS_KEYWORDS if keyword.lower() in text)
    transcript_depth = min(len(video.transcript_text) / 6000.0, 1.0)
    availability = 1.0 if video.transcript_text.strip() else 0.0
    lens = min(max(video.lens_fit, 0.0) / 1.5, 1.0)
    return min(
        availability * 0.25
        + transcript_depth * 0.25
        + min(keyword_hits / 6.0, 1.0) * 0.30
        + lens * 0.20,
        1.0,
    )


def _diverse_top(ordered: list[RadarVideo], top_n: int) -> list[RadarVideo]:
    selected: list[RadarVideo] = []
    used_pools: set[str] = set()
    for video in ordered:
        if len(selected) >= top_n:
            break
        if video.pool in used_pools:
            continue
        selected.append(video)
        used_pools.add(video.pool)
    if len(selected) >= top_n:
        return selected
    selected_ids = {video.video_id for video in selected}
    for video in ordered:
        if len(selected) >= top_n:
            break
        if video.video_id in selected_ids:
            continue
        selected.append(video)
    return selected
