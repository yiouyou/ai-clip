from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ai_clip.core.models import Platform, TranscriptSegment
from ai_clip.core.run_status import RunStage, WorkflowRunStatus


class ChannelSpec(BaseModel):
    platform: Platform
    url: str
    name: str = ""
    pool: str = "general"
    role: str = "signal"
    tags: list[str] = Field(default_factory=list)
    priority: float = 1.0
    lens_fit: float = 1.0
    max_duration_sec: float | None = None
    cookies: str = ""


class RadarVideo(BaseModel):
    video_id: str
    url: str
    platform: Platform
    channel_url: str = ""
    channel_name: str = ""
    pool: str = "general"
    role: str = "signal"
    title: str = ""
    uploader: str = ""
    tags: list[str] = Field(default_factory=list)
    priority: float = 1.0
    lens_fit: float = 1.0
    duration_sec: float = 0.0
    published_date: str = ""
    age_days: float = 0.0
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    share_count: int | None = None
    favorite_count: int | None = None
    coin_count: int | None = None
    danmaku_count: int | None = None
    score: float = 0.0
    score_reasons: list[str] = Field(default_factory=list)
    score_components: dict[str, float] = Field(default_factory=dict)
    transcript_text: str = ""
    transcript_language: str = ""
    transcript_source: str = ""
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    content_status: str = "pending"
    content_error: str = ""
    content_cache_path: str = ""
    content_attempts: list[str] = Field(default_factory=list)


class RadarSnapshot(BaseModel):
    collected_at: str
    video: RadarVideo


class RadarCandidates(BaseModel):
    date: str
    top_n: int
    shortlist_n: int = 0
    ranking_phase: str = "final"
    videos: list[RadarVideo] = Field(default_factory=list)


class RadarFeedbackEvent(BaseModel):
    date: str
    video_id: str
    decision: Literal["accept", "reject"]
    reason: str = ""
    title: str = ""
    topic: str = ""
    pool: str = "general"
    platform: Platform
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ZackSelection(BaseModel):
    date: str
    selected_video_id: str
    selected_index: int = 1
    selected_video: RadarVideo
    topic: str = ""
    angle: str = ""
    why_selected: str = ""
    fact_risk: str = "medium"
    research_focus: list[str] = Field(default_factory=list)
    backup_video_ids: list[str] = Field(default_factory=list)


class ZackDraft(BaseModel):
    date: str
    title: str = ""
    markdown: str = ""
    videos: list[RadarVideo] = Field(default_factory=list)


class RadarRunResult(BaseModel):
    date: str
    collected: int
    candidates_path: str
    selection_path: str = ""
    brief_path: str
    draft_path: str
    run_status_path: str = ""
    review_path: str = ""
    revised_draft_path: str = ""
    verification_path: str = ""


class RadarBackfillResult(BaseModel):
    end_date: str
    days: int
    collected: int
    output_dir: str
    files: list[str] = Field(default_factory=list)


class ChannelCollectResult(BaseModel):
    platform: Platform
    url: str
    name: str = ""
    status: str = "pending"
    count: int = 0
    duration_sec: float = 0.0
    error: str = ""
    video_ids: list[str] = Field(default_factory=list)


class RadarCollectReport(BaseModel):
    collected_at: str
    snapshots: list[RadarSnapshot] = Field(default_factory=list)
    channels: list[ChannelCollectResult] = Field(default_factory=list)


RadarRunStage = RunStage


class RadarRunStatus(WorkflowRunStatus):
    workflow: str = "daily-radar"
    date: str


