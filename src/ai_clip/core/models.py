"""Data contract shared across every pipeline stage.

Each stage reads the previous artifact and writes the next one, all as JSON on
disk under a per-project directory. Keeping these models stable is what lets any
stage be re-run, cached, or swapped for a different implementation.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Platform(StrEnum):
    youtube = "youtube"
    bilibili = "bilibili"
    douyin = "douyin"
    kuaishou = "kuaishou"
    tiktok = "tiktok"
    unknown = "unknown"


class Candidate(BaseModel):
    """A discovered video, before download. Ranked by `virality`."""

    url: str
    platform: Platform = Platform.unknown
    title: str = ""
    uploader: str = ""
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    duration_sec: float = 0.0
    age_days: float = 0.0
    virality: float = 0.0


class CandidateList(BaseModel):
    """Output of the discover stage."""

    topic: str = ""
    platform: Platform = Platform.unknown
    candidates: list[Candidate] = Field(default_factory=list)


class Clip(BaseModel):
    """Output of the download stage."""

    clip_id: str
    source_url: str
    platform: Platform = Platform.unknown
    video_path: str
    title: str = ""
    duration_sec: float = 0.0
    meta: dict = Field(default_factory=dict)


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """Output of the extract stage."""

    clip_id: str
    language: str = ""
    text: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    audio_path: str | None = None


class Intent(StrEnum):
    info = "info"  # knowledge-first (neutral, explain)
    emotion = "emotion"  # opinionated take: stance + emotion from news/events
    sales = "sales"  # product promo: pain -> agitate -> product -> proof -> CTA


class ProductProfile(BaseModel):
    """Reusable product description for the `sales` intent (loaded from YAML)."""

    name: str = ""
    description: str = ""
    audience: str = ""
    selling_points: list[str] = Field(default_factory=list)
    cta: str = ""


class ViralAnalysis(BaseModel):
    """Output of the analyze stage: the reusable "why it worked" formula.

    Intent-specific fields are populated only for the matching intent."""

    clip_id: str
    intent: Intent = Intent.info
    hook: str = ""
    structure: list[str] = Field(default_factory=list)
    emotion_curve: list[str] = Field(default_factory=list)
    formula: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
    # emotion intent:
    stance: str = ""
    # sales intent:
    pain_points: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)


class VideoFormat(StrEnum):
    talking_head = "talking_head"  # narration + optional b-roll stills
    slideshow = "slideshow"  # image cards + on-screen captions + narration
    remix = "remix"  # segments cut from the source clip + narration
    montage = "montage"  # fully AI-generated multi-shot drama


class Shot(BaseModel):
    """One storyboard shot. The filename fields are the contract that lets the
    assemble stage pick up assets regardless of how they were produced
    (ComfyUI API, or a human downloading from a website). For `remix`, the shot
    instead points at a [source_start, source_end] span of the source clip."""

    index: int
    duration_sec: float = 3.0
    shot_type: str = ""
    image_prompt: str = ""
    video_prompt: str = ""
    voiceover: str = ""
    caption: str = ""  # on-screen text (slideshow)
    image_file: str = ""
    video_file: str = ""
    source_start: float | None = None  # remix: cut start in source clip (sec)
    source_end: float | None = None  # remix: cut end in source clip (sec)

    @property
    def is_source_segment(self) -> bool:
        return self.source_start is not None and self.source_end is not None

    def expected_files(self) -> list[str]:
        if self.is_source_segment:
            return []
        return [f for f in (self.image_file, self.video_file) if f]


class Storyboard(BaseModel):
    """Output of the storyboard step; input to the assemble step."""

    project: str
    format: VideoFormat = VideoFormat.talking_head
    theme: str = ""
    source_clip_id: str | None = None
    aspect_ratio: str = "9:16"
    shots: list[Shot] = Field(default_factory=list)
