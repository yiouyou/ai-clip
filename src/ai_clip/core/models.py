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


class ViralAnalysis(BaseModel):
    """Output of the analyze stage: the reusable "why it worked" formula."""

    clip_id: str
    hook: str = ""
    structure: list[str] = Field(default_factory=list)
    emotion_curve: list[str] = Field(default_factory=list)
    formula: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class Shot(BaseModel):
    """One storyboard shot. The filename fields are the contract that lets the
    assemble stage pick up assets regardless of how they were produced
    (ComfyUI API, or a human downloading from a website)."""

    index: int
    duration_sec: float = 3.0
    shot_type: str = ""
    image_prompt: str = ""
    video_prompt: str = ""
    voiceover: str = ""
    image_file: str = ""
    video_file: str = ""

    def expected_files(self) -> list[str]:
        return [f for f in (self.image_file, self.video_file) if f]


class Storyboard(BaseModel):
    """Output of the storyboard step; input to the assemble step."""

    project: str
    theme: str = ""
    source_clip_id: str | None = None
    aspect_ratio: str = "9:16"
    shots: list[Shot] = Field(default_factory=list)
