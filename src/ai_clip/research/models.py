from __future__ import annotations

from pydantic import BaseModel, Field

from ai_clip.source_research.models import ResearchQuery, SearchResult


class ProjectResearchReport(BaseModel):
    clip_id: str = ""
    theme: str = ""
    search_calls: int = 0
    queries: list[ResearchQuery] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)
    markdown: str = ""
