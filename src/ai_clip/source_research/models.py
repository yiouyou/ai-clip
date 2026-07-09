from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    query: str
    angle: str = ""
    rationale: str = ""


class SearchResult(BaseModel):
    query: str
    angle: str = ""
    title: str = ""
    url: str = ""
    content: str = ""
    score: float = 0.0
    source: str = "tavily"


class SourceResearchReport(BaseModel):
    date: str
    selected_video_id: str = ""
    topic: str = ""
    angle: str = ""
    search_calls: int
    queries: list[ResearchQuery] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)
    markdown: str = ""
