from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    severity: str = "medium"
    category: str = ""
    detail: str = ""
    suggestion: str = ""


class ReviewerResult(BaseModel):
    role: str
    model: str = ""
    ok: bool = False
    verdict: str = ""
    summary: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    raw: str = ""
    error: str = ""


class PairReviewReport(BaseModel):
    artifact: str
    source_path: str
    producer_model: str = ""
    status: str = "failed"
    reviewers: list[ReviewerResult] = Field(default_factory=list)
