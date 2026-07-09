from __future__ import annotations

import httpx

from ai_clip.core.config import SourceResearchConfig
from ai_clip.source_research.models import SearchResult


class SourceResearchError(RuntimeError):
    pass


def tavily_search(query: str, cfg: SourceResearchConfig) -> list[SearchResult]:
    if not cfg.tavily_api_key:
        raise SourceResearchError("TAVILY_API_KEY is empty")
    response = httpx.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {cfg.tavily_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": cfg.search_depth,
            "max_results": max(cfg.max_results, 1),
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=cfg.timeout,
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue
        results.append(SearchResult(
            query=query,
            title=str(item.get("title") or ""),
            url=str(item.get("url") or ""),
            content=str(item.get("content") or ""),
            score=float(item.get("score") or 0.0),
        ))
    return results
