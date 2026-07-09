from __future__ import annotations

import json

from ai_clip.core.config import Config
from ai_clip.core.llm import chat, extract_json
from ai_clip.radar.models import RadarVideo, ZackSelection
from ai_clip.source_research.client import tavily_search
from ai_clip.source_research.models import ResearchQuery, SearchResult, SourceResearchReport
from ai_clip.source_research.prompts import (
    QUERY_SYSTEM,
    QUERY_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)
from ai_clip.zack_selection.selector import selection_for_prompt

_MAX_SEARCHES_HARD_CAP = 3
_DEFAULT_FOCUS = [
    ("event_facts", "Verify who, what, when, official statements, and mainstream reporting."),
    ("structural_background", "Find mechanisms, data, institutional context, incentives, or history."),
    ("counterclaims_risk", "Check disputed claims, title bait, exaggeration, and alternative explanations."),
]


def generate_source_research(selection: ZackSelection, cfg: Config) -> SourceResearchReport:
    max_searches = _search_count(cfg.source_research.max_searches)
    focus = _focus_plan(selection, max_searches)
    query_text = chat(
        cfg.llm,
        system=QUERY_SYSTEM,
        user=QUERY_USER.format(
            date=selection.date,
            max_searches=max_searches,
            selection=selection_for_prompt(selection),
            source=_source_material(selection.selected_video),
            focus=_focus_for_prompt(focus),
        ),
    )
    queries = _queries_for_focus(_parse_queries(query_text), focus, selection)
    results: list[SearchResult] = []
    for query in queries:
        results.extend(_search_with_angle(query, cfg))

    markdown = chat(
        cfg.llm,
        system=SYNTHESIS_SYSTEM,
        user=SYNTHESIS_USER.format(
            date=selection.date,
            selection=selection_for_prompt(selection),
            source=_source_material(selection.selected_video),
            results=_search_results_for_prompt(results),
        ),
    )
    return SourceResearchReport(
        date=selection.date,
        selected_video_id=selection.selected_video_id,
        topic=selection.topic,
        angle=selection.angle,
        search_calls=len(queries),
        queries=queries,
        results=results,
        markdown=markdown.strip(),
    )


def _search_count(value: int) -> int:
    return min(max(value, 1), _MAX_SEARCHES_HARD_CAP)


def _parse_queries(text: str) -> list[ResearchQuery]:
    data = extract_json(text)
    out = []
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        out.append(ResearchQuery(
            query=query,
            angle=str(item.get("angle") or "").strip(),
            rationale=str(item.get("rationale") or ""),
        ))
    return out


def _queries_for_focus(
    queries: list[ResearchQuery],
    focus: list[tuple[str, str]],
    selection: ZackSelection,
) -> list[ResearchQuery]:
    remaining = list(queries)
    out: list[ResearchQuery] = []
    for angle, description in focus:
        match_index = _find_query_for_angle(remaining, angle)
        if match_index is None and remaining:
            match_index = 0
        if match_index is None:
            out.append(_fallback_query(selection, angle, description))
            continue
        query = remaining.pop(match_index)
        out.append(query.model_copy(update={
            "angle": angle,
            "rationale": query.rationale or description,
        }))
    return out


def _find_query_for_angle(queries: list[ResearchQuery], angle: str) -> int | None:
    normalized = angle.strip().lower()
    for i, query in enumerate(queries):
        if query.angle.strip().lower() == normalized:
            return i
    return None


def _fallback_query(selection: ZackSelection, angle: str, description: str) -> ResearchQuery:
    topic = selection.topic or selection.selected_video.title
    return ResearchQuery(
        angle=angle,
        query=f"{topic} {description}",
        rationale=f"Fallback query for {angle}: {description}",
    )


def _search_with_angle(query: ResearchQuery, cfg: Config) -> list[SearchResult]:
    return [
        result.model_copy(update={"angle": query.angle})
        for result in tavily_search(query.query, cfg.source_research)
    ]


def _focus_plan(selection: ZackSelection, max_searches: int) -> list[tuple[str, str]]:
    custom = _custom_focus(selection.research_focus)
    focus = custom + [item for item in _DEFAULT_FOCUS if item[0] not in {angle for angle, _ in custom}]
    return focus[:max_searches]


def _custom_focus(items: list[str]) -> list[tuple[str, str]]:
    out = []
    seen: set[str] = set()
    for item in items:
        angle, description = _split_focus(item)
        if angle in seen:
            continue
        seen.add(angle)
        out.append((angle, description))
    return out


def _split_focus(item: str) -> tuple[str, str]:
    raw = item.strip()
    if ":" in raw:
        angle, _, description = raw.partition(":")
        angle = _normalize_angle(angle)
        return angle, description.strip() or raw
    angle = _normalize_angle(raw)
    return angle, raw


def _normalize_angle(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "event_facts"


def _focus_for_prompt(focus: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {angle}: {description}" for angle, description in focus)


def _source_material(video: RadarVideo) -> str:
    transcript = video.transcript_text[:3000] if video.transcript_text else "(no transcript)"
    return "\n".join([
        f"title: {video.title}",
        f"url: {video.url}",
        f"platform: {video.platform}",
        f"channel: {video.channel_name or video.uploader}",
        f"score: {video.score}",
        f"score_reasons: {'; '.join(video.score_reasons)}",
        "transcript:",
        transcript,
    ])


def _search_results_for_prompt(results: list[SearchResult]) -> str:
    rows = []
    for i, result in enumerate(results, start=1):
        rows.append(json.dumps({
            "index": i,
            "query": result.query,
            "angle": result.angle,
            "title": result.title,
            "url": result.url,
            "content": result.content[:1200],
            "score": result.score,
        }, ensure_ascii=False))
    return "\n".join(rows)
