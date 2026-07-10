from __future__ import annotations

from collections.abc import Callable
import json

from ai_clip.core.llm import extract_json
from ai_clip.source_research.models import ResearchQuery, SearchResult

MAX_SEARCHES_HARD_CAP = 3
Focus = tuple[str, str]
FallbackQuery = Callable[[str, str], ResearchQuery]
Search = Callable[[str], list[SearchResult]]


def search_count(value: int) -> int:
    return min(max(value, 1), MAX_SEARCHES_HARD_CAP)


def parse_queries(text: str) -> list[ResearchQuery]:
    data = extract_json(text)
    queries = []
    for item in data.get("queries", []):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        queries.append(ResearchQuery(
            query=query,
            angle=str(item.get("angle") or "").strip(),
            rationale=str(item.get("rationale") or "").strip(),
        ))
    return queries


def align_queries(
    queries: list[ResearchQuery],
    focus: list[Focus],
    fallback: FallbackQuery,
) -> list[ResearchQuery]:
    remaining = list(queries)
    matched: dict[str, ResearchQuery] = {}
    for angle, _ in focus:
        match_index = _find_query_for_angle(remaining, angle)
        if match_index is not None:
            matched[angle] = remaining.pop(match_index)

    aligned: list[ResearchQuery] = []
    for angle, description in focus:
        query = matched.get(angle)
        if query is None and remaining:
            query = remaining.pop(0)
        if query is None:
            aligned.append(fallback(angle, description))
            continue
        aligned.append(query.model_copy(update={
            "angle": angle,
            "rationale": query.rationale or description,
        }))
    return aligned


def execute_searches(queries: list[ResearchQuery], search: Search) -> list[SearchResult]:
    results: list[SearchResult] = []
    for query in queries:
        results.extend(
            result.model_copy(update={"angle": query.angle})
            for result in search(query.query)
        )
    return results


def merge_focus(custom: list[str], defaults: list[Focus], limit: int) -> list[Focus]:
    custom_focus = _custom_focus(custom)
    custom_angles = {angle for angle, _ in custom_focus}
    return (custom_focus + [item for item in defaults if item[0] not in custom_angles])[:limit]


def focus_for_prompt(focus: list[Focus]) -> str:
    return "\n".join(f"- {angle}: {description}" for angle, description in focus)


def search_results_for_prompt(results: list[SearchResult]) -> str:
    rows = []
    for index, result in enumerate(results, start=1):
        rows.append(json.dumps({
            "index": index,
            "query": result.query,
            "angle": result.angle,
            "title": result.title,
            "url": result.url,
            "content": result.content[:1200],
            "score": result.score,
        }, ensure_ascii=False))
    return "\n".join(rows) or "(no search results)"


def _find_query_for_angle(queries: list[ResearchQuery], angle: str) -> int | None:
    normalized = angle.strip().lower()
    for index, query in enumerate(queries):
        if query.angle.strip().lower() == normalized:
            return index
    return None


def _custom_focus(items: list[str]) -> list[Focus]:
    focus = []
    seen: set[str] = set()
    for item in items:
        angle, description = _split_focus(item)
        if angle in seen:
            continue
        seen.add(angle)
        focus.append((angle, description))
    return focus


def _split_focus(item: str) -> Focus:
    raw = item.strip()
    if ":" in raw:
        angle, _, description = raw.partition(":")
        return _normalize_angle(angle), description.strip() or raw
    return _normalize_angle(raw), raw


def _normalize_angle(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or "event_facts"
