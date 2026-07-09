from __future__ import annotations

import json

from ai_clip.core.config import Config
from ai_clip.core.llm import chat, extract_json
from ai_clip.core.models import Transcript, ViralAnalysis
from ai_clip.research.models import ProjectResearchReport
from ai_clip.research.prompts import QUERY_SYSTEM, QUERY_USER, SYNTHESIS_SYSTEM, SYNTHESIS_USER
from ai_clip.source_research.client import tavily_search
from ai_clip.source_research.models import ResearchQuery, SearchResult

_MAX_SEARCHES_HARD_CAP = 3
_DEFAULT_FOCUS = [
    ("event_facts", "Verify factual claims, named entities, dates, and source context."),
    ("structural_background", "Find mechanisms, data, history, incentives, or institutions."),
    ("original_lens", "Find material useful for biology, ecology, or complex-systems analogies."),
]


def generate_project_research(
    transcript: Transcript,
    cfg: Config,
    analysis: ViralAnalysis | None = None,
    theme: str = "",
) -> ProjectResearchReport:
    max_searches = _search_count(cfg.source_research.max_searches)
    focus = _DEFAULT_FOCUS[:max_searches]
    query_text = chat(
        cfg.llm,
        system=QUERY_SYSTEM,
        user=QUERY_USER.format(
            theme=theme or "(not provided)",
            max_searches=max_searches,
            analysis=_analysis_for_prompt(analysis),
            transcript=_transcript_for_prompt(transcript),
            focus=_focus_for_prompt(focus),
        ),
    )
    queries = _queries_for_focus(_parse_queries(query_text), focus, transcript, theme)
    results: list[SearchResult] = []
    for query in queries:
        results.extend(_search_with_angle(query, cfg))

    markdown = chat(
        cfg.llm,
        system=SYNTHESIS_SYSTEM,
        user=SYNTHESIS_USER.format(
            theme=theme or "(not provided)",
            analysis=_analysis_for_prompt(analysis),
            transcript=_transcript_for_prompt(transcript),
            results=_search_results_for_prompt(results),
        ),
    )
    return ProjectResearchReport(
        clip_id=transcript.clip_id,
        theme=theme,
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
            rationale=str(item.get("rationale") or "").strip(),
        ))
    return out


def _queries_for_focus(
    queries: list[ResearchQuery],
    focus: list[tuple[str, str]],
    transcript: Transcript,
    theme: str,
) -> list[ResearchQuery]:
    remaining = list(queries)
    out: list[ResearchQuery] = []
    for angle, description in focus:
        match_index = _find_query_for_angle(remaining, angle)
        if match_index is None and remaining:
            match_index = 0
        if match_index is None:
            out.append(_fallback_query(transcript, theme, angle, description))
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


def _fallback_query(
    transcript: Transcript,
    theme: str,
    angle: str,
    description: str,
) -> ResearchQuery:
    topic = theme.strip() or transcript.text[:80].strip() or transcript.clip_id
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


def _analysis_for_prompt(analysis: ViralAnalysis | None) -> str:
    if analysis is None:
        return "(no analysis available)"
    return json.dumps({
        "hook": analysis.hook,
        "structure": analysis.structure,
        "formula": analysis.formula,
        "stance": analysis.stance,
        "notes": analysis.notes,
    }, ensure_ascii=False, indent=2)


def _transcript_for_prompt(transcript: Transcript) -> str:
    text = transcript.text.strip()
    if text:
        return text[:4000]
    rows = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments]
    return "\n".join(rows)[:4000] or "(empty transcript)"


def _focus_for_prompt(focus: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {angle}: {description}" for angle, description in focus)


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
    return "\n".join(rows) or "(no search results)"
