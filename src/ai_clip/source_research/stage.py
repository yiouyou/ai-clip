from __future__ import annotations

from ai_clip.core.config import Config
from ai_clip.core.llm import chat
from ai_clip.radar.models import RadarVideo, ZackSelection
from ai_clip.source_research.client import tavily_search
from ai_clip.source_research.models import ResearchQuery, SourceResearchReport
from ai_clip.source_research.prompts import (
    QUERY_SYSTEM,
    QUERY_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)
from ai_clip.research_engine import (
    align_queries,
    execute_searches,
    focus_for_prompt,
    merge_focus,
    parse_queries,
    search_count,
    search_results_for_prompt,
)
from ai_clip.zack_selection.selector import selection_for_prompt

_DEFAULT_FOCUS = [
    ("event_facts", "Verify who, what, when, official statements, and mainstream reporting."),
    ("structural_background", "Find mechanisms, data, institutional context, incentives, or history."),
    ("counterclaims_risk", "Check disputed claims, title bait, exaggeration, and alternative explanations."),
]


def generate_source_research(selection: ZackSelection, cfg: Config) -> SourceResearchReport:
    max_searches = search_count(cfg.source_research.max_searches)
    focus = _focus_plan(selection, max_searches)
    query_text = chat(
        cfg.llm,
        system=QUERY_SYSTEM,
        user=QUERY_USER.format(
            date=selection.date,
            max_searches=max_searches,
            selection=selection_for_prompt(selection),
            source=_source_material(selection.selected_video),
            focus=focus_for_prompt(focus),
        ),
    )
    queries = align_queries(
        parse_queries(query_text),
        focus,
        lambda angle, description: _fallback_query(selection, angle, description),
    )
    results = execute_searches(
        queries,
        lambda query: tavily_search(query, cfg.source_research),
    )

    markdown = chat(
        cfg.llm,
        system=SYNTHESIS_SYSTEM,
        user=SYNTHESIS_USER.format(
            date=selection.date,
            selection=selection_for_prompt(selection),
            source=_source_material(selection.selected_video),
            results=search_results_for_prompt(results),
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


def _fallback_query(selection: ZackSelection, angle: str, description: str) -> ResearchQuery:
    topic = selection.topic or selection.selected_video.title
    return ResearchQuery(
        angle=angle,
        query=f"{topic} {description}",
        rationale=f"Fallback query for {angle}: {description}",
    )


def _focus_plan(selection: ZackSelection, max_searches: int) -> list[tuple[str, str]]:
    return merge_focus(selection.research_focus, _DEFAULT_FOCUS, max_searches)


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
