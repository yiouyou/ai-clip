from __future__ import annotations

import json

from ai_clip.core.config import Config
from ai_clip.core.llm import chat
from ai_clip.core.models import Transcript, ViralAnalysis
from ai_clip.research.models import ProjectResearchReport
from ai_clip.research.prompts import QUERY_SYSTEM, QUERY_USER, SYNTHESIS_SYSTEM, SYNTHESIS_USER
from ai_clip.research_engine import (
    align_queries,
    execute_searches,
    focus_for_prompt,
    parse_queries,
    search_count,
    search_results_for_prompt,
)
from ai_clip.source_research.client import tavily_search
from ai_clip.source_research.models import ResearchQuery

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
    return _generate_research(
        transcript=transcript,
        cfg=cfg,
        analysis=analysis,
        theme=theme,
    )


def generate_topic_research(theme: str, cfg: Config) -> ProjectResearchReport:
    if not theme.strip():
        raise ValueError("topic research requires a non-empty theme")
    return _generate_research(transcript=None, cfg=cfg, analysis=None, theme=theme)


def _generate_research(
    transcript: Transcript | None,
    cfg: Config,
    analysis: ViralAnalysis | None,
    theme: str,
) -> ProjectResearchReport:
    max_searches = search_count(cfg.source_research.max_searches)
    focus = _DEFAULT_FOCUS[:max_searches]
    query_text = chat(
        cfg.llm,
        system=QUERY_SYSTEM,
        user=QUERY_USER.format(
            theme=theme or "(not provided)",
            max_searches=max_searches,
            analysis=_analysis_for_prompt(analysis),
            transcript=_transcript_for_prompt(transcript),
            focus=focus_for_prompt(focus),
        ),
    )
    queries = align_queries(
        parse_queries(query_text),
        focus,
        lambda angle, description: _fallback_query(transcript, theme, angle, description),
    )
    results = execute_searches(
        queries,
        lambda query: tavily_search(query, cfg.source_research),
    )

    markdown = chat(
        cfg.llm,
        system=SYNTHESIS_SYSTEM,
        user=SYNTHESIS_USER.format(
            theme=theme or "(not provided)",
            analysis=_analysis_for_prompt(analysis),
            transcript=_transcript_for_prompt(transcript),
            results=search_results_for_prompt(results),
        ),
    )
    return ProjectResearchReport(
        clip_id=transcript.clip_id if transcript else "",
        theme=theme,
        search_calls=len(queries),
        queries=queries,
        results=results,
        markdown=markdown.strip(),
    )


def _fallback_query(
    transcript: Transcript | None,
    theme: str,
    angle: str,
    description: str,
) -> ResearchQuery:
    topic = theme.strip()
    if not topic and transcript is not None:
        topic = transcript.text[:80].strip() or transcript.clip_id
    return ResearchQuery(
        angle=angle,
        query=f"{topic} {description}",
        rationale=f"Fallback query for {angle}: {description}",
    )


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


def _transcript_for_prompt(transcript: Transcript | None) -> str:
    if transcript is None:
        return "(theme-only research; no source transcript)"
    text = transcript.text.strip()
    if text:
        return text[:4000]
    rows = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments]
    return "\n".join(rows)[:4000] or "(empty transcript)"
