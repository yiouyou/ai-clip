from ai_clip.radar.backfill import run_backfill
from ai_clip.radar.stage import (
    run_collect,
    run_draft_script,
    run_enrich_content,
    run_rank_candidates,
    run_rerank_candidates,
    run_research_evidence,
    run_select_topic,
)
from ai_clip.radar.workflow import run_all

__all__ = [
    "DAILY_RADAR_STAGES",
    "run_all",
    "run_backfill",
    "run_collect",
    "run_draft_script",
    "run_enrich_content",
    "run_rank_candidates",
    "run_rerank_candidates",
    "run_research_evidence",
    "run_select_topic",
]


def __getattr__(name: str):
    if name != "DAILY_RADAR_STAGES":
        raise AttributeError(name)
    from ai_clip.registry import REGISTRY

    workflow = REGISTRY.workflow("daily-radar")
    flags = {"research": True, "review": True, "rewrite": True}
    return tuple(REGISTRY.stage(stage) for stage in workflow.stage_names(flags))
