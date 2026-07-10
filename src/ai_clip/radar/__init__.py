from ai_clip.radar.backfill import run_backfill
from ai_clip.radar.stage import (
    run_collect,
    run_content_rerank,
    run_source_content,
    run_source_research,
    run_zack_draft,
    run_zack_ranking,
    run_zack_selection,
)
from ai_clip.radar.workflow import run_all

__all__ = [
    "DAILY_RADAR_STAGES",
    "run_all",
    "run_backfill",
    "run_collect",
    "run_content_rerank",
    "run_source_content",
    "run_source_research",
    "run_zack_draft",
    "run_zack_ranking",
    "run_zack_selection",
]


def __getattr__(name: str):
    if name != "DAILY_RADAR_STAGES":
        raise AttributeError(name)
    from ai_clip.registry import REGISTRY

    workflow = REGISTRY.workflow("daily-radar")
    flags = {"research": True, "review": True, "rewrite": True}
    return tuple(REGISTRY.stage(stage) for stage in workflow.stage_names(flags))
