from __future__ import annotations

from ai_clip.core.config import Config
from ai_clip.core.stages import (
    StageInvocation,
    StageResult,
    execute_workflow,
    stage_execution,
)
from ai_clip.radar import stage as radar_stages
from ai_clip.radar.models import RadarRunResult
from ai_clip.radar.ops import RadarRunLock
from ai_clip.radar.research_policy import automatic_research_searches
from ai_clip.radar.status import begin_run, finalize_run_usage, track_stage
from ai_clip.radar.storage import RadarPaths
from ai_clip.radar.time import today_in_tz


def run_all(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
    research: bool = False,
    force_collect: bool = False,
    review: bool = False,
    rewrite: bool = False,
) -> RadarRunResult:
    from ai_clip.registry import REGISTRY

    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    collected = 0
    review_path = ""
    revised_draft_path = ""
    verification_path = ""
    review_report = None
    expected_review_path = str(paths.reviews_dir / f"{date}_zack_draft_review.json")
    expected_revised_path = str(paths.draft_revised_md)
    expected_verification_path = str(paths.reviews_dir / f"{date}_zack_draft_verify.json")

    def workflow_tracker(invocation: StageInvocation):
        return track_stage(paths, invocation.name, dict(invocation.inputs))

    def collect_stage():
        nonlocal collected
        collected = radar_stages.run_collect(cfg, date, force=force_collect)
        return StageResult(value=collected)

    def ranking_stage():
        return StageResult(value=radar_stages.run_zack_ranking(cfg, date, top_n))

    def content_stage():
        return StageResult(value=radar_stages.run_source_content(cfg, date))

    def content_rerank_stage():
        return StageResult(value=radar_stages.run_content_rerank(cfg, date))

    def selection_stage():
        selection = radar_stages.run_zack_selection(cfg, date)
        if not research:
            searches = automatic_research_searches(selection, cfg)
            if searches:
                cfg.source_research.max_searches = searches
                flags["research"] = True
        return StageResult(value=selection)

    def research_stage():
        return StageResult(value=radar_stages.run_source_research(cfg, date))

    def draft_stage():
        return StageResult(value=radar_stages.run_zack_draft(cfg, date))

    def pair_review_stage():
        nonlocal review_path, review_report
        from ai_clip.pair.stage import review_artifact

        review_report = review_artifact(cfg, "radar", "zack_draft", run_date=date)
        review_path = expected_review_path
        return StageResult(
            value=review_report,
            outputs={"review": review_path},
            metrics={
                "status": review_report.status,
                "reviewers": len(review_report.reviewers),
            },
        )

    def pair_rewrite_stage():
        nonlocal revised_draft_path
        from ai_clip.pair.stage import rewrite_reviewed_artifact

        if review_report is None:
            raise RuntimeError("pair-rewrite requires pair-review in the same workflow run")
        if review_report.status == "blocked":
            return StageResult(
                status="skipped",
                metrics={"reason": "pair-review blocked"},
            )
        revised = rewrite_reviewed_artifact(
            cfg,
            "radar",
            "zack_draft",
            review_report,
            run_date=date,
        )
        revised_draft_path = str(revised)
        return StageResult(
            value=revised,
            outputs={"revised-draft": revised_draft_path},
        )

    def pair_verify_stage():
        nonlocal verification_path
        from ai_clip.pair.stage import verify_rewritten_artifact

        if not revised_draft_path:
            return StageResult(
                status="skipped",
                metrics={"reason": "pair-rewrite produced no draft"},
            )
        report = verify_rewritten_artifact(
            cfg,
            "radar",
            "zack_draft",
            run_date=date,
        )
        verification_path = expected_verification_path
        return StageResult(
            value=report,
            outputs={"verification": verification_path},
            metrics={
                "status": report.status,
                "reviewers": len(report.reviewers),
                "bounded": True,
            },
        )

    executions = {
        "collect": stage_execution(
            "collect",
            collect_stage,
            {"date": date, "force": str(force_collect)},
        ),
        "zack-ranking": stage_execution(
            "zack-ranking",
            ranking_stage,
            {"date": date, "top_n": str(top_n or cfg.radar.top_n)},
        ),
        "source-content": stage_execution("source-content", content_stage, {"date": date}),
        "content-rerank": stage_execution("content-rerank", content_rerank_stage, {"date": date}),
        "zack-selection": stage_execution("zack-selection", selection_stage, {"date": date}),
        "source-research": stage_execution("source-research", research_stage, {"date": date}),
        "zack-draft": stage_execution("zack-draft", draft_stage, {"date": date}),
        "pair-review": stage_execution(
            "pair-review",
            pair_review_stage,
            {"draft": str(paths.draft_md)},
            tracker_factory=workflow_tracker,
        ),
        "pair-rewrite": stage_execution(
            "pair-rewrite",
            pair_rewrite_stage,
            {"draft": str(paths.draft_md), "review": expected_review_path},
            tracker_factory=workflow_tracker,
        ),
        "pair-verify": stage_execution(
            "pair-verify",
            pair_verify_stage,
            {"revised-draft": expected_revised_path},
            tracker_factory=workflow_tracker,
        ),
    }
    flags = {
        "research": research,
        "review": review or rewrite,
        "rewrite": rewrite,
    }
    with RadarRunLock(paths):
        begin_run(paths)
        try:
            execute_workflow(REGISTRY.workflow("daily-radar"), executions, flags)
        finally:
            finalize_run_usage(paths)
    return RadarRunResult(
        date=date,
        collected=collected,
        candidates_path=str(paths.candidates_json),
        selection_path=str(paths.selection_json),
        brief_path=str(paths.brief_md),
        draft_path=str(paths.draft_md),
        run_status_path=str(paths.run_status_json),
        review_path=review_path,
        revised_draft_path=revised_draft_path,
        verification_path=verification_path,
    )
