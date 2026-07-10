from __future__ import annotations

from ai_clip.core.config import Config
from ai_clip.core.stages import execute_workflow
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

    def collect_stage():
        nonlocal collected
        collected = radar_stages.run_collect(cfg, date, force=force_collect)

    def ranking_stage():
        radar_stages.run_zack_ranking(cfg, date, top_n)

    def content_stage():
        radar_stages.run_source_content(cfg, date)

    def content_rerank_stage():
        radar_stages.run_content_rerank(cfg, date)

    def selection_stage():
        selection = radar_stages.run_zack_selection(cfg, date)
        if not research:
            searches = automatic_research_searches(selection, cfg)
            if searches:
                cfg.source_research.max_searches = searches
                flags["research"] = True
        return selection

    def research_stage():
        radar_stages.run_source_research(cfg, date)

    def draft_stage():
        radar_stages.run_zack_draft(cfg, date)

    def pair_review_stage():
        nonlocal review_path, review_report
        from ai_clip.pair.stage import review_artifact

        with track_stage(
            paths,
            "pair-review",
            inputs={"draft": str(paths.draft_md)},
        ) as stage:
            review_report = review_artifact(cfg, "radar", "zack_draft", run_date=date)
            review_path = str(paths.reviews_dir / f"{date}_zack_draft_review.json")
            stage.set(
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
        with track_stage(
            paths,
            "pair-rewrite",
            inputs={"draft": str(paths.draft_md), "review": review_path},
        ) as stage:
            if review_report.status == "blocked":
                stage.set(status="skipped", metrics={"reason": "pair-review blocked"})
                return
            revised = rewrite_reviewed_artifact(
                cfg,
                "radar",
                "zack_draft",
                review_report,
                run_date=date,
            )
            revised_draft_path = str(revised)
            stage.set(outputs={"revised-draft": revised_draft_path})

    def pair_verify_stage():
        nonlocal verification_path
        from ai_clip.pair.stage import verify_rewritten_artifact

        with track_stage(
            paths,
            "pair-verify",
            inputs={"revised-draft": revised_draft_path},
        ) as stage:
            if not revised_draft_path:
                stage.set(status="skipped", metrics={"reason": "pair-rewrite produced no draft"})
                return
            report = verify_rewritten_artifact(
                cfg,
                "radar",
                "zack_draft",
                run_date=date,
            )
            verification_path = str(paths.reviews_dir / f"{date}_zack_draft_verify.json")
            stage.set(
                outputs={"verification": verification_path},
                metrics={
                    "status": report.status,
                    "reviewers": len(report.reviewers),
                    "bounded": True,
                },
            )

    handlers = {
        "collect": collect_stage,
        "zack-ranking": ranking_stage,
        "source-content": content_stage,
        "content-rerank": content_rerank_stage,
        "zack-selection": selection_stage,
        "source-research": research_stage,
        "zack-draft": draft_stage,
        "pair-review": pair_review_stage,
        "pair-rewrite": pair_rewrite_stage,
        "pair-verify": pair_verify_stage,
    }
    flags = {
        "research": research,
        "review": review or rewrite,
        "rewrite": rewrite,
    }
    with RadarRunLock(paths):
        begin_run(paths)
        try:
            execute_workflow(REGISTRY.workflow("daily-radar"), handlers, flags)
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
