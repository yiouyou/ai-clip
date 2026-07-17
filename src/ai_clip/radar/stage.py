from __future__ import annotations

import json
from pathlib import Path

from ai_clip.core import billing
from ai_clip.core.artifacts import artifact_matches, write_artifact_manifest
from ai_clip.core.config import Config
from ai_clip.radar.collect import (
    collect_channels_with_timeout,
    collect_channels_with_diagnostics,
    load_channels,
)
from ai_clip.radar.feedback import apply_feedback_events, read_feedback_events
from ai_clip.radar.models import (
    RadarCandidates,
    RadarCollectReport,
    RadarSnapshot,
    ZackDraft,
    ZackSelection,
)
from ai_clip.radar.storage import (
    RadarPaths,
    dedupe_snapshots,
    latest_previous_by_video,
    read_snapshots,
    write_text_atomic,
    write_json_model,
    write_snapshots,
)
from ai_clip.radar.status import mark_stale, track_stage
from ai_clip.radar.time import today_in_tz
from ai_clip.source_content import add_source_content
from ai_clip.source_research import SourceResearchReport, generate_source_research
from ai_clip.zack_draft import generate_zack_draft, render_brief
from ai_clip.zack_ranking import load_ranking_feedback, rank_videos, rerank_by_content
from ai_clip.zack_selection import generate_zack_selection, render_selection


def run_collect(cfg: Config, date: str | None = None, force: bool = False) -> int:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "collect",
        inputs={
            "channels": cfg.radar.channels_path,
            "date": date,
            "force": str(force),
        },
    ) as stage:
        existing = dedupe_snapshots(read_snapshots(paths.snapshot_jsonl))
        if existing and not force:
            stage.set(
                status="skipped",
                outputs={"snapshots": str(paths.snapshot_jsonl)},
                metrics={"snapshots": len(existing), "reused": True},
            )
            return len(existing)
        channels = load_channels(cfg.radar.channels_path)
        if cfg.radar.channel_timeout_sec <= 0:
            report = collect_channels_with_diagnostics(channels, cfg.radar)
        else:
            report = collect_channels_with_timeout(
                channels,
                cfg.radar,
                cfg.radar.channel_timeout_sec,
            )
        preserved = 0
        if force:
            merged, preserved = _merge_forced_snapshots(existing, report)
        else:
            merged = dedupe_snapshots(existing + report.snapshots)
        write_snapshots(paths.snapshot_jsonl, merged)
        write_json_model(paths.collect_report_json, report.model_copy(update={"snapshots": merged}))
        failed = sum(1 for channel in report.channels if channel.status != "succeeded")
        stage.set(
            outputs={
                "snapshots": str(paths.snapshot_jsonl),
                "collect-report": str(paths.collect_report_json),
            },
            metrics={
                "snapshots": len(merged),
                "channels": len(report.channels),
                "channels_failed": failed,
                "force": force,
                "snapshots_preserved": preserved,
            },
        )
    if force:
        mark_stale(paths, _downstream_stage_names(paths), "force-collect replaced snapshots")
    return len(merged)


def _merge_forced_snapshots(
    existing: list[RadarSnapshot],
    report: RadarCollectReport,
) -> tuple[list[RadarSnapshot], int]:
    preserve_urls = {
        channel.url
        for channel in report.channels
        if channel.status != "succeeded"
    }
    preserved = [
        snapshot
        for snapshot in existing
        if snapshot.video.channel_url in preserve_urls
    ]
    return dedupe_snapshots(preserved + report.snapshots), len(preserved)


def run_zack_ranking(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
) -> RadarCandidates:
    date = date or today_in_tz(cfg.radar.timezone)
    top_n = top_n or cfg.radar.top_n
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "zack-ranking",
        inputs={
            "snapshots": str(paths.snapshot_jsonl),
            "feedback": cfg.radar.feedback_path,
            "top_n": str(top_n),
        },
    ) as stage:
        snapshots = dedupe_snapshots(read_snapshots(paths.snapshot_jsonl))
        previous = latest_previous_by_video(paths.root, date)
        feedback = apply_feedback_events(
            load_ranking_feedback(cfg.radar.feedback_path),
            read_feedback_events(paths.feedback_events_jsonl),
        )
        shortlist_n = max(top_n, cfg.radar.shortlist_n)
        ranked = rank_videos(snapshots, previous, shortlist_n, feedback=feedback)
        candidates = RadarCandidates(
            date=date,
            top_n=top_n,
            shortlist_n=len(ranked),
            ranking_phase="metadata-shortlist",
            videos=ranked,
        )
        write_json_model(paths.shortlist_json, candidates)
        _write_manifest(
            paths.shortlist_json,
            stage="zack-ranking",
            inputs=_radar_inputs(
                paths.snapshot_jsonl,
                Path(cfg.radar.feedback_path),
                paths.feedback_events_jsonl,
            ),
            params={"top_n": str(top_n), "shortlist_n": str(shortlist_n)},
        )
        stage.set(
            outputs={"shortlist": str(paths.shortlist_json)},
            metrics={
                "snapshots": len(snapshots),
                "shortlist": len(candidates.videos),
                "target_candidates": top_n,
                "feedback_loaded": Path(cfg.radar.feedback_path).exists(),
                "feedback_events": len(read_feedback_events(paths.feedback_events_jsonl)),
            },
        )
        return candidates


def run_source_content(cfg: Config, date: str | None = None) -> RadarCandidates:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "source-content",
        inputs={
            "shortlist": str(paths.shortlist_json),
            "channels": cfg.radar.channels_path,
        },
    ) as stage:
        candidates = _read_candidates(paths.shortlist_json)
        channels = load_channels(cfg.radar.channels_path)
        radar_whisper = cfg.whisper.model_copy(update={"model_size": cfg.radar.transcribe_model_size})
        enriched = add_source_content(
            candidates.videos,
            channels,
            paths.source_content_dir,
            whisper=radar_whisper,
            transcribe_missing=cfg.radar.transcribe_missing,
        )
        updated = candidates.model_copy(update={"videos": enriched})
        write_json_model(paths.shortlist_json, updated)
        _write_manifest(
            paths.shortlist_json,
            stage="source-content",
            inputs=_radar_inputs(
                paths.snapshot_jsonl,
                Path(cfg.radar.feedback_path),
                paths.feedback_events_jsonl,
                Path(cfg.radar.channels_path),
            ),
            params={
                "top_n": str(candidates.top_n),
                "shortlist_n": str(candidates.shortlist_n or len(candidates.videos)),
                "transcribe_missing": str(cfg.radar.transcribe_missing),
                "transcribe_model_size": cfg.radar.transcribe_model_size,
            },
        )
        available = sum(1 for video in updated.videos if video.transcript_text)
        missing = sum(1 for video in updated.videos if not video.transcript_text)
        cached = sum(1 for video in updated.videos if video.content_status == "cached")
        stage.set(
            outputs={
                "shortlist": str(paths.shortlist_json),
                "source-content": str(paths.source_content_dir),
            },
            metrics={
                "videos": len(updated.videos),
                "scripts_available": available,
                "scripts_missing": missing,
                "scripts_cached": cached,
            },
        )
        return updated


def run_content_rerank(cfg: Config, date: str | None = None) -> RadarCandidates:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "content-rerank",
        inputs={"shortlist": str(paths.shortlist_json)},
    ) as stage:
        shortlist = _read_candidates(paths.shortlist_json)
        reranked = rerank_by_content(shortlist.videos, shortlist.top_n)
        candidates = shortlist.model_copy(update={
            "ranking_phase": "content-reranked",
            "videos": reranked,
        })
        write_json_model(paths.candidates_json, candidates)
        _write_manifest(
            paths.candidates_json,
            stage="content-rerank",
            inputs=[paths.shortlist_json],
            params={"top_n": str(shortlist.top_n)},
        )
        stage.set(
            outputs={"candidates": str(paths.candidates_json)},
            metrics={
                "shortlist": len(shortlist.videos),
                "candidates": len(candidates.videos),
                "scripts_available": sum(1 for video in shortlist.videos if video.transcript_text),
            },
        )
        return candidates


def run_zack_selection(cfg: Config, date: str | None = None) -> ZackSelection:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "zack-selection",
        inputs={"candidates": str(paths.candidates_json)},
    ) as stage:
        if paths.shortlist_json.exists():
            shortlist = _read_candidates(paths.shortlist_json)
            if not artifact_matches(
                paths.candidates_json,
                inputs=[paths.shortlist_json],
                params={"top_n": str(shortlist.top_n)},
            ):
                run_content_rerank(cfg, date)
        candidates = _read_candidates(paths.candidates_json)
        if not candidates.videos:
            raise ValueError(
                f"no candidates available for zack-selection on {date}; "
                "collect/ranking produced an empty candidate set"
            )
        with billing.account(paths.root, "zack_selection"):
            selection = generate_zack_selection(candidates, cfg.llm)
        write_json_model(paths.selection_json, selection)
        write_text_atomic(paths.selection_md, render_selection(selection), encoding="utf-8")
        _write_manifest(
            paths.selection_json,
            stage="zack-selection",
            inputs=_radar_inputs(paths.candidates_json),
            model=cfg.llm.model,
        )
        _write_manifest(
            paths.selection_md,
            stage="zack-selection",
            inputs=_radar_inputs(paths.candidates_json),
            model=cfg.llm.model,
        )
        stage.set(
            outputs={
                "selection-json": str(paths.selection_json),
                "selection-md": str(paths.selection_md),
            },
            metrics={
                "selected_index": selection.selected_index,
                "fact_risk": selection.fact_risk,
            },
        )
        return selection


def run_source_research(cfg: Config, date: str | None = None) -> SourceResearchReport:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "source-research",
        inputs={"selection": str(paths.selection_json)},
    ) as stage:
        selection = (
            _read_selection(paths.selection_json)
            if paths.selection_json.exists()
            else run_zack_selection(cfg, date)
        )
        with billing.account(paths.root, "source_research"):
            report = generate_source_research(selection, cfg)
        write_json_model(paths.research_json, report)
        write_text_atomic(paths.research_md, report.markdown, encoding="utf-8")
        _write_manifest(
            paths.research_json,
            stage="source-research",
            inputs=_radar_inputs(paths.selection_json),
            params=_source_research_params(cfg),
            model=cfg.llm.model,
        )
        _write_manifest(
            paths.research_md,
            stage="source-research",
            inputs=_radar_inputs(paths.selection_json),
            params=_source_research_params(cfg),
            model=cfg.llm.model,
        )
        stage.set(
            outputs={
                "research-json": str(paths.research_json),
                "research-md": str(paths.research_md),
            },
            metrics={"search_calls": report.search_calls},
        )
        return report


def run_zack_draft(cfg: Config, date: str | None = None) -> ZackDraft:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "zack-draft",
        inputs={
            "candidates": str(paths.candidates_json),
            "selection": str(paths.selection_json),
            "research": str(paths.research_md),
        },
    ) as stage:
        candidates = _read_candidates(paths.candidates_json)
        selection = (
            _read_selection(paths.selection_json)
            if paths.selection_json.exists()
            else run_zack_selection(cfg, date)
        )
        research_markdown = _current_source_research(paths, cfg)
        with billing.account(paths.root, "zack_draft"):
            markdown = generate_zack_draft(
                candidates,
                cfg.llm,
                selection=selection,
                research_markdown=research_markdown,
            )
        draft = ZackDraft(
            date=date,
            title=f"今日选题雷达 {date}",
            markdown=markdown,
            videos=candidates.videos,
        )
        write_text_atomic(paths.brief_md, render_brief(candidates), encoding="utf-8")
        write_text_atomic(paths.draft_md, markdown, encoding="utf-8")
        draft_inputs = _radar_inputs(
            paths.candidates_json,
            paths.selection_json,
            *( [paths.research_md] if research_markdown else [] ),
        )
        _write_manifest(
            paths.brief_md,
            stage="zack-draft",
            inputs=_radar_inputs(paths.candidates_json),
            model=cfg.llm.model,
        )
        _write_manifest(
            paths.draft_md,
            stage="zack-draft",
            inputs=draft_inputs,
            params={"research_used": str(bool(research_markdown))},
            model=cfg.llm.model,
        )
        stage.set(
            outputs={
                "brief": str(paths.brief_md),
                "draft": str(paths.draft_md),
            },
            metrics={"videos": len(candidates.videos), "research_used": bool(research_markdown)},
        )
        return draft


def _downstream_stage_names(paths: RadarPaths) -> list[str]:
    names = [
        "zack-ranking",
        "source-content",
        "content-rerank",
        "zack-selection",
        "zack-draft",
    ]
    if paths.research_json.exists() or paths.research_md.exists():
        names.append("source-research")
    review_path = paths.reviews_dir / f"{paths.date}_zack_draft_review.json"
    if review_path.exists():
        names.append("pair-review")
    if paths.draft_revised_md.exists():
        names.append("pair-rewrite")
    return names


def _read_candidates(path: Path) -> RadarCandidates:
    return RadarCandidates.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_selection(path: Path) -> ZackSelection:
    return ZackSelection.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _radar_inputs(*paths: Path) -> list[Path]:
    return list(paths)


def _source_research_params(cfg: Config) -> dict[str, str]:
    return {"max_searches": str(cfg.source_research.max_searches)}


def _current_source_research(paths: RadarPaths, cfg: Config) -> str:
    if not artifact_matches(
        paths.research_md,
        inputs=[paths.selection_json],
        params=_source_research_params(cfg),
        model=cfg.llm.model,
    ):
        return ""
    return paths.research_md.read_text(encoding="utf-8")


def _write_manifest(
    path: Path,
    *,
    stage: str,
    inputs: list[Path],
    params: dict[str, str] | None = None,
    model: str = "",
) -> None:
    write_artifact_manifest(
        path,
        stage=stage,
        inputs=inputs,
        params=params,
        model=model,
    )

