from __future__ import annotations

import json
import multiprocessing as mp
import queue
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ai_clip.core import billing
from ai_clip.core.artifacts import write_artifact_manifest
from ai_clip.core.config import Config
from ai_clip.core.config import RadarConfig
from ai_clip.core.stages import StageSpec
from ai_clip.radar.collect import (
    collect_channel_with_diagnostics,
    collect_channels,
    collect_channels_with_diagnostics,
    load_channels,
)
from ai_clip.radar.models import (
    ChannelCollectResult,
    RadarBackfillResult,
    RadarCandidates,
    RadarCollectReport,
    RadarRunResult,
    RadarSnapshot,
    ZackDraft,
    ZackSelection,
)
from ai_clip.radar.ops import RadarRunLock
from ai_clip.radar.storage import (
    RadarPaths,
    dedupe_snapshots,
    latest_previous_by_video,
    read_snapshots,
    write_text_atomic,
    write_json_model,
    write_snapshots,
)
from ai_clip.radar.status import is_stage_stale, mark_stale, track_stage
from ai_clip.source_content import add_source_content
from ai_clip.source_research import SourceResearchReport, generate_source_research
from ai_clip.zack_draft import generate_zack_draft, render_brief
from ai_clip.zack_ranking import load_ranking_feedback, rank_videos
from ai_clip.zack_selection import generate_zack_selection, render_selection


DAILY_RADAR_STAGES = (
    StageSpec(
        name="collect",
        description="Collect channel metadata snapshots.",
        inputs=("channels",),
        outputs=("snapshots", "collect-report"),
    ),
    StageSpec(
        name="zack-ranking",
        description="Rank snapshots into candidate topics.",
        inputs=("snapshots", "feedback"),
        outputs=("candidates",),
    ),
    StageSpec(
        name="source-content",
        description="Fetch subtitles or transcripts for ranked candidates.",
        inputs=("candidates", "channels"),
        outputs=("candidates", "source-content"),
    ),
    StageSpec(
        name="zack-selection",
        description="Choose one topic from the ranked candidates.",
        inputs=("candidates",),
        outputs=("selection-json", "selection-md"),
    ),
    StageSpec(
        name="source-research",
        description="Search and synthesize research for the selected topic.",
        inputs=("selection",),
        outputs=("research-json", "research-md"),
        optional=True,
    ),
    StageSpec(
        name="zack-draft",
        description="Generate the daily talking-head draft.",
        inputs=("candidates", "selection", "research"),
        outputs=("brief", "draft"),
    ),
    StageSpec(
        name="pair-review",
        description="Review the zack draft with two distinct models.",
        inputs=("draft",),
        outputs=("review"),
        optional=True,
    ),
    StageSpec(
        name="pair-rewrite",
        description="Rewrite the zack draft from the pair-review report.",
        inputs=("draft", "review"),
        outputs=("revised-draft"),
        optional=True,
    ),
)


def today_in_tz(tz_name: str) -> str:
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        if tz_name == "Asia/Shanghai":
            tz = timezone(timedelta(hours=8))
        else:
            raise
    return datetime.now(tz).date().isoformat()


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
        existing = dedupe_snapshots(read_snapshots(paths.existing_snapshot_jsonl()))
        if existing and not force:
            stage.set(
                status="skipped",
                outputs={"snapshots": str(paths.existing_snapshot_jsonl())},
                metrics={"snapshots": len(existing), "reused": True},
            )
            return len(existing)
        channels = load_channels(cfg.radar.channels_path)
        if cfg.radar.channel_timeout_sec <= 0:
            report = collect_channels_with_diagnostics(channels, cfg.radar)
        else:
            report = _collect_channels_with_timeout(
                channels,
                cfg.radar,
                cfg.radar.channel_timeout_sec,
            )
        merged = dedupe_snapshots(report.snapshots if force else existing + report.snapshots)
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
            },
        )
    if force:
        mark_stale(paths, _downstream_stage_names(paths), "force-collect replaced snapshots")
    return len(merged)


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
            "snapshots": str(paths.existing_snapshot_jsonl()),
            "feedback": cfg.radar.feedback_path,
            "top_n": str(top_n),
        },
    ) as stage:
        snapshots = dedupe_snapshots(read_snapshots(paths.existing_snapshot_jsonl()))
        previous = latest_previous_by_video(paths.root, date, legacy_root=paths.legacy_root)
        feedback = load_ranking_feedback(cfg.radar.feedback_path)
        ranked = rank_videos(snapshots, previous, top_n, feedback=feedback)
        candidates = RadarCandidates(date=date, top_n=top_n, videos=ranked)
        write_json_model(paths.candidates_json, candidates)
        _write_manifest(
            paths.candidates_json,
            stage="zack-ranking",
            inputs=_radar_inputs(paths.existing_snapshot_jsonl(), Path(cfg.radar.feedback_path)),
            params={"top_n": str(top_n)},
        )
        stage.set(
            outputs={"candidates": str(paths.candidates_json)},
            metrics={
                "snapshots": len(snapshots),
                "candidates": len(candidates.videos),
                "feedback_loaded": Path(cfg.radar.feedback_path).exists(),
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
            "candidates": str(paths.existing_candidates_json()),
            "channels": cfg.radar.channels_path,
        },
    ) as stage:
        candidates = _read_candidates(paths.existing_candidates_json())
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
        write_json_model(paths.candidates_json, updated)
        _write_manifest(
            paths.candidates_json,
            stage="source-content",
            inputs=_radar_inputs(
                paths.existing_snapshot_jsonl(),
                Path(cfg.radar.feedback_path),
                Path(cfg.radar.channels_path),
            ),
            params={
                "top_n": str(candidates.top_n),
                "transcribe_missing": str(cfg.radar.transcribe_missing),
                "transcribe_model_size": cfg.radar.transcribe_model_size,
            },
        )
        available = sum(1 for video in updated.videos if video.transcript_text)
        missing = sum(1 for video in updated.videos if not video.transcript_text)
        cached = sum(1 for video in updated.videos if video.content_status == "cached")
        stage.set(
            outputs={
                "candidates": str(paths.candidates_json),
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


def run_zack_selection(cfg: Config, date: str | None = None) -> ZackSelection:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    paths.ensure()
    with track_stage(
        paths,
        "zack-selection",
        inputs={"candidates": str(paths.existing_candidates_json())},
    ) as stage:
        candidates = _read_candidates(paths.existing_candidates_json())
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
            inputs=_radar_inputs(paths.existing_candidates_json()),
            model=cfg.llm.model,
        )
        _write_manifest(
            paths.selection_md,
            stage="zack-selection",
            inputs=_radar_inputs(paths.existing_candidates_json()),
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
            params={"max_searches": str(cfg.source_research.max_searches)},
            model=cfg.llm.model,
        )
        _write_manifest(
            paths.research_md,
            stage="source-research",
            inputs=_radar_inputs(paths.selection_json),
            params={"max_searches": str(cfg.source_research.max_searches)},
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
            "candidates": str(paths.existing_candidates_json()),
            "selection": str(paths.selection_json),
            "research": str(paths.research_md),
        },
    ) as stage:
        candidates = _read_candidates(paths.existing_candidates_json())
        selection = (
            _read_selection(paths.selection_json)
            if paths.selection_json.exists()
            else run_zack_selection(cfg, date)
        )
        research_markdown = (
            paths.research_md.read_text(encoding="utf-8")
            if paths.research_md.exists() and not is_stage_stale(paths, "source-research")
            else ""
        )
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
            paths.existing_candidates_json(),
            paths.selection_json,
            *( [paths.research_md] if research_markdown else [] ),
        )
        _write_manifest(
            paths.brief_md,
            stage="zack-draft",
            inputs=_radar_inputs(paths.existing_candidates_json()),
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


def run_all(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
    research: bool = False,
    force_collect: bool = False,
    review: bool = False,
    rewrite: bool = False,
) -> RadarRunResult:
    date = date or today_in_tz(cfg.radar.timezone)
    paths = RadarPaths(cfg.data_dir, date)
    with RadarRunLock(paths):
        collected = run_collect(cfg, date, force=force_collect)
        run_zack_ranking(cfg, date, top_n)
        run_source_content(cfg, date)
        run_zack_selection(cfg, date)
        if research:
            run_source_research(cfg, date)
        run_zack_draft(cfg, date)
        review_path = ""
        revised_draft_path = ""
        if review or rewrite:
            review_path, revised_draft_path = _run_pair_review(
                cfg,
                date,
                rewrite=rewrite,
            )
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
        )


def run_backfill(
    cfg: Config,
    days: int = 7,
    end_date: str | None = None,
    top_n: int | None = None,
    channel_limit: int | None = None,
    channel_timeout: int = 30,
) -> RadarBackfillResult:
    end_date = end_date or today_in_tz(cfg.radar.timezone)
    top_n = top_n or cfg.radar.top_n
    days = max(days, 1)
    end = datetime.fromisoformat(end_date).date()
    wanted_dates = [(end - timedelta(days=offset)).isoformat() for offset in range(days)]
    wanted = set(wanted_dates)

    radar_cfg = cfg.radar.model_copy(update={
        "since_days": 0,
        "channel_limit": channel_limit or cfg.radar.channel_limit,
    })
    channels = load_channels(cfg.radar.channels_path)
    if channel_timeout <= 0:
        collected_snapshots = collect_channels(channels, radar_cfg)
    else:
        collected_snapshots = _collect_channels_with_timeout(
            channels,
            radar_cfg,
            channel_timeout,
        ).snapshots
    snapshots = [
        snapshot for snapshot in collected_snapshots if snapshot.video.published_date in wanted
    ]

    paths = RadarPaths(cfg.data_dir, end_date)
    paths.ensure()
    out_dir = paths.backfill_run_dir(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    by_date: dict[str, list[RadarSnapshot]] = {date: [] for date in reversed(wanted_dates)}
    for snapshot in snapshots:
        by_date.setdefault(snapshot.video.published_date, []).append(snapshot)

    summary_lines = [f"# Radar Backfill Top {top_n}: {end_date}", ""]
    for date, day_snapshots in by_date.items():
        ranked = rank_videos(day_snapshots, previous={}, top_n=top_n)
        candidates = RadarCandidates(date=date, top_n=top_n, videos=ranked)
        json_path = out_dir / f"{date}_top{top_n}.json"
        md_path = out_dir / f"{date}_top{top_n}.md"
        write_json_model(json_path, candidates)
        md_path.write_text(render_brief(candidates), encoding="utf-8")
        files.extend([str(json_path), str(md_path)])
        summary_lines += [f"## {date}", ""]
        if not ranked:
            summary_lines += ["- No candidates collected.", ""]
            continue
        for i, video in enumerate(ranked, start=1):
            summary_lines.append(
                f"{i}. {video.title} | {video.platform} | score={video.score} | {video.url}"
            )
        summary_lines.append("")

    summary_path = out_dir / f"{end_date}_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    files.append(str(summary_path))
    return RadarBackfillResult(
        end_date=end_date,
        days=days,
        collected=len(snapshots),
        output_dir=str(out_dir),
        files=files,
    )


def _collect_channels_with_timeout(
    channels: list,
    radar_cfg: RadarConfig,
    timeout_sec: int,
) -> RadarCollectReport:
    collected_at = datetime.now(timezone.utc).isoformat()
    snapshots: list[RadarSnapshot] = []
    results: list[ChannelCollectResult] = []
    seen: set[str] = set()
    ctx = mp.get_context("spawn")
    pending = list(channels)
    running: dict[int, tuple[mp.Process, object, object, float]] = {}
    max_workers = max(int(radar_cfg.channel_workers or 1), 1)
    while pending or running:
        while pending and len(running) < max_workers:
            channel = pending.pop(0)
            q = ctx.Queue()
            proc = ctx.Process(target=_collect_channel_process, args=(channel, radar_cfg, q))
            proc.start()
            running[proc.pid or id(proc)] = (proc, q, channel, time.monotonic())

        for key, (proc, q, channel, start) in list(running.items()):
            try:
                raw = q.get_nowait()
            except queue.Empty:
                if time.monotonic() - start >= timeout_sec:
                    proc.terminate()
                    proc.join(timeout=5)
                    results.append(ChannelCollectResult(
                        platform=channel.platform,
                        url=channel.url,
                        name=channel.name,
                        status="timeout",
                        duration_sec=round(timeout_sec, 3),
                        error=f"channel collect exceeded {timeout_sec}s",
                    ))
                    del running[key]
                elif not proc.is_alive():
                    proc.join(timeout=1)
                    results.append(ChannelCollectResult(
                        platform=channel.platform,
                        url=channel.url,
                        name=channel.name,
                        status="failed",
                        duration_sec=round(max(time.monotonic() - start, 0.0), 3),
                        error=f"collector process exited with code {proc.exitcode}",
                    ))
                    del running[key]
                continue
            proc.join(timeout=5)
            del running[key]
            result = ChannelCollectResult.model_validate(raw["result"])
            results.append(result)
            for raw_video in raw["videos"]:
                video = RadarSnapshot.model_validate({
                    "collected_at": collected_at,
                    "video": raw_video,
                }).video
                if video.video_id in seen:
                    continue
                seen.add(video.video_id)
                snapshots.append(RadarSnapshot(collected_at=collected_at, video=video))
        if running:
            time.sleep(0.05)
    return RadarCollectReport(collected_at=collected_at, snapshots=snapshots, channels=results)


def _collect_channel_process(channel, radar_cfg: RadarConfig, q) -> None:
    videos, result = collect_channel_with_diagnostics(channel, radar_cfg)
    q.put({
        "videos": [video.model_dump(mode="json") for video in videos],
        "result": result.model_dump(mode="json"),
    })


def _run_pair_review(cfg: Config, date: str, rewrite: bool = False) -> tuple[str, str]:
    from ai_clip.pair.stage import review_artifact, rewrite_reviewed_artifact

    paths = RadarPaths(cfg.data_dir, date)
    review_path = ""
    revised_path = ""
    with track_stage(
        paths,
        "pair-review",
        inputs={"draft": str(paths.existing_draft_md())},
    ) as stage:
        report = review_artifact(cfg, "radar", "zack_draft", run_date=date)
        review_path = str(paths.reviews_dir / f"{date}_zack_draft_review.json")
        stage.set(
            outputs={"review": review_path},
            metrics={"status": report.status, "reviewers": len(report.reviewers)},
        )
    if rewrite:
        with track_stage(
            paths,
            "pair-rewrite",
            inputs={"draft": str(paths.existing_draft_md()), "review": review_path},
        ) as stage:
            revised = rewrite_reviewed_artifact(
                cfg,
                "radar",
                "zack_draft",
                report,
                run_date=date,
            )
            revised_path = str(revised)
            stage.set(outputs={"revised-draft": revised_path})
    return review_path, revised_path


def _downstream_stage_names(paths: RadarPaths) -> list[str]:
    names = ["zack-ranking", "source-content", "zack-selection", "zack-draft"]
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




