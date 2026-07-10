from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

from ai_clip.core.run_lock import RunLock
from ai_clip.radar.models import RadarCandidates, RadarRunStatus
from ai_clip.radar.status import radar_status_store
from ai_clip.radar.storage import RadarPaths, read_snapshots


@dataclass(frozen=True)
class RadarStatusSummary:
    date: str
    status: str
    run_status_path: str
    run_id: str = ""
    attempt: int = 0
    collect_report_path: str = ""
    stages: list[dict[str, object]] = field(default_factory=list)
    channel_counts: dict[str, int] = field(default_factory=dict)
    channel_failures: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    usage: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairResult:
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


class RadarRunLock(RunLock):
    def __init__(
        self,
        paths: RadarPaths,
        stale_after_minutes: int = 180,
    ) -> None:
        super().__init__(
            paths.root / "locks" / f"{paths.date}.lock",
            f"daily-radar is already running for {paths.date}",
            stale_after_minutes=stale_after_minutes,
            metadata={"date": paths.date},
        )


def read_radar_status(paths: RadarPaths) -> RadarStatusSummary:
    status = _read_run_status(paths)
    collect_report = _read_json(paths.collect_report_json)
    channel_counts: dict[str, int] = {}
    channel_failures: list[str] = []
    for item in collect_report.get("channels", []) if isinstance(collect_report, dict) else []:
        state = str(item.get("status") or "unknown")
        channel_counts[state] = channel_counts.get(state, 0) + 1
        if state != "succeeded":
            label = str(item.get("name") or item.get("url") or "channel")
            error = str(item.get("error") or "")
            channel_failures.append(f"{label}: {state}{f' - {error}' if error else ''}")
    artifacts = {
        "snapshots": str(paths.snapshot_jsonl),
        "shortlist": str(paths.shortlist_json),
        "candidates": str(paths.candidates_json),
        "selection": str(paths.selection_json),
        "research": str(paths.research_md),
        "draft": str(paths.draft_md),
        "revised_draft": str(paths.draft_revised_md),
        "verification": str(paths.reviews_dir / f"{paths.date}_zack_draft_verify.json"),
        "feedback": str(paths.feedback_events_jsonl),
    }
    return RadarStatusSummary(
        date=paths.date,
        status=status.status,
        run_status_path=str(paths.run_status_json),
        run_id=status.run_id,
        attempt=status.attempt,
        collect_report_path=str(paths.collect_report_json) if paths.collect_report_json.exists() else "",
        stages=[
            {
                "name": stage.name,
                "status": stage.status,
                "duration": str(stage.duration_sec),
                "error": stage.error,
                "started_at": stage.started_at,
                "finished_at": stage.finished_at,
                "inputs": stage.inputs,
                "outputs": stage.outputs,
                "metrics": stage.metrics,
            }
            for stage in status.stages
        ],
        channel_counts=channel_counts,
        channel_failures=channel_failures,
        artifacts=artifacts,
        usage=status.usage,
    )


def repair_radar_date(paths: RadarPaths, apply: bool = False) -> RepairResult:
    removed: list[str] = []
    kept: list[str] = []
    candidates = _empty_candidates(paths.candidates_json)
    shortlist = _empty_candidates(paths.shortlist_json)
    snapshots_empty = paths.snapshot_jsonl.exists() and not read_snapshots(paths.snapshot_jsonl)
    status = _read_run_status(paths)
    failed_or_stale = status.status in {"failed", "stale", "pending"}

    targets: list[Path] = []
    if snapshots_empty and failed_or_stale:
        targets.append(paths.snapshot_jsonl)
    if candidates and failed_or_stale:
        targets.append(paths.candidates_json)
    if shortlist and failed_or_stale:
        targets.append(paths.shortlist_json)
    if not targets:
        return RepairResult()
    for path in targets:
        if apply:
            path.unlink(missing_ok=True)
            removed.append(str(path))
        else:
            kept.append(str(path))
    return RepairResult(removed=removed, kept=kept)


def _read_run_status(paths: RadarPaths) -> RadarRunStatus:
    if not paths.run_status_json.exists():
        return RadarRunStatus(date=paths.date, status="missing")
    return radar_status_store(paths).read()


def _read_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _empty_candidates(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        candidates = RadarCandidates.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError):
        return True
    return not candidates.videos
