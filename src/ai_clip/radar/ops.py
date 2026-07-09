from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

from ai_clip.radar.models import RadarCandidates, RadarRunStatus
from ai_clip.radar.storage import RadarPaths, read_snapshots, write_text_atomic


@dataclass(frozen=True)
class RadarStatusSummary:
    date: str
    status: str
    run_status_path: str
    collect_report_path: str = ""
    stages: list[dict[str, str]] = field(default_factory=list)
    channel_counts: dict[str, int] = field(default_factory=dict)
    channel_failures: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairResult:
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


class RadarRunLock(AbstractContextManager["RadarRunLock"]):
    def __init__(
        self,
        paths: RadarPaths,
        stale_after_minutes: int = 180,
    ) -> None:
        self.paths = paths
        self.stale_after_minutes = stale_after_minutes
        self.path = paths.root / "locks" / f"{paths.date}.lock"
        self.acquired = False

    def __enter__(self) -> "RadarRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not _lock_is_stale(self.path, self.stale_after_minutes):
            raise RuntimeError(f"daily-radar is already running for {self.paths.date}: {self.path}")
        payload = {
            "pid": os.getpid(),
            "date": self.paths.date,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        write_text_atomic(self.path, json.dumps(payload, indent=2), encoding="utf-8")
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False


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
        "candidates": str(paths.candidates_json),
        "selection": str(paths.selection_json),
        "research": str(paths.research_md),
        "draft": str(paths.draft_md),
        "revised_draft": str(paths.draft_revised_md),
    }
    return RadarStatusSummary(
        date=paths.date,
        status=status.status,
        run_status_path=str(paths.run_status_json),
        collect_report_path=str(paths.collect_report_json) if paths.collect_report_json.exists() else "",
        stages=[
            {
                "name": stage.name,
                "status": stage.status,
                "duration": str(stage.duration_sec),
                "error": stage.error,
            }
            for stage in status.stages
        ],
        channel_counts=channel_counts,
        channel_failures=channel_failures,
        artifacts=artifacts,
    )


def repair_radar_date(paths: RadarPaths, apply: bool = False) -> RepairResult:
    removed: list[str] = []
    kept: list[str] = []
    candidates = _empty_candidates(paths.candidates_json)
    snapshots_empty = paths.snapshot_jsonl.exists() and not read_snapshots(paths.snapshot_jsonl)
    status = _read_run_status(paths)
    failed_or_stale = status.status in {"failed", "stale", "pending"}

    targets: list[Path] = []
    if snapshots_empty and failed_or_stale:
        targets.append(paths.snapshot_jsonl)
    if candidates and failed_or_stale:
        targets.append(paths.candidates_json)
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
    data = _read_json(paths.run_status_json)
    if not isinstance(data, dict):
        return RadarRunStatus(date=paths.date, status="invalid")
    return RadarRunStatus.model_validate(data)


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


def _lock_is_stale(path: Path, stale_after_minutes: int) -> bool:
    data = _read_json(path)
    if not isinstance(data, dict):
        return True
    started_at = str(data.get("started_at") or "")
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return True
    age = datetime.now(timezone.utc) - started.astimezone(timezone.utc)
    if age > timedelta(minutes=stale_after_minutes):
        return True
    pid = data.get("pid")
    return isinstance(pid, int) and not _pid_exists(pid)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_exists(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_pid_exists(pid: int) -> bool:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True
