from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import re
from types import TracebackType

from ai_clip.radar.models import RadarRunStage, RadarRunStatus
from ai_clip.radar.storage import RadarPaths, write_json_model


class StageTracker(AbstractContextManager["StageTracker"]):
    def __init__(
        self,
        paths: RadarPaths,
        name: str,
        inputs: dict[str, str] | None = None,
    ) -> None:
        self.paths = paths
        self.name = name
        self.inputs = inputs or {}
        self.outputs: dict[str, str] = {}
        self.metrics: dict[str, str | int | float | bool] = {}
        self.status = "succeeded"
        self.error = ""
        self._started_at = ""

    def set(
        self,
        *,
        status: str | None = None,
        outputs: dict[str, str] | None = None,
        metrics: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        if status is not None:
            self.status = status
        if outputs:
            self.outputs.update(outputs)
        if metrics:
            self.metrics.update(metrics)

    def __enter__(self) -> "StageTracker":
        self._started_at = _now()
        status = _read_status(self.paths)
        status.started_at = status.started_at or self._started_at
        status.updated_at = self._started_at
        _upsert_stage(
            status,
            RadarRunStage(
                name=self.name,
                status="running",
                started_at=self._started_at,
                inputs=self.inputs,
            ),
        )
        _write_status(self.paths, status)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        finished_at = _now()
        if exc_value is not None:
            self.status = "failed"
            self.error = _sanitize_error(exc_value)
        status = _read_status(self.paths)
        started_at = self._started_at or finished_at
        _upsert_stage(
            status,
            RadarRunStage(
                name=self.name,
                status=self.status,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=_duration(started_at, finished_at),
                inputs=self.inputs,
                outputs=self.outputs,
                metrics=self.metrics,
                error=self.error,
            ),
        )
        status.updated_at = finished_at
        status.status = _overall_status(status)
        _write_status(self.paths, status)
        return False


def track_stage(
    paths: RadarPaths,
    name: str,
    inputs: dict[str, str] | None = None,
) -> StageTracker:
    return StageTracker(paths, name, inputs)


def mark_stale(
    paths: RadarPaths,
    stage_names: list[str],
    reason: str,
    outputs: dict[str, str] | None = None,
) -> None:
    status = _read_status(paths)
    now = _now()
    status.started_at = status.started_at or now
    for name in stage_names:
        _upsert_stage(
            status,
            RadarRunStage(
                name=name,
                status="stale",
                finished_at=now,
                outputs=outputs or {},
                metrics={"reason": reason},
            ),
        )
    status.updated_at = now
    status.status = _overall_status(status)
    _write_status(paths, status)


def is_stage_stale(paths: RadarPaths, stage_name: str) -> bool:
    status = _read_status(paths)
    return any(stage.name == stage_name and stage.status == "stale" for stage in status.stages)


def _read_status(paths: RadarPaths) -> RadarRunStatus:
    if not paths.run_status_json.exists():
        return RadarRunStatus(date=paths.date)
    try:
        data = json.loads(paths.run_status_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return RadarRunStatus(date=paths.date)
    return RadarRunStatus.model_validate(data)


def _write_status(paths: RadarPaths, status: RadarRunStatus) -> None:
    write_json_model(paths.run_status_json, status)


def _upsert_stage(status: RadarRunStatus, stage: RadarRunStage) -> None:
    status.stages = [item for item in status.stages if item.name != stage.name]
    status.stages.append(stage)


def _overall_status(status: RadarRunStatus) -> str:
    stage_statuses = [stage.status for stage in status.stages]
    if any(item == "failed" for item in stage_statuses):
        return "failed"
    if any(item == "running" for item in stage_statuses):
        return "running"
    if any(item == "stale" for item in stage_statuses):
        return "stale"
    if stage_statuses and all(item in {"succeeded", "skipped"} for item in stage_statuses):
        return "succeeded"
    return "pending"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration(started_at: str, finished_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at)
        finish = datetime.fromisoformat(finished_at)
    except ValueError:
        return 0.0
    return round(max((finish - start).total_seconds(), 0.0), 3)


def _sanitize_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(r"(?i)(api[_-]?key|authorization|cookie|token)=\S+", r"\1=<redacted>", text)
    return text[:500]
