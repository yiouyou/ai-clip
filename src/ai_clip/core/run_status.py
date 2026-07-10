from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from types import TracebackType
from typing import Generic, TypeVar
import uuid

from pydantic import BaseModel, Field

from ai_clip.core.artifacts import write_model
from ai_clip.core.paths import ProjectPaths


class RunStage(BaseModel):
    name: str
    status: str = "pending"
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, str | int | float | bool] = Field(default_factory=dict)
    error: str = ""


class WorkflowRunStatus(BaseModel):
    workflow: str
    project: str = ""
    date: str = ""
    run_id: str = ""
    attempt: int = 0
    status: str = "pending"
    started_at: str = ""
    updated_at: str = ""
    stages: list[RunStage] = Field(default_factory=list)
    usage: dict[str, object] = Field(default_factory=dict)


StatusT = TypeVar("StatusT", bound=WorkflowRunStatus)


class RunStatusStore(Generic[StatusT]):
    """Read and atomically write one workflow status document."""

    def __init__(
        self,
        path: Path,
        model_cls: type[StatusT],
        default_factory: Callable[[], StatusT],
    ) -> None:
        self.path = path
        self.model_cls = model_cls
        self.default_factory = default_factory

    def read(self) -> StatusT:
        if not self.path.exists():
            return self.default_factory()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return self.model_cls.model_validate(data)
        except (OSError, ValueError, json.JSONDecodeError):
            status = self.default_factory()
            status.status = "invalid"
            return status

    def write(self, status: StatusT) -> None:
        write_model(self.path, status)

    def begin(self) -> StatusT:
        previous = self.read()
        if previous.stages:
            _mark_running_stale(previous, "previous run did not finish")
            self.archive(previous)
        run = self.default_factory()
        run.run_id = _new_run_id()
        run.attempt = max(previous.attempt, 0) + 1
        run.status = "pending"
        run.started_at = _now()
        run.updated_at = run.started_at
        self.write(run)
        return run

    def archive(self, status: StatusT) -> Path:
        run_id = status.run_id or _new_run_id()
        path = self.path.parent / "history" / self.path.stem / f"{run_id}.json"
        write_model(path, status)
        return path


class WorkflowStageTracker(AbstractContextManager["WorkflowStageTracker"]):
    def __init__(
        self,
        store: RunStatusStore,
        stage: str,
        inputs: dict[str, str] | None = None,
    ) -> None:
        self.store = store
        self.stage = stage
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

    def __enter__(self) -> "WorkflowStageTracker":
        self._started_at = _now()
        run = self.store.read()
        if not run.run_id:
            run.run_id = _new_run_id()
            run.attempt = max(run.attempt, 1)
        run.started_at = run.started_at or self._started_at
        run.updated_at = self._started_at
        _upsert_stage(
            run,
            RunStage(
                name=self.stage,
                status="running",
                started_at=self._started_at,
                inputs=self.inputs,
            ),
        )
        run.status = _overall_status(run)
        self.store.write(run)
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
        run = self.store.read()
        started_at = self._started_at or finished_at
        _upsert_stage(
            run,
            RunStage(
                name=self.stage,
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
        run.updated_at = finished_at
        run.status = _overall_status(run)
        self.store.write(run)
        return False


def project_status_store(paths: ProjectPaths, workflow: str) -> RunStatusStore[WorkflowRunStatus]:
    return RunStatusStore(
        paths.run_status_json(workflow),
        WorkflowRunStatus,
        lambda: WorkflowRunStatus(workflow=workflow, project=paths.project),
    )


def track_workflow_stage(
    paths: ProjectPaths,
    workflow: str,
    stage: str,
    inputs: dict[str, str] | None = None,
) -> WorkflowStageTracker:
    return WorkflowStageTracker(project_status_store(paths, workflow), stage, inputs)


def begin_workflow_run(paths: ProjectPaths, workflow: str) -> WorkflowRunStatus:
    return project_status_store(paths, workflow).begin()


def read_workflow_status(paths: ProjectPaths, workflow: str) -> WorkflowRunStatus:
    return project_status_store(paths, workflow).read()


def write_workflow_status(paths: ProjectPaths, run: WorkflowRunStatus) -> None:
    project_status_store(paths, run.workflow).write(run)


def update_run_usage(store: RunStatusStore[StatusT], root: Path) -> StatusT:
    from ai_clip.core.billing import summarize

    run = store.read()
    summary = summarize(root, since=run.started_at)
    run.usage = {
        "total": summary["total"],
        "by_stage": summary["by_stage"],
        "by_model": summary["by_model"],
        "by_kind": summary["by_kind"],
    }
    store.write(run)
    return run


def mark_stages_stale(
    store: RunStatusStore[StatusT],
    stage_names: list[str],
    reason: str,
    outputs: dict[str, str] | None = None,
) -> StatusT:
    run = store.read()
    now = _now()
    run.started_at = run.started_at or now
    for name in stage_names:
        _upsert_stage(
            run,
            RunStage(
                name=name,
                status="stale",
                finished_at=now,
                outputs=outputs or {},
                metrics={"reason": reason},
            ),
        )
    run.updated_at = now
    run.status = _overall_status(run)
    store.write(run)
    return run


def stage_is_stale(store: RunStatusStore, stage_name: str) -> bool:
    run = store.read()
    return any(stage.name == stage_name and stage.status == "stale" for stage in run.stages)


def mark_stale_running_stages(
    paths: ProjectPaths,
    workflow: str,
    *,
    older_than_minutes: int = 180,
    reason: str = "previous run did not finish",
) -> WorkflowRunStatus:
    store = project_status_store(paths, workflow)
    run = store.read()
    if not run.stages:
        return run
    now = _now()
    changed = False
    for stage in run.stages:
        if stage.status != "running":
            continue
        if not _is_old_running_stage(stage, older_than_minutes):
            continue
        stage.status = "stale"
        stage.finished_at = now
        stage.duration_sec = _duration(stage.started_at or now, now)
        stage.error = reason
        changed = True
    if changed:
        run.updated_at = now
        run.status = _overall_status(run)
        store.write(run)
    return run


def _mark_running_stale(run: WorkflowRunStatus, reason: str) -> bool:
    now = _now()
    changed = False
    for stage in run.stages:
        if stage.status != "running":
            continue
        stage.status = "stale"
        stage.finished_at = now
        stage.duration_sec = _duration(stage.started_at or now, now)
        stage.error = reason
        changed = True
    if changed:
        run.updated_at = now
        run.status = _overall_status(run)
    return changed


def _upsert_stage(run: WorkflowRunStatus, stage: RunStage) -> None:
    run.stages = [item for item in run.stages if item.name != stage.name]
    run.stages.append(stage)


def _overall_status(run: WorkflowRunStatus) -> str:
    statuses = [stage.status for stage in run.stages]
    if any(item == "failed" for item in statuses):
        return "failed"
    if any(item == "running" for item in statuses):
        return "running"
    if any(item == "stale" for item in statuses):
        return "stale"
    if statuses and all(item in {"succeeded", "skipped"} for item in statuses):
        return "succeeded"
    return "pending"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _duration(started_at: str, finished_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at)
        finish = datetime.fromisoformat(finished_at)
    except ValueError:
        return 0.0
    return round(max((finish - start).total_seconds(), 0.0), 3)


def _is_old_running_stage(stage: RunStage, older_than_minutes: int) -> bool:
    if older_than_minutes <= 0:
        return True
    if not stage.started_at:
        return True
    try:
        started_at = datetime.fromisoformat(stage.started_at)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - started_at > timedelta(minutes=older_than_minutes)


def _sanitize_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    text = re.sub(r"(?i)(api[_-]?key|authorization|cookie|token)=\S+", r"\1=<redacted>", text)
    return text[:500]
