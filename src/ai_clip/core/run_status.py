from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timedelta, timezone
import json
import re
from types import TracebackType

from pydantic import BaseModel, Field

from ai_clip.core.paths import ProjectPaths, write_model


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
    project: str
    status: str = "pending"
    started_at: str = ""
    updated_at: str = ""
    stages: list[RunStage] = Field(default_factory=list)


class WorkflowStageTracker(AbstractContextManager["WorkflowStageTracker"]):
    def __init__(
        self,
        paths: ProjectPaths,
        workflow: str,
        stage: str,
        inputs: dict[str, str] | None = None,
    ) -> None:
        self.paths = paths
        self.workflow = workflow
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
        run = read_workflow_status(self.paths, self.workflow)
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
        write_workflow_status(self.paths, run)
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
        run = read_workflow_status(self.paths, self.workflow)
        _upsert_stage(
            run,
            RunStage(
                name=self.stage,
                status=self.status,
                started_at=self._started_at or finished_at,
                finished_at=finished_at,
                duration_sec=_duration(self._started_at or finished_at, finished_at),
                inputs=self.inputs,
                outputs=self.outputs,
                metrics=self.metrics,
                error=self.error,
            ),
        )
        run.updated_at = finished_at
        run.status = _overall_status(run)
        write_workflow_status(self.paths, run)
        return False


def track_workflow_stage(
    paths: ProjectPaths,
    workflow: str,
    stage: str,
    inputs: dict[str, str] | None = None,
) -> WorkflowStageTracker:
    return WorkflowStageTracker(paths, workflow, stage, inputs)


def read_workflow_status(paths: ProjectPaths, workflow: str) -> WorkflowRunStatus:
    path = paths.run_status_json(workflow)
    if not path.exists():
        return WorkflowRunStatus(workflow=workflow, project=paths.project)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return WorkflowRunStatus(workflow=workflow, project=paths.project, status="invalid")
    return WorkflowRunStatus.model_validate(data)


def write_workflow_status(paths: ProjectPaths, run: WorkflowRunStatus) -> None:
    write_model(paths.run_status_json(run.workflow), run)


def mark_stale_running_stages(
    paths: ProjectPaths,
    workflow: str,
    *,
    older_than_minutes: int = 180,
    reason: str = "previous run did not finish",
) -> WorkflowRunStatus:
    run = read_workflow_status(paths, workflow)
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
        write_workflow_status(paths, run)
    return run


def _upsert_stage(run: WorkflowRunStatus, stage: RunStage) -> None:
    run.stages = [item for item in run.stages if item.name != stage.name]
    run.stages.append(stage)


def _overall_status(run: WorkflowRunStatus) -> str:
    statuses = [stage.status for stage in run.stages]
    if any(item == "failed" for item in statuses):
        return "failed"
    if any(item == "stale" for item in statuses):
        return "stale"
    if any(item == "running" for item in statuses):
        return "running"
    if statuses and all(item in {"succeeded", "skipped"} for item in statuses):
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
