from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import re
from typing import Any, Generic, Literal, Protocol, TypeVar


StageStatus = Literal["succeeded", "skipped", "waiting"]
MetricValue = str | int | float | bool
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class StageInvocation:
    name: str
    inputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.name)


@dataclass(frozen=True)
class StageResult(Generic[ResultT]):
    value: ResultT | None = None
    status: StageStatus = "succeeded"
    outputs: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "skipped", "waiting"}:
            raise ValueError(f"invalid completed stage status: {self.status!r}")


class StageTracker(Protocol):
    def set(
        self,
        *,
        status: str | None = None,
        outputs: dict[str, str] | None = None,
        metrics: dict[str, MetricValue] | None = None,
    ) -> None: ...


StageTrackerFactory = Callable[[StageInvocation], AbstractContextManager[StageTracker]]


@dataclass(frozen=True)
class StageExecution(Generic[ResultT]):
    invocation: StageInvocation
    handler: Callable[[], StageResult[ResultT]] = field(compare=False)
    tracker_factory: StageTrackerFactory | None = field(default=None, compare=False)


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    optional: bool = False
    runner: Callable[..., Any] | None = field(default=None, compare=False)
    tool_name: str | None = None
    tool_params: Mapping[str, str] = field(default_factory=dict)
    cli_exposed: bool = True


@dataclass(frozen=True)
class WorkflowStep:
    stage: str
    when: str = ""

    def enabled(self, flags: Mapping[str, bool]) -> bool:
        return not self.when or bool(flags.get(self.when, False))


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str
    steps: tuple[WorkflowStep, ...]
    cli_exposed: bool = True
    status_name: str = ""

    @property
    def status_key(self) -> str:
        return self.status_name or self.name

    def stage_names(self, flags: Mapping[str, bool] | None = None) -> tuple[str, ...]:
        active_flags = flags or {}
        return tuple(step.stage for step in self.steps if step.enabled(active_flags))


class StageRegistry:
    def __init__(self) -> None:
        self._stages: dict[str, StageSpec] = {}
        self._workflows: dict[str, WorkflowSpec] = {}

    def register_stage(self, spec: StageSpec) -> StageSpec:
        _validate_name(spec.name)
        if spec.name in self._stages:
            raise ValueError(f"duplicate stage: {spec.name}")
        self._stages[spec.name] = spec
        return spec

    def register_workflow(self, spec: WorkflowSpec) -> WorkflowSpec:
        _validate_name(spec.name)
        if spec.status_name and (
            not spec.status_name.strip() or " " in spec.status_name
        ):
            raise ValueError(f"invalid workflow status name: {spec.status_name!r}")
        if spec.name in self._workflows:
            raise ValueError(f"duplicate workflow: {spec.name}")
        unknown = [step.stage for step in spec.steps if step.stage not in self._stages]
        if unknown:
            raise ValueError(f"workflow {spec.name!r} references unknown stages: {unknown}")
        self._workflows[spec.name] = spec
        return spec

    def stage(self, name: str) -> StageSpec:
        try:
            return self._stages[name]
        except KeyError as exc:
            raise KeyError(f"unknown stage {name!r}; available: {sorted(self._stages)}") from exc

    def workflow(self, name: str) -> WorkflowSpec:
        try:
            return self._workflows[name]
        except KeyError as exc:
            raise KeyError(f"unknown workflow {name!r}; available: {sorted(self._workflows)}") from exc

    def stages(self) -> tuple[StageSpec, ...]:
        return tuple(self._stages.values())

    def workflows(self) -> tuple[WorkflowSpec, ...]:
        return tuple(self._workflows.values())


def execute_workflow(
    spec: WorkflowSpec,
    executions: Mapping[str, StageExecution[Any]],
    flags: dict[str, bool] | None = None,
    tracker_factory: StageTrackerFactory | None = None,
) -> dict[str, StageResult[Any]]:
    runtime_flags = flags if flags is not None else {}
    results: dict[str, StageResult[Any]] = {}
    for step in spec.steps:
        if not step.enabled(runtime_flags):
            continue
        try:
            execution = executions[step.stage]
        except KeyError as exc:
            raise KeyError(
                f"workflow {spec.name!r} has no execution for stage {step.stage!r}"
            ) from exc
        if execution.invocation.name != step.stage:
            raise ValueError(
                f"workflow stage {step.stage!r} received invocation for "
                f"{execution.invocation.name!r}"
            )
        results[step.stage] = execute_stage(execution, tracker_factory)
    return results


def execute_stage(
    execution: StageExecution[ResultT],
    tracker_factory: StageTrackerFactory | None = None,
) -> StageResult[ResultT]:
    active_tracker_factory = execution.tracker_factory or tracker_factory
    if active_tracker_factory is None:
        return _require_stage_result(execution)
    with active_tracker_factory(execution.invocation) as tracker:
        result = _require_stage_result(execution)
        tracker.set(
            status=result.status,
            outputs=dict(result.outputs),
            metrics=dict(result.metrics),
        )
        return result


def stage_execution(
    name: str,
    handler: Callable[[], StageResult[ResultT]],
    inputs: Mapping[str, str] | None = None,
    *,
    tracker_factory: StageTrackerFactory | None = None,
) -> StageExecution[ResultT]:
    return StageExecution(
        invocation=StageInvocation(name=name, inputs=inputs or {}),
        handler=handler,
        tracker_factory=tracker_factory,
    )


def _require_stage_result(execution: StageExecution[ResultT]) -> StageResult[ResultT]:
    result = execution.handler()
    if not isinstance(result, StageResult):
        raise TypeError(
            f"stage {execution.invocation.name!r} returned {type(result).__name__}; "
            "expected StageResult"
        )
    return result


def _validate_name(name: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name) is None:
        raise ValueError(f"registry names must use kebab-case: {name!r}")
