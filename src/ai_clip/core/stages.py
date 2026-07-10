from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    optional: bool = False
    run: Callable[..., Any] | None = field(default=None, compare=False)
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
    handlers: Mapping[str, Callable[[], Any]],
    flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    runtime_flags = flags if flags is not None else {}
    results: dict[str, Any] = {}
    for step in spec.steps:
        if not step.enabled(runtime_flags):
            continue
        try:
            handler = handlers[step.stage]
        except KeyError as exc:
            raise KeyError(
                f"workflow {spec.name!r} has no handler for stage {step.stage!r}"
            ) from exc
        results[step.stage] = handler()
    return results


def _validate_name(name: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name) is None:
        raise ValueError(f"registry names must use kebab-case: {name!r}")
