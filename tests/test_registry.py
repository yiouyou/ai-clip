import pytest
from typer.testing import CliRunner

from ai_clip import cli
from ai_clip.core.stages import (
    StageRegistry,
    StageSpec,
    WorkflowSpec,
    WorkflowStep,
    execute_workflow,
)
from ai_clip.registry import REGISTRY
from ai_clip.tools import all_tools


def test_registry_uses_canonical_kebab_case_names():
    assert REGISTRY.stage("source-research").tool_name == "source_research"
    assert REGISTRY.workflow("source-draft").status_key == "source_draft"
    assert REGISTRY.workflow("daily-radar").stage_names({"research": False}) == (
        "collect",
        "zack-ranking",
        "source-content",
        "content-rerank",
        "zack-selection",
        "zack-draft",
    )


def test_daily_radar_optional_steps_follow_flags():
    stages = REGISTRY.workflow("daily-radar").stage_names({
        "research": True,
        "review": True,
        "rewrite": True,
    })
    assert stages[-5:] == (
        "source-research",
        "zack-draft",
        "pair-review",
        "pair-rewrite",
        "pair-verify",
    )


def test_registry_rejects_duplicate_and_unknown_stages():
    registry = StageRegistry()
    with pytest.raises(ValueError, match="kebab-case"):
        registry.register_stage(StageSpec(name="not_valid", description="invalid"))
    registry.register_stage(StageSpec(name="one", description="one"))
    with pytest.raises(ValueError, match="duplicate stage"):
        registry.register_stage(StageSpec(name="one", description="duplicate"))
    with pytest.raises(ValueError, match="unknown stages"):
        registry.register_workflow(WorkflowSpec(
            name="flow",
            description="flow",
            steps=(WorkflowStep("missing"),),
        ))


def test_execute_workflow_uses_declared_order_and_dynamic_flags():
    spec = WorkflowSpec(
        name="flow",
        description="flow",
        steps=(WorkflowStep("one"), WorkflowStep("two", when="ready")),
    )
    flags = {"ready": False}
    calls = []

    def one():
        calls.append("one")
        flags["ready"] = True

    execute_workflow(spec, {"one": one, "two": lambda: calls.append("two")}, flags)

    assert calls == ["one", "two"]


def test_tool_registry_is_derived_from_stage_registry():
    expected = {spec.tool_name for spec in REGISTRY.stages() if spec.tool_name}
    assert {tool.name for tool in all_tools()} == expected


def test_cli_exposes_registered_stages_and_workflows():
    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for spec in REGISTRY.stages():
        if spec.cli_exposed:
            assert spec.name in result.output
    for spec in REGISTRY.workflows():
        if spec.cli_exposed:
            assert spec.name in result.output


def test_daily_radar_compatibility_stage_list_comes_from_registry():
    from ai_clip.radar import DAILY_RADAR_STAGES

    assert tuple(stage.name for stage in DAILY_RADAR_STAGES) == REGISTRY.workflow(
        "daily-radar"
    ).stage_names({"research": True, "review": True, "rewrite": True})
