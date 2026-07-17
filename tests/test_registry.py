from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from ai_clip import cli
from ai_clip.core.stages import (
    StageRegistry,
    StageResult,
    StageSpec,
    WorkflowSpec,
    WorkflowStep,
    execute_workflow,
    stage_execution,
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


def test_original_uses_topic_research_and_gates_paid_stages_on_assets():
    workflow = REGISTRY.workflow("original")

    assert workflow.stage_names({"research": True, "assets-ready": False}) == (
        "topic-research",
        "storyboard",
        "assets",
    )
    assert workflow.stage_names({"research": True, "assets-ready": True})[-2:] == (
        "voiceover",
        "assemble",
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
        return StageResult()

    execute_workflow(
        spec,
        {
            "one": stage_execution("one", one),
            "two": stage_execution("two", lambda: StageResult(value=calls.append("two"))),
        },
        flags,
    )

    assert calls == ["one", "two"]


def test_execute_workflow_rejects_mismatched_invocation_and_raw_results():
    spec = WorkflowSpec(
        name="flow",
        description="flow",
        steps=(WorkflowStep("one"),),
    )

    with pytest.raises(ValueError, match="invocation for 'two'"):
        execute_workflow(spec, {"one": stage_execution("two", StageResult)})

    with pytest.raises(TypeError, match="expected StageResult"):
        execute_workflow(spec, {"one": stage_execution("one", lambda: "raw")})

    with pytest.raises(ValueError, match="kebab-case"):
        stage_execution("not_valid", StageResult)


def test_execute_workflow_applies_result_metadata_to_tracker():
    spec = WorkflowSpec(
        name="flow",
        description="flow",
        steps=(WorkflowStep("one"),),
    )
    seen = {}

    class Tracker:
        def set(self, **kwargs):
            seen["result"] = kwargs

    @contextmanager
    def tracker_factory(invocation):
        seen["invocation"] = invocation
        yield Tracker()

    results = execute_workflow(
        spec,
        {
            "one": stage_execution(
                "one",
                lambda: StageResult(
                    value="payload",
                    status="waiting",
                    outputs={"draft": "draft.md"},
                    metrics={"missing": 2},
                ),
                {"source": "source.md"},
            ),
        },
        tracker_factory=tracker_factory,
    )

    assert results["one"].value == "payload"
    assert seen["invocation"].inputs == {"source": "source.md"}
    assert seen["result"] == {
        "status": "waiting",
        "outputs": {"draft": "draft.md"},
        "metrics": {"missing": 2},
    }


def test_stage_result_rejects_runtime_only_statuses():
    with pytest.raises(ValueError, match="invalid completed stage status"):
        StageResult(status="failed")


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
