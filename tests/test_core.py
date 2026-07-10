from pathlib import Path

import pytest
from pydantic import BaseModel

from ai_clip.core.artifacts import (
    ArtifactStore,
    artifact_matches,
    artifact_is_stale,
    artifact_manifest_is_stale,
    artifact_manifest_path,
    read_artifact_manifest,
    write_artifact_manifest,
    write_text_atomic,
)
from ai_clip.core.artifact_status import project_artifact_statuses
from ai_clip.core.config import load_config
from ai_clip.core.device import whisper_runtime
from ai_clip.core.models import Clip, Platform, Shot
from ai_clip.core.paths import ProjectPaths, read_model, write_model
from ai_clip.core.run_status import (
    RunStage,
    WorkflowRunStatus,
    begin_workflow_run,
    mark_stale_running_stages,
    read_workflow_status,
    track_workflow_stage,
    write_workflow_status,
)
from ai_clip.core.run_lock import RunLock


class ExampleArtifact(BaseModel):
    name: str
    count: int


def test_clip_roundtrip(tmp_path: Path):
    clip = Clip(clip_id="abc", source_url="u", platform=Platform.youtube, video_path="v.mp4")
    p = tmp_path / "clip.json"
    write_model(p, clip)
    assert read_model(p, Clip) == clip


def test_write_text_atomic_replaces_existing_file(tmp_path: Path):
    path = tmp_path / "nested" / "artifact.txt"
    write_text_atomic(path, "old")
    write_text_atomic(path, "new")

    assert path.read_text(encoding="utf-8") == "new"
    assert list(path.parent.glob("*.tmp")) == []


def test_artifact_store_model_roundtrip(tmp_path: Path):
    store = ArtifactStore(tmp_path / "project")
    artifact = ExampleArtifact(name="topic", count=3)

    path = store.write_model("nested", "artifact.json", model=artifact)

    assert path == tmp_path / "project" / "nested" / "artifact.json"
    assert store.exists("nested", "artifact.json")
    assert store.read_model("nested", "artifact.json", model_cls=ExampleArtifact) == artifact


def test_artifact_manifest_detects_stale_inputs(tmp_path: Path):
    source = tmp_path / "source.txt"
    artifact = tmp_path / "artifact.txt"
    source.write_text("v1", encoding="utf-8")
    artifact.write_text("out", encoding="utf-8")

    manifest = write_artifact_manifest(
        artifact,
        stage="test",
        inputs=[source],
        params={"mode": "unit"},
        model="model-x",
    )

    assert artifact_manifest_path(artifact).exists()
    assert manifest.stage == "test"
    assert read_artifact_manifest(artifact).model == "model-x"
    assert not artifact_is_stale(artifact, [source])
    assert not artifact_manifest_is_stale(artifact)

    source.write_text("v2 changed", encoding="utf-8")

    assert artifact_is_stale(artifact, [source])
    assert artifact_manifest_is_stale(artifact)


def test_artifact_matches_invocation_metadata(tmp_path: Path):
    source = tmp_path / "source.txt"
    artifact = tmp_path / "artifact.txt"
    source.write_text("source", encoding="utf-8")
    artifact.write_text("result", encoding="utf-8")
    write_artifact_manifest(
        artifact,
        stage="draft",
        inputs=[source],
        params={"intent": "info"},
        model="model-a",
        config_hash="prompt-v1",
    )

    assert artifact_matches(
        artifact,
        inputs=[source],
        params={"intent": "info"},
        model="model-a",
        config_hash="prompt-v1",
    )
    assert not artifact_matches(artifact, inputs=[source], params={"intent": "emotion"})
    assert not artifact_matches(artifact, inputs=[source], model="model-b")
    assert not artifact_matches(artifact, inputs=[source], config_hash="prompt-v2")


def test_corrupt_artifact_manifest_is_stale(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("result", encoding="utf-8")
    artifact_manifest_path(artifact).write_text("not-json", encoding="utf-8")

    assert artifact_is_stale(artifact)
    assert artifact_manifest_is_stale(artifact)
    assert not artifact_matches(artifact)


def test_artifact_store_manifest_helpers(tmp_path: Path):
    store = ArtifactStore(tmp_path / "project")
    source = store.write_text("source.txt", content="v1")
    store.write_text("artifact.txt", content="out")

    store.write_manifest("artifact.txt", stage="unit", inputs=[source])

    assert not store.is_stale("artifact.txt", inputs=[source])
    source.write_text("v2 changed", encoding="utf-8")
    assert store.is_stale("artifact.txt", inputs=[source])


def test_project_artifact_statuses(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    pp.transcript_json.write_text("{}", encoding="utf-8")
    pp.research_md.write_text("research", encoding="utf-8")
    write_artifact_manifest(pp.research_md, stage="research", inputs=[pp.transcript_json])

    statuses = {item.name: item.status for item in project_artifact_statuses(pp)}

    assert statuses["research"] == "fresh"
    assert statuses["storyboard"] == "missing"
    assert statuses["source_draft"] == "missing"

    pp.transcript_json.write_text('{"changed": true}', encoding="utf-8")
    statuses = {item.name: item.status for item in project_artifact_statuses(pp)}

    assert statuses["research"] == "stale"


def test_mark_stale_running_workflow_stages(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    write_workflow_status(
        pp,
        WorkflowRunStatus(
            workflow="source_draft",
            project="demo",
            status="running",
            stages=[
                RunStage(
                    name="extract",
                    status="running",
                    started_at="2026-01-01T00:00:00+00:00",
                )
            ],
        ),
    )

    run = mark_stale_running_stages(pp, "source_draft", older_than_minutes=0)

    assert run.status == "stale"
    assert run.stages[0].status == "stale"
    assert run.stages[0].error == "previous run did not finish"


def test_begin_workflow_run_archives_previous_attempt(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    first = begin_workflow_run(pp, "source_draft")
    with track_workflow_stage(pp, "source_draft", "download"):
        pass
    completed = read_workflow_status(pp, "source_draft")

    second = begin_workflow_run(pp, "source_draft")

    assert first.run_id == completed.run_id
    assert second.run_id != first.run_id
    assert second.attempt == 2
    history = pp.runs_dir / "history" / "source_draft" / f"{first.run_id}.json"
    archived = WorkflowRunStatus.model_validate_json(history.read_text(encoding="utf-8"))
    assert archived.status == "succeeded"
    assert archived.stages[0].name == "download"


def test_begin_workflow_run_archives_interrupted_stage_as_stale(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    first = begin_workflow_run(pp, "source_draft")
    write_workflow_status(
        pp,
        first.model_copy(update={
            "status": "running",
            "stages": [RunStage(name="extract", status="running")],
        }),
    )

    begin_workflow_run(pp, "source_draft")

    history = pp.runs_dir / "history" / "source_draft" / f"{first.run_id}.json"
    archived = WorkflowRunStatus.model_validate_json(history.read_text(encoding="utf-8"))
    assert archived.status == "stale"
    assert archived.stages[0].status == "stale"
    assert archived.stages[0].error == "previous run did not finish"


def test_run_lock_blocks_concurrent_owner_and_releases(tmp_path: Path):
    path = tmp_path / "workflow.lock"

    with RunLock(path, "workflow already running"):
        assert path.exists()
        with pytest.raises(RuntimeError, match="already running"):
            with RunLock(path, "workflow already running"):
                pass

    assert not path.exists()


def test_shot_expected_files():
    shot = Shot(index=1, image_file="shot_01.png", video_file="shot_01.mp4")
    assert shot.expected_files() == ["shot_01.png", "shot_01.mp4"]
    assert Shot(index=2).expected_files() == []


def test_project_paths_ensure(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    assert pp.prompts_dir.is_dir()
    assert pp.assets_dir.is_dir()
    assert pp.reviews_dir.is_dir()
    assert pp.storyboard_json == tmp_path / "demo" / "storyboard.json"


def test_config_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AICLIP_LLM_MODEL", "qwen-max")
    monkeypatch.setenv("AICLIP_DATA_DIR", "/tmp/x")
    monkeypatch.setenv("NEWAPP_URL", "https://newapp.example/v1")
    monkeypatch.setenv("NEWAPP_API_KEY", "new-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setenv("AICLIP_RADAR_CHANNELS", "config/my-channels.yaml")
    monkeypatch.setenv("AICLIP_RADAR_FEEDBACK", "config/my-feedback.yaml")
    monkeypatch.setenv("AICLIP_RADAR_CHANNEL_WORKERS", "2")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    monkeypatch.setenv("AICLIP_SOURCE_RESEARCH_MAX_SEARCHES", "3")
    cfg = load_config(tmp_path / "missing.yaml")  # missing -> defaults + env
    assert cfg.llm.model == "qwen-max"
    assert cfg.data_dir == "/tmp/x"
    assert cfg.pair.base_url == "https://newapp.example/v1"
    assert cfg.pair.api_key == "new-key"
    assert cfg.pair.deepseek_api_key == "deep-key"
    assert cfg.radar.channels_path == "config/my-channels.yaml"
    assert cfg.radar.feedback_path == "config/my-feedback.yaml"
    assert cfg.radar.channel_workers == 2
    assert cfg.source_research.tavily_api_key == "tavily-key"
    assert cfg.source_research.max_searches == 3


def test_config_ignores_removed_scout_key(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
scout:
  channels_path: config/legacy-channels.yaml
  top_n: 2
""",
        encoding="utf-8",
    )

    cfg = load_config(config)

    assert cfg.radar.channels_path == "config/channels.yaml"
    assert cfg.radar.top_n == 3


def test_config_ignores_removed_scout_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AICLIP_SCOUT_CHANNELS", "config/legacy-env-channels.yaml")

    cfg = load_config(tmp_path / "missing.yaml")

    assert cfg.radar.channels_path == "config/channels.yaml"


def test_whisper_runtime_cpu(monkeypatch):
    monkeypatch.setattr("ai_clip.core.device.has_cuda", lambda: False)
    assert whisper_runtime("medium", "auto") == ("cpu", "int8")


def test_whisper_runtime_gpu(monkeypatch):
    monkeypatch.setattr("ai_clip.core.device.has_cuda", lambda: True)
    assert whisper_runtime("large-v3", "auto") == ("cuda", "float16")
    assert whisper_runtime("large-v3", "int8") == ("cuda", "int8")


