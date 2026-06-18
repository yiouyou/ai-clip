from pathlib import Path

from ai_clip.core.config import load_config
from ai_clip.core.device import whisper_runtime
from ai_clip.core.models import Clip, Platform, Shot
from ai_clip.core.paths import ProjectPaths, read_model, write_model


def test_clip_roundtrip(tmp_path: Path):
    clip = Clip(clip_id="abc", source_url="u", platform=Platform.youtube, video_path="v.mp4")
    p = tmp_path / "clip.json"
    write_model(p, clip)
    assert read_model(p, Clip) == clip


def test_shot_expected_files():
    shot = Shot(index=1, image_file="shot_01.png", video_file="shot_01.mp4")
    assert shot.expected_files() == ["shot_01.png", "shot_01.mp4"]
    assert Shot(index=2).expected_files() == []


def test_project_paths_ensure(tmp_path: Path):
    pp = ProjectPaths(tmp_path, "demo")
    pp.ensure()
    assert pp.prompts_dir.is_dir()
    assert pp.assets_dir.is_dir()
    assert pp.storyboard_json == tmp_path / "demo" / "storyboard.json"


def test_config_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AICLIP_LLM_MODEL", "qwen-max")
    monkeypatch.setenv("AICLIP_DATA_DIR", "/tmp/x")
    cfg = load_config(tmp_path / "missing.yaml")  # missing -> defaults + env
    assert cfg.llm.model == "qwen-max"
    assert cfg.data_dir == "/tmp/x"


def test_whisper_runtime_cpu(monkeypatch):
    monkeypatch.setattr("ai_clip.core.device.has_cuda", lambda: False)
    assert whisper_runtime("medium", "auto") == ("cpu", "int8")


def test_whisper_runtime_gpu(monkeypatch):
    monkeypatch.setattr("ai_clip.core.device.has_cuda", lambda: True)
    assert whisper_runtime("large-v3", "auto") == ("cuda", "float16")
    assert whisper_runtime("large-v3", "int8") == ("cuda", "int8")
