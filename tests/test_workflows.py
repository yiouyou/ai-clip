import pytest

from ai_clip import workflows
from ai_clip.core.config import Config
from ai_clip.core.models import Shot, Storyboard, ViralAnalysis, VideoFormat


def _record(calls, name, ret=None):
    def fn(*a, **k):
        calls.append(name)
        return ret
    return fn


def test_transcribe_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(workflows.pipeline, "run_download", _record(calls, "download"))
    monkeypatch.setattr(workflows.pipeline, "run_extract", _record(calls, "extract"))
    monkeypatch.setattr(
        workflows.pipeline, "run_export",
        lambda c, p: (tmp_path / "a.srt", tmp_path / "a.txt"),
    )
    r = workflows.transcribe(Config(), "p", "url")
    assert calls == ["download", "extract"]
    assert r["srt"].endswith("a.srt")


def test_teardown_order(monkeypatch):
    calls = []
    monkeypatch.setattr(workflows.pipeline, "run_download", _record(calls, "download"))
    monkeypatch.setattr(workflows.pipeline, "run_extract", _record(calls, "extract"))
    monkeypatch.setattr(
        workflows.pipeline, "run_analyze",
        lambda c, p: ViralAnalysis(clip_id="x", hook="H", formula="F"),
    )
    r = workflows.teardown(Config(), "p", "url")
    assert calls == ["download", "extract"]
    assert r["hook"] == "H" and r["formula"] == "F"


def test_remix_full_auto(monkeypatch):
    calls = []
    for name in ["run_download", "run_extract", "run_analyze", "run_voiceover"]:
        monkeypatch.setattr(workflows.pipeline, name, _record(calls, name))
    monkeypatch.setattr(
        workflows.pipeline, "run_storyboard",
        lambda *a, **k: calls.append("run_storyboard"),
    )
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")
    r = workflows.remix(Config(), "p", "url", "theme")
    assert r["output"] == "out.mp4"
    assert calls == [
        "run_download", "run_extract", "run_analyze", "run_storyboard", "run_voiceover",
    ]


def test_original_rejects_remix():
    with pytest.raises(ValueError):
        workflows.original(Config(), "p", "theme", fmt=VideoFormat.remix)


def test_original_needs_assets(monkeypatch):
    sb = Storyboard(project="p", shots=[Shot(index=1, image_file="shot_01.png", image_prompt="x")])
    monkeypatch.setattr(workflows.pipeline, "run_storyboard", lambda *a, **k: sb)
    monkeypatch.setattr(workflows.pipeline, "run_assets", lambda c, p: 0)
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", lambda c, p: {})
    monkeypatch.setattr(workflows, "check_assets", lambda sb, d: ["shot_01 (shot_01.png)"])
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "should_not_run")
    r = workflows.original(Config(), "p", "theme")
    assert r["status"] == "needs_assets"
    assert r["missing"] == ["shot_01 (shot_01.png)"]


def test_original_done_when_assets_ready(monkeypatch):
    sb = Storyboard(project="p", shots=[Shot(index=1)])
    monkeypatch.setattr(workflows.pipeline, "run_storyboard", lambda *a, **k: sb)
    monkeypatch.setattr(workflows.pipeline, "run_assets", lambda c, p: 1)
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", lambda c, p: {})
    monkeypatch.setattr(workflows, "check_assets", lambda sb, d: [])
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")
    r = workflows.original(Config(), "p", "theme")
    assert r["status"] == "done" and r["output"] == "out.mp4"
