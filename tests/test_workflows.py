import pytest
import json

from ai_clip import workflows
from ai_clip.core.config import Config
from ai_clip.core.models import Clip, Intent, Platform, Shot, Storyboard, Transcript, ViralAnalysis, VideoFormat


def _record(calls, name, ret=None):
    def fn(*a, **k):
        calls.append(name)
        return ret
    return fn


def test_transcribe_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        workflows.pipeline,
        "run_download",
        _record(calls, "download", Clip(clip_id="p", source_url="url", platform=Platform.youtube, video_path="v.mp4")),
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "run_extract",
        _record(calls, "extract", Transcript(clip_id="p", language="zh", segments=[])),
    )
    monkeypatch.setattr(
        workflows.pipeline, "run_export",
        lambda c, p: (tmp_path / "a.srt", tmp_path / "a.txt"),
    )
    r = workflows.transcribe(Config(data_dir=str(tmp_path)), "p", "url")
    assert calls == ["download", "extract"]
    assert r["srt"].endswith("a.srt")
    assert r["run_status"].endswith("runs\\transcribe.json") or r["run_status"].endswith("runs/transcribe.json")
    status = json.loads((tmp_path / "p" / "runs" / "transcribe.json").read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert [stage["name"] for stage in status["stages"]] == ["download", "extract", "export"]
    assert status["usage"]["total"]["calls"] == 0


def test_teardown_order(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(workflows.pipeline, "run_download", _record(calls, "download"))
    monkeypatch.setattr(workflows.pipeline, "run_extract", _record(calls, "extract"))
    monkeypatch.setattr(
        workflows.pipeline, "run_analyze",
        lambda c, p, i=None: ViralAnalysis(clip_id="x", hook="H", formula="F"),
    )
    r = workflows.teardown(Config(data_dir=str(tmp_path)), "p", "url")
    assert calls == ["download", "extract"]
    assert r["hook"] == "H" and r["formula"] == "F"
    assert r["run_status"].endswith("teardown.json")


def test_source_draft_workflow_runs_pipeline_stages(monkeypatch, tmp_path):
    calls = []

    def fake_download(cfg, project, url):
        calls.append(("download", project, url))

    def fake_extract(cfg, project, use_subtitles=False):
        calls.append(("extract", project, use_subtitles))

    def fake_analyze(cfg, project, intent):
        calls.append(("analyze", project, intent))
        return ViralAnalysis(clip_id=project, intent=intent, hook="hook", formula="formula")

    def fake_source_draft(
        cfg,
        project,
        intent=Intent.info,
        stance="",
        use_research=True,
        research_theme="",
        allow_untracked_research=True,
    ):
        calls.append(("source_draft", project, intent, stance))
        return "data/demo/source_draft.md"

    monkeypatch.setattr("ai_clip.workflows.pipeline.run_download", fake_download)
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_extract", fake_extract)
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_analyze", fake_analyze)
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_source_draft", fake_source_draft)

    result = workflows.source_draft(
        Config(data_dir=str(tmp_path)),
        "demo",
        "https://example.com/video",
        intent=Intent.emotion,
        stance="complex systems",
        use_subtitles=True,
    )

    assert result == {
        "workflow": "source_draft",
        "hook": "hook",
        "draft": "data/demo/source_draft.md",
        "run_status": str(tmp_path / "demo" / "runs" / "source_draft.json"),
    }
    assert calls == [
        ("download", "demo", "https://example.com/video"),
        ("extract", "demo", True),
        ("analyze", "demo", Intent.emotion),
        ("source_draft", "demo", Intent.emotion, "complex systems"),
    ]


def test_source_draft_can_run_research_before_draft(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_download", _record(calls, "download"))
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_extract", _record(calls, "extract"))
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.run_analyze",
        lambda c, p, i: calls.append("analyze")
        or ViralAnalysis(clip_id=p, intent=i, hook="hook", formula="formula"),
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.run_research",
        lambda c, p, theme="": calls.append(("research", theme)) or "research.md",
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.run_source_draft",
        lambda c, p, intent=Intent.info, stance="", **kwargs: calls.append("source_draft")
        or "source_draft.md",
    )

    result = workflows.source_draft(
        Config(data_dir=str(tmp_path)),
        "demo",
        "https://example.com/video",
        research=True,
        theme="biology lens",
    )

    assert result["draft"] == "source_draft.md"
    assert calls == ["download", "extract", "analyze", ("research", "biology lens"), "source_draft"]


def test_source_draft_reuses_existing_artifacts_by_default(monkeypatch, tmp_path):
    calls = []
    root = tmp_path / "demo"
    root.mkdir()
    (root / "clip.json").write_text(
        '{"clip_id":"demo","source_url":"url","platform":"youtube","video_path":"v.mp4"}',
        encoding="utf-8",
    )
    (root / "transcript.json").write_text(
        '{"clip_id":"demo","language":"zh","text":"text","segments":[]}',
        encoding="utf-8",
    )
    (root / "analysis.json").write_text(
        '{"clip_id":"demo","intent":"info","hook":"hook","formula":"formula"}',
        encoding="utf-8",
    )
    (root / "research.md").write_text("research", encoding="utf-8")
    (root / "source_draft.md").write_text("draft", encoding="utf-8")

    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.load_current_download",
        lambda *args, **kwargs: Clip(
            clip_id="demo",
            source_url="https://example.com/video",
            platform=Platform.youtube,
            video_path="v.mp4",
        ),
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.load_current_extract",
        lambda *args, **kwargs: Transcript(clip_id="demo", language="zh", text="text"),
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.load_current_analysis",
        lambda *args, **kwargs: ViralAnalysis(
            clip_id="demo",
            intent=Intent.info,
            hook="hook",
            formula="formula",
        ),
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.load_current_research",
        lambda *args, **kwargs: root / "research.md",
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.load_current_source_draft",
        lambda *args, **kwargs: root / "source_draft.md",
    )

    for name in [
        "run_download",
        "run_extract",
        "run_analyze",
        "run_research",
        "run_source_draft",
    ]:
        monkeypatch.setattr("ai_clip.workflows.pipeline." + name, _record(calls, name))

    result = workflows.source_draft(
        Config(data_dir=str(tmp_path)),
        "demo",
        "https://example.com/video",
        research=True,
    )

    assert result["hook"] == "hook"
    assert result["draft"] == str(root / "source_draft.md")
    assert calls == []


def test_source_draft_no_resume_forces_stages(monkeypatch, tmp_path):
    calls = []
    root = tmp_path / "demo"
    root.mkdir()
    (root / "clip.json").write_text(
        '{"clip_id":"demo","source_url":"url","platform":"youtube","video_path":"v.mp4"}',
        encoding="utf-8",
    )
    (root / "transcript.json").write_text(
        '{"clip_id":"demo","language":"zh","text":"text","segments":[]}',
        encoding="utf-8",
    )
    (root / "analysis.json").write_text(
        '{"clip_id":"demo","intent":"info","hook":"old","formula":"old"}',
        encoding="utf-8",
    )
    (root / "source_draft.md").write_text("old", encoding="utf-8")

    monkeypatch.setattr("ai_clip.workflows.pipeline.run_download", _record(calls, "download"))
    monkeypatch.setattr("ai_clip.workflows.pipeline.run_extract", _record(calls, "extract"))
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.run_analyze",
        lambda c, p, i: calls.append("analyze")
        or ViralAnalysis(clip_id=p, intent=i, hook="new", formula="new"),
    )
    monkeypatch.setattr(
        "ai_clip.workflows.pipeline.run_source_draft",
        lambda c, p, intent=Intent.info, stance="", **kwargs: calls.append("source_draft")
        or "source_draft.md",
    )

    result = workflows.source_draft(
        Config(data_dir=str(tmp_path)),
        "demo",
        "https://example.com/video",
        resume=False,
    )

    assert result["hook"] == "new"
    assert result["draft"] == "source_draft.md"
    assert calls == ["download", "extract", "analyze", "source_draft"]


def test_source_draft_changed_url_reruns_every_downstream_stage(monkeypatch, tmp_path):
    calls = []
    root = tmp_path / "demo"
    root.mkdir()
    (root / "clip.json").write_text(
        json.dumps({
            "clip_id": "demo",
            "source_url": "https://example.com/old",
            "platform": "youtube",
            "video_path": str(root / "old.mp4"),
        }),
        encoding="utf-8",
    )
    (root / "transcript.json").write_text(
        '{"clip_id":"demo","language":"zh","text":"old","segments":[]}',
        encoding="utf-8",
    )
    (root / "analysis.json").write_text(
        '{"clip_id":"demo","intent":"info","hook":"old","formula":"old"}',
        encoding="utf-8",
    )
    (root / "source_draft.md").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        workflows.pipeline,
        "run_download",
        lambda *args, **kwargs: calls.append("download") or Clip(
            clip_id="demo",
            source_url="https://example.com/new",
            platform=Platform.youtube,
            video_path=str(root / "new.mp4"),
        ),
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "run_extract",
        lambda *args, **kwargs: calls.append("extract") or Transcript(clip_id="demo"),
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "run_analyze",
        lambda *args, **kwargs: calls.append("analyze") or ViralAnalysis(clip_id="demo"),
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "run_source_draft",
        lambda *args, **kwargs: calls.append("source_draft") or root / "source_draft.md",
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "load_current_extract",
        lambda *args, **kwargs: pytest.fail("extract reuse must not run after download reruns"),
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "load_current_analysis",
        lambda *args, **kwargs: pytest.fail("analysis reuse must not run after extract reruns"),
    )

    workflows.source_draft(
        Config(data_dir=str(tmp_path)),
        "demo",
        "https://example.com/new",
    )

    assert calls == ["download", "extract", "analyze", "source_draft"]


def test_source_draft_blocks_concurrent_project_run(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    lock_path = tmp_path / "demo" / "runs" / "locks" / "source_draft.lock"

    from ai_clip.core.run_lock import RunLock

    with RunLock(lock_path, "held"):
        with pytest.raises(RuntimeError, match="source_draft is already running"):
            workflows.source_draft(cfg, "demo", "https://example.com/video")


def test_remix_full_auto(monkeypatch, tmp_path):
    calls = []
    for name in ["run_download", "run_extract", "run_analyze"]:
        monkeypatch.setattr(workflows.pipeline, name, _record(calls, name))
    monkeypatch.setattr(
        workflows.pipeline,
        "run_voiceover",
        _record(calls, "run_voiceover", {}),
    )
    monkeypatch.setattr(
        workflows.pipeline, "run_storyboard",
        lambda *a, **k: calls.append("run_storyboard") or Storyboard(
            project="p",
            format=VideoFormat.remix,
            shots=[Shot(index=1)],
        ),
    )
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")
    r = workflows.remix(Config(data_dir=str(tmp_path)), "p", "url", "theme")
    assert r["output"] == "out.mp4"
    assert r["run_status"].endswith("remix.json")
    assert calls == [
        "run_download", "run_extract", "run_analyze", "run_storyboard", "run_voiceover",
    ]


def test_remix_can_run_research_before_storyboard(monkeypatch, tmp_path):
    calls = []
    for name in ["run_download", "run_extract", "run_analyze"]:
        monkeypatch.setattr(workflows.pipeline, name, _record(calls, name))
    monkeypatch.setattr(workflows.pipeline, "run_research", _record(calls, "run_research", "r.md"))
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", _record(calls, "run_voiceover", {}))
    monkeypatch.setattr(
        workflows.pipeline,
        "run_storyboard",
        lambda *a, **k: calls.append("run_storyboard") or Storyboard(
            project="p",
            format=VideoFormat.remix,
            shots=[Shot(index=1)],
        ),
    )
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")

    r = workflows.remix(Config(data_dir=str(tmp_path)), "p", "url", "theme", research=True)

    assert r["output"] == "out.mp4"
    assert calls == [
        "run_download",
        "run_extract",
        "run_analyze",
        "run_research",
        "run_storyboard",
        "run_voiceover",
    ]


def test_original_rejects_remix():
    with pytest.raises(ValueError):
        workflows.original(Config(), "p", "theme", fmt=VideoFormat.remix)


def test_original_needs_assets(monkeypatch, tmp_path):
    sb = Storyboard(project="p", shots=[Shot(index=1, image_file="shot_01.png", image_prompt="x")])
    monkeypatch.setattr(workflows.pipeline, "run_storyboard", lambda *a, **k: sb)
    monkeypatch.setattr(workflows.pipeline, "run_assets", lambda c, p: 0)
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", lambda c, p: {})
    monkeypatch.setattr(workflows, "check_assets", lambda sb, d: ["shot_01 (shot_01.png)"])
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "should_not_run")
    r = workflows.original(Config(data_dir=str(tmp_path)), "p", "theme")
    assert r["status"] == "needs_assets"
    assert r["missing"] == ["shot_01 (shot_01.png)"]
    assert r["run_status"].endswith("original.json")


def test_original_can_run_research_before_storyboard(monkeypatch, tmp_path):
    calls = []
    sb = Storyboard(project="p", shots=[Shot(index=1)])
    monkeypatch.setattr(
        workflows.pipeline,
        "run_research",
        lambda c, p, theme="": calls.append(("research", theme)) or "research.md",
    )
    monkeypatch.setattr(
        workflows.pipeline,
        "run_storyboard",
        lambda *a, **k: calls.append("storyboard") or sb,
    )
    monkeypatch.setattr(workflows.pipeline, "run_assets", lambda c, p: 1)
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", lambda c, p: {})
    monkeypatch.setattr(workflows, "check_assets", lambda sb, d: [])
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")

    result = workflows.original(Config(data_dir=str(tmp_path)), "p", "theme", research=True)

    assert result["status"] == "done"
    assert calls == [("research", "theme"), "storyboard"]


def test_original_done_when_assets_ready(monkeypatch, tmp_path):
    sb = Storyboard(project="p", shots=[Shot(index=1)])
    monkeypatch.setattr(workflows.pipeline, "run_storyboard", lambda *a, **k: sb)
    monkeypatch.setattr(workflows.pipeline, "run_assets", lambda c, p: 1)
    monkeypatch.setattr(workflows.pipeline, "run_voiceover", lambda c, p: {})
    monkeypatch.setattr(workflows, "check_assets", lambda sb, d: [])
    monkeypatch.setattr(workflows.pipeline, "run_assemble", lambda c, p: "out.mp4")
    r = workflows.original(Config(data_dir=str(tmp_path)), "p", "theme")
    assert r["status"] == "done" and r["output"] == "out.mp4"
    assert r["run_status"].endswith("original.json")

