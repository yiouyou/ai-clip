from pathlib import Path

from ai_clip import pipeline
from ai_clip.core.config import Config
from ai_clip.core.models import Clip, Intent, Platform, Transcript, ViralAnalysis
from ai_clip.core.paths import ProjectPaths, write_model
from ai_clip.research.models import ProjectResearchReport


def _project(cfg: Config, project: str = "demo") -> ProjectPaths:
    paths = ProjectPaths(cfg.data_dir, project)
    paths.ensure()
    return paths


def test_download_resume_rejects_a_different_url(monkeypatch, tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path))
    paths = _project(cfg)
    video = paths.root / "video.mp4"
    video.write_bytes(b"video")
    clip = Clip(
        clip_id="demo",
        source_url="https://example.com/one",
        platform=Platform.youtube,
        video_path=str(video),
    )
    monkeypatch.setattr(pipeline, "download_stage", lambda *args, **kwargs: clip)

    pipeline.run_download(cfg, "demo", clip.source_url)

    assert pipeline.load_current_download(cfg, "demo", clip.source_url) == clip
    assert pipeline.load_current_download(cfg, "demo", "https://example.com/two") is None


def test_analysis_resume_rejects_changed_intent_and_model(monkeypatch, tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.model = "model-a"
    paths = _project(cfg)
    write_model(paths.transcript_json, Transcript(clip_id="demo", text="source"))
    monkeypatch.setattr(
        pipeline,
        "analyze_stage",
        lambda transcript, llm, intent: ViralAnalysis(clip_id="demo", intent=intent),
    )

    pipeline.run_analyze(cfg, "demo", Intent.info)

    assert pipeline.load_current_analysis(cfg, "demo", Intent.info) is not None
    assert pipeline.load_current_analysis(cfg, "demo", Intent.emotion) is None
    cfg.llm.model = "model-b"
    assert pipeline.load_current_analysis(cfg, "demo", Intent.info) is None


def test_source_draft_resume_rejects_changed_parameters(monkeypatch, tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.model = "model-a"
    paths = _project(cfg)
    write_model(paths.transcript_json, Transcript(clip_id="demo", text="source"))
    write_model(paths.analysis_json, ViralAnalysis(clip_id="demo", intent=Intent.info))
    monkeypatch.setattr(
        "ai_clip.produce.source_draft.generate_source_draft",
        lambda **kwargs: "# draft",
    )

    pipeline.run_source_draft(
        cfg,
        "demo",
        intent=Intent.info,
        stance="systems",
        use_research=False,
    )

    assert pipeline.load_current_source_draft(
        cfg,
        "demo",
        Intent.info,
        "systems",
        use_research=False,
    ) == paths.source_draft_md
    assert pipeline.load_current_source_draft(
        cfg,
        "demo",
        Intent.info,
        "different",
        use_research=False,
    ) is None
    cfg.llm.model = "model-b"
    assert pipeline.load_current_source_draft(
        cfg,
        "demo",
        Intent.info,
        "systems",
        use_research=False,
    ) is None


def test_research_resume_rejects_changed_theme_and_upstream(monkeypatch, tmp_path: Path):
    cfg = Config(data_dir=str(tmp_path))
    paths = _project(cfg)
    write_model(paths.transcript_json, Transcript(clip_id="demo", text="source"))
    write_model(paths.analysis_json, ViralAnalysis(clip_id="demo"))
    monkeypatch.setattr(
        "ai_clip.research.generate_project_research",
        lambda **kwargs: ProjectResearchReport(
            clip_id="demo",
            theme="theme-a",
            markdown="# research",
        ),
    )

    pipeline.run_research(cfg, "demo", theme="theme-a")

    assert pipeline.load_current_research(cfg, "demo", "theme-a") == paths.research_md
    assert pipeline.load_current_research(cfg, "demo", "theme-b") is None
    write_model(paths.transcript_json, Transcript(clip_id="demo", text="changed"))
    assert pipeline.load_current_research(cfg, "demo", "theme-a") is None
