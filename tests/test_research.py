import json

from ai_clip import pipeline
from ai_clip.core.artifacts import artifact_manifest_path, read_artifact_manifest
from ai_clip.core.config import Config
from ai_clip.core.models import Storyboard, Transcript, ViralAnalysis, VideoFormat
from ai_clip.core.paths import ProjectPaths, read_model, write_model
from ai_clip.research.models import ProjectResearchReport
from ai_clip.research.stage import generate_project_research
from ai_clip.source_research.models import SearchResult


def test_generate_project_research_clamps_searches_and_synthesizes(monkeypatch):
    cfg = Config()
    cfg.source_research.tavily_api_key = "tavily"
    cfg.source_research.max_searches = 9
    calls = {"queries": [], "searches": []}

    def fake_chat(cfg, system, user):
        if "Return JSON only" in user:
            calls["queries"].append(user)
            return json.dumps({
                "queries": [
                    {
                        "angle": "event_facts",
                        "query": "official event facts",
                        "rationale": "facts",
                    }
                ]
            })
        assert "official event facts" in user
        return "# Research Brief\n\n## Confirmed Facts\n- ok"

    def fake_search(query, cfg):
        calls["searches"].append(query)
        return [SearchResult(query=query, title="Source", url="https://example.com", content="ok")]

    monkeypatch.setattr("ai_clip.research.stage.chat", fake_chat)
    monkeypatch.setattr("ai_clip.research.stage.tavily_search", fake_search)

    report = generate_project_research(
        transcript=Transcript(clip_id="clip", text="source transcript"),
        analysis=ViralAnalysis(clip_id="clip", formula="hook formula"),
        cfg=cfg,
        theme="AI and biology",
    )

    assert report.clip_id == "clip"
    assert report.search_calls == 3
    assert [query.angle for query in report.queries] == [
        "event_facts",
        "structural_background",
        "original_lens",
    ]
    assert len(calls["searches"]) == 3
    assert report.markdown.startswith("# Research Brief")


def test_run_research_writes_project_artifacts(monkeypatch, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    pp = ProjectPaths(cfg.data_dir, "demo")
    pp.ensure()
    write_model(pp.transcript_json, Transcript(clip_id="demo", text="transcript"))
    write_model(pp.analysis_json, ViralAnalysis(clip_id="demo", formula="formula"))

    def fake_generate(transcript, cfg, analysis=None, theme=""):
        assert transcript.clip_id == "demo"
        assert analysis and analysis.formula == "formula"
        assert theme == "theme"
        return ProjectResearchReport(clip_id="demo", theme=theme, markdown="# Research Brief\n\nok")

    monkeypatch.setattr("ai_clip.research.generate_project_research", fake_generate)

    path = pipeline.run_research(cfg, "demo", theme="theme")

    assert path == pp.research_md
    assert pp.research_md.read_text(encoding="utf-8") == "# Research Brief\n\nok"
    assert read_model(pp.research_json, ProjectResearchReport).theme == "theme"
    assert artifact_manifest_path(pp.research_md).exists()
    manifest = read_artifact_manifest(pp.research_md)
    assert manifest.stage == "research"
    assert manifest.params["theme"] == "theme"
    assert str(pp.transcript_json) in manifest.inputs


def test_run_storyboard_writes_manifest_with_research_input(monkeypatch, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    pp = ProjectPaths(cfg.data_dir, "demo")
    pp.ensure()
    write_model(pp.transcript_json, Transcript(clip_id="demo", text="transcript"))
    write_model(pp.analysis_json, ViralAnalysis(clip_id="demo", formula="formula"))
    pp.research_md.write_text("edited research", encoding="utf-8")

    def fake_storyboard(**kwargs):
        assert kwargs["research_markdown"] == "edited research"
        return Storyboard(project="demo", format=VideoFormat.talking_head)

    monkeypatch.setattr("ai_clip.pipeline.generate_storyboard", fake_storyboard)

    sb = pipeline.run_storyboard(cfg, "demo", "theme")

    assert sb.project == "demo"
    manifest = read_artifact_manifest(pp.storyboard_json)
    assert manifest.stage == "storyboard"
    assert str(pp.research_md) in manifest.inputs
    assert manifest.params["theme"] == "theme"


def test_run_source_draft_injects_existing_research(monkeypatch, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    pp = ProjectPaths(cfg.data_dir, "demo")
    pp.ensure()
    write_model(pp.transcript_json, Transcript(clip_id="demo", text="transcript"))
    write_model(pp.analysis_json, ViralAnalysis(clip_id="demo", formula="formula"))
    pp.research_md.write_text("edited research", encoding="utf-8")

    def fake_draft(transcript, analysis, cfg, intent, stance="", research_markdown=""):
        assert research_markdown == "edited research"
        return "# draft"

    monkeypatch.setattr("ai_clip.produce.source_draft.generate_source_draft", fake_draft)

    path = pipeline.run_source_draft(cfg, "demo")

    assert path == pp.source_draft_md
    assert pp.source_draft_md.read_text(encoding="utf-8") == "# draft"
    manifest = read_artifact_manifest(pp.source_draft_md)
    assert manifest.stage == "source_draft"
    assert str(pp.research_md) in manifest.inputs
