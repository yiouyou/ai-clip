import pytest

from ai_clip import tools
from ai_clip.core.config import Config


def test_public_tool_name_contract_is_explicit():
    names = {t.name for t in tools.all_tools()}
    assert names == {
        "download",
        "extract",
        "analyze",
        "source_research",
        "storyboard",
        "assets",
        "voiceover",
        "assemble",
        "review_artifact",
        "rewrite_artifact",
        "verify_artifact",
        "source_draft",
        "collect",
        "rank_candidates",
        "enrich_content",
        "rerank_candidates",
        "select_topic",
        "research_evidence",
        "draft_script",
    }


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        tools.get("nope")


def test_tool_metadata_complete():
    for t in tools.all_tools():
        assert t.description
        assert t.params


def test_invoke_dispatches(monkeypatch):
    seen = {}

    def fake_download(cfg, project, url):
        seen["args"] = (project, url)
        return "clip"

    fake = tools.Tool("download", "fake", fake_download, {"project": "p"})
    monkeypatch.setitem(tools._TOOLS, "download", fake)
    out = tools.invoke("download", Config(), project="p", url="u")
    assert out == "clip"
    assert seen["args"] == ("p", "u")


