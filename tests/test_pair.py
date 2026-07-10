import json

import pytest

from ai_clip.core.config import Config
from ai_clip.core.artifacts import artifact_manifest_path
from ai_clip.artifact_catalog import resolve_review_artifact
from ai_clip.pair.models import PairReviewReport, ReviewerResult
from ai_clip.pair.client import configured_models
from ai_clip.pair.stage import (
    PairReviewError,
    pair_artifact_freshness,
    review_artifact,
    rewrite_reviewed_artifact,
    verify_rewritten_artifact,
)


def _reply(verdict="pass"):
    return json.dumps({
        "verdict": verdict,
        "summary": "ok",
        "issues": [],
    })


def test_configured_models_includes_newapp_and_deepseek():
    cfg = Config()
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "new"
    cfg.pair.models = ["gpt-5.5"]
    cfg.pair.deepseek_api_key = "deep"
    models = configured_models(cfg.pair)
    assert [m.model for m in models] == ["gpt-5.5", "deepseek-4-pro"]
    assert models[0].base_url == "https://newapp.example/v1"
    assert models[1].base_url == "https://api.deepseek.com/v1"


def test_pair_review_uses_distinct_models_and_writes_report(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "storyboard.md").write_text("# storyboard", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.model = "producer"
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["producer", "logic-model", "style-model"]

    seen = []

    def fake_chat(model, **kwargs):
        seen.append(model.model)
        return _reply()

    monkeypatch.setattr("ai_clip.pair.stage.client.chat", fake_chat)
    report = review_artifact(cfg, project, "storyboard")
    assert report.status == "passed"
    assert [r.model for r in report.reviewers] == ["logic-model", "style-model"]
    assert seen == ["logic-model", "style-model"]
    assert (root / "reviews" / "storyboard_review.json").exists()


def test_pair_review_supports_source_draft(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "source_draft.md").write_text("# draft", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["m1", "m2"]

    monkeypatch.setattr("ai_clip.pair.stage.client.chat", lambda *a, **k: _reply())
    report = review_artifact(cfg, project, "source_draft")

    assert report.status == "passed"
    assert (root / "reviews" / "source_draft_review.json").exists()


def test_pair_review_supports_zack_draft(monkeypatch, tmp_path):
    zack_draft = tmp_path / "radar" / "drafts" / "2026-01-02.md"
    zack_draft.parent.mkdir(parents=True)
    zack_draft.write_text("# zack draft", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["m1", "m2"]

    monkeypatch.setattr("ai_clip.pair.stage.client.chat", lambda *a, **k: _reply())
    report = review_artifact(cfg, "radar", "zack_draft", run_date="2026-01-02")

    assert report.artifact == "zack_draft"
    assert report.status == "passed"
    assert (tmp_path / "radar" / "reviews" / "2026-01-02_zack_draft_review.json").exists()


@pytest.mark.parametrize("artifact", ["radar_draft", "scout_draft"])
def test_pair_review_rejects_removed_legacy_aliases(tmp_path, artifact):
    cfg = Config(data_dir=str(tmp_path))
    with pytest.raises(PairReviewError, match="unknown artifact"):
        review_artifact(cfg, "radar", artifact, run_date="2026-01-02")


def test_pair_rewrite_writes_revised_source_draft(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "source_draft.md").write_text("# draft", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.api_key = "key"
    report = PairReviewReport(
        artifact="source_draft",
        source_path=str(root / "source_draft.md"),
        producer_model="producer",
        status="needs_review",
        reviewers=[ReviewerResult(role="logic", ok=True, verdict="revise", summary="fix")],
    )

    monkeypatch.setattr("ai_clip.pair.stage.llm_mod.chat", lambda *a, **k: "# revised")
    out = rewrite_reviewed_artifact(cfg, project, "source_draft", report)

    assert out == root / "source_draft.revised.md"
    assert out.read_text(encoding="utf-8") == "# revised"


def test_pair_rewrite_rejects_blocked_report(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "source_draft.md").write_text("# draft", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    report = PairReviewReport(
        artifact="source_draft",
        source_path=str(root / "source_draft.md"),
        producer_model="producer",
        status="blocked",
        reviewers=[ReviewerResult(role="logic", ok=False, error="model unavailable")],
    )

    monkeypatch.setattr("ai_clip.pair.stage.llm_mod.chat", lambda *a, **k: "# revised")

    with pytest.raises(PairReviewError, match="pair-review is blocked"):
        rewrite_reviewed_artifact(cfg, project, "source_draft", report)

    assert not (root / "source_draft.revised.md").exists()


def test_pair_rewrite_writes_revised_script(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "script.md").write_text("# script", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.api_key = "key"
    report = PairReviewReport(
        artifact="script",
        source_path=str(root / "script.md"),
        producer_model="producer",
        status="needs_review",
        reviewers=[ReviewerResult(role="style", ok=True, verdict="revise", summary="tighten")],
    )

    monkeypatch.setattr("ai_clip.pair.stage.llm_mod.chat", lambda *a, **k: "# revised script")
    out = rewrite_reviewed_artifact(cfg, project, "script", report)

    assert out == root / "script.revised.md"
    assert out.read_text(encoding="utf-8") == "# revised script"


def test_pair_rewrite_writes_revised_research(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "research.md").write_text("# research", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.api_key = "key"
    report = PairReviewReport(
        artifact="research",
        source_path=str(root / "research.md"),
        producer_model="producer",
        status="needs_review",
        reviewers=[ReviewerResult(role="logic", ok=True, verdict="revise", summary="source risk")],
    )

    monkeypatch.setattr("ai_clip.pair.stage.llm_mod.chat", lambda *a, **k: "# revised research")
    out = rewrite_reviewed_artifact(cfg, project, "research", report)

    assert out == root / "research.revised.md"
    assert out.read_text(encoding="utf-8") == "# revised research"


def test_pair_review_falls_back_to_next_distinct_model(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "script.md").write_text("script", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["m1", "m2", "m3"]

    def fake_chat(model, **kwargs):
        if model.model == "m2":
            raise RuntimeError("temporary")
        return _reply()

    monkeypatch.setattr("ai_clip.pair.stage.client.chat", fake_chat)
    report = review_artifact(cfg, project, "script")
    assert [r.model for r in report.reviewers] == ["m1", "m3"]
    assert all(r.ok for r in report.reviewers)


def test_pair_review_requires_two_distinct_models(tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    (root / "storyboard.md").write_text("# storyboard", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["same"]
    cfg.pair.deepseek_models = []
    with pytest.raises(PairReviewError, match="two distinct"):
        review_artifact(cfg, project, "storyboard")


def test_pair_review_rewrite_verify_freshness_chain(monkeypatch, tmp_path):
    project = "demo"
    root = tmp_path / project
    root.mkdir()
    source = root / "source_draft.md"
    source.write_text("# draft", encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    cfg.llm.model = "producer"
    cfg.pair.base_url = "https://newapp.example/v1"
    cfg.pair.api_key = "key"
    cfg.pair.models = ["logic", "style"]
    monkeypatch.setattr("ai_clip.pair.stage.client.chat", lambda *a, **k: _reply("revise"))

    report = review_artifact(cfg, project, "source_draft")
    monkeypatch.setattr("ai_clip.pair.stage.llm_mod.chat", lambda *a, **k: "# revised")
    revised = rewrite_reviewed_artifact(cfg, project, "source_draft", report)
    monkeypatch.setattr("ai_clip.pair.stage.client.chat", lambda *a, **k: _reply("pass"))
    verification = verify_rewritten_artifact(cfg, project, "source_draft")
    target = resolve_review_artifact(cfg, project, "source_draft")

    assert verification.kind == "verify"
    assert verification.source_path == str(revised)
    assert artifact_manifest_path(target.review).exists()
    assert artifact_manifest_path(revised).exists()
    assert target.verification is not None
    assert artifact_manifest_path(target.verification).exists()
    assert pair_artifact_freshness(cfg, target) == {
        "review": True,
        "rewrite": True,
        "verify": True,
    }

    source.write_text("# changed", encoding="utf-8")
    assert pair_artifact_freshness(cfg, target) == {
        "review": False,
        "rewrite": False,
        "verify": False,
    }



