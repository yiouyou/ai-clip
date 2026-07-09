from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Intent, Transcript, ViralAnalysis
from ai_clip.produce.source_draft import generate_source_draft


def test_generate_source_draft_injects_research(monkeypatch):
    seen = {}

    def fake_chat(cfg, system, user):
        seen["user"] = user
        return "# draft"

    monkeypatch.setattr("ai_clip.produce.source_draft.llm_mod.chat", fake_chat)

    out = generate_source_draft(
        transcript=Transcript(clip_id="demo", text="source transcript"),
        analysis=ViralAnalysis(clip_id="demo", formula="formula"),
        cfg=LLMConfig(api_key="x"),
        intent=Intent.info,
        research_markdown="confirmed research detail",
    )

    assert out == "# draft"
    assert "confirmed research detail" in seen["user"]
