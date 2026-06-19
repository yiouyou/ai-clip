from ai_clip.core import billing


def test_llm_cost_known_model():
    # deepseek-v4-pro = (0.435, 0.87) per 1M
    assert billing.llm_cost("deepseek-v4-pro", 1_000_000, 0) == 0.435
    assert round(billing.llm_cost("deepseek-v4-pro", 0, 1_000_000), 4) == 0.87


def test_llm_cost_unknown_model_is_zero():
    assert billing.llm_cost("mystery", 1_000_000, 1_000_000) == 0.0


def test_record_noop_without_context(tmp_path):
    # outside an account() block, recording is a no-op (no file written)
    billing.record_llm("gpt-5.5", 100, 100)
    assert not (tmp_path / "cost.jsonl").exists()


def test_account_records_and_summarizes(tmp_path):
    with billing.account(tmp_path, "analyze"):
        billing.record_llm("deepseek-v4-pro", 1_000_000, 1_000_000)
    with billing.account(tmp_path, "voiceover"):
        billing.record_tts("mimo", 500)

    s = billing.summarize(tmp_path)
    assert s["total"]["calls"] == 2
    assert s["total"]["input_tokens"] == 1_000_000
    assert s["total"]["chars"] == 500
    # 0.435 + 0.87 = 1.305 for the llm call
    assert round(s["by_stage"]["analyze"], 4) == 1.305
    assert "deepseek-v4-pro" in s["by_model"]


def test_summarize_missing_file(tmp_path):
    s = billing.summarize(tmp_path / "nope")
    assert s["total"]["cost"] == 0.0
    assert s["items"] == []
