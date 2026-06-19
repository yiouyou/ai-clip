import json

from ai_clip.analyze import analyzer
from ai_clip.analyze.prompts import build_user_prompt
from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Intent, ProductProfile, Transcript, VideoFormat
from ai_clip.produce import storyboard as sb_mod
from ai_clip.produce.formats.base import GenerateArgs, intent_block


def test_build_user_prompt_intent_keys():
    assert '"stance"' in build_user_prompt("t", Intent.emotion)
    assert '"pain_points"' in build_user_prompt("t", Intent.sales)
    assert '"stance"' not in build_user_prompt("t", Intent.info)


def test_intent_block_emotion_uses_stance():
    args = GenerateArgs(project="p", theme="t", cfg=LLMConfig(), intent=Intent.emotion,
                        stance="AI 取代工作令人焦虑")
    block = intent_block(args)
    assert "emotion" in block and "AI 取代工作令人焦虑" in block


def test_intent_block_sales_uses_product():
    prod = ProductProfile(name="麻将手风预报", description="预测手气", audience="牌友",
                          selling_points=["每日评分"], cta="快来下载")
    args = GenerateArgs(project="p", theme="t", cfg=LLMConfig(), intent=Intent.sales, product=prod)
    block = intent_block(args)
    assert "麻将手风预报" in block and "快来下载" in block


def test_intent_block_sales_without_product_is_generic():
    args = GenerateArgs(project="p", theme="t", cfg=LLMConfig(), intent=Intent.sales)
    assert "generic" in intent_block(args)


def test_analyze_emotion_populates_stance(monkeypatch):
    reply = json.dumps({"hook": "h", "formula": "f", "stance": "AI 被高估了"})
    monkeypatch.setattr(analyzer.llm_mod, "chat", lambda *a, **k: reply)
    a = analyzer.analyze(Transcript(clip_id="c", text="x"), LLMConfig(api_key="k"), Intent.emotion)
    assert a.intent == Intent.emotion
    assert a.stance == "AI 被高估了"


def test_analyze_sales_populates_pain_points(monkeypatch):
    reply = json.dumps({"hook": "h", "formula": "f",
                        "pain_points": ["手气差"], "objections": ["怕是迷信"]})
    monkeypatch.setattr(analyzer.llm_mod, "chat", lambda *a, **k: reply)
    a = analyzer.analyze(Transcript(clip_id="c", text="x"), LLMConfig(api_key="k"), Intent.sales)
    assert a.pain_points == ["手气差"]
    assert a.objections == ["怕是迷信"]


def test_storyboard_threads_sales_product_into_prompt(monkeypatch):
    captured = {}

    def fake_chat(cfg, system, user):
        captured["user"] = user
        return json.dumps({"lines": [{"voiceover": "v", "duration_sec": 3}]})

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    prod = ProductProfile(name="麻将手风预报", cta="下载它")
    sb_mod.generate_storyboard(
        "p", "麻将", LLMConfig(api_key="k"), fmt=VideoFormat.talking_head,
        intent=Intent.sales, product=prod,
    )
    assert "麻将手风预报" in captured["user"]
    assert "下载它" in captured["user"]
