import json
from pathlib import Path

import pytest

from ai_clip.analyze import analyzer
from ai_clip.core import llm as llm_mod
from ai_clip.core.config import AssetsConfig, LLMConfig
from ai_clip.core.models import Transcript, TranscriptSegment, VideoFormat
from ai_clip.produce import storyboard as sb_mod
from ai_clip.produce.assets.comfyui import ComfyUIError, ComfyUIProvider
from ai_clip.produce.assets.factory import resolve_image_provider
from ai_clip.produce.assets.prompt_only import PromptOnlyProvider

_ANALYSIS_REPLY = json.dumps(
    {
        "hook": "前3秒抛出反常识问题",
        "structure": ["问题", "反转", "结论"],
        "emotion_curve": ["好奇", "惊讶"],
        "formula": "反常识钩子 + 三段式 + 行动号召",
        "scores": {"hook_strength": 0.9, "retention": 0.8},
        "notes": "节奏快",
    }
)

_TALKING_HEAD_REPLY = json.dumps({
    "lines": [
        {"voiceover": "你知道吗", "duration_sec": 4, "broll_prompt": "城市夜景"},
        {"voiceover": "答案很简单", "duration_sec": 3, "broll_prompt": ""},
    ]
})

_SLIDESHOW_REPLY = json.dumps({
    "cards": [
        {"caption": "第一招", "voiceover": "先做这个", "image_prompt": "图1", "duration_sec": 3},
        {"caption": "第二招", "voiceover": "再做那个", "image_prompt": "图2", "duration_sec": 3},
    ]
})

_REMIX_REPLY = json.dumps({
    "spans": [
        {"source_start": 1.0, "source_end": 4.0, "voiceover": "高能片段一"},
        {"source_start": 5.0, "source_end": 999.0, "voiceover": "高能片段二"},  # clamp
        {"source_start": 8.0, "source_end": 8.0, "voiceover": "零长度丢弃"},   # dropped
    ]
})


def test_analyze_with_mocked_llm(monkeypatch):
    monkeypatch.setattr(analyzer.llm_mod, "chat", lambda *a, **k: _ANALYSIS_REPLY)
    t = Transcript(clip_id="c1", text="some transcript text")
    result = analyzer.analyze(t, LLMConfig(api_key="x"))
    assert result.hook.startswith("前3秒")
    assert result.scores["hook_strength"] == 0.9
    assert result.formula


def test_analyze_empty_transcript():
    with pytest.raises(ValueError):
        analyzer.analyze(Transcript(clip_id="c1", text="  "), LLMConfig(api_key="x"))


def test_talking_head_format(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: _TALKING_HEAD_REPLY)
    sb = sb_mod.generate_storyboard(
        "demo", "理财", LLMConfig(api_key="x"), fmt=VideoFormat.talking_head
    )
    assert sb.format == VideoFormat.talking_head
    assert len(sb.shots) == 2
    assert sb.shots[0].voiceover == "你知道吗"
    assert sb.shots[0].image_file == "shot_01.png"  # has b-roll
    assert sb.shots[1].image_file == ""  # no b-roll -> no asset expected
    assert sb.shots[1].expected_files() == []

    prompts = tmp_path / "prompts"
    sb_mod.write_storyboard_files(sb, prompts, tmp_path / "storyboard.md")
    assert (prompts / "shot_01_image.txt").read_text(encoding="utf-8") == "城市夜景"
    assert not (prompts / "shot_02_image.txt").exists()


def test_slideshow_format(monkeypatch):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: _SLIDESHOW_REPLY)
    sb = sb_mod.generate_storyboard(
        "demo", "技巧", LLMConfig(api_key="x"), fmt=VideoFormat.slideshow
    )
    assert sb.format == VideoFormat.slideshow
    assert sb.shots[0].caption == "第一招"
    assert sb.shots[0].image_file == "shot_01.png"


def test_remix_format_clamps_and_drops(monkeypatch):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: _REMIX_REPLY)
    transcript = Transcript(
        clip_id="c1",
        segments=[
            TranscriptSegment(start=0.0, end=4.0, text="a"),
            TranscriptSegment(start=4.0, end=10.0, text="b"),
        ],
    )
    sb = sb_mod.generate_storyboard(
        "demo", "盘点", LLMConfig(api_key="x"),
        fmt=VideoFormat.remix, transcript=transcript,
    )
    assert sb.format == VideoFormat.remix
    assert len(sb.shots) == 2  # zero-length span dropped
    assert sb.shots[0].is_source_segment
    assert sb.shots[0].expected_files() == []  # remix needs no generated assets
    assert sb.shots[1].source_end == 10.0  # clamped to transcript max


def test_remix_requires_transcript(monkeypatch):
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: _REMIX_REPLY)
    with pytest.raises(ValueError):
        sb_mod.generate_storyboard(
            "demo", "x", LLMConfig(api_key="x"), fmt=VideoFormat.remix
        )


def test_factory_prompt_only():
    p = resolve_image_provider(AssetsConfig(image_provider="prompt_only"))
    assert isinstance(p, PromptOnlyProvider)
    assert p.name == "prompt_only"


def test_factory_auto_falls_back(monkeypatch):
    # ComfyUI unreachable -> auto must fall back to prompt_only
    monkeypatch.setattr(ComfyUIProvider, "is_available", staticmethod(lambda *a, **k: False))
    p = resolve_image_provider(AssetsConfig(image_provider="auto"))
    assert isinstance(p, PromptOnlyProvider)


def test_factory_comfyui_explicit_unavailable_raises(monkeypatch):
    monkeypatch.setattr(ComfyUIProvider, "is_available", staticmethod(lambda *a, **k: False))
    with pytest.raises(RuntimeError):
        resolve_image_provider(AssetsConfig(image_provider="comfyui"))


def test_comfyui_inject_prompt():
    template = {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    from ai_clip.core.models import Shot

    graph = provider._inject_prompt("hello")
    assert graph["6"]["inputs"]["text"] == "hello"
    _ = Shot  # silence unused in some runners


def test_comfyui_inject_prompt_missing_placeholder():
    provider = ComfyUIProvider("http://x", {"6": {"inputs": {"text": "fixed"}}})
    with pytest.raises(ComfyUIError):
        provider._inject_prompt("hello")
