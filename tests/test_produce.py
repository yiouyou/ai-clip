import json
from pathlib import Path

import pytest

from ai_clip.analyze import analyzer
from ai_clip.core.config import AssetsConfig, LLMConfig
from ai_clip.core.models import Transcript, ViralAnalysis
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

_STORYBOARD_REPLY = json.dumps(
    {
        "shots": [
            {
                "duration_sec": 3,
                "shot_type": "close-up",
                "image_prompt": "城市夜景霓虹特写",
                "video_prompt": "镜头缓慢推进",
                "voiceover": "你知道吗",
            },
            {
                "duration_sec": 4,
                "shot_type": "wide",
                "image_prompt": "骑行者穿过街道",
                "video_prompt": "跟拍运镜",
                "voiceover": "答案很简单",
            },
        ]
    }
)


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


def test_generate_storyboard_and_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sb_mod.llm_mod, "chat", lambda *a, **k: _STORYBOARD_REPLY)
    analysis = ViralAnalysis(clip_id="c1", formula="反常识钩子")
    sb = sb_mod.generate_storyboard("demo", "夜骑", LLMConfig(api_key="x"), analysis)

    assert len(sb.shots) == 2
    # filename contract
    assert sb.shots[0].image_file == "shot_01.png"
    assert sb.shots[0].video_file == "shot_01.mp4"
    assert sb.source_clip_id == "c1"

    prompts = tmp_path / "prompts"
    md = tmp_path / "storyboard.md"
    sb_mod.write_storyboard_files(sb, prompts, md)
    assert (prompts / "shot_01_image.txt").read_text(encoding="utf-8") == "城市夜景霓虹特写"
    assert "Shot 01" in md.read_text(encoding="utf-8")


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
