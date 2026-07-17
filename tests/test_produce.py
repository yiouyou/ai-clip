import json
import subprocess
from pathlib import Path

import httpx
import pytest

from ai_clip.analyze import analyzer
from ai_clip.core import llm as llm_mod
from ai_clip import pipeline
from ai_clip.core.artifacts import artifact_manifest_path
from ai_clip.core.async_jobs import (
    async_job_request_hash,
    async_job_state_path,
    new_async_job_state,
    read_async_job_state,
    write_async_job_state,
)
from ai_clip.core.config import AssetsConfig, Config, LLMConfig
from ai_clip.core.models import AssetEngine, Shot, Storyboard, Transcript, TranscriptSegment, VideoFormat
from ai_clip.core.paths import ProjectPaths, write_model
from ai_clip.produce import storyboard as sb_mod
from ai_clip.produce.assets.comfyui import ComfyUIError, ComfyUIProvider
from ai_clip.produce.assets.factory import resolve_image_provider
from ai_clip.produce.assets.prompt_only import PromptOnlyProvider
from ai_clip.produce.assets.smart_illustrator import SmartIllustratorProvider

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
        {
            "voiceover": "你知道吗",
            "duration_sec": 4,
            "broll_prompt": "城市夜景",
            "asset_engine": "smart_illustrator",
        },
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
    seen = {}

    def fake_chat(cfg, system, user):
        seen["user"] = user
        return _TALKING_HEAD_REPLY

    monkeypatch.setattr(llm_mod, "chat", fake_chat)
    sb = sb_mod.generate_storyboard(
        "demo",
        "理财",
        LLMConfig(api_key="x"),
        fmt=VideoFormat.talking_head,
        research_markdown="confirmed detail",
    )
    assert sb.format == VideoFormat.talking_head
    assert len(sb.shots) == 2
    assert sb.shots[0].voiceover == "你知道吗"
    assert sb.shots[0].image_file == "shot_01.png"  # has b-roll
    assert sb.shots[0].asset_engine == AssetEngine.smart_illustrator
    assert sb.shots[1].image_file == ""  # no b-roll -> no asset expected
    assert sb.shots[1].expected_files() == []
    assert "confirmed detail" in seen["user"]

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


def test_factory_asset_engine_selects_smart_illustrator(monkeypatch, tmp_path):
    script = tmp_path / "generate-image.ts"
    script.write_text("// ok", encoding="utf-8")
    monkeypatch.setattr(SmartIllustratorProvider, "is_available", staticmethod(lambda *a, **k: True))
    cfg = AssetsConfig(smart_illustrator_script=str(script))
    p = resolve_image_provider(cfg, engine=AssetEngine.smart_illustrator)
    assert isinstance(p, SmartIllustratorProvider)


def test_factory_asset_engine_hint_falls_back_in_auto(monkeypatch):
    monkeypatch.setattr(SmartIllustratorProvider, "is_available", staticmethod(lambda *a, **k: False))
    p = resolve_image_provider(AssetsConfig(image_provider="auto"), engine=AssetEngine.gemini)
    assert isinstance(p, PromptOnlyProvider)


def test_smart_illustrator_provider_writes_prompt_and_calls_script(monkeypatch, tmp_path):
    script = tmp_path / "generate-image.ts"
    script.write_text("// ok", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda name: "npx" if name == "npx" else None)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = SmartIllustratorProvider(AssetsConfig(smart_illustrator_script=str(script)))
    out = provider.generate(
        Shot(index=1, image_prompt="信息图", image_file="shot_01.png"),
        tmp_path / "assets",
    )
    assert out.read_bytes() == b"png"
    assert (tmp_path / "assets" / "source" / "shot_01_smart_illustrator_prompt.txt").exists()
    assert "--prompt-file" in calls[0][0]


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


def test_comfyui_persists_client_prompt_id_before_submit(monkeypatch, tmp_path):
    import ai_clip.produce.assets.comfyui as comfy

    template = {"6": {"inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    shot = Shot(index=1, image_prompt="hello", image_file="shot_01.png")
    out = tmp_path / shot.image_file

    def fake_post(url, json, timeout):
        state = read_async_job_state(out)
        assert state is not None
        assert state.status == "submitting"
        assert json["prompt_id"] == state.remote_id
        return httpx.Response(
            200,
            json={"prompt_id": state.remote_id},
            request=httpx.Request("POST", url),
        )

    def fake_get(url, **kwargs):
        if "/history/" in url:
            prompt_id = url.rsplit("/", 1)[-1]
            payload = {
                prompt_id: {"outputs": {"9": {"images": [{"filename": "out.png"}]}}}
            }
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
        return httpx.Response(200, content=b"PNG", request=httpx.Request("GET", url))

    monkeypatch.setattr(comfy.httpx, "post", fake_post)
    monkeypatch.setattr(comfy.httpx, "get", fake_get)

    assert provider.generate(shot, tmp_path).read_bytes() == b"PNG"
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "succeeded"


def test_comfyui_resumes_known_prompt_without_resubmitting(monkeypatch, tmp_path):
    import ai_clip.produce.assets.comfyui as comfy

    template = {"6": {"inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    shot = Shot(index=1, image_prompt="hello", image_file="shot_01.png")
    out = tmp_path / shot.image_file
    graph = provider._inject_prompt(shot.image_prompt)
    state = new_async_job_state(
        provider=provider.name,
        request_hash=async_job_request_hash(provider.name, provider.base_url, {"prompt": graph}),
        output_path=out,
        remote_id="00000000-0000-0000-0000-000000000001",
        status="submitted",
    )
    write_async_job_state(out, state)
    monkeypatch.setattr(
        comfy.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resubmit")),
    )

    def fake_get(url, **kwargs):
        if "/history/" in url:
            payload = {
                state.remote_id: {
                    "outputs": {"9": {"images": [{"filename": "out.png"}]}}
                }
            }
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
        return httpx.Response(200, content=b"RESUMED", request=httpx.Request("GET", url))

    monkeypatch.setattr(comfy.httpx, "get", fake_get)

    assert provider.generate(shot, tmp_path).read_bytes() == b"RESUMED"


def test_comfyui_rejects_changed_request_while_job_is_active(monkeypatch, tmp_path):
    import ai_clip.produce.assets.comfyui as comfy

    template = {"6": {"inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    out = tmp_path / "shot_01.png"
    state = new_async_job_state(
        provider=provider.name,
        request_hash="old-request",
        output_path=out,
        remote_id="00000000-0000-0000-0000-000000000001",
        status="running",
    )
    write_async_job_state(out, state)
    monkeypatch.setattr(
        comfy.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resubmit")),
    )

    with pytest.raises(ComfyUIError, match="conflicts"):
        provider.generate(
            Shot(index=1, image_prompt="new", image_file=out.name),
            tmp_path,
        )


def test_comfyui_recovers_after_submission_response_is_lost(monkeypatch, tmp_path):
    import ai_clip.produce.assets.comfyui as comfy

    template = {"6": {"inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    shot = Shot(index=1, image_prompt="hello", image_file="shot_01.png")
    out = tmp_path / shot.image_file

    def fake_post(url, json, timeout):
        raise httpx.ReadTimeout("response lost", request=httpx.Request("POST", url))

    def fake_get(url, **kwargs):
        if "/history/" in url:
            state = read_async_job_state(out)
            assert state is not None
            assert state.status == "unknown"
            payload = {
                state.remote_id: {
                    "outputs": {"9": {"images": [{"filename": "out.png"}]}}
                }
            }
            return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
        return httpx.Response(200, content=b"RECOVERED", request=httpx.Request("GET", url))

    monkeypatch.setattr(comfy.httpx, "post", fake_post)
    monkeypatch.setattr(comfy.httpx, "get", fake_get)

    assert provider.generate(shot, tmp_path).read_bytes() == b"RECOVERED"
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "succeeded"


def test_comfyui_records_remote_failure(monkeypatch, tmp_path):
    import ai_clip.produce.assets.comfyui as comfy

    template = {"6": {"inputs": {"text": "AICLIP_PROMPT"}}}
    provider = ComfyUIProvider("http://x", template)
    shot = Shot(index=1, image_prompt="hello", image_file="shot_01.png")
    out = tmp_path / shot.image_file

    def fake_post(url, json, timeout):
        return httpx.Response(
            200,
            json={"prompt_id": json["prompt_id"]},
            request=httpx.Request("POST", url),
        )

    def fake_get(url, **kwargs):
        prompt_id = url.rsplit("/", 1)[-1]
        payload = {prompt_id: {"outputs": {}, "status": {"status_str": "error"}}}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(comfy.httpx, "post", fake_post)
    monkeypatch.setattr(comfy.httpx, "get", fake_get)

    with pytest.raises(ComfyUIError, match="failed"):
        provider.generate(shot, tmp_path)
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "failed"


def test_run_assets_recovers_output_written_before_manifest(monkeypatch, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    paths = ProjectPaths(tmp_path, "demo")
    paths.ensure()
    out = paths.assets_dir / "shot_01.png"
    out.write_bytes(b"generated")
    state = new_async_job_state(
        provider="fake",
        request_hash="request",
        output_path=out,
        remote_id="remote",
        status="succeeded",
    )
    write_async_job_state(out, state)
    write_model(
        paths.storyboard_json,
        Storyboard(
            project="demo",
            shots=[Shot(index=1, image_file=out.name, image_prompt="a")],
        ),
    )

    class FakeProvider:
        name = "fake"

        def cache_params(self):
            return {"provider": self.name}

        def generate(self, shot, assets_dir):
            return assets_dir / shot.image_file

    monkeypatch.setattr(
        "ai_clip.produce.assets.factory.resolve_image_provider",
        lambda *a, **k: FakeProvider(),
    )

    assert pipeline.run_assets(cfg, "demo") == 1
    assert artifact_manifest_path(out).exists()
    assert async_job_state_path(out).exists()


def test_run_assets_reuses_matching_generation_and_refreshes_changed_prompt(
    monkeypatch,
    tmp_path,
):
    cfg = Config(data_dir=str(tmp_path))
    paths = ProjectPaths(tmp_path, "demo")
    paths.ensure()
    calls = []

    class FakeProvider:
        name = "fake"

        def cache_params(self):
            return {"provider": self.name, "model": "v1"}

        def generate(self, shot, assets_dir):
            calls.append(shot.image_prompt)
            out = assets_dir / shot.image_file
            out.write_bytes(shot.image_prompt.encode())
            return out

    monkeypatch.setattr(
        "ai_clip.produce.assets.factory.resolve_image_provider",
        lambda *a, **k: FakeProvider(),
    )
    write_model(
        paths.storyboard_json,
        Storyboard(project="demo", shots=[Shot(index=1, image_file="shot_01.png", image_prompt="a")]),
    )

    assert pipeline.run_assets(cfg, "demo") == 1
    assert pipeline.run_assets(cfg, "demo") == 0
    write_model(
        paths.storyboard_json,
        Storyboard(project="demo", shots=[Shot(index=1, image_file="shot_01.png", image_prompt="b")]),
    )
    assert pipeline.run_assets(cfg, "demo") == 1

    assert calls == ["a", "b"]
    assert artifact_manifest_path(paths.assets_dir / "shot_01.png").exists()


def test_run_assets_preserves_human_files_and_removes_generated_orphans(monkeypatch, tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    paths = ProjectPaths(tmp_path, "demo")
    paths.ensure()
    human = paths.assets_dir / "shot_01.png"
    human.write_bytes(b"human")
    write_model(
        paths.storyboard_json,
        Storyboard(project="demo", shots=[Shot(index=1, image_file=human.name, image_prompt="a")]),
    )
    monkeypatch.setattr(
        "ai_clip.produce.assets.factory.resolve_image_provider",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("human asset must not resolve provider")),
    )
    assert pipeline.run_assets(cfg, "demo") == 0
    assert human.read_bytes() == b"human"

    from ai_clip.produce.assets.cache import record_generated_asset

    generated = paths.assets_dir / "shot_02.png"
    generated.write_bytes(b"generated")
    record_generated_asset(generated, {"provider": "fake"})
    write_async_job_state(
        generated,
        new_async_job_state(
            provider="fake",
            request_hash="request",
            output_path=generated,
            remote_id="remote",
            status="succeeded",
        ),
    )
    write_model(paths.storyboard_json, Storyboard(project="demo", shots=[]))
    pipeline.run_assets(cfg, "demo")

    assert human.exists()
    assert not generated.exists()
    assert not artifact_manifest_path(generated).exists()
    assert not async_job_state_path(generated).exists()
