import base64
import shutil
from pathlib import Path

import pytest

from ai_clip.core.config import TTSConfig, load_config
from ai_clip.core.models import Shot, Storyboard
from ai_clip.produce.tts import mimo
from ai_clip.produce.tts.mimo import MimoTTS, TTSError
from ai_clip.produce.voiceover import generate_voiceover, voice_filename

ffmpeg_available = shutil.which("ffmpeg") is not None


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_audio_payload(text: str):
    data = base64.b64encode(f"WAV::{text}".encode()).decode()
    return {"choices": [{"message": {"audio": {"data": data}}}]}


def test_mimo_requires_key():
    with pytest.raises(TTSError):
        MimoTTS(TTSConfig(api_key=""))


def test_mimo_clone_requires_reference():
    tts = MimoTTS(TTSConfig(api_key="k", model="mimo-v2.5-tts-voiceclone"))
    with pytest.raises(TTSError):
        tts.synthesize("hello", "out.wav")


def test_mimo_preset_synthesize(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp(_fake_audio_payload(json["messages"][1]["content"]))

    monkeypatch.setattr(mimo.httpx, "post", fake_post)
    tts = MimoTTS(TTSConfig(api_key="k", model="mimo-v2.5-tts", voice="Chloe"))
    out = tts.synthesize("你好世界", tmp_path / "v.wav", style="gentle")

    assert out.read_bytes() == b"WAV::\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"
    assert captured["headers"]["api-key"] == "k"
    assert captured["json"]["audio"]["voice"] == "Chloe"
    assert captured["json"]["messages"][0]["content"] == "gentle"
    assert captured["json"]["messages"][1]["content"] == "你好世界"


def test_mimo_clone_builds_data_uri(monkeypatch, tmp_path: Path):
    ref = tmp_path / "ref.mp3"
    ref.write_bytes(b"\x00\x01\x02reference")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["voice"] = json["audio"]["voice"]
        return _FakeResp(_fake_audio_payload("x"))

    monkeypatch.setattr(mimo.httpx, "post", fake_post)
    tts = MimoTTS(
        TTSConfig(api_key="k", model="mimo-v2.5-tts-voiceclone"), reference_path=ref
    )
    tts.synthesize("line", tmp_path / "v.wav")
    assert captured["voice"].startswith("data:audio/mpeg;base64,")


def test_generate_voiceover_skips_empty(tmp_path: Path):
    class FakeTTS:
        def __init__(self):
            self.calls = []

        def synthesize(self, text, out_path, style=""):
            self.calls.append(text)
            Path(out_path).write_bytes(b"x")
            return Path(out_path)

    sb = Storyboard(
        project="p",
        shots=[
            Shot(index=1, voiceover="说话内容"),
            Shot(index=2, voiceover="   "),  # empty -> skipped
        ],
    )
    tts = FakeTTS()
    produced = generate_voiceover(sb, tts, tmp_path)
    assert set(produced) == {1}
    assert (tmp_path / voice_filename(1)).exists()
    assert tts.calls == ["说话内容"]


def test_env_key_resolution(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AICLIP_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dskey")
    monkeypatch.setenv("MIMO_API_KEY", "mimokey")
    cfg = load_config(tmp_path / "missing.yaml")  # defaults: deepseek base_url
    assert cfg.llm.api_key == "dskey"
    assert cfg.tts.api_key == "mimokey"
