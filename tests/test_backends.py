from pathlib import Path

import pytest

from ai_clip.produce.backends import MoneyPrinterBackend, ProduceSpec
from ai_clip.produce.backends.moneyprinter import MoneyPrinterError, _ASPECT


def test_request_body_maps_spec():
    be = MoneyPrinterBackend("http://x")
    body = be._request_body(ProduceSpec(theme="麻将", out_path=Path("o.mp4"),
                                        aspect_ratio="9:16", voice_name="V", language="zh"))
    assert body["video_subject"] == "麻将"
    assert body["video_aspect"] == _ASPECT["9:16"]
    assert body["voice_name"] == "V"
    assert body["video_language"] == "zh"
    assert body["subtitle_enabled"] is True


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


def test_produce_polls_then_downloads(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m

    posted = {}

    def fake_post(url, json, timeout):
        posted["url"] = url
        return _Resp({"data": {"task_id": "t1"}})

    states = iter([
        {"data": {"state": 1}},                         # still running
        {"data": {"videos": ["http://x/api/v1/download/tasks/t1/final.mp4"]}},
    ])
    monkeypatch.setattr(m.httpx, "post", fake_post)
    monkeypatch.setattr(m.httpx, "get", lambda url, timeout: _Resp(next(states)))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): return None
        def iter_bytes(self): yield b"MP4DATA"

    monkeypatch.setattr(m.httpx, "stream", lambda method, url, timeout: _Stream())

    out = MoneyPrinterBackend("http://x").produce(
        ProduceSpec(theme="t", out_path=tmp_path / "o.mp4")
    )
    assert out.read_bytes() == b"MP4DATA"
    assert posted["url"].endswith("/api/v1/videos")


def test_produce_raises_on_failed_state(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m
    monkeypatch.setattr(m.httpx, "post", lambda url, json, timeout: _Resp({"data": {"task_id": "t"}}))
    monkeypatch.setattr(m.httpx, "get", lambda url, timeout: _Resp({"data": {"state": -1}}))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    with pytest.raises(MoneyPrinterError):
        MoneyPrinterBackend("http://x").produce(ProduceSpec(theme="t", out_path=tmp_path / "o.mp4"))
