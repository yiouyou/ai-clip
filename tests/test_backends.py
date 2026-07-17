from pathlib import Path

import httpx
import pytest

from ai_clip.core.async_jobs import (
    async_job_request_hash,
    new_async_job_state,
    read_async_job_state,
    write_async_job_state,
)
from ai_clip.produce.backends import MoneyPrinterBackend, ProduceSpec
from ai_clip.produce.backends.moneyprinter import MoneyPrinterError


def test_request_body_maps_spec():
    be = MoneyPrinterBackend("http://x")
    body = be._request_body(ProduceSpec(theme="麻将", out_path=Path("o.mp4"),
                                        aspect_ratio="9:16", voice_name="V", language="zh"))
    assert body["video_subject"] == "麻将"
    assert body["video_aspect"] == "9:16"
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
    out = tmp_path / "o.mp4"
    with pytest.raises(MoneyPrinterError):
        MoneyPrinterBackend("http://x").produce(ProduceSpec(theme="t", out_path=out))
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "failed"


def test_produce_resumes_known_task_without_resubmitting(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m

    out = tmp_path / "o.mp4"
    spec = ProduceSpec(theme="t", out_path=out)
    backend = MoneyPrinterBackend("http://x")
    body = backend._request_body(spec)
    state = new_async_job_state(
        provider=backend.name,
        request_hash=async_job_request_hash(backend.name, backend.base_url, body),
        output_path=out,
        remote_id="existing-task",
        status="submitted",
    )
    write_async_job_state(out, state)
    monkeypatch.setattr(
        m.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resubmit")),
    )
    monkeypatch.setattr(
        m.httpx,
        "get",
        lambda url, timeout: _Resp({"data": {"videos": ["/video.mp4"]}}),
    )

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): return None
        def iter_bytes(self): yield b"RESUMED"

    monkeypatch.setattr(m.httpx, "stream", lambda method, url, timeout: _Stream())

    assert backend.produce(spec).read_bytes() == b"RESUMED"
    completed = read_async_job_state(out)
    assert completed is not None
    assert completed.status == "succeeded"


def test_produce_blocks_retry_when_submission_outcome_is_unknown(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m

    out = tmp_path / "o.mp4"
    spec = ProduceSpec(theme="t", out_path=out)
    calls = 0

    def ambiguous_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(
            "response lost",
            request=httpx.Request("POST", "http://x/api/v1/videos"),
        )

    monkeypatch.setattr(m.httpx, "post", ambiguous_post)
    backend = MoneyPrinterBackend("http://x")
    with pytest.raises(MoneyPrinterError, match="outcome is unknown"):
        backend.produce(spec)
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "unknown"
    assert state.remote_id == ""

    with pytest.raises(MoneyPrinterError, match="verify the server"):
        backend.produce(spec)
    assert calls == 1


def test_produce_records_connect_timeout_as_safe_failure(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m

    out = tmp_path / "o.mp4"

    def connect_timeout(*args, **kwargs):
        raise httpx.ConnectTimeout(
            "not connected",
            request=httpx.Request("POST", "http://x/api/v1/videos"),
        )

    monkeypatch.setattr(m.httpx, "post", connect_timeout)
    with pytest.raises(MoneyPrinterError, match="was not submitted"):
        MoneyPrinterBackend("http://x").produce(ProduceSpec(theme="t", out_path=out))
    state = read_async_job_state(out)
    assert state is not None
    assert state.status == "failed"


def test_produce_rejects_changed_request_while_task_is_active(monkeypatch, tmp_path):
    import ai_clip.produce.backends.moneyprinter as m

    out = tmp_path / "o.mp4"
    state = new_async_job_state(
        provider="moneyprinter",
        request_hash="old-request",
        output_path=out,
        remote_id="active-task",
        status="running",
    )
    write_async_job_state(out, state)
    monkeypatch.setattr(
        m.httpx,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resubmit")),
    )

    with pytest.raises(MoneyPrinterError, match="conflicts"):
        MoneyPrinterBackend("http://x").produce(ProduceSpec(theme="new", out_path=out))


def test_storyboard_to_clip_json_maps_source_spans():
    from ai_clip.core.models import Shot, Storyboard
    from ai_clip.produce.backends import storyboard_to_clip_json

    sb = Storyboard(project="p", shots=[
        Shot(index=1, source_start=1.0, source_end=4.0, voiceover="一"),
        Shot(index=2, voiceover="无源,跳过"),  # not a source segment -> skipped
    ])
    clips = storyboard_to_clip_json(sb)
    assert len(clips) == 1
    assert clips[0]["timestamp"] == "00:00:01-00:00:04"
    assert clips[0]["narration"] == "一"
    assert clips[0]["_id"] == 1
    assert clips[0]["OST"] == 0


def test_narrato_missing_repo_raises(tmp_path):
    from ai_clip.produce.backends.narrato import NarratoBackend, NarratoError

    be = NarratoBackend(tmp_path / "nope", "python")
    with pytest.raises(NarratoError):
        be.produce_remix("src.mp4", [{"timestamp": "0-1", "narration": "x"}], tmp_path / "o.mp4")
