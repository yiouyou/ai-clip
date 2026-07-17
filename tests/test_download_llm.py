import httpx
import pytest

from ai_clip.core import llm as llm_mod
from ai_clip.core import billing
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Platform
from ai_clip.download.downloader import detect_platform, make_clip_id


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=x", Platform.youtube),
        ("https://youtu.be/x", Platform.youtube),
        ("https://www.bilibili.com/video/BV1", Platform.bilibili),
        ("https://v.douyin.com/abc", Platform.douyin),
        ("https://www.kuaishou.com/x", Platform.kuaishou),
        ("https://www.tiktok.com/@a/video/1", Platform.tiktok),
        ("https://example.com/x", Platform.unknown),
    ],
)
def test_detect_platform(url, expected):
    assert detect_platform(url) == expected


def test_make_clip_id_stable():
    assert make_clip_id("u") == make_clip_id("u")
    assert len(make_clip_id("u")) == 12


def test_extract_json_fenced():
    assert llm_mod.extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_raw():
    assert llm_mod.extract_json('noise {"a": 2} tail') == {"a": 2}


def test_extract_json_error():
    with pytest.raises(llm_mod.LLMError):
        llm_mod.extract_json("no json here")


def test_extract_json_rejects_malformed_object_with_category():
    with pytest.raises(llm_mod.LLMError) as caught:
        llm_mod.extract_json('{"broken": }')

    assert caught.value.category.value == "invalid_response"


def test_chat_requires_key():
    with pytest.raises(llm_mod.LLMError):
        llm_mod.chat(LLMConfig(api_key=""), "s", "u")


def test_chat_retries_rate_limit_and_records_attempts(monkeypatch, tmp_path):
    calls = []

    def fake_post(url, **kwargs):
        request = httpx.Request("POST", url)
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(
            200,
            request=request,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    monkeypatch.setattr(llm_mod.httpx, "post", fake_post)
    monkeypatch.setattr("ai_clip.core.retry.time.sleep", lambda _: None)
    with billing.account(tmp_path, "test"):
        result = llm_mod.chat(
            LLMConfig(api_key="key", model="test", max_attempts=2),
            "system",
            "user",
        )

    assert result == "ok"
    assert len(calls) == 2
    usage = billing.summarize(tmp_path)
    assert usage["total"]["calls"] == 1
    assert usage["total"]["attempts"] == 2
    assert usage["total"]["retries"] == 1
