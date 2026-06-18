import pytest

from ai_clip.core import llm as llm_mod
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


def test_chat_requires_key():
    with pytest.raises(llm_mod.LLMError):
        llm_mod.chat(LLMConfig(api_key=""), "s", "u")
