from ai_clip.core.models import Shot
from ai_clip.produce.captions import shot_text, wrap_text


def test_shot_text_prefers_caption():
    assert shot_text(Shot(index=1, caption="标题", voiceover="旁白")) == "标题"
    assert shot_text(Shot(index=1, voiceover="旁白")) == "旁白"
    assert shot_text(Shot(index=1)) == ""


def test_wrap_cjk_by_visual_width():
    # CJK char = 2 units; max_units=4 -> 2 chars per line
    assert wrap_text("一二三四五", max_units=4) == "一二\n三四\n五"


def test_wrap_does_not_split_ascii_word():
    # "Anthropic" must never break mid-word even on a tight budget
    out = wrap_text("说说Anthropic公司", max_units=4)
    assert "Anthropic" in out
    assert "Anthrop\n" not in out


def test_wrap_keeps_existing_newlines():
    assert wrap_text("a\nb", max_units=20) == "a\nb"


def test_wrap_ascii_words_space_separated():
    out = wrap_text("hello world foo", max_units=12)
    assert out.splitlines()[0] == "hello world"
