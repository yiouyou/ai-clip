from ai_clip.core.models import Shot
from ai_clip.produce.captions import shot_text, wrap_text


def test_shot_text_prefers_caption():
    assert shot_text(Shot(index=1, caption="标题", voiceover="旁白")) == "标题"
    assert shot_text(Shot(index=1, voiceover="旁白")) == "旁白"
    assert shot_text(Shot(index=1)) == ""


def test_wrap_text_by_width():
    assert wrap_text("一二三四五", width=2) == "一二\n三四\n五"


def test_wrap_text_keeps_existing_newlines():
    assert wrap_text("a\nb", width=10) == "a\nb"
