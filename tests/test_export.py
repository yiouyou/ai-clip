from pathlib import Path

from ai_clip.core.models import Transcript, TranscriptSegment
from ai_clip.extract.export import _fmt_ts, to_srt, to_txt, write_srt, write_txt


def _transcript() -> Transcript:
    return Transcript(
        clip_id="c",
        text="一二 三四",
        segments=[
            TranscriptSegment(start=0.0, end=1.5, text="一二"),
            TranscriptSegment(start=1.5, end=3.25, text="三四"),
        ],
    )


def test_fmt_ts():
    assert _fmt_ts(0) == "00:00:00,000"
    assert _fmt_ts(1.5) == "00:00:01,500"
    assert _fmt_ts(3661.123) == "01:01:01,123"
    assert _fmt_ts(-5) == "00:00:00,000"


def test_to_srt():
    srt = to_srt(_transcript())
    assert "1\n00:00:00,000 --> 00:00:01,500\n一二" in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\n三四" in srt


def test_to_txt_uses_segments():
    assert to_txt(_transcript()) == "一二\n三四"


def test_to_txt_fallback_to_text():
    t = Transcript(clip_id="c", text="整段文字")
    assert to_txt(t) == "整段文字"


def test_write_files(tmp_path: Path):
    t = _transcript()
    srt = write_srt(t, tmp_path / "a.srt")
    txt = write_txt(t, tmp_path / "a.txt")
    assert srt.read_text(encoding="utf-8").startswith("1\n")
    assert txt.read_text(encoding="utf-8") == "一二\n三四"
