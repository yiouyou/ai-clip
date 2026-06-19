from ai_clip.extract.subtitles import parse_vtt

_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.500
Hello and welcome

00:00:03.500 --> 00:00:06.000
<c>to the show</c>

00:00:06.000 --> 00:00:06.000
empty-duration dropped

00:00:07.000 --> 00:00:09.000
Hello and welcome
"""


def test_parse_vtt_timestamps_and_text():
    segs = parse_vtt(_VTT)
    assert segs[0].start == 1.0
    assert segs[0].end == 3.5
    assert segs[0].text == "Hello and welcome"


def test_parse_vtt_strips_tags():
    segs = parse_vtt(_VTT)
    assert segs[1].text == "to the show"


def test_parse_vtt_drops_zero_duration():
    texts = [s.text for s in parse_vtt(_VTT)]
    assert "empty-duration dropped" not in texts


def test_parse_vtt_dedupes_rolling_lines():
    # the repeated "Hello and welcome" at a different time is kept (different key),
    # but exact dupes (same start/end/text) are collapsed.
    segs = parse_vtt(_VTT)
    assert sum(s.text == "Hello and welcome" for s in segs) == 2
