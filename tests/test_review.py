from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.review import apply_script_md, to_script_md


def _remix_sb() -> Storyboard:
    return Storyboard(
        project="p", format=VideoFormat.remix,
        shots=[
            Shot(index=1, source_start=5.0, source_end=12.0, voiceover="原始第一段"),
            Shot(index=2, source_start=30.0, source_end=38.0, voiceover="原始第二段"),
        ],
    )


def test_export_contains_narration_and_timestamps():
    md = to_script_md(_remix_sb())
    assert "## shot 01  [5-12]" in md
    assert "原始第一段" in md


def test_roundtrip_edit_narration():
    sb = _remix_sb()
    md = to_script_md(sb)
    md = md.replace("原始第一段", "改写后的第一段")
    out = apply_script_md(sb, md)
    assert out.shots[0].voiceover == "改写后的第一段"
    assert out.shots[1].voiceover == "原始第二段"


def test_roundtrip_edit_timestamp_and_clamp():
    sb = _remix_sb()
    md = "## shot 01  [3-9]\n第一段\n\n## shot 02  [30-999]\n第二段\n"
    out = apply_script_md(sb, md, source_max=80.0)
    assert out.shots[0].source_start == 3.0
    assert out.shots[0].source_end == 9.0
    assert out.shots[0].duration_sec == 6.0
    assert out.shots[1].source_end == 80.0  # clamped to source_max


def test_deleting_a_block_drops_the_shot():
    sb = _remix_sb()
    md = "## shot 02  [30-38]\n只保留第二段\n"
    out = apply_script_md(sb, md)
    assert len(out.shots) == 1
    assert out.shots[0].index == 2


def test_talking_head_has_no_timestamp_header():
    sb = Storyboard(project="p", format=VideoFormat.talking_head,
                    shots=[Shot(index=1, voiceover="一句口播", image_file="shot_01.png")])
    md = to_script_md(sb)
    assert "## shot 01" in md and "[" not in md.split("## shot 01")[1].splitlines()[0]
    # editing narration keeps the image_file (non-narration field) intact
    out = apply_script_md(sb, md.replace("一句口播", "改后的口播"))
    assert out.shots[0].voiceover == "改后的口播"
    assert out.shots[0].image_file == "shot_01.png"
