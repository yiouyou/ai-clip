import pytest

from ai_clip import pipeline
from ai_clip.core.config import Config
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.core.paths import ProjectPaths, read_model, write_model
from ai_clip.produce.review import ReviewValidationError, apply_script_md, to_script_md


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


def test_slideshow_roundtrip_edits_caption_and_narration():
    sb = Storyboard(
        project="p",
        format=VideoFormat.slideshow,
        target_duration_sec=10,
        shots=[
            Shot(
                index=1,
                duration_sec=5,
                caption="原始标题",
                voiceover="原始口播",
                image_file="shot_01.png",
            )
        ],
    )

    md = to_script_md(sb)
    assert "Caption: 原始标题" in md
    assert "Storyboard duration: 5s / target 10s" in md
    edited = md.replace("Caption: 原始标题", "Caption: 新标题").replace(
        "原始口播", "新口播"
    )
    out = apply_script_md(sb, edited)

    assert out.shots[0].caption == "新标题"
    assert out.shots[0].voiceover == "新口播"
    assert out.shots[0].image_file == "shot_01.png"


def test_slideshow_roundtrip_can_clear_caption():
    sb = Storyboard(
        project="p",
        format=VideoFormat.slideshow,
        shots=[Shot(index=1, caption="删除我", voiceover="保留口播")],
    )

    out = apply_script_md(sb, to_script_md(sb).replace("Caption: 删除我", "Caption:"))

    assert out.shots[0].caption == ""
    assert out.shots[0].voiceover == "保留口播"


def test_remix_apply_rejects_total_duration_over_target():
    sb = Storyboard(
        project="p",
        format=VideoFormat.remix,
        target_duration_sec=10,
        shots=[
            Shot(index=1, duration_sec=4, source_start=0, source_end=4),
            Shot(index=2, duration_sec=4, source_start=10, source_end=14),
        ],
    )
    edited = "## shot 01 [0-6]\n第一段\n\n## shot 02 [10-16]\n第二段\n"

    with pytest.raises(ReviewValidationError, match="12s exceeds target 10s"):
        apply_script_md(sb, edited)


def test_pipeline_rejected_review_does_not_overwrite_storyboard(tmp_path):
    cfg = Config(data_dir=str(tmp_path))
    paths = ProjectPaths(tmp_path, "p")
    paths.ensure()
    original = Storyboard(
        project="p",
        format=VideoFormat.remix,
        target_duration_sec=5,
        shots=[Shot(index=1, duration_sec=5, source_start=0, source_end=5)],
    )
    write_model(paths.storyboard_json, original)
    paths.script_md.write_text("## shot 01 [0-8]\n超预算\n", encoding="utf-8")

    with pytest.raises(ReviewValidationError):
        pipeline.run_review_apply(cfg, "p")

    assert read_model(paths.storyboard_json, Storyboard) == original
