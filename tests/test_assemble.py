"""Integration test for the assemble stage against a real ffmpeg."""

import shutil
from pathlib import Path

import pytest

from ai_clip.core.ffmpeg import probe_duration, run
from ai_clip.core.models import Shot, Storyboard
from ai_clip.produce.assemble import MissingAssetsError, assemble, check_assets

ffmpeg_available = shutil.which("ffmpeg") is not None
pytestmark = pytest.mark.skipif(not ffmpeg_available, reason="ffmpeg not on PATH")


def _make_png(path: Path, color: str):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x568", "-frames:v", "1", str(path)])


def _storyboard() -> Storyboard:
    return Storyboard(
        project="t",
        aspect_ratio="9:16",
        shots=[
            Shot(index=1, duration_sec=1.0, image_file="shot_01.png", video_file="shot_01.mp4"),
            Shot(index=2, duration_sec=1.0, image_file="shot_02.png", video_file="shot_02.mp4"),
        ],
    )


def test_check_assets_reports_missing(tmp_path: Path):
    sb = _storyboard()
    assets = tmp_path / "assets"
    assets.mkdir()
    assert len(check_assets(sb, assets)) == 2  # nothing present


def test_assemble_raises_on_missing(tmp_path: Path):
    sb = _storyboard()
    assets = tmp_path / "assets"
    assets.mkdir()
    with pytest.raises(MissingAssetsError):
        assemble(sb, assets, tmp_path / "out.mp4")


def test_assemble_from_images(tmp_path: Path):
    sb = _storyboard()
    assets = tmp_path / "assets"
    assets.mkdir()
    _make_png(assets / "shot_01.png", "red")
    _make_png(assets / "shot_02.png", "blue")

    assert check_assets(sb, assets) == []
    out = assemble(sb, assets, tmp_path / "out.mp4")
    assert out.exists()
    # two 1s shots -> ~2s output
    assert probe_duration(out) >= 1.8


def _make_source_video(path: Path, seconds: float):
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x568:duration={seconds}:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ])


def test_assemble_remix_cuts_source(tmp_path: Path):
    source = tmp_path / "source.mp4"
    _make_source_video(source, 10.0)
    sb = Storyboard(
        project="t",
        format="remix",
        aspect_ratio="9:16",
        shots=[
            Shot(index=1, duration_sec=2.0, source_start=1.0, source_end=3.0),
            Shot(index=2, duration_sec=2.0, source_start=5.0, source_end=7.0),
        ],
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    assert check_assets(sb, assets) == []  # source shots need no assets
    out = assemble(sb, assets, tmp_path / "out.mp4", source_video=source)
    assert out.exists()
    assert probe_duration(out) >= 3.6  # ~2s + 2s


def test_assemble_remix_requires_source(tmp_path: Path):
    sb = Storyboard(
        project="t", format="remix",
        shots=[Shot(index=1, source_start=0.0, source_end=2.0)],
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    with pytest.raises(MissingAssetsError):
        assemble(sb, assets, tmp_path / "out.mp4", source_video=None)


def _make_wav(path: Path, seconds: float):
    run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
        "-ar", "44100", str(path),
    ])


def test_assemble_with_voiceover_extends_duration(tmp_path: Path):
    """A 3s narration on a 1s shot should stretch the shot to fit the audio."""
    sb = Storyboard(
        project="t",
        aspect_ratio="9:16",
        shots=[Shot(index=1, duration_sec=1.0, image_file="shot_01.png", video_file="shot_01.mp4")],
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    _make_png(assets / "shot_01.png", "green")

    voice = tmp_path / "voice"
    voice.mkdir()
    _make_wav(voice / "shot_01.wav", 3.0)

    out = assemble(sb, assets, tmp_path / "out.mp4", voice_dir=voice)
    assert out.exists()
    assert probe_duration(out) >= 2.8  # driven by the 3s narration, not the 1s shot


def _make_mp4(path: Path, seconds: float, color: str):
    run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x568:d={seconds}:r=30",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ])


def test_check_assets_ignores_narration_only_and_source():
    sb = Storyboard(
        project="t",
        shots=[
            Shot(index=1, voiceover="just narration"),  # no asset expected -> ok
            Shot(index=2, source_start=0.0, source_end=2.0),  # source -> ok
            Shot(index=3, image_file="shot_03.png"),  # expects asset
        ],
    )
    missing = check_assets(sb, Path("/nonexistent"))
    assert missing == ["shot_03 (shot_03.png)"]


def test_assemble_mixed_media(tmp_path: Path):
    """One timeline mixing: still image, b-roll video, source-segment (remix),
    and a narration-only shot — all carrying cloned-style voiceover."""
    source = tmp_path / "source.mp4"
    _make_mp4(source, 6.0, "white")

    sb = Storyboard(
        project="t",
        aspect_ratio="9:16",
        shots=[
            Shot(index=1, duration_sec=1.0, image_file="shot_01.png"),     # still
            Shot(index=2, duration_sec=1.0, video_file="shot_02.mp4"),     # b-roll video
            Shot(index=3, source_start=1.0, source_end=3.0),               # remix span
            Shot(index=4, duration_sec=1.0, voiceover="narration only"),   # solid bg
        ],
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    _make_png(assets / "shot_01.png", "red")
    _make_mp4(assets / "shot_02.mp4", 2.0, "blue")

    voice = tmp_path / "voice"
    voice.mkdir()
    for i in (1, 2, 4):
        _make_wav(voice / f"shot_{i:02d}.wav", 1.0)

    assert check_assets(sb, assets) == []
    out = assemble(sb, assets, tmp_path / "out.mp4", voice_dir=voice, source_video=source)
    assert out.exists()
    # 1 + 1 + 2(span) + 1 = ~5s
    assert probe_duration(out) >= 4.5
