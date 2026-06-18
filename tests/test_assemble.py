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
