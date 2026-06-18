"""Assemble stage: turn collected assets into a single MP4.

For each shot we prefer assets/<video_file>; if absent we fall back to a still
assets/<image_file> shown for the shot duration. Every segment is normalized to
the same resolution/fps/codec (with a silent audio track) so the concat demuxer
can stitch them losslessly. Assets may originate from ComfyUI or a human — the
filename contract is all this stage knows about.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_clip.core.ffmpeg import ensure_ffmpeg, run
from ai_clip.core.models import Storyboard

_RESOLUTIONS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
_FPS = 30


class MissingAssetsError(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("missing assets: " + ", ".join(missing))


def check_assets(sb: Storyboard, assets_dir: Path) -> list[str]:
    """Return the list of shots that have neither a video nor an image asset."""
    missing: list[str] = []
    for shot in sb.shots:
        has_video = shot.video_file and (assets_dir / shot.video_file).exists()
        has_image = shot.image_file and (assets_dir / shot.image_file).exists()
        if not (has_video or has_image):
            missing.append(f"shot_{shot.index:02d} ({shot.video_file}|{shot.image_file})")
    return missing


def assemble(sb: Storyboard, assets_dir: Path, out_path: Path) -> Path:
    ensure_ffmpeg()
    missing = check_assets(sb, assets_dir)
    if missing:
        raise MissingAssetsError(missing)

    w, h = _RESOLUTIONS.get(sb.aspect_ratio, _RESOLUTIONS["9:16"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        segments = [
            _normalize_shot(shot, assets_dir, tmpdir, w, h)
            for shot in sb.shots
        ]
        concat_file = tmpdir / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in segments), encoding="utf-8"
        )
        run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(out_path),
        ])
    return out_path


def _normalize_shot(shot, assets_dir: Path, tmpdir: Path, w: int, h: int) -> Path:
    seg = tmpdir / f"seg_{shot.index:02d}.mp4"
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={_FPS},format=yuv420p"
    )
    video = assets_dir / shot.video_file if shot.video_file else None
    if video and video.exists():
        args = ["ffmpeg", "-y", "-i", str(video), "-t", str(shot.duration_sec)]
    else:
        image = assets_dir / shot.image_file
        args = ["ffmpeg", "-y", "-loop", "1", "-i", str(image), "-t", str(shot.duration_sec)]

    args += [
        "-f", "lavfi", "-t", str(shot.duration_sec), "-i", "anullsrc=r=44100:cl=stereo",
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(_FPS),
        "-c:a", "aac", "-shortest", str(seg),
    ]
    run(args)
    return seg
