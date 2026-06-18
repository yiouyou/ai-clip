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

from ai_clip.core.ffmpeg import ensure_ffmpeg, probe_duration, run
from ai_clip.core.models import Shot, Storyboard

_RESOLUTIONS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
_FPS = 30


class MissingAssetsError(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("missing assets: " + ", ".join(missing))


def check_assets(sb: Storyboard, assets_dir: Path) -> list[str]:
    """Return the list of shots that have no usable input. Source-segment (remix)
    shots are satisfied by the source clip, not files in assets/."""
    missing: list[str] = []
    for shot in sb.shots:
        if shot.is_source_segment:
            continue
        has_video = shot.video_file and (assets_dir / shot.video_file).exists()
        has_image = shot.image_file and (assets_dir / shot.image_file).exists()
        if not (has_video or has_image):
            missing.append(f"shot_{shot.index:02d} ({shot.video_file}|{shot.image_file})")
    return missing


def assemble(
    sb: Storyboard,
    assets_dir: Path,
    out_path: Path,
    voice_dir: Path | None = None,
    source_video: Path | None = None,
) -> Path:
    ensure_ffmpeg()
    missing = check_assets(sb, assets_dir)
    if missing:
        raise MissingAssetsError(missing)
    if any(s.is_source_segment for s in sb.shots) and not (
        source_video and Path(source_video).exists()
    ):
        raise MissingAssetsError(["source video required for remix shots"])

    w, h = _RESOLUTIONS.get(sb.aspect_ratio, _RESOLUTIONS["9:16"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        segments = [
            _normalize_shot(shot, assets_dir, tmpdir, w, h, voice_dir, source_video)
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


def _shot_duration(shot: Shot, voice_path: Path | None) -> float:
    """A shot lasts at least its configured length, extended to fit narration."""
    if voice_path and voice_path.exists():
        return max(shot.duration_sec, probe_duration(voice_path))
    return shot.duration_sec


def _normalize_shot(
    shot: Shot,
    assets_dir: Path,
    tmpdir: Path,
    w: int,
    h: int,
    voice_dir: Path | None,
    source_video: Path | None = None,
) -> Path:
    seg = tmpdir / f"seg_{shot.index:02d}.mp4"
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={_FPS},format=yuv420p"
    )
    voice_path = (voice_dir / f"shot_{shot.index:02d}.wav") if voice_dir else None
    duration = _shot_duration(shot, voice_path)

    if shot.is_source_segment:
        # Remix: cut the span out of the source clip.
        args = [
            "ffmpeg", "-y", "-ss", str(shot.source_start), "-to", str(shot.source_end),
            "-i", str(source_video),
        ]
    else:
        video = assets_dir / shot.video_file if shot.video_file else None
        if video and video.exists():
            args = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(video)]
        else:
            image = assets_dir / shot.image_file
            args = ["ffmpeg", "-y", "-loop", "1", "-i", str(image)]

    if voice_path and voice_path.exists():
        # Narration track; pad with silence so the shot reaches `duration`.
        args += ["-i", str(voice_path), "-map", "0:v:0", "-map", "1:a:0", "-af", "apad"]
    elif shot.is_source_segment:
        # Keep the source clip's own audio.
        args += ["-map", "0:v:0", "-map", "0:a:0?"]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-map", "0:v:0", "-map", "1:a:0"]

    args += [
        "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(_FPS),
        "-c:a", "aac", "-ar", "44100",
        str(seg),
    ]
    run(args)
    return seg
