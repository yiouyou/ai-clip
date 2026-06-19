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
from ai_clip.produce.captions import (
    drawtext_filter,
    prepare_font,
    shot_text,
    wrap_text,
)

_RESOLUTIONS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080)}
_FPS = 30
_BG_COLOR = "black"  # fallback background for narration-only shots (no b-roll)


class MissingAssetsError(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__("missing assets: " + ", ".join(missing))


def check_assets(sb: Storyboard, assets_dir: Path) -> list[str]:
    """Return shots that *expect* an asset but are missing it. A shot that expects
    no files (source segment, or a narration-only talking-head line) is fine — it
    renders from the source clip or a solid background respectively."""
    missing: list[str] = []
    for shot in sb.shots:
        expected = shot.expected_files()
        if not expected:
            continue
        if not any((assets_dir / f).exists() for f in expected):
            missing.append(f"shot_{shot.index:02d} ({'|'.join(expected)})")
    return missing


def assemble(
    sb: Storyboard,
    assets_dir: Path,
    out_path: Path,
    voice_dir: Path | None = None,
    source_video: Path | None = None,
    burn_captions: bool = False,
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
        font_name = prepare_font(tmpdir) if burn_captions else None
        segments = [
            _normalize_shot(
                shot, assets_dir, tmpdir, w, h, voice_dir, source_video, font_name
            )
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
    font_name: str | None = None,
) -> Path:
    seg = tmpdir / f"seg_{shot.index:02d}.mp4"
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={_FPS},format=yuv420p"
    )
    if font_name and shot_text(shot):
        text_name = f"cap_{shot.index:02d}.txt"
        (tmpdir / text_name).write_text(wrap_text(shot_text(shot)), encoding="utf-8")
        vf = f"{vf},{drawtext_filter(font_name, text_name, w)}"
    voice_path = (voice_dir / f"shot_{shot.index:02d}.wav") if voice_dir else None
    duration = _shot_duration(shot, voice_path)

    video = assets_dir / shot.video_file if shot.video_file else None
    image = assets_dir / shot.image_file if shot.image_file else None
    if shot.is_source_segment:
        # Remix: cut the span out of the source clip.
        args = [
            "ffmpeg", "-y", "-ss", str(shot.source_start), "-to", str(shot.source_end),
            "-i", str(source_video),
        ]
    elif video and video.exists():
        args = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(video)]
    elif image and image.exists():
        args = ["ffmpeg", "-y", "-loop", "1", "-i", str(image)]
    else:
        # Narration-only shot (e.g. talking-head line with no b-roll): solid bg.
        args = ["ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c={_BG_COLOR}:s={w}x{h}:r={_FPS}"]

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
    # cwd=tmpdir so drawtext can reference the font/text by colon-free relative name.
    run(args, cwd=tmpdir)
    return seg
