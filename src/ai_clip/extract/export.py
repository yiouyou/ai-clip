"""Export a Transcript to .srt subtitles or plain .txt."""

from __future__ import annotations

from pathlib import Path

from ai_clip.core.models import Transcript


def _fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for i, seg in enumerate(transcript.segments, start=1):
        blocks.append(
            f"{i}\n{_fmt_ts(seg.start)} --> {_fmt_ts(seg.end)}\n{seg.text}\n"
        )
    return "\n".join(blocks)


def to_txt(transcript: Transcript) -> str:
    if transcript.segments:
        return "\n".join(seg.text for seg in transcript.segments)
    return transcript.text


def write_srt(transcript: Transcript, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_srt(transcript), encoding="utf-8")
    return p


def write_txt(transcript: Transcript, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_txt(transcript), encoding="utf-8")
    return p
