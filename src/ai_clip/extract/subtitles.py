"""Use a video's existing (or auto-generated) subtitles instead of running
whisper — far faster and usually accurate enough for long videos.

Downloads subtitles via yt-dlp as WebVTT, then parses cues into segments.
"""

from __future__ import annotations

import re
from pathlib import Path

from ai_clip.core.models import TranscriptSegment

_PREFERRED_LANGS = ["en", "en-US", "en-GB", "zh", "zh-Hans", "zh-CN", "zh-Hant"]
_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
_TAG = re.compile(r"<[^>]+>")


def _ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    seen: set[tuple[float, float, str]] = set()
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        cue_line = next((ln for ln in lines if "-->" in ln), None)
        if not cue_line:
            continue
        stamps = _TS.findall(cue_line)
        if len(stamps) < 2:
            continue
        start = _ts_to_sec(*stamps[0])
        end = _ts_to_sec(*stamps[1])
        body_lines = lines[lines.index(cue_line) + 1:]
        body = _TAG.sub("", " ".join(body_lines)).strip()
        if not body or end <= start:
            continue
        key = (round(start, 2), round(end, 2), body)
        if key in seen:  # auto-captions repeat rolling lines
            continue
        seen.add(key)
        segments.append(TranscriptSegment(start=start, end=end, text=body))
    return segments


def fetch_subtitle_segments(
    url: str, out_dir: str | Path, langs: list[str] | None = None
) -> tuple[list[TranscriptSegment], str] | None:
    """Download subtitles for `url` and return (segments, language), or None if
    no usable subtitles exist."""
    import yt_dlp  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    langs = langs or _PREFERRED_LANGS
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": str(out / "subs.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)

    for lang in langs:
        for cand in (out.glob(f"subs.{lang}.vtt")):
            segs = parse_vtt(cand.read_text(encoding="utf-8", errors="replace"))
            if segs:
                return segs, lang
    # fall back to any downloaded vtt
    for cand in sorted(out.glob("subs.*.vtt")):
        segs = parse_vtt(cand.read_text(encoding="utf-8", errors="replace"))
        if segs:
            lang = cand.name.split(".")[1] if "." in cand.name else ""
            return segs, lang
    return None
