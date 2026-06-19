"""On-screen caption text for a shot + an ffmpeg drawtext filter.

The text shown is the shot's caption (slideshow) if present, otherwise its
voiceover line. Long lines are wrapped by character count to fit a vertical
frame; CJK has no spaces so we wrap purely on width.

To dodge the notorious Windows drive-colon escaping problem inside ffmpeg
filtergraphs, the caller copies the font and writes the text file into the same
working directory and runs ffmpeg with cwd set there, so we reference both by
plain (colon-free) relative names.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ai_clip.core.fonts import find_cjk_font
from ai_clip.core.models import Shot

_WRAP_CHARS = 16  # roughly fits a 1080-wide 9:16 frame at the chosen font size
_FONT_NAME = "capfont.ttf"


def shot_text(shot: Shot) -> str:
    return (shot.caption or shot.voiceover or "").strip()


def wrap_text(text: str, width: int = _WRAP_CHARS) -> str:
    out, line = [], ""
    for ch in text:
        if ch == "\n":
            out.append(line)
            line = ""
            continue
        line += ch
        if len(line) >= width:
            out.append(line)
            line = ""
    if line:
        out.append(line)
    return "\n".join(out)


def prepare_font(workdir: Path) -> str | None:
    """Copy the CJK font into workdir under a colon-free name. Returns the name."""
    font = find_cjk_font()
    if not font:
        return None
    dest = workdir / _FONT_NAME
    if not dest.exists():
        shutil.copyfile(font, dest)
    return _FONT_NAME


def drawtext_filter(font_name: str, text_name: str, frame_w: int) -> str:
    """drawtext using colon-free relative names (resolved against ffmpeg cwd)."""
    fontsize = max(36, frame_w // 22)
    return (
        f"drawtext=fontfile={font_name}:textfile={text_name}:"
        f"fontcolor=white:fontsize={fontsize}:line_spacing=8:"
        f"box=1:boxcolor=black@0.5:boxborderw=16:"
        f"x=(w-text_w)/2:y=h-text_h-h/8"
    )
