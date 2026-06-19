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

# Visual-width budget per line, in "units" where a CJK char = 2 and ASCII = 1.
# ~34 units ≈ 17 CJK chars, a comfortable single line on a 1080-wide 9:16 frame.
_WRAP_UNITS = 34
_FONT_NAME = "capfont.ttf"

_ASCII_WORD = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'’-._/"
)


def shot_text(shot: Shot) -> str:
    return (shot.caption or shot.voiceover or "").strip()


def _char_units(ch: str) -> int:
    return 1 if ch in _ASCII_WORD or ch == " " else 2


def _tokenize(text: str) -> list[str]:
    """Split into tokens: each ASCII word stays whole; each CJK char is its own
    token. This stops English words being broken mid-word across lines."""
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if ch in _ASCII_WORD:
            buf += ch
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            if ch != " ":  # spaces are implicit separators between tokens
                tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


def _token_units(tok: str) -> int:
    return sum(_char_units(c) for c in tok)


def wrap_text(text: str, max_units: int = _WRAP_UNITS) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line, used = "", 0
        for tok in _tokenize(paragraph):
            tu = _token_units(tok)
            is_ascii = tok[0] in _ASCII_WORD
            sep = " " if (line and is_ascii and line[-1] not in " ") else ""
            if used + len(sep) + tu > max_units and line:
                lines.append(line)
                line, used = "", 0
                sep = ""
            line += sep + tok
            used += len(sep) + tu
        if line:
            lines.append(line)
    return "\n".join(lines)


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
    fontsize = max(28, frame_w // 30)  # ~36px on a 1080-wide frame
    return (
        f"drawtext=fontfile={font_name}:textfile={text_name}:"
        f"fontcolor=white:fontsize={fontsize}:line_spacing=6:"
        f"borderw=4:bordercolor=black@0.9:"
        f"box=1:boxcolor=black@0.45:boxborderw=10:"
        f"x=(w-text_w)/2:y=h*0.80-text_h/2"
    )
