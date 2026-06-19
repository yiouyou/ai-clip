"""Locate a CJK-capable font for burning captions, and escape paths for ffmpeg."""

from __future__ import annotations

import functools
from pathlib import Path

_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simsun.ttc",
    # Linux (Noto / WenQuanYi)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
]


@functools.lru_cache(maxsize=1)
def find_cjk_font() -> str | None:
    for c in _CANDIDATES:
        if Path(c).exists():
            return c
    return None


def escape_for_filter(path: str) -> str:
    """Escape a path for use inside an ffmpeg filtergraph (Windows drive colon)."""
    return path.replace("\\", "/").replace(":", "\\:")
