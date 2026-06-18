"""Format-aware produce: each viral archetype has its own storyboard generator.

All generators share the signature in base.GenerateArgs and return a Storyboard,
so the storyboard step / CLI / agent treat them interchangeably.
"""

from __future__ import annotations

from ai_clip.core.models import VideoFormat
from ai_clip.produce.formats import montage, remix, slideshow, talking_head

_GENERATORS = {
    VideoFormat.talking_head: talking_head.generate,
    VideoFormat.slideshow: slideshow.generate,
    VideoFormat.remix: remix.generate,
    VideoFormat.montage: montage.generate,
}


def get_generator(fmt: VideoFormat):
    if fmt not in _GENERATORS:
        raise ValueError(f"unknown format: {fmt}")
    return _GENERATORS[fmt]


__all__ = ["get_generator"]
