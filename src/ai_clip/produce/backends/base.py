"""Optional external produce backends (theme -> finished mp4).

These are alternatives to the self-built storyboard→voiceover→assemble path,
wrapping mature projects via their HTTP API. They take a theme and return a
finished video, so the comparison is apples-to-apples with `ai-clip original`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ProduceSpec:
    theme: str
    out_path: Path
    aspect_ratio: str = "9:16"
    voice_name: str = ""
    language: str = ""
    subtitle: bool = True
    paragraphs: int = 1


class ProduceBackend(Protocol):
    name: str

    def produce(self, spec: ProduceSpec) -> Path:
        """Generate a video for spec.theme and write it to spec.out_path."""
        ...
