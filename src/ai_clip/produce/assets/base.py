"""Asset provider protocol. An image provider turns a Shot's prompt into a file
at assets/<shot.image_file>. prompt_only is a no-op (a human fills assets/);
comfyui generates via local models. Both honor the same filename contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ai_clip.core.models import Shot


class ImageProvider(Protocol):
    name: str

    def cache_params(self) -> dict[str, str]: ...

    def generate(self, shot: Shot, assets_dir: Path) -> Path | None:
        """Produce assets_dir/<shot.image_file>, or return None to defer to a
        human (prompt_only mode)."""
        ...
