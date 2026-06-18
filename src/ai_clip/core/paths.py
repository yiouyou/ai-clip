"""Per-project directory layout and JSON artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ProjectPaths:
    """All artifacts for one project live under data_dir/<project>/."""

    def __init__(self, data_dir: str | Path, project: str) -> None:
        self.root = Path(data_dir) / project
        self.project = project

    @property
    def clip_json(self) -> Path:
        return self.root / "clip.json"

    @property
    def transcript_json(self) -> Path:
        return self.root / "transcript.json"

    @property
    def analysis_json(self) -> Path:
        return self.root / "analysis.json"

    @property
    def storyboard_json(self) -> Path:
        return self.root / "storyboard.json"

    @property
    def storyboard_md(self) -> Path:
        return self.root / "storyboard.md"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def output_mp4(self) -> Path:
        return self.root / "output.mp4"

    def ensure(self) -> None:
        for d in (self.root, self.prompts_dir, self.assets_dir):
            d.mkdir(parents=True, exist_ok=True)


def write_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def read_model[T: BaseModel](path: Path, model_cls: type[T]) -> T:
    data = json.loads(path.read_text(encoding="utf-8"))
    return model_cls.model_validate(data)
