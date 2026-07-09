"""Per-project directory layout and JSON artifact helpers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ai_clip.core.artifacts import (
    ArtifactStore,
    read_model as _read_model,
    write_model as _write_model,
)


class ProjectPaths:
    """All artifacts for one project live under data_dir/<project>/."""

    def __init__(self, data_dir: str | Path, project: str) -> None:
        self.root = Path(data_dir) / project
        self.project = project

    @property
    def store(self) -> ArtifactStore:
        return ArtifactStore(self.root)

    @property
    def candidates_json(self) -> Path:
        return self.root / "candidates.json"

    @property
    def clip_json(self) -> Path:
        return self.root / "clip.json"

    @property
    def transcript_json(self) -> Path:
        return self.root / "transcript.json"

    @property
    def transcript_srt(self) -> Path:
        return self.root / "transcript.srt"

    @property
    def transcript_txt(self) -> Path:
        return self.root / "transcript.txt"

    @property
    def analysis_json(self) -> Path:
        return self.root / "analysis.json"

    @property
    def research_json(self) -> Path:
        return self.root / "research.json"

    @property
    def research_md(self) -> Path:
        return self.root / "research.md"

    @property
    def storyboard_json(self) -> Path:
        return self.root / "storyboard.json"

    @property
    def storyboard_md(self) -> Path:
        return self.root / "storyboard.md"

    @property
    def script_md(self) -> Path:
        return self.root / "script.md"

    @property
    def source_draft_md(self) -> Path:
        return self.root / "source_draft.md"

    @property
    def source_draft_revised_md(self) -> Path:
        return self.root / "source_draft.revised.md"

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def reviews_dir(self) -> Path:
        return self.root / "reviews"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def run_status_json(self, workflow: str) -> Path:
        safe = workflow.replace("/", "-").replace("\\", "-")
        return self.runs_dir / f"{safe}.json"

    @property
    def voice_dir(self) -> Path:
        return self.root / "voice"

    @property
    def reference_audio(self) -> Path:
        return self.root / "voice_reference.mp3"

    @property
    def output_mp4(self) -> Path:
        return self.root / "output.mp4"

    def ensure(self) -> None:
        for d in (
            self.root,
            self.prompts_dir,
            self.assets_dir,
            self.voice_dir,
            self.reviews_dir,
            self.runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def write_model(path: Path, model: BaseModel) -> None:
    _write_model(path, model)


def read_model[T: BaseModel](path: Path, model_cls: type[T]) -> T:
    return _read_model(path, model_cls)
