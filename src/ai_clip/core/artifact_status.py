from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_clip.core.artifacts import artifact_is_stale, artifact_manifest_path
from ai_clip.core.paths import ProjectPaths


@dataclass(frozen=True)
class ProjectArtifactStatus:
    name: str
    path: Path
    status: str
    manifest: Path


def project_artifact_statuses(paths: ProjectPaths) -> list[ProjectArtifactStatus]:
    specs = [
        ("research", paths.research_md, _existing(paths.transcript_json, paths.analysis_json)),
        (
            "storyboard",
            paths.storyboard_json,
            _existing(paths.analysis_json, paths.transcript_json, paths.research_md),
        ),
        (
            "source_draft",
            paths.source_draft_md,
            _existing(paths.transcript_json, paths.analysis_json, paths.research_md),
        ),
    ]
    return [
        ProjectArtifactStatus(
            name=name,
            path=path,
            status=_status(path, inputs),
            manifest=artifact_manifest_path(path),
        )
        for name, path, inputs in specs
    ]


def _status(path: Path, inputs: list[Path]) -> str:
    if not path.exists():
        return "missing"
    return "stale" if artifact_is_stale(path, inputs) else "fresh"


def _existing(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]
