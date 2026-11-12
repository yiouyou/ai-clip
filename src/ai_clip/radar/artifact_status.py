from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_is_stale, artifact_manifest_path
from ai_clip.radar.storage import RadarPaths


@dataclass(frozen=True)
class RadarArtifactStatus:
    name: str
    path: Path
    status: str
    manifest: Path


def radar_artifact_statuses(paths: RadarPaths) -> list[RadarArtifactStatus]:
    specs = [
        ("shortlist", paths.shortlist_json),
        ("candidates", paths.candidates_json),
        ("selection", paths.selection_json),
        ("evidence-research", paths.research_md),
        ("editorial-draft", paths.draft_md),
        ("review-artifact", paths.draft_review_json),
        ("rewrite-artifact", paths.draft_revised_md),
        ("verify-artifact", paths.draft_verification_json),
    ]
    return [
        RadarArtifactStatus(
            name=name,
            path=path,
            status=_status(path),
            manifest=artifact_manifest_path(path),
        )
        for name, path in specs
    ]


def _status(path: Path) -> str:
    if not path.exists():
        return "missing"
    return "stale" if artifact_manifest_is_stale(path) else "fresh"
