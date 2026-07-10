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
        ("source_research", paths.research_md),
        ("zack_draft", paths.draft_md),
        ("pair_review", paths.reviews_dir / f"{paths.date}_zack_draft_review.json"),
        ("pair_rewrite", paths.draft_revised_md),
        ("pair_verify", paths.reviews_dir / f"{paths.date}_zack_draft_verify.json"),
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
