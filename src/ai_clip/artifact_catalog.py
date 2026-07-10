from __future__ import annotations

from ai_clip.core.artifacts import ArtifactRef
from ai_clip.core.config import Config
from ai_clip.core.paths import ProjectPaths
from ai_clip.radar.storage import RadarPaths

def resolve_review_artifact(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None = None,
) -> ArtifactRef:
    """Resolve workflow-specific storage into a neutral review artifact reference."""
    name = artifact
    if name == "zack_draft":
        if not run_date:
            raise ValueError("zack_draft review requires --date")
        paths = RadarPaths(cfg.data_dir, run_date)
        return ArtifactRef(
            name=name,
            source=paths.draft_md,
            review=paths.reviews_dir / f"{run_date}_zack_draft_review.json",
            revised=paths.draft_revised_md,
            verification=paths.reviews_dir / f"{run_date}_zack_draft_verify.json",
            billing_root=paths.root,
        )

    paths = ProjectPaths(cfg.data_dir, project)
    paths.ensure()
    sources = {
        "analysis": paths.analysis_json,
        "research": paths.research_md,
        "script": paths.script_md,
        "source_draft": paths.source_draft_md,
        "storyboard": paths.storyboard_md,
    }
    revised = {
        "research": paths.root / "research.revised.md",
        "script": paths.root / "script.revised.md",
        "source_draft": paths.source_draft_revised_md,
    }
    if name not in sources:
        allowed = ", ".join(sorted([*sources, "zack_draft"]))
        raise ValueError(f"unknown artifact {name!r}; expected one of: {allowed}")
    return ArtifactRef(
        name=name,
        source=sources[name],
        review=paths.reviews_dir / f"{name}_review.json",
        revised=revised.get(name),
        verification=paths.reviews_dir / f"{name}_verify.json",
        billing_root=paths.root,
    )
