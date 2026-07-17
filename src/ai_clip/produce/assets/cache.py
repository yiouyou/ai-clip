from __future__ import annotations

from pathlib import Path

from ai_clip.core.artifacts import (
    artifact_manifest_path,
    artifact_matches,
    read_artifact_manifest,
    write_artifact_manifest,
)
from ai_clip.core.async_jobs import async_job_state_path
from ai_clip.core.models import Shot
from ai_clip.produce.assets.base import ImageProvider


def asset_params(shot: Shot, provider: ImageProvider) -> dict[str, str]:
    return {
        "prompt": shot.image_prompt,
        "engine": shot.asset_engine.value if shot.asset_engine else "",
        **provider.cache_params(),
    }


def asset_needs_generation(path: Path, params: dict[str, str]) -> bool:
    if not path.exists():
        return True
    if not artifact_manifest_path(path).exists():
        return async_job_state_path(path).exists()
    return not artifact_matches(path, params=params)


def record_generated_asset(path: Path, params: dict[str, str]) -> None:
    write_artifact_manifest(path, stage="assets", params=params)


def remove_orphaned_generated_assets(assets_dir: Path, expected: set[str]) -> list[Path]:
    removed = []
    for path in assets_dir.glob("shot_*.*"):
        if path.name.endswith(".meta.json") or path.name in expected:
            continue
        manifest_path = artifact_manifest_path(path)
        if not manifest_path.exists():
            continue
        try:
            manifest = read_artifact_manifest(path)
        except (OSError, ValueError):
            continue
        if manifest.stage != "assets":
            continue
        path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        async_job_state_path(path).unlink(missing_ok=True)
        removed.append(path)
    return removed
