from __future__ import annotations

from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_path
from ai_clip.core.run_status import WorkflowRunStatus


def build_run_view(status: WorkflowRunStatus, status_path: Path) -> dict[str, object]:
    artifacts = []
    for stage in status.stages:
        for name, raw_path in stage.outputs.items():
            path = Path(raw_path)
            manifest = artifact_manifest_path(path)
            artifacts.append({
                "stage": stage.name,
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "manifest": str(manifest) if manifest.exists() else "",
            })
    return {
        "workflow": status.workflow,
        "project": status.project,
        "date": status.date,
        "run_id": status.run_id,
        "attempt": status.attempt,
        "status": status.status,
        "started_at": status.started_at,
        "updated_at": status.updated_at,
        "run_status_path": str(status_path),
        "history_dir": str(status_path.parent / "history" / status_path.stem),
        "stages": [stage.model_dump(mode="json") for stage in status.stages],
        "artifacts": artifacts,
        "usage": status.usage,
    }
