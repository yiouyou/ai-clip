from __future__ import annotations

from ai_clip.core.run_status import (
    RunStatusStore,
    WorkflowStageTracker,
    mark_stages_stale,
    stage_is_stale,
    update_run_usage,
)
from ai_clip.radar.models import RadarRunStatus
from ai_clip.radar.storage import RadarPaths


def radar_status_store(paths: RadarPaths) -> RunStatusStore[RadarRunStatus]:
    return RunStatusStore(
        paths.run_status_json,
        RadarRunStatus,
        lambda: RadarRunStatus(date=paths.date),
    )


def track_stage(
    paths: RadarPaths,
    name: str,
    inputs: dict[str, str] | None = None,
) -> WorkflowStageTracker:
    return WorkflowStageTracker(radar_status_store(paths), name, inputs)


def begin_run(paths: RadarPaths) -> RadarRunStatus:
    return radar_status_store(paths).begin()


def mark_stale(
    paths: RadarPaths,
    stage_names: list[str],
    reason: str,
    outputs: dict[str, str] | None = None,
) -> None:
    mark_stages_stale(
        radar_status_store(paths),
        stage_names,
        reason,
        outputs,
    )


def is_stage_stale(paths: RadarPaths, stage_name: str) -> bool:
    return stage_is_stale(radar_status_store(paths), stage_name)


def read_status(paths: RadarPaths) -> RadarRunStatus:
    return radar_status_store(paths).read()


def finalize_run_usage(paths: RadarPaths) -> RadarRunStatus:
    return update_run_usage(radar_status_store(paths), paths.root)
