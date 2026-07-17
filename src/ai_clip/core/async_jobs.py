"""Durable state for external asynchronous generation jobs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ai_clip.core.artifacts import read_model, write_model

AsyncJobStatus = Literal[
    "submitting",
    "submitted",
    "running",
    "succeeded",
    "failed",
    "unknown",
]

ACTIVE_JOB_STATUSES = frozenset({"submitting", "submitted", "running", "unknown"})


class AsyncJobState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    provider: str
    request_hash: str
    output_path: str
    remote_id: str = ""
    status: AsyncJobStatus
    created_at: str
    updated_at: str
    error: str = ""


def async_job_state_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.job.json")


def async_job_request_hash(provider: str, base_url: str, payload: dict) -> str:
    canonical = json.dumps(
        {
            "provider": provider,
            "base_url": base_url.rstrip("/"),
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def new_async_job_state(
    *,
    provider: str,
    request_hash: str,
    output_path: Path,
    status: AsyncJobStatus,
    remote_id: str = "",
) -> AsyncJobState:
    now = datetime.now(timezone.utc).isoformat()
    return AsyncJobState(
        provider=provider,
        request_hash=request_hash,
        output_path=str(output_path),
        remote_id=remote_id,
        status=status,
        created_at=now,
        updated_at=now,
    )


def read_async_job_state(output_path: Path) -> AsyncJobState | None:
    path = async_job_state_path(output_path)
    if not path.exists():
        return None
    return read_model(path, AsyncJobState)


def write_async_job_state(output_path: Path, state: AsyncJobState) -> None:
    write_model(async_job_state_path(output_path), state)


def transition_async_job(
    output_path: Path,
    state: AsyncJobState,
    status: AsyncJobStatus,
    *,
    remote_id: str | None = None,
    error: str = "",
) -> AsyncJobState:
    updated = state.model_copy(
        update={
            "status": status,
            "remote_id": state.remote_id if remote_id is None else remote_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
    )
    write_async_job_state(output_path, updated)
    return updated
