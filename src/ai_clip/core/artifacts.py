"""Shared artifact read/write primitives.

This module intentionally stays small. Project and radar workflows own their
directory layout; ArtifactStore only centralizes the mechanics of resolving
paths and writing durable text/JSON/model artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ArtifactRef:
    """Resolved paths for a reviewable artifact, independent of workflow layout."""

    name: str
    source: Path
    review: Path
    billing_root: Path
    revised: Path | None = None
    verification: Path | None = None


class ArtifactInput(BaseModel):
    path: str
    exists: bool
    mtime_ns: int = 0
    size: int = 0


class ArtifactManifest(BaseModel):
    artifact: str
    stage: str
    created_at: str
    inputs: dict[str, ArtifactInput] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    model: str = ""
    config_hash: str = ""


def write_text_atomic(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding=encoding)
    tmp.replace(path)


def write_json_atomic(path: Path, data: Any, indent: int = 2) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model(path: Path, model: BaseModel) -> None:
    write_text_atomic(path, model.model_dump_json(indent=2), encoding="utf-8")


def read_model[T: BaseModel](path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate(read_json(path))


def artifact_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.meta.json")


def snapshot_input(path: Path) -> ArtifactInput:
    if not path.exists():
        return ArtifactInput(path=str(path), exists=False)
    stat = path.stat()
    return ArtifactInput(
        path=str(path),
        exists=True,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )


def snapshot_inputs(paths: list[Path] | tuple[Path, ...]) -> dict[str, ArtifactInput]:
    return {str(path): snapshot_input(path) for path in paths}


def write_artifact_manifest(
    path: Path,
    *,
    stage: str,
    inputs: list[Path] | tuple[Path, ...] = (),
    params: dict[str, str] | None = None,
    model: str = "",
    config_hash: str = "",
) -> ArtifactManifest:
    manifest = ArtifactManifest(
        artifact=str(path),
        stage=stage,
        created_at=datetime.now(timezone.utc).isoformat(),
        inputs=snapshot_inputs(inputs),
        params=params or {},
        model=model,
        config_hash=config_hash,
    )
    write_model(artifact_manifest_path(path), manifest)
    return manifest


def read_artifact_manifest(path: Path) -> ArtifactManifest:
    return read_model(artifact_manifest_path(path), ArtifactManifest)


def artifact_is_stale(path: Path, inputs: list[Path] | tuple[Path, ...] = ()) -> bool:
    if not path.exists():
        return True
    manifest_path = artifact_manifest_path(path)
    if not manifest_path.exists():
        return True
    try:
        manifest = read_artifact_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    current = snapshot_inputs(inputs)
    return _inputs_changed(manifest.inputs, current)


def artifact_manifest_is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    manifest_path = artifact_manifest_path(path)
    if not manifest_path.exists():
        return True
    try:
        manifest = read_artifact_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    current = {
        key: snapshot_input(Path(item.path))
        for key, item in manifest.inputs.items()
    }
    return _inputs_changed(manifest.inputs, current)


def artifact_matches(
    path: Path,
    *,
    inputs: list[Path] | tuple[Path, ...] = (),
    params: dict[str, str] | None = None,
    model: str | None = None,
    config_hash: str | None = None,
) -> bool:
    """Return whether an artifact was produced for the expected invocation."""
    if artifact_is_stale(path, inputs):
        return False
    try:
        manifest = read_artifact_manifest(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if params is not None and manifest.params != params:
        return False
    if model is not None and manifest.model != model:
        return False
    if config_hash is not None and manifest.config_hash != config_hash:
        return False
    return True


def _inputs_changed(
    previous: dict[str, ArtifactInput],
    current: dict[str, ArtifactInput],
) -> bool:
    if set(previous) != set(current):
        return True
    for key, current_item in current.items():
        if previous[key].model_dump() != current_item.model_dump():
            return True
    return False


class ArtifactStore:
    """Resolve and read/write artifacts under a workflow root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, *parts: str | Path) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str | Path) -> bool:
        return self.path(*parts).exists()

    def write_text(
        self,
        *parts: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        path = self.path(*parts)
        write_text_atomic(path, content, encoding=encoding)
        return path

    def read_text(self, *parts: str | Path, encoding: str = "utf-8") -> str:
        return self.path(*parts).read_text(encoding=encoding)

    def write_json(self, *parts: str | Path, data: Any, indent: int = 2) -> Path:
        path = self.path(*parts)
        write_json_atomic(path, data, indent=indent)
        return path

    def read_json(self, *parts: str | Path) -> Any:
        return read_json(self.path(*parts))

    def write_model(self, *parts: str | Path, model: BaseModel) -> Path:
        path = self.path(*parts)
        write_model(path, model)
        return path

    def read_model[T: BaseModel](self, *parts: str | Path, model_cls: type[T]) -> T:
        return read_model(self.path(*parts), model_cls)

    def write_manifest(
        self,
        *parts: str | Path,
        stage: str,
        inputs: list[Path] | tuple[Path, ...] = (),
        params: dict[str, str] | None = None,
        model: str = "",
        config_hash: str = "",
    ) -> ArtifactManifest:
        return write_artifact_manifest(
            self.path(*parts),
            stage=stage,
            inputs=inputs,
            params=params,
            model=model,
            config_hash=config_hash,
        )

    def is_stale(self, *parts: str | Path, inputs: list[Path] | tuple[Path, ...] = ()) -> bool:
        return artifact_is_stale(self.path(*parts), inputs)
