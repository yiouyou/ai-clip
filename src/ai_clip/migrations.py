from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_path, write_text_atomic

_RADAR_STAGE_NAMES = {
    "zack-ranking": "rank-candidates",
    "source-content": "enrich-content",
    "content-rerank": "rerank-candidates",
    "zack-selection": "select-topic",
    "source-research": "research-evidence",
    "zack-draft": "draft-script",
    "pair-review": "review-artifact",
    "pair-rewrite": "rewrite-artifact",
    "pair-verify": "verify-artifact",
    "artifact-review": "review-artifact",
    "artifact-rewrite": "rewrite-artifact",
    "artifact-verify": "verify-artifact",
}
_PROJECT_STAGE_NAMES = {
    "research": "source-research",
    "source_draft": "source-draft",
    "pair-review": "review-artifact",
    "pair-rewrite": "rewrite-artifact",
    "pair-verify": "verify-artifact",
    "artifact-review": "review-artifact",
    "artifact-rewrite": "rewrite-artifact",
    "artifact-verify": "verify-artifact",
}
_COMMON_BILLING_STAGE_NAMES = {
    "pair_review": "artifact_review",
    "pair_rewrite": "artifact_rewrite",
    "pair_verify": "artifact_verify",
}
_RADAR_BILLING_STAGE_NAMES = {
    **_COMMON_BILLING_STAGE_NAMES,
    "zack_selection": "select_topic",
    "source_research": "research_evidence",
    "zack_draft": "draft_script",
}
_ARTIFACT_NAMES = {
    "source_draft": "source-draft",
    "zack_draft": "editorial-draft",
}
_WORKFLOW_NAMES = {
    "source_draft": "source-draft",
    "daily_radar": "daily-radar",
}
_PATH_REPLACEMENTS = (
    ("source_draft.md", "source-draft.md"),
    ("source_draft.revised.md", "source-draft.revised.md"),
    ("voice_reference.mp3", "voice-reference.mp3"),
    ("_zack_draft_review.json", ".editorial-draft.review.json"),
    ("_zack_draft_verify.json", ".editorial-draft.verify.json"),
    ("source_draft_review.json", "source-draft.review.json"),
    ("source_draft_verify.json", "source-draft.verify.json"),
)


@dataclass(frozen=True)
class NamingMigration:
    source: Path
    destination: Path
    status: str


def migrate_artifact_names(data_dir: str | Path, *, apply: bool = False) -> list[NamingMigration]:
    root = Path(data_dir)
    moves = [
        (source, destination)
        for source, destination in _project_moves(root) + _radar_moves(root / "radar")
        if source.exists()
    ]
    conflicts = {source for source, destination in moves if destination.exists()}
    if apply and conflicts:
        return [
            NamingMigration(
                source,
                destination,
                "conflict" if source in conflicts else "planned",
            )
            for source, destination in moves
        ]
    results = []
    for source, destination in moves:
        status = "conflict" if destination.exists() else "planned"
        if apply and status == "planned":
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            status = "migrated"
        results.append(NamingMigration(source, destination, status))
    results.extend(_migrate_metadata(root, apply=apply))
    results.extend(_migrate_cost_ledgers(root, apply=apply))
    return results


def _project_moves(root: Path) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
    moves: list[tuple[Path, Path]] = []
    for project in root.iterdir():
        if not project.is_dir() or project.name == "radar":
            continue
        for legacy, canonical in (
            ("source_draft.md", "source-draft.md"),
            ("source_draft.revised.md", "source-draft.revised.md"),
            ("voice_reference.mp3", "voice-reference.mp3"),
            ("runs/source_draft.json", "runs/source-draft.json"),
            ("runs/history/source_draft", "runs/history/source-draft"),
        ):
            _append_with_manifest(moves, project / legacy, project / canonical)
        reviews = project / "reviews"
        if reviews.exists():
            for source in reviews.glob("*_review.json"):
                artifact = source.name.removesuffix("_review.json").replace("_", "-")
                _append_with_manifest(moves, source, reviews / f"{artifact}.review.json")
            for source in reviews.glob("*_verify.json"):
                artifact = source.name.removesuffix("_verify.json").replace("_", "-")
                _append_with_manifest(moves, source, reviews / f"{artifact}.verify.json")
    return moves


def _radar_moves(root: Path) -> list[tuple[Path, Path]]:
    reviews = root / "reviews"
    if not reviews.exists():
        return []
    moves: list[tuple[Path, Path]] = []
    for source in reviews.glob("*_zack_draft_review.json"):
        date = source.name.removesuffix("_zack_draft_review.json")
        _append_with_manifest(moves, source, reviews / f"{date}.editorial-draft.review.json")
    for source in reviews.glob("*_zack_draft_verify.json"):
        date = source.name.removesuffix("_zack_draft_verify.json")
        _append_with_manifest(moves, source, reviews / f"{date}.editorial-draft.verify.json")
    return moves


def _append_with_manifest(
    moves: list[tuple[Path, Path]],
    source: Path,
    destination: Path,
) -> None:
    moves.append((source, destination))
    moves.append((artifact_manifest_path(source), artifact_manifest_path(destination)))


def _migrate_metadata(root: Path, *, apply: bool) -> list[NamingMigration]:
    pending: list[tuple[Path, dict, dict]] = []
    for path in _metadata_paths(root):
        radar = _is_within(path, root / "radar")
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        normalized = _normalize_document(current, radar=radar)
        if normalized == current:
            continue
        pending.append((path, current, normalized))
    if apply:
        pending.sort(key=lambda item: item[0].name.endswith(".meta.json"))
        for path, current, normalized in pending:
            if path.name.endswith(".meta.json") and _manifest_inputs_need_refresh(
                current,
                normalized,
            ):
                _refresh_manifest_inputs(normalized, root)
            write_text_atomic(
                path,
                json.dumps(normalized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    status = "migrated" if apply else "planned"
    return [NamingMigration(path, path, status) for path, _, _ in pending]


def _metadata_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: set[Path] = set()
    for project in root.iterdir():
        if not project.is_dir():
            continue
        paths.update((project / "runs").glob("**/*.json"))
        paths.update((project / "reviews").glob("*.json"))
        paths.update(project.glob("*.meta.json"))
        paths.update((project / "reviews").glob("*.meta.json"))
    radar = root / "radar"
    if radar.exists():
        paths.update(radar.glob("**/*.meta.json"))
    return sorted(path for path in paths if path.is_file())


def _normalize_document(value, *, radar: bool):
    if not isinstance(value, dict):
        return value
    normalized = copy.deepcopy(value)
    stage_names = _RADAR_STAGE_NAMES if radar else _PROJECT_STAGE_NAMES
    if isinstance(normalized.get("workflow"), str):
        normalized["workflow"] = _WORKFLOW_NAMES.get(
            normalized["workflow"],
            normalized["workflow"],
        )
    if isinstance(normalized.get("stage"), str):
        normalized["stage"] = stage_names.get(normalized["stage"], normalized["stage"])
    if isinstance(normalized.get("artifact"), str):
        normalized["artifact"] = _normalize_artifact(normalized["artifact"])
    params = normalized.get("params")
    if isinstance(params, dict) and isinstance(params.get("artifact"), str):
        params["artifact"] = _normalize_artifact(params["artifact"])
    stages = normalized.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            if isinstance(stage.get("name"), str):
                stage["name"] = stage_names.get(stage["name"], stage["name"])
            outputs = stage.get("outputs")
            if isinstance(outputs, dict):
                stage["outputs"] = {
                    key: _normalize_path(item) if isinstance(item, str) else item
                    for key, item in outputs.items()
                }
    inputs = normalized.get("inputs")
    if isinstance(inputs, dict):
        normalized["inputs"] = _normalize_manifest_inputs(inputs)
    usage = normalized.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("by_stage"), dict):
        billing_names = (
            _RADAR_BILLING_STAGE_NAMES if radar else _COMMON_BILLING_STAGE_NAMES
        )
        usage["by_stage"] = {
            billing_names.get(name, name): amount
            for name, amount in usage["by_stage"].items()
        }
    return normalized


def _normalize_artifact(value: str) -> str:
    return _ARTIFACT_NAMES.get(value, _normalize_path(value))


def _normalize_path(value: str) -> str:
    normalized = value
    for legacy, canonical in _PATH_REPLACEMENTS:
        normalized = normalized.replace(legacy, canonical)
    return normalized


def _normalize_manifest_inputs(inputs: dict) -> dict:
    normalized = {}
    for key, item in inputs.items():
        normalized_key = _normalize_path(key) if isinstance(key, str) else key
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            item["path"] = _normalize_path(item["path"])
        normalized[normalized_key] = item
    return normalized


def _manifest_inputs_need_refresh(current: dict, normalized: dict) -> bool:
    current_params = current.get("params", {})
    normalized_params = normalized.get("params", {})
    artifact_changed = (
        isinstance(current_params, dict)
        and isinstance(normalized_params, dict)
        and current_params.get("artifact") != normalized_params.get("artifact")
    )
    return artifact_changed or current.get("inputs") != normalized.get("inputs")


def _refresh_manifest_inputs(manifest: dict, root: Path) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        return
    refreshed = {}
    for key, item in inputs.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            refreshed[key] = item
            continue
        path_text = item["path"]
        path = _resolve_data_path(path_text, root)
        if path.exists():
            stat = path.stat()
            refreshed[key] = {
                "path": path_text,
                "exists": True,
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        else:
            refreshed[key] = {"path": path_text, "exists": False, "mtime_ns": 0, "size": 0}
    manifest["inputs"] = refreshed


def _resolve_data_path(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root.resolve().parent / path


def _migrate_cost_ledgers(root: Path, *, apply: bool) -> list[NamingMigration]:
    results = []
    for path in sorted(root.glob("**/cost.jsonl")):
        radar = _is_within(path, root / "radar")
        billing_names = (
            _RADAR_BILLING_STAGE_NAMES if radar else _COMMON_BILLING_STAGE_NAMES
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        records = []
        try:
            for line in lines:
                records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        changed = False
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("stage"), str):
                continue
            stage = record["stage"]
            canonical = billing_names.get(stage, stage)
            if canonical != stage:
                record["stage"] = canonical
                changed = True
        if not changed:
            continue
        status = "planned"
        if apply:
            content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
            write_text_atomic(path, content, encoding="utf-8")
            status = "migrated"
        results.append(NamingMigration(path, path, status))
    return results


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
