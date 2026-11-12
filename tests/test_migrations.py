import json
from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_is_stale, artifact_manifest_path
from ai_clip.migrations import migrate_artifact_names


def test_naming_migration_is_explicit_and_idempotent(tmp_path: Path):
    project = tmp_path / "demo"
    reviews = project / "reviews"
    radar_reviews = tmp_path / "radar" / "reviews"
    reviews.mkdir(parents=True)
    radar_reviews.mkdir(parents=True)

    source_draft = project / "source_draft.md"
    source_draft.write_text("draft", encoding="utf-8")
    artifact_manifest_path(source_draft).write_text("{}", encoding="utf-8")
    project_review = reviews / "source_draft_review.json"
    project_review.write_text("{}", encoding="utf-8")
    radar_review = radar_reviews / "2026-01-02_zack_draft_review.json"
    radar_review.write_text(
        json.dumps({
            "artifact": "zack_draft",
            "stage": "pair-review",
            "summary": "pair-review found a zack-draft issue",
            "reviewers": [{
                "raw": "pair-review raw evidence",
                "issues": [{"detail": "zack-draft detail"}],
            }],
        }),
        encoding="utf-8",
    )
    project_runs = project / "runs"
    project_runs.mkdir()
    project_run = project_runs / "source_draft.json"
    project_run.write_text(
        json.dumps({
            "workflow": "source_draft",
            "stages": [
                {"name": "research"},
                {"name": "source_draft", "outputs": {"draft": str(source_draft)}},
            ],
            "usage": {"by_stage": {"pair_review": 1.25}},
        }),
        encoding="utf-8",
    )
    radar_runs = tmp_path / "radar" / "runs"
    radar_runs.mkdir()
    radar_run = radar_runs / "2026-01-02.json"
    radar_run.write_text(
        json.dumps({
            "workflow": "daily_radar",
            "stages": [
                {"name": "zack-ranking"},
                {"name": "source-content"},
                {"name": "pair-review", "metrics": {"reason": "pair-review blocked"}},
            ],
            "usage": {"by_stage": {"zack_selection": 2.5, "source_research": 0.5}},
        }),
        encoding="utf-8",
    )
    revised = tmp_path / "radar" / "drafts" / "2026-01-02.revised.md"
    revised.parent.mkdir()
    revised.write_text("revised", encoding="utf-8")
    review_stat = radar_review.stat()
    artifact_manifest_path(revised).write_text(
        json.dumps({
            "artifact": str(revised),
            "stage": "pair-rewrite",
            "created_at": "2026-01-02T00:00:00+00:00",
            "inputs": {
                str(radar_review): {
                    "path": str(radar_review),
                    "exists": True,
                    "mtime_ns": review_stat.st_mtime_ns,
                    "size": review_stat.st_size,
                },
            },
            "params": {"artifact": "zack_draft", "prompt_version": "1"},
        }),
        encoding="utf-8",
    )
    project_cost = project / "cost.jsonl"
    project_cost.write_text(
        '\n'.join([
            json.dumps({"stage": "source_research", "cost": 1}),
            json.dumps({"stage": "pair_review", "cost": 2}),
        ]) + '\n',
        encoding="utf-8",
    )
    radar_cost = tmp_path / "radar" / "cost.jsonl"
    radar_cost.write_text(
        '\n'.join([
            json.dumps({"stage": "source_research", "cost": 3}),
            json.dumps({"stage": "zack_draft", "cost": 4}),
        ]) + '\n',
        encoding="utf-8",
    )

    preview = migrate_artifact_names(tmp_path)

    assert preview
    assert {item.status for item in preview} == {"planned"}
    assert source_draft.exists()
    assert not (project / "source-draft.md").exists()

    applied = migrate_artifact_names(tmp_path, apply=True)

    assert {item.status for item in applied} == {"migrated"}
    assert (project / "source-draft.md").read_text(encoding="utf-8") == "draft"
    assert artifact_manifest_path(project / "source-draft.md").exists()
    assert (reviews / "source-draft.review.json").exists()
    migrated_radar_review = radar_reviews / "2026-01-02.editorial-draft.review.json"
    assert migrated_radar_review.exists()
    assert json.loads(migrated_radar_review.read_text(encoding="utf-8")) == {
        "artifact": "editorial-draft",
        "stage": "review-artifact",
        "summary": "pair-review found a zack-draft issue",
        "reviewers": [{
            "raw": "pair-review raw evidence",
            "issues": [{"detail": "zack-draft detail"}],
        }],
    }
    migrated_project_run = project_runs / "source-draft.json"
    project_status = json.loads(migrated_project_run.read_text(encoding="utf-8"))
    assert project_status["workflow"] == "source-draft"
    assert [stage["name"] for stage in project_status["stages"]] == [
        "source-research",
        "source-draft",
    ]
    assert project_status["stages"][1]["outputs"]["draft"].endswith("source-draft.md")
    assert project_status["usage"]["by_stage"] == {"artifact_review": 1.25}
    radar_status = json.loads(radar_run.read_text(encoding="utf-8"))
    assert radar_status["workflow"] == "daily-radar"
    assert [stage["name"] for stage in radar_status["stages"]] == [
        "rank-candidates",
        "enrich-content",
        "review-artifact",
    ]
    assert radar_status["stages"][2]["metrics"]["reason"] == "pair-review blocked"
    assert radar_status["usage"]["by_stage"] == {
        "select_topic": 2.5,
        "research_evidence": 0.5,
    }
    revised_manifest = json.loads(
        artifact_manifest_path(revised).read_text(encoding="utf-8")
    )
    assert revised_manifest["stage"] == "rewrite-artifact"
    assert revised_manifest["params"]["artifact"] == "editorial-draft"
    assert not artifact_manifest_is_stale(revised)
    project_cost_records = [
        json.loads(line) for line in project_cost.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in project_cost_records] == [
        "source_research",
        "artifact_review",
    ]
    radar_cost_records = [
        json.loads(line) for line in radar_cost.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["stage"] for record in radar_cost_records] == [
        "research_evidence",
        "draft_script",
    ]
    assert migrate_artifact_names(tmp_path, apply=True) == []


def test_naming_migration_never_overwrites_destination(tmp_path: Path):
    project = tmp_path / "demo"
    project.mkdir()
    legacy = project / "source_draft.md"
    canonical = project / "source-draft.md"
    legacy.write_text("legacy", encoding="utf-8")
    canonical.write_text("canonical", encoding="utf-8")
    legacy_voice = project / "voice_reference.mp3"
    legacy_voice.write_bytes(b"voice")

    results = migrate_artifact_names(tmp_path, apply=True)

    conflict = next(item for item in results if item.source == legacy)
    assert conflict.status == "conflict"
    assert legacy.read_text(encoding="utf-8") == "legacy"
    assert canonical.read_text(encoding="utf-8") == "canonical"
    assert legacy_voice.exists()
    assert not (project / "voice-reference.mp3").exists()
    assert next(item for item in results if item.source == legacy_voice).status == "planned"
