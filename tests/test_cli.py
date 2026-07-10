import json

from typer.testing import CliRunner

from ai_clip import cli
from ai_clip.core.doctor import DoctorCheck
from ai_clip.pair.models import PairReviewReport, ReviewerResult


def test_daily_radar_is_top_level_workflow_command(monkeypatch):
    runner = CliRunner()

    def fake_daily_radar(
        cfg,
        date=None,
        top_n=None,
        research=False,
        force_collect=False,
        review=False,
        rewrite=False,
    ):
        assert date == "2026-01-02"
        assert top_n == 3
        assert research is False
        assert force_collect is False
        assert review is False
        assert rewrite is False
        return {
            "workflow": "daily_radar",
            "date": "2026-01-02",
            "collected": 4,
            "draft": "draft.md",
            "run_status": "run.json",
        }

    monkeypatch.setattr("ai_clip.cli.workflows.daily_radar", fake_daily_radar)
    result = runner.invoke(cli.app, ["daily-radar", "--date", "2026-01-02", "--top", "3"])

    assert result.exit_code == 0
    assert "daily-radar" in result.output
    assert "draft.md" in result.output


def test_daily_radar_can_enable_source_research(monkeypatch):
    runner = CliRunner()
    seen = {}

    def fake_daily_radar(
        cfg,
        date=None,
        top_n=None,
        research=False,
        force_collect=False,
        review=False,
        rewrite=False,
    ):
        seen["research"] = research
        seen["force_collect"] = force_collect
        seen["max_searches"] = cfg.source_research.max_searches
        seen["review"] = review
        seen["rewrite"] = rewrite
        return {
            "workflow": "daily_radar",
            "date": date,
            "collected": 1,
            "draft": "draft.md",
            "review": "review.json",
            "revised_draft": "draft.revised.md",
        }

    monkeypatch.setattr("ai_clip.cli.workflows.daily_radar", fake_daily_radar)
    result = runner.invoke(
        cli.app,
        [
            "daily-radar",
            "--date",
            "2026-01-02",
            "--research",
            "--research-searches",
            "1",
            "--force-collect",
            "--rewrite",
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "research": True,
        "force_collect": True,
        "max_searches": 1,
        "review": True,
        "rewrite": True,
    }
    assert "review.json" in result.output
    assert "draft.revised.md" in result.output


def test_daily_radar_json_uses_standard_envelope(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "ai_clip.cli.workflows.daily_radar",
        lambda *a, **k: {
            "workflow": "daily_radar",
            "date": "2026-01-02",
            "collected": 4,
            "draft": "draft.md",
        },
    )

    result = runner.invoke(cli.app, ["daily-radar", "--date", "2026-01-02", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema_version"] == 1
    assert payload["command"] == "daily-radar"
    assert payload["result"]["draft"] == "draft.md"


def test_collect_command_can_force(monkeypatch):
    runner = CliRunner()
    seen = {}

    def fake_run_collect(cfg, date=None, force=False):
        seen["date"] = date
        seen["force"] = force
        return 2

    monkeypatch.setattr("ai_clip.cli.pipeline.run_collect", fake_run_collect)
    result = runner.invoke(
        cli.app,
        ["collect", "--date", "2026-01-02", "--force-collect"],
    )

    assert result.exit_code == 0
    assert seen == {"date": "2026-01-02", "force": True}
    assert "2 snapshot" in result.output


def test_source_research_command(monkeypatch, tmp_path):
    runner = CliRunner()

    class Report:
        date = "2026-01-02"
        search_calls = 2

    def fake_run_source_research(cfg, date=None):
        assert date == "2026-01-02"
        assert cfg.source_research.max_searches == 2
        return Report()

    monkeypatch.setattr("ai_clip.cli.pipeline.run_source_research", fake_run_source_research)
    result = runner.invoke(
        cli.app,
        [
            "source-research",
            "--date",
            "2026-01-02",
            "--max-searches",
            "2",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert "source-research" in result.output
    assert "2 search(es)" in result.output


def test_research_command(monkeypatch, tmp_path):
    runner = CliRunner()
    seen = {}

    def fake_run_research(cfg, project, theme=""):
        seen["project"] = project
        seen["theme"] = theme
        seen["max_searches"] = cfg.source_research.max_searches
        return tmp_path / project / "research.md"

    monkeypatch.setattr("ai_clip.cli.pipeline.run_research", fake_run_research)
    result = runner.invoke(
        cli.app,
        [
            "research",
            "-p",
            "demo",
            "--theme",
            "AI and biology",
            "--max-searches",
            "1",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert seen == {"project": "demo", "theme": "AI and biology", "max_searches": 1}
    assert "research" in result.output
    assert "research.md" in result.output


def test_source_draft_command_can_enable_research(monkeypatch, tmp_path):
    runner = CliRunner()
    seen = {}

    def fake_source_draft(
        cfg,
        project,
        url,
        intent,
        stance="",
        use_subtitles=False,
        research=False,
        theme="",
        resume=True,
    ):
        seen.update({
            "project": project,
            "url": url,
            "research": research,
            "theme": theme,
            "max_searches": cfg.source_research.max_searches,
            "whisper_model": cfg.whisper.model_size,
            "resume": resume,
        })
        return {"draft": "source_draft.md"}

    monkeypatch.setattr("ai_clip.cli.workflows.source_draft", fake_source_draft)
    result = runner.invoke(
        cli.app,
        [
            "source-draft",
            "https://example.com/video",
            "-p",
            "demo",
            "--research",
            "--theme",
            "biology lens",
            "--research-searches",
            "1",
            "--whisper-model",
            "small",
            "--no-resume",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "project": "demo",
        "url": "https://example.com/video",
        "research": True,
        "theme": "biology lens",
        "max_searches": 1,
        "whisper_model": "small",
        "resume": False,
    }
    assert "source_draft.md" in result.output


def test_original_command_can_enable_research(monkeypatch, tmp_path):
    runner = CliRunner()
    seen = {}

    def fake_original(
        cfg,
        project,
        theme,
        fmt,
        intent,
        stance="",
        product=None,
        duration=30.0,
        n_shots=6,
        research=False,
    ):
        seen.update({
            "project": project,
            "theme": theme,
            "research": research,
            "max_searches": cfg.source_research.max_searches,
        })
        return {"status": "done", "output": "out.mp4"}

    monkeypatch.setattr("ai_clip.cli.workflows.original", fake_original)
    result = runner.invoke(
        cli.app,
        [
            "original",
            "--theme",
            "biology lens",
            "-p",
            "demo",
            "--research",
            "--research-searches",
            "2",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "project": "demo",
        "theme": "biology lens",
        "research": True,
        "max_searches": 2,
    }
    assert "out.mp4" in result.output


def test_zack_selection_command(monkeypatch, tmp_path):
    runner = CliRunner()

    class Selection:
        selected_index = 2
        topic = "selected topic"

    def fake_run_zack_selection(cfg, date=None):
        assert date == "2026-01-02"
        return Selection()

    monkeypatch.setattr("ai_clip.cli.pipeline.run_zack_selection", fake_run_zack_selection)
    result = runner.invoke(
        cli.app,
        [
            "zack-selection",
            "--date",
            "2026-01-02",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
    )

    assert result.exit_code == 0
    assert "zack-selection" in result.output
    assert "selected topic" in result.output


def test_radar_run_is_not_registered():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["radar", "run"])

    assert result.exit_code != 0


def test_module_stage_commands_are_registered():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "collect" in result.output
    assert "zack-ranking" in result.output
    assert "source-content" in result.output
    assert "content-rerank" in result.output
    assert "zack-selection" in result.output
    assert "source-research" in result.output
    assert "zack-draft" in result.output
    assert "doctor" in result.output
    assert "radar-status" in result.output
    assert "radar-repair" in result.output


def test_scout_group_is_not_registered():
    runner = CliRunner()
    result = runner.invoke(cli.app, ["scout", "collect"])

    assert result.exit_code != 0


def test_doctor_command_prints_checks(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "ai_clip.cli.run_doctor",
        lambda cfg: [DoctorCheck(name="data_dir", status="pass", detail="ok")],
    )
    monkeypatch.setattr("ai_clip.cli.doctor_exit_code", lambda checks: 0)

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "data_dir" in result.output
    assert "pass" in result.output


def test_doctor_json_uses_standard_envelope(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "ai_clip.cli.run_doctor",
        lambda cfg: [DoctorCheck(name="data_dir", status="pass", detail="ok")],
    )
    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "schema_version": 1,
        "command": "doctor",
        "status": "succeeded",
        "result": {
            "checks": [{"name": "data_dir", "status": "pass", "detail": "ok", "hint": ""}],
            "exit_code": 0,
        },
    }


def test_radar_status_command_reads_run_artifacts(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")
    run_dir = tmp_path / "radar" / "runs"
    report_dir = tmp_path / "radar" / "collect-reports"
    run_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    (run_dir / "2026-01-02.json").write_text(
        json.dumps({
            "workflow": "daily-radar",
            "date": "2026-01-02",
            "status": "failed",
            "stages": [
                {"name": "collect", "status": "failed", "duration_sec": 1.2, "error": "boom"}
            ],
        }),
        encoding="utf-8",
    )
    (report_dir / "2026-01-02.json").write_text(
        json.dumps({
            "collected_at": "2026-01-02T00:00:00+00:00",
            "channels": [{"name": "X", "status": "timeout", "error": "slow"}],
        }),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["radar-status", "--date", "2026-01-02", "--config", str(config)],
    )

    assert result.exit_code == 0
    assert "failed" in result.output
    assert "collect" in result.output
    assert "timeout" in result.output
    assert "candidates" in result.output
    assert "missing" in result.output


def test_radar_status_json_includes_run_identity(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")
    run_dir = tmp_path / "radar" / "runs"
    run_dir.mkdir(parents=True)
    (run_dir / "2026-01-02.json").write_text(
        json.dumps({
            "workflow": "daily-radar",
            "date": "2026-01-02",
            "run_id": "run-2",
            "attempt": 2,
            "status": "succeeded",
            "stages": [{"name": "collect", "status": "succeeded"}],
        }),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["radar-status", "--date", "2026-01-02", "--config", str(config), "--json"],
    )

    assert result.exit_code == 0
    envelope = json.loads(result.output)
    assert envelope["schema_version"] == 1
    assert envelope["command"] == "radar-status"
    payload = envelope["result"]
    assert payload["run_id"] == "run-2"
    assert payload["attempt"] == 2
    assert payload["stages"][0]["name"] == "collect"
    assert {item["name"] for item in payload["artifact_freshness"]} == {
        "candidates",
        "shortlist",
        "selection",
        "source_research",
        "zack_draft",
        "pair_review",
        "pair_rewrite",
        "pair_verify",
    }


def test_project_status_handles_missing_storyboard(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["status", "-p", "demo", "--config", str(config)],
    )

    assert result.exit_code == 0
    assert "research" in result.output
    assert "storyboard.json missing" in result.output


def test_project_status_json_handles_missing_storyboard(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["status", "-p", "demo", "--config", str(config), "--json"],
    )

    assert result.exit_code == 0
    envelope = json.loads(result.output)
    assert envelope["schema_version"] == 1
    assert envelope["command"] == "status"
    payload = envelope["result"]
    assert payload["project"] == "demo"
    assert payload["storyboard"] == {
        "path": str(tmp_path / "demo" / "storyboard.json"),
        "exists": False,
        "shots": 0,
    }
    assert payload["missing_assets"] == []
    assert payload["assets_ready"] is None
    assert payload["runs"] == []
    assert {item["name"]: item["status"] for item in payload["artifacts"]} == {
        "research": "missing",
        "storyboard": "missing",
        "source_draft": "missing",
    }


def test_radar_repair_dry_run_and_apply(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")
    (tmp_path / "radar" / "runs").mkdir(parents=True)
    (tmp_path / "radar" / "snapshots").mkdir(parents=True)
    (tmp_path / "radar" / "candidates").mkdir(parents=True)
    (tmp_path / "radar" / "runs" / "2026-01-02.json").write_text(
        json.dumps({"workflow": "daily-radar", "date": "2026-01-02", "status": "failed"}),
        encoding="utf-8",
    )
    snapshot = tmp_path / "radar" / "snapshots" / "2026-01-02.jsonl"
    candidates = tmp_path / "radar" / "candidates" / "2026-01-02.json"
    snapshot.write_text("", encoding="utf-8")
    candidates.write_text(
        '{"date": "2026-01-02", "top_n": 3, "videos": []}',
        encoding="utf-8",
    )

    dry = runner.invoke(
        cli.app,
        ["radar-repair", "--date", "2026-01-02", "--config", str(config)],
    )
    applied = runner.invoke(
        cli.app,
        ["radar-repair", "--date", "2026-01-02", "--config", str(config), "--apply"],
    )

    assert dry.exit_code == 0
    assert "would remove" in dry.output
    assert applied.exit_code == 0
    assert not snapshot.exists()
    assert not candidates.exists()


def test_radar_feedback_records_explicit_decision(monkeypatch, tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")
    seen = {}

    class Event:
        decision = "accept"
        video_id = "youtube:1"

    def fake_record(paths, decision, video_id="", reason=""):
        seen.update({
            "date": paths.date,
            "decision": decision,
            "video_id": video_id,
            "reason": reason,
        })
        return Event()

    monkeypatch.setattr("ai_clip.radar.feedback.record_feedback", fake_record)
    result = runner.invoke(
        cli.app,
        [
            "radar-feedback",
            "accept",
            "--date",
            "2026-01-02",
            "--video-id",
            "youtube:1",
            "--reason",
            "good angle",
            "--config",
            str(config),
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "date": "2026-01-02",
        "decision": "accept",
        "video_id": "youtube:1",
        "reason": "good angle",
    }


def test_run_status_json_navigates_project_run_artifacts(tmp_path):
    runner = CliRunner()
    config = tmp_path / "config.yaml"
    config.write_text(f"data_dir: {tmp_path.as_posix()}\n", encoding="utf-8")
    output = tmp_path / "demo" / "source_draft.md"
    output.parent.mkdir(parents=True)
    output.write_text("draft", encoding="utf-8")
    run_path = tmp_path / "demo" / "runs" / "source_draft.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps({
            "workflow": "source_draft",
            "project": "demo",
            "run_id": "run-1",
            "attempt": 1,
            "status": "succeeded",
            "stages": [{
                "name": "source-draft",
                "status": "succeeded",
                "outputs": {"draft": str(output)},
            }],
            "usage": {"total": {"calls": 1}},
        }),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "run-status",
            "--workflow",
            "source-draft",
            "--project",
            "demo",
            "--json",
            "--config",
            str(config),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "run-status"
    assert payload["result"]["run_id"] == "run-1"
    assert payload["result"]["artifacts"][0]["path"] == str(output)
    assert payload["result"]["artifacts"][0]["exists"] is True


def test_pair_review_json_reports_bounded_quality_chain(monkeypatch, tmp_path):
    runner = CliRunner()
    review = PairReviewReport(
        artifact="source_draft",
        source_path="source_draft.md",
        status="needs_review",
        reviewers=[ReviewerResult(role="logic", model="m1", ok=True, verdict="revise")],
    )
    verification = review.model_copy(update={
        "kind": "verify",
        "source_path": "source_draft.revised.md",
        "status": "passed",
    })
    monkeypatch.setattr("ai_clip.cli.pipeline.run_pair_review", lambda *a, **k: review)
    revised = tmp_path / "demo" / "source_draft.revised.md"
    monkeypatch.setattr("ai_clip.cli.pipeline.run_pair_rewrite", lambda *a, **k: revised)
    monkeypatch.setattr("ai_clip.cli.pipeline.run_pair_verify", lambda *a, **k: verification)

    result = runner.invoke(
        cli.app,
        ["pair-review", "-p", "demo", "--artifact", "source_draft", "--rewrite", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "pair-review"
    assert payload["result"]["review_status"] == "needs_review"
    assert payload["result"]["revised"] == str(revised)
    assert payload["result"]["verification_status"] == "passed"



