"""ai-clip command line. Each stage is a subcommand; `remix` and `original`
chain them. Human-in-the-loop is explicit: `storyboard` pauses for asset
creation, `assemble` resumes once assets/ is filled."""

from __future__ import annotations

import json as json_mod

import typer
from rich.console import Console
from rich.table import Table

from ai_clip.core.artifact_status import project_artifact_statuses
from ai_clip.core.config import load_config, load_product
from ai_clip.core.doctor import doctor_exit_code, run_doctor
from ai_clip.core.llm import LLMError
from ai_clip.produce.assemble import MissingAssetsError, check_assets
from ai_clip.core.models import Intent, Platform, Storyboard, VideoFormat
from ai_clip.core.paths import ProjectPaths, read_model
from ai_clip import pipeline, workflows

app = typer.Typer(
    add_completion=False,
    help="Download, analyze and produce short videos.",
    pretty_exceptions_show_locals=False,
)
console = Console()


def _cfg(config: str | None):
    return load_config(config)


def _require_daily_radar(workflow: str) -> None:
    if workflow != "daily-radar":
        console.print(f"[red]unsupported workflow[/] {workflow!r}; only daily-radar is supported")
        raise typer.Exit(1)


def _radar_paths(cfg, date: str | None):
    from ai_clip.radar.stage import today_in_tz
    from ai_clip.radar.storage import RadarPaths

    return RadarPaths(cfg.data_dir, date or today_in_tz(cfg.radar.timezone))


def _project_status_payload(pp: ProjectPaths) -> dict:
    artifacts = [
        {
            "name": item.name,
            "status": item.status,
            "path": str(item.path),
            "manifest": str(item.manifest),
        }
        for item in project_artifact_statuses(pp)
    ]
    payload = {
        "project": pp.project,
        "root": str(pp.root),
        "artifacts": artifacts,
        "storyboard": {
            "path": str(pp.storyboard_json),
            "exists": pp.storyboard_json.exists(),
            "shots": 0,
        },
        "assets_dir": str(pp.assets_dir),
        "missing_assets": [],
        "assets_ready": None,
    }
    if not pp.storyboard_json.exists():
        return payload
    sb = read_model(pp.storyboard_json, Storyboard)
    missing = check_assets(sb, pp.assets_dir)
    payload["storyboard"]["shots"] = len(sb.shots)
    payload["missing_assets"] = missing
    payload["assets_ready"] = not missing
    return payload


@app.command("doctor")
def doctor(
    config: str = typer.Option(None, "--config"),
):
    """Check local prerequisites and configuration without making paid API calls."""
    checks = run_doctor(_cfg(config))
    table = Table("Check", "Status", "Detail", "Hint")
    for check in checks:
        style = "green" if check.status == "pass" else "red" if check.status == "fail" else "yellow"
        table.add_row(check.name, f"[{style}]{check.status}[/]", check.detail, check.hint)
    console.print(table)
    code = doctor_exit_code(checks)
    if code:
        raise typer.Exit(code)


@app.command("collect")
def collect(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    force: bool = typer.Option(
        False,
        "--force",
        "--force-collect",
        help="ignore existing snapshots and fetch channels again",
    ),
    channel_timeout: int = typer.Option(
        None, "--channel-timeout", help="seconds before skipping a channel; 0 disables"
    ),
    channel_workers: int = typer.Option(
        None, "--channel-workers", help="parallel channel collectors when timeout is enabled"
    ),
    config: str = typer.Option(None, "--config"),
):
    """Collect metadata snapshots for configured YouTube/Bilibili channels."""
    _require_daily_radar(workflow)
    cfg = _cfg(config)
    if channel_timeout is not None:
        cfg.radar.channel_timeout_sec = channel_timeout
    if channel_workers is not None:
        cfg.radar.channel_workers = channel_workers
    count = pipeline.run_collect(cfg, date, force=force)
    console.print(f"[green]collect[/] {count} snapshot(s)")


@app.command("zack-ranking")
def zack_ranking(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    top: int = typer.Option(None, "--top"),
    config: str = typer.Option(None, "--config"),
):
    """Rank daily-radar snapshots with Zack's topic-selection policy."""
    _require_daily_radar(workflow)
    candidates = pipeline.run_zack_ranking(_cfg(config), date, top)
    console.print(f"[green]zack-ranking[/] top {len(candidates.videos)} for {candidates.date}")


@app.command("source-content")
def source_content(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    config: str = typer.Option(None, "--config"),
):
    """Fetch available scripts/subtitles for ranked candidates."""
    _require_daily_radar(workflow)
    candidates = pipeline.run_source_content(_cfg(config), date)
    with_script = sum(1 for video in candidates.videos if video.transcript_text)
    console.print(f"[green]source-content[/] scripts {with_script}/{len(candidates.videos)}")


@app.command("zack-selection")
def zack_selection(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    config: str = typer.Option(None, "--config"),
):
    """Select one daily-radar topic before research and drafting."""
    _require_daily_radar(workflow)
    selection = pipeline.run_zack_selection(_cfg(config), date)
    console.print(
        f"[green]zack-selection[/] #{selection.selected_index} {selection.topic}"
    )


@app.command("source-research")
def source_research(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    max_searches: int = typer.Option(None, "--max-searches", help="Tavily searches; clamped to 1-3"),
    config: str = typer.Option(None, "--config"),
):
    """Research event details and safe framing for the selected daily-radar topic."""
    _require_daily_radar(workflow)
    cfg = _cfg(config)
    if max_searches is not None:
        cfg.source_research.max_searches = max_searches
    try:
        report = pipeline.run_source_research(cfg, date)
    except Exception as exc:
        console.print(f"[red]source-research failed[/] — {exc}")
        raise typer.Exit(1) from exc
    from ai_clip.radar.storage import RadarPaths

    paths = RadarPaths(cfg.data_dir, report.date)
    console.print(
        f"[green]source-research[/] {report.search_calls} search(es) -> {paths.research_md}"
    )


@app.command("zack-draft")
def zack_draft(
    workflow: str = typer.Option("daily-radar", "--workflow"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    config: str = typer.Option(None, "--config"),
):
    """Generate today's topic brief and original talking-head drafts."""
    _require_daily_radar(workflow)
    cfg = _cfg(config)
    try:
        draft = pipeline.run_zack_draft(cfg, date)
    except LLMError as exc:
        console.print(f"[red]zack-draft failed[/] — {exc}")
        raise typer.Exit(1) from exc
    from ai_clip.radar.storage import RadarPaths

    paths = RadarPaths(cfg.data_dir, draft.date)
    console.print(f"[green]zack-draft[/] -> {paths.draft_md}")


@app.command("radar-status")
def radar_status(
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    config: str = typer.Option(None, "--config"),
):
    """Show daily-radar stage status and per-channel collect diagnostics."""
    from ai_clip.radar.ops import read_radar_status
    from ai_clip.radar.artifact_status import radar_artifact_statuses

    cfg = _cfg(config)
    paths = _radar_paths(cfg, date)
    summary = read_radar_status(paths)
    console.print(f"[green]daily-radar[/] {summary.date}: {summary.status}")
    console.print(f"run-status: {summary.run_status_path}")
    if summary.collect_report_path:
        console.print(f"collect-report: {summary.collect_report_path}")
    if summary.stages:
        table = Table("Stage", "Status", "Duration", "Error")
        for stage in summary.stages:
            status = stage["status"]
            style = "green" if status in {"succeeded", "skipped"} else "red" if status == "failed" else "yellow"
            table.add_row(stage["name"], f"[{style}]{status}[/]", stage["duration"], stage["error"])
        console.print(table)
    if summary.channel_counts:
        counts = ", ".join(f"{name}={count}" for name, count in sorted(summary.channel_counts.items()))
        console.print(f"channels: {counts}")
    for failure in summary.channel_failures[:10]:
        console.print(f"[yellow]channel[/] {failure}")
    artifact_statuses = radar_artifact_statuses(paths)
    if artifact_statuses:
        table = Table("Artifact", "Freshness", "Path")
        for item in artifact_statuses:
            style = "green" if item.status == "fresh" else "yellow" if item.status == "stale" else "dim"
            table.add_row(item.name, f"[{style}]{item.status}[/]", str(item.path))
        console.print(table)
    for name, path in summary.artifacts.items():
        console.print(f"{name}: {path}")


@app.command("radar-repair")
def radar_repair(
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    apply: bool = typer.Option(False, "--apply", help="remove clearly invalid artifacts"),
    config: str = typer.Option(None, "--config"),
):
    """Dry-run or apply conservative cleanup for failed/stale daily-radar artifacts."""
    from ai_clip.radar.ops import repair_radar_date

    cfg = _cfg(config)
    paths = _radar_paths(cfg, date)
    result = repair_radar_date(paths, apply=apply)
    verb = "removed" if apply else "would remove"
    for path in result.removed or result.kept:
        console.print(f"[green]{verb}[/] {path}" if apply else f"[yellow]{verb}[/] {path}")
    if not result.removed and not result.kept:
        console.print("[green]nothing to repair[/]")


@app.command("daily-radar-backfill")
def daily_radar_backfill(
    days: int = typer.Option(7, "--days"),
    top: int = typer.Option(3, "--top"),
    limit: int = typer.Option(None, "--limit", help="videos to inspect per channel"),
    channel_timeout: int = typer.Option(30, "--channel-timeout", help="seconds before skipping a channel"),
    end_date: str = typer.Option(None, "--end-date", help="YYYY-MM-DD; defaults to today"),
    config: str = typer.Option(None, "--config"),
):
    """Collect recent metadata and save daily Top N files for retrospective review."""
    result = pipeline.run_daily_radar_backfill(
        _cfg(config),
        days=days,
        end_date=end_date,
        top_n=top,
        channel_limit=limit,
        channel_timeout=channel_timeout,
    )
    console.print(
        f"[green]daily-radar-backfill[/] {result.days}d collected {result.collected} -> "
        f"{result.output_dir}"
    )


@app.command("daily-radar")
def daily_radar(
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today"),
    top: int = typer.Option(None, "--top"),
    force_collect: bool = typer.Option(
        False,
        "--force-collect",
        help="ignore existing snapshots and fetch channels again",
    ),
    research: bool = typer.Option(False, "--research", help="run source-research before zack-draft"),
    research_searches: int = typer.Option(
        None, "--research-searches", help="Tavily searches when --research is set; clamped to 1-3"
    ),
    channel_timeout: int = typer.Option(
        None, "--channel-timeout", help="seconds before skipping a channel; 0 disables"
    ),
    channel_workers: int = typer.Option(
        None, "--channel-workers", help="parallel channel collectors when timeout is enabled"
    ),
    review: bool = typer.Option(False, "--review", help="run pair-review on zack-draft"),
    rewrite: bool = typer.Option(False, "--rewrite", help="write a revised draft after pair-review"),
    config: str = typer.Option(None, "--config"),
):
    """Run the daily topic radar workflow through selection and drafting."""
    cfg = _cfg(config)
    if channel_timeout is not None:
        cfg.radar.channel_timeout_sec = channel_timeout
    if channel_workers is not None:
        cfg.radar.channel_workers = channel_workers
    if research_searches is not None:
        cfg.source_research.max_searches = research_searches
    try:
        result = workflows.daily_radar(
            cfg,
            date,
            top,
            research=research,
            force_collect=force_collect,
            review=review or rewrite,
            rewrite=rewrite,
        )
    except RuntimeError as exc:
        console.print(f"[red]daily-radar failed[/] — {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"[green]daily-radar[/] {result['date']}: collected {result['collected']} -> "
        f"{result['draft']}"
    )
    if result.get("run_status"):
        console.print(f"[green]run-status[/] -> {result['run_status']}")
    if result.get("review"):
        console.print(f"[green]pair-review[/] -> {result['review']}")
    if result.get("revised_draft"):
        console.print(f"[green]pair-rewrite[/] -> {result['revised_draft']}")


@app.command()
def discover(
    topic: str,
    project: str = typer.Option(..., "--project", "-p"),
    platform: Platform = typer.Option(Platform.youtube, "--platform"),
    channel: str = typer.Option(None, "--channel"),
    since_days: int = typer.Option(7, "--since-days"),
    limit: int = typer.Option(15, "--limit"),
    top_n: int = typer.Option(5, "--top"),
    config: str = typer.Option(None, "--config"),
):
    """Search a platform for recently-spreading clips on a topic, ranked by virality."""
    result = pipeline.run_discover(
        _cfg(config), project, topic, platform, channel, since_days, limit, top_n
    )
    if not result.candidates:
        console.print("[yellow]no candidates found[/] (try widening --since-days)")
        return
    for i, c in enumerate(result.candidates, start=1):
        console.print(
            f"[green]{i}.[/] v={c.virality:,.0f} · {c.view_count:,} views · "
            f"{c.age_days:.1f}d · {c.title[:60]}\n   {c.url}"
        )


@app.command()
def download(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Download a source clip with yt-dlp."""
    clip = pipeline.run_download(_cfg(config), project, url)
    console.print(f"[green]downloaded[/] {clip.platform} -> {clip.video_path}")


@app.command()
def extract(
    project: str = typer.Option(..., "--project", "-p"),
    subs: bool = typer.Option(False, "--subs", help="use the video's subtitles instead of whisper"),
    config: str = typer.Option(None, "--config"),
):
    """Transcribe via faster-whisper, or reuse the video's subtitles with --subs."""
    t = pipeline.run_extract(_cfg(config), project, use_subtitles=subs)
    console.print(f"[green]transcribed[/] {len(t.segments)} segments, lang={t.language}")


@app.command()
def export(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Export the transcript to .srt and .txt."""
    srt, txt = pipeline.run_export(_cfg(config), project)
    console.print(f"[green]exported[/] {srt} , {txt}")


@app.command()
def analyze(
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    config: str = typer.Option(None, "--config"),
):
    """Reverse-engineer the viral formula via LLM (intent: info|emotion|sales)."""
    a = pipeline.run_analyze(_cfg(config), project, intent)
    console.print(f"[green]analyzed[/] ({a.intent}) hook: {a.hook[:80]}")


@app.command()
def research(
    project: str = typer.Option(..., "--project", "-p"),
    theme: str = typer.Option("", "--theme"),
    max_searches: int = typer.Option(
        None,
        "--max-searches",
        help="Tavily searches for this project research stage; clamped to 1-3",
    ),
    config: str = typer.Option(None, "--config"),
):
    """Research source facts/context before storyboard; writes editable research.md."""
    cfg = _cfg(config)
    if max_searches is not None:
        cfg.source_research.max_searches = max_searches
    try:
        path = pipeline.run_research(cfg, project, theme=theme)
    except Exception as exc:
        console.print(f"[red]research failed[/] — {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]research[/] -> {path}")


@app.command()
def storyboard(
    project: str = typer.Option(..., "--project", "-p"),
    theme: str = typer.Option(..., "--theme"),
    fmt: VideoFormat = typer.Option(VideoFormat.talking_head, "--format", "-f"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """Generate a storyboard for the chosen format (talking_head|slideshow|remix|montage)."""
    cfg = _cfg(config)
    sb = pipeline.run_storyboard(
        cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
        product=load_product(product), duration_sec=duration, n_shots=shots,
    )
    pp = ProjectPaths(cfg.data_dir, project)
    console.print(f"[green]storyboard[/] ({sb.format}) {len(sb.shots)} shots -> {pp.storyboard_md}")
    if sb.format == VideoFormat.remix:
        console.print("[yellow]Next:[/] `ai-clip voiceover` then `ai-clip assemble` (no manual assets).")
    else:
        console.print(
            "[yellow]Next:[/] create assets (即梦/Gemini/ComfyUI) into "
            f"{pp.assets_dir} per storyboard.md, then `ai-clip voiceover` + `ai-clip assemble`."
        )


@app.command()
def status(
    project: str = typer.Option(..., "--project", "-p"),
    json_output: bool = typer.Option(False, "--json", help="emit machine-readable JSON"),
    config: str = typer.Option(None, "--config"),
):
    """Show project artifact freshness and which shot assets are still missing."""
    pp = ProjectPaths(_cfg(config).data_dir, project)
    payload = _project_status_payload(pp)
    if json_output:
        console.file.write(json_mod.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        console.file.flush()
        return
    table = Table("Artifact", "Status", "Path")
    for item in payload["artifacts"]:
        style = "green" if item["status"] == "fresh" else "yellow" if item["status"] == "stale" else "dim"
        table.add_row(item["name"], f"[{style}]{item['status']}[/]", item["path"])
    console.print(table)
    if not payload["storyboard"]["exists"]:
        console.print("[yellow]storyboard.json missing; run `ai-clip storyboard` before asset status[/]")
        return
    missing = payload["missing_assets"]
    shots = payload["storyboard"]["shots"]
    if missing:
        console.print(f"[yellow]missing {len(missing)}/{shots}:[/]")
        for m in missing:
            console.print(f"  - {m}")
    else:
        console.print(f"[green]all {shots} shots have assets — ready to assemble[/]")


@app.command()
def assets(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Generate missing image assets with the configured provider."""
    generated = pipeline.run_assets(_cfg(config), project)
    console.print(f"[green]assets[/] generated {generated} image(s)")


@app.command()
def review(
    project: str = typer.Option(..., "--project", "-p"),
    apply: bool = typer.Option(False, "--apply", help="parse edited script.md back into storyboard"),
    config: str = typer.Option(None, "--config"),
):
    """Export an editable script.md (the 文案), or --apply edits back to storyboard."""
    cfg = _cfg(config)
    if apply:
        sb = pipeline.run_review_apply(cfg, project)
        console.print(f"[green]applied[/] script -> storyboard ({len(sb.shots)} shots)")
    else:
        path = pipeline.run_review_export(cfg, project)
        console.print(
            f"[green]script[/] -> {path}\n"
            "[yellow]Edit the narration (and remix timestamps), then:[/] "
            f"ai-clip review -p {project} --apply"
        )


@app.command("pair-review")
def pair_review(
    project: str = typer.Option("radar", "--project", "-p"),
    artifact: str = typer.Option("storyboard", "--artifact"),
    date: str = typer.Option(None, "--date", help="YYYY-MM-DD, required for zack_draft"),
    rewrite: bool = typer.Option(False, "--rewrite", help="write a revised draft after review"),
    config: str = typer.Option(None, "--config"),
):
    """Run two-model review over a text artifact."""
    cfg = _cfg(config)
    try:
        report = pipeline.run_pair_review(cfg, project, artifact, run_date=date)
    except Exception as exc:
        console.print(f"[red]pair-review failed[/] — {exc}")
        raise typer.Exit(1) from exc
    if report.artifact == "zack_draft":
        from ai_clip.radar.storage import RadarPaths

        out = RadarPaths(cfg.data_dir, date or "").reviews_dir / f"{date}_zack_draft_review.json"
    else:
        out = ProjectPaths(cfg.data_dir, project).reviews_dir / f"{report.artifact}_review.json"
    models = ", ".join(r.model or "none" for r in report.reviewers)
    console.print(f"[green]pair-review[/] {report.status} via {models} -> {out}")
    if rewrite:
        try:
            revised = pipeline.run_pair_rewrite(
                cfg,
                project,
                artifact,
                report,
                run_date=date,
            )
        except Exception as exc:
            console.print(f"[red]pair-rewrite failed[/] — {exc}")
            raise typer.Exit(1) from exc
        console.print(f"[green]pair-rewrite[/] -> {revised}")


@app.command()
def voiceover(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Synthesize per-shot narration via MiMo TTS (clones the source voice by default)."""
    produced = pipeline.run_voiceover(_cfg(config), project)
    console.print(f"[green]voiceover[/] synthesized {len(produced)} shot(s)")


@app.command()
def assemble(
    project: str = typer.Option(..., "--project", "-p"),
    captions: bool = typer.Option(False, "--captions"),
    config: str = typer.Option(None, "--config"),
):
    """Stitch collected assets into the final MP4."""
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    try:
        out = pipeline.run_assemble(cfg, project)
    except MissingAssetsError as exc:
        console.print(f"[red]cannot assemble[/] — {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]assembled[/] -> {out}")


# ---- composed workflows (W1-W5) ----

@app.command()
def transcribe(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """W1 提文案: download -> extract -> export srt/txt."""
    r = workflows.transcribe(_cfg(config), project, url)
    console.print(f"[green]transcribed[/] -> {r['srt']} , {r['txt']}")


@app.command()
def teardown(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    config: str = typer.Option(None, "--config"),
):
    """W2 爆款拆解: download -> extract -> analyze (intent: info|emotion|sales)."""
    r = workflows.teardown(_cfg(config), project, url, intent)
    console.print(f"[green]teardown[/] hook: {r['hook'][:80]}")
    console.print(f"formula: {r['formula'][:120]}")


@app.command("source-draft")
def source_draft(
    url: str,
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    subs: bool = typer.Option(False, "--subs", help="use the video's subtitles instead of whisper"),
    whisper_model: str = typer.Option(
        None,
        "--whisper-model",
        help="override whisper model for this run, e.g. tiny|base|small|medium|large-v3",
    ),
    research: bool = typer.Option(False, "--research", help="run project research before drafting"),
    theme: str = typer.Option("", "--theme", help="research theme; defaults to source-derived context"),
    research_searches: int = typer.Option(
        None, "--research-searches", help="Tavily searches when --research is set; clamped to 1-3"
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="reuse existing artifacts when present"),
    config: str = typer.Option(None, "--config"),
):
    """Single source video -> original talking-head draft in the creator's preferred lens."""
    cfg = _cfg(config)
    if whisper_model:
        cfg.whisper.model_size = whisper_model
    if research_searches is not None:
        cfg.source_research.max_searches = research_searches
    r = workflows.source_draft(
        cfg,
        project,
        url,
        intent=intent,
        stance=stance,
        use_subtitles=subs,
        research=research,
        theme=theme,
        resume=resume,
    )
    console.print(f"[green]source-draft[/] -> {r['draft']}")


@app.command()
def remix(
    url: str,
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    captions: bool = typer.Option(False, "--captions"),
    subs: bool = typer.Option(False, "--subs", help="use the video's subtitles instead of whisper"),
    research: bool = typer.Option(False, "--research", help="run project research before storyboard"),
    research_searches: int = typer.Option(
        None, "--research-searches", help="Tavily searches when --research is set; clamped to 1-3"
    ),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """W3 二创(全自动): download -> ... -> remix storyboard -> cloned voiceover -> mp4."""
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    if research_searches is not None:
        cfg.source_research.max_searches = research_searches
    r = workflows.remix(
        cfg, project, url, theme, intent=intent, stance=stance,
        product=load_product(product), duration=duration, n_shots=shots,
        use_subtitles=subs, research=research,
    )
    console.print(f"[green]remix done[/] -> {r['output']}")


@app.command()
def original(
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    fmt: VideoFormat = typer.Option(VideoFormat.talking_head, "--format", "-f"),
    intent: Intent = typer.Option(Intent.info, "--intent", "-i"),
    stance: str = typer.Option("", "--stance"),
    product: str = typer.Option(None, "--product"),
    captions: bool = typer.Option(False, "--captions"),
    research: bool = typer.Option(False, "--research", help="run project research before storyboard"),
    research_searches: int = typer.Option(
        None, "--research-searches", help="Tavily searches when --research is set; clamped to 1-3"
    ),
    duration: float = typer.Option(30.0, "--duration"),
    shots: int = typer.Option(6, "--shots"),
    config: str = typer.Option(None, "--config"),
):
    """W4/W5 原创: storyboard -> assets(ComfyUI if available) -> voiceover -> assemble."""
    if fmt == VideoFormat.remix:
        console.print("[red]remix format needs a source clip; use `ai-clip remix`.[/]")
        raise typer.Exit(1)
    cfg = _cfg(config)
    cfg.burn_captions = captions or cfg.burn_captions
    if research_searches is not None:
        cfg.source_research.max_searches = research_searches
    r = workflows.original(
        cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
        product=load_product(product), duration=duration, n_shots=shots,
        research=research,
    )
    if r["status"] == "done":
        console.print(f"[green]original done[/] -> {r['output']}")
    else:
        console.print(
            f"[yellow]need assets[/] ({len(r['missing'])} missing, {r['generated']} generated). "
            f"Fill {r['assets_dir']} per storyboard.md, then `ai-clip assemble`."
        )


@app.command()
def cost(
    project: str = typer.Option(..., "--project", "-p"),
    config: str = typer.Option(None, "--config"),
):
    """Show metered spend (LLM tokens + TTS chars) for a project."""
    from ai_clip.core import billing

    pp = ProjectPaths(_cfg(config).data_dir, project)
    s = billing.summarize(pp.root)
    t = s["total"]
    console.print(
        f"[green]total ${t['cost']:.4f}[/] over {t['calls']} call(s) | "
        f"in {t['input_tokens']:,} / out {t['output_tokens']:,} tok | {t['chars']:,} chars"
    )
    for stage, c in s["by_stage"].items():
        console.print(f"  stage {stage}: ${c:.4f}")
    for model, c in s["by_model"].items():
        console.print(f"  model {model}: ${c:.4f}")


@app.command()
def mpt(
    theme: str = typer.Option(..., "--theme"),
    project: str = typer.Option(..., "--project", "-p"),
    voice: str = typer.Option("", "--voice"),
    config: str = typer.Option(None, "--config"),
):
    """External backend: generate a theme video via MoneyPrinterTurbo's API."""
    from ai_clip.produce.backends import MoneyPrinterBackend, ProduceSpec

    cfg = _cfg(config)
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    backend = MoneyPrinterBackend(cfg.produce.moneyprinter_url)
    if not MoneyPrinterBackend.is_available(cfg.produce.moneyprinter_url):
        console.print(
            f"[red]MoneyPrinterTurbo not reachable[/] at {cfg.produce.moneyprinter_url}. "
            "Start it (docker compose up in the MoneyPrinterTurbo repo)."
        )
        raise typer.Exit(1)
    out = backend.produce(ProduceSpec(
        theme=theme, out_path=pp.output_mp4, aspect_ratio=cfg.aspect_ratio, voice_name=voice,
    ))
    console.print(f"[green]mpt done[/] -> {out}")


if __name__ == "__main__":
    app()
