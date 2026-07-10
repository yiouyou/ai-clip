"""Composed workflows: thin orchestrations over the verified pipeline stages.

Each returns a small result dict describing what was produced and whether a
human step is still required (creating assets for generated formats).
"""

from __future__ import annotations

from ai_clip import pipeline
from ai_clip.core.config import Config
from ai_clip.core.models import Intent, Platform, ProductProfile, VideoFormat, ViralAnalysis
from ai_clip.core.paths import ProjectPaths, read_model
from ai_clip.core.run_status import mark_stale_running_stages, track_workflow_stage
from ai_clip.produce.assemble import check_assets


def _paths(cfg: Config, project: str) -> ProjectPaths:
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    return pp


def _run_status_path(cfg: Config, project: str, workflow: str) -> str:
    return str(_paths(cfg, project).run_status_json(workflow))


def _stage(cfg: Config, project: str, workflow: str, name: str, inputs: dict[str, str] | None = None):
    return track_workflow_stage(_paths(cfg, project), workflow, name, inputs)


def transcribe(cfg: Config, project: str, url: str) -> dict:
    """W1 提文案: download -> extract -> export srt/txt."""
    workflow = "transcribe"
    with _stage(cfg, project, workflow, "download", {"url": url}) as stage:
        clip = pipeline.run_download(cfg, project, url)
        stage.set(outputs={"clip": clip.video_path})
    with _stage(cfg, project, workflow, "extract") as stage:
        transcript = pipeline.run_extract(cfg, project)
        stage.set(metrics={"segments": len(transcript.segments), "language": transcript.language})
    with _stage(cfg, project, workflow, "export") as stage:
        srt, txt = pipeline.run_export(cfg, project)
        stage.set(outputs={"srt": str(srt), "txt": str(txt)})
    return {
        "workflow": workflow,
        "srt": str(srt),
        "txt": str(txt),
        "run_status": _run_status_path(cfg, project, workflow),
    }


def teardown(
    cfg: Config, project: str, url: str, intent: Intent = Intent.info
) -> dict:
    """W2 爆款拆解: download -> extract -> analyze (intent-aware)."""
    workflow = "teardown"
    with _stage(cfg, project, workflow, "download", {"url": url}):
        pipeline.run_download(cfg, project, url)
    with _stage(cfg, project, workflow, "extract"):
        pipeline.run_extract(cfg, project)
    with _stage(cfg, project, workflow, "analyze", {"intent": intent.value}) as stage:
        analysis = pipeline.run_analyze(cfg, project, intent)
        stage.set(metrics={"intent": analysis.intent})
    return {
        "workflow": workflow,
        "hook": analysis.hook,
        "formula": analysis.formula,
        "run_status": _run_status_path(cfg, project, workflow),
    }


def source_draft(
    cfg: Config,
    project: str,
    url: str,
    intent: Intent = Intent.info,
    stance: str = "",
    use_subtitles: bool = False,
    research: bool = False,
    theme: str = "",
    resume: bool = True,
) -> dict:
    """W7 单视频原创口播: download -> extract -> analyze -> [research] -> source_draft."""
    workflow = "source_draft"
    pp = _paths(cfg, project)
    mark_stale_running_stages(pp, workflow, older_than_minutes=0)
    if resume and pp.clip_json.exists():
        with _stage(cfg, project, workflow, "download", {"url": url}) as stage:
            stage.set(
                status="skipped",
                outputs={"clip": str(pp.clip_json)},
                metrics={"reused": True},
            )
    else:
        with _stage(cfg, project, workflow, "download", {"url": url}) as stage:
            clip = pipeline.run_download(cfg, project, url)
            stage.set(outputs={"clip": getattr(clip, "video_path", str(pp.clip_json))})
    with _stage(
        cfg,
        project,
        workflow,
        "extract",
        {"use_subtitles": str(use_subtitles)},
    ) as stage:
        if resume and pp.transcript_json.exists():
            stage.set(
                status="skipped",
                outputs={"transcript": str(pp.transcript_json)},
                metrics={"reused": True},
            )
        else:
            transcript = pipeline.run_extract(cfg, project, use_subtitles=use_subtitles)
            stage.set(
                outputs={"transcript": str(pp.transcript_json)},
                metrics={
                    "segments": len(getattr(transcript, "segments", [])),
                    "language": getattr(transcript, "language", ""),
                },
            )
    if resume and pp.analysis_json.exists():
        with _stage(cfg, project, workflow, "analyze", {"intent": intent.value}) as stage:
            analysis = read_model(pp.analysis_json, ViralAnalysis)
            stage.set(
                status="skipped",
                outputs={"analysis": str(pp.analysis_json)},
                metrics={"intent": analysis.intent, "reused": True},
            )
    else:
        with _stage(cfg, project, workflow, "analyze", {"intent": intent.value}) as stage:
            analysis = pipeline.run_analyze(cfg, project, intent)
            stage.set(
                outputs={"analysis": str(pp.analysis_json)},
                metrics={"intent": analysis.intent},
            )
    if research:
        if resume and pp.research_md.exists():
            with _stage(cfg, project, workflow, "research", {"theme": theme}) as stage:
                research_md = pp.research_md
                stage.set(
                    status="skipped",
                    outputs={"research": str(research_md)},
                    metrics={"reused": True},
                )
        else:
            with _stage(cfg, project, workflow, "research", {"theme": theme}) as stage:
                research_md = pipeline.run_research(cfg, project, theme=theme)
                stage.set(outputs={"research": str(research_md)})
    if resume and pp.source_draft_md.exists():
        with _stage(cfg, project, workflow, "source-draft", {"stance": stance}) as stage:
            draft = pp.source_draft_md
            stage.set(
                status="skipped",
                outputs={"draft": str(draft)},
                metrics={"reused": True},
            )
    else:
        with _stage(cfg, project, workflow, "source-draft", {"stance": stance}) as stage:
            draft = pipeline.run_source_draft(cfg, project, intent=intent, stance=stance)
            stage.set(outputs={"draft": str(draft)})
    return {
        "workflow": workflow,
        "hook": analysis.hook,
        "draft": str(draft),
        "run_status": _run_status_path(cfg, project, workflow),
    }


def remix(
    cfg: Config, project: str, url: str, theme: str,
    intent: Intent = Intent.info, stance: str = "",
    product: ProductProfile | None = None,
    duration: float = 30.0, n_shots: int = 6,
    use_subtitles: bool = False,
    research: bool = False,
) -> dict:
    """W3 二创(全自动): download -> extract -> analyze -> remix storyboard ->
    voiceover(clone) -> assemble. Needs no manual assets."""
    workflow = "remix"
    with _stage(cfg, project, workflow, "download", {"url": url}):
        pipeline.run_download(cfg, project, url)
    with _stage(cfg, project, workflow, "extract", {"use_subtitles": str(use_subtitles)}):
        pipeline.run_extract(cfg, project, use_subtitles=use_subtitles)
    with _stage(cfg, project, workflow, "analyze", {"intent": intent.value}):
        pipeline.run_analyze(cfg, project, intent)
    if research:
        with _stage(cfg, project, workflow, "research", {"theme": theme}) as stage:
            research_md = pipeline.run_research(cfg, project, theme=theme)
            stage.set(outputs={"research": str(research_md)})
    with _stage(
        cfg,
        project,
        workflow,
        "storyboard",
        {"theme": theme, "duration": str(duration), "shots": str(n_shots)},
    ) as stage:
        sb = pipeline.run_storyboard(
            cfg, project, theme, fmt=VideoFormat.remix, intent=intent,
            stance=stance, product=product, duration_sec=duration, n_shots=n_shots,
        )
        stage.set(metrics={"shots": len(sb.shots), "format": sb.format.value})
    with _stage(cfg, project, workflow, "voiceover") as stage:
        produced = pipeline.run_voiceover(cfg, project)
        stage.set(metrics={"voiceovers": len(produced)})
    with _stage(cfg, project, workflow, "assemble") as stage:
        out = pipeline.run_assemble(cfg, project)
        stage.set(outputs={"output": str(out)})
    return {"workflow": workflow, "output": str(out), "run_status": _run_status_path(cfg, project, workflow)}


def original(
    cfg: Config, project: str, theme: str,
    fmt: VideoFormat = VideoFormat.talking_head,
    intent: Intent = Intent.info, stance: str = "",
    product: ProductProfile | None = None,
    duration: float = 30.0, n_shots: int = 6,
    research: bool = False,
) -> dict:
    """W4 原创 / W5 全自动本地: storyboard -> assets(ComfyUI if available) ->
    voiceover -> assemble. If assets are still missing (prompt_only), stop and
    ask the human to fill assets/ then run `ai-clip assemble`."""
    if fmt == VideoFormat.remix:
        raise ValueError("remix needs a source clip; use the remix workflow")

    workflow = "original"
    if research:
        with _stage(cfg, project, workflow, "research", {"theme": theme}) as stage:
            research_md = pipeline.run_research(cfg, project, theme=theme)
            stage.set(outputs={"research": str(research_md)})
    with _stage(
        cfg,
        project,
        workflow,
        "storyboard",
        {"theme": theme, "format": fmt.value, "duration": str(duration), "shots": str(n_shots)},
    ) as stage:
        sb = pipeline.run_storyboard(
            cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
            product=product, duration_sec=duration, n_shots=n_shots,
        )
        stage.set(metrics={"shots": len(sb.shots), "format": sb.format.value})
    with _stage(cfg, project, workflow, "assets") as stage:
        generated = pipeline.run_assets(cfg, project)
        stage.set(metrics={"generated": generated})
    with _stage(cfg, project, workflow, "voiceover") as stage:
        produced = pipeline.run_voiceover(cfg, project)
        stage.set(metrics={"voiceovers": len(produced)})

    pp = _paths(cfg, project)
    missing = check_assets(sb, pp.assets_dir)
    if missing:
        with _stage(cfg, project, workflow, "assemble") as stage:
            stage.set(status="skipped", metrics={"missing_assets": len(missing)})
        return {
            "workflow": workflow, "status": "needs_assets",
            "generated": generated, "missing": missing,
            "assets_dir": str(pp.assets_dir),
            "run_status": _run_status_path(cfg, project, workflow),
        }
    with _stage(cfg, project, workflow, "assemble") as stage:
        out = pipeline.run_assemble(cfg, project)
        stage.set(outputs={"output": str(out)})
    return {
        "workflow": workflow,
        "status": "done",
        "output": str(out),
        "run_status": _run_status_path(cfg, project, workflow),
    }


def daily_radar(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
    research: bool = False,
    force_collect: bool = False,
    review: bool = False,
    rewrite: bool = False,
) -> dict:
    """W6 每日选题雷达: collect -> ranking -> content -> selection -> [research] -> draft."""
    result = pipeline.run_daily_radar(
        cfg,
        date,
        top_n,
        research=research,
        force_collect=force_collect,
        review=review,
        rewrite=rewrite,
    )
    return {
        "workflow": "daily_radar",
        "date": result.date,
        "collected": result.collected,
        "candidates": result.candidates_path,
        "selection": result.selection_path,
        "brief": result.brief_path,
        "draft": result.draft_path,
        "run_status": result.run_status_path,
        "review": result.review_path,
        "revised_draft": result.revised_draft_path,
    }


def discover_top_url(
    cfg: Config, project: str, topic: str,
    platform: Platform = Platform.youtube, since_days: int = 7,
) -> str | None:
    """Helper: discover and return the single most-viral candidate URL."""
    result = pipeline.run_discover(cfg, project, topic, platform, since_days=since_days)
    return result.candidates[0].url if result.candidates else None
