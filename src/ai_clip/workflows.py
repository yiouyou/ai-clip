"""Composed workflows: thin orchestrations over the verified pipeline stages.

Each returns a small result dict describing what was produced and whether a
human step is still required (creating assets for generated formats).
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from ai_clip import pipeline
from ai_clip.core.config import Config
from ai_clip.core.stages import (
    StageExecution,
    StageInvocation,
    StageResult,
    execute_workflow,
    stage_execution,
)
from ai_clip.core.models import Intent, Platform, ProductProfile, VideoFormat
from ai_clip.core.paths import ProjectPaths
from ai_clip.core.run_lock import RunLock
from ai_clip.core.run_status import (
    begin_workflow_run,
    mark_stale_running_stages,
    project_status_store,
    track_workflow_stage,
    update_run_usage,
)
from ai_clip.produce.assemble import check_assets
from ai_clip.registry import REGISTRY


def _paths(cfg: Config, project: str) -> ProjectPaths:
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    return pp


def _run_status_path(cfg: Config, project: str, workflow: str) -> str:
    return str(_paths(cfg, project).run_status_json(workflow))


def _stage(cfg: Config, project: str, workflow: str, name: str, inputs: dict[str, str] | None = None):
    return track_workflow_stage(_paths(cfg, project), workflow, name, inputs)


def _begin(cfg: Config, project: str, workflow: str) -> None:
    begin_workflow_run(_paths(cfg, project), workflow)


def _execute(
    cfg: Config,
    project: str,
    workflow: str,
    executions: dict[str, StageExecution[Any]],
    flags: dict[str, bool] | None = None,
) -> dict[str, StageResult[Any]]:
    def tracker_factory(invocation: StageInvocation):
        return _stage(
            cfg,
            project,
            REGISTRY.workflow(workflow).status_key,
            invocation.name,
            dict(invocation.inputs),
        )

    return execute_workflow(
        REGISTRY.workflow(workflow),
        executions,
        flags,
        tracker_factory,
    )


def _project_locked(workflow: str):
    status_key = REGISTRY.workflow(workflow).status_key

    def decorate(func):
        @wraps(func)
        def locked(cfg: Config, project: str, *args, **kwargs):
            paths = _paths(cfg, project)
            lock = RunLock(
                paths.runs_dir / "locks" / f"{status_key}.lock",
                f"{status_key} is already running for project {project}",
                metadata={"workflow": status_key, "project": project},
            )
            with lock:
                try:
                    return func(cfg, project, *args, **kwargs)
                finally:
                    update_run_usage(project_status_store(paths, status_key), paths.root)

        return locked

    return decorate


@_project_locked("transcribe")
def transcribe(cfg: Config, project: str, url: str) -> dict:
    """W1 提文案: download -> extract -> export srt/txt."""
    workflow = REGISTRY.workflow("transcribe").status_key
    _begin(cfg, project, workflow)
    srt = None
    txt = None

    def download_stage():
        clip = pipeline.run_download(cfg, project, url)
        return StageResult(value=clip, outputs={"clip": clip.video_path})

    def extract_stage():
        transcript = pipeline.run_extract(cfg, project)
        return StageResult(
            value=transcript,
            metrics={"segments": len(transcript.segments), "language": transcript.language},
        )

    def export_stage():
        nonlocal srt, txt
        srt, txt = pipeline.run_export(cfg, project)
        return StageResult(value=(srt, txt), outputs={"srt": str(srt), "txt": str(txt)})

    _execute(
        cfg,
        project,
        "transcribe",
        {
            "download": stage_execution("download", download_stage, {"url": url}),
            "extract": stage_execution("extract", extract_stage),
            "export": stage_execution("export", export_stage),
        },
    )
    return {
        "workflow": workflow,
        "srt": str(srt),
        "txt": str(txt),
        "run_status": _run_status_path(cfg, project, workflow),
    }


@_project_locked("teardown")
def teardown(
    cfg: Config, project: str, url: str, intent: Intent = Intent.info
) -> dict:
    """W2 爆款拆解: download -> extract -> analyze (intent-aware)."""
    workflow = REGISTRY.workflow("teardown").status_key
    _begin(cfg, project, workflow)
    analysis = None

    def download_stage():
        return StageResult(value=pipeline.run_download(cfg, project, url))

    def extract_stage():
        return StageResult(value=pipeline.run_extract(cfg, project))

    def analyze_stage():
        nonlocal analysis
        analysis = pipeline.run_analyze(cfg, project, intent)
        return StageResult(value=analysis, metrics={"intent": analysis.intent})

    _execute(
        cfg,
        project,
        "teardown",
        {
            "download": stage_execution("download", download_stage, {"url": url}),
            "extract": stage_execution("extract", extract_stage),
            "analyze": stage_execution(
                "analyze",
                analyze_stage,
                {"intent": intent.value},
            ),
        },
    )
    if analysis is None:
        raise RuntimeError("teardown workflow did not produce analysis")
    return {
        "workflow": workflow,
        "hook": analysis.hook,
        "formula": analysis.formula,
        "run_status": _run_status_path(cfg, project, workflow),
    }


@_project_locked("source-draft")
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
    workflow = REGISTRY.workflow("source-draft").status_key
    pp = _paths(cfg, project)
    mark_stale_running_stages(pp, workflow, older_than_minutes=0)
    _begin(cfg, project, workflow)
    analysis = None
    draft = None
    download_reused = False
    extract_reused = False
    analysis_reused = False
    research_reused = False

    def download_stage():
        nonlocal download_reused
        clip = pipeline.load_current_download(cfg, project, url) if resume else None
        download_reused = clip is not None
        if clip is not None:
            return StageResult(
                value=clip,
                status="skipped",
                outputs={"clip": str(pp.clip_json)},
                metrics={"reused": True},
            )
        clip = pipeline.run_download(cfg, project, url)
        return StageResult(
            value=clip,
            outputs={"clip": getattr(clip, "video_path", str(pp.clip_json))},
        )

    def extract_stage():
        nonlocal extract_reused
        transcript = (
            pipeline.load_current_extract(cfg, project, use_subtitles=use_subtitles)
            if resume and download_reused
            else None
        )
        extract_reused = transcript is not None
        if transcript is not None:
            return StageResult(
                value=transcript,
                status="skipped",
                outputs={"transcript": str(pp.transcript_json)},
                metrics={"reused": True},
            )
        transcript = pipeline.run_extract(cfg, project, use_subtitles=use_subtitles)
        return StageResult(
            value=transcript,
            outputs={"transcript": str(pp.transcript_json)},
            metrics={
                "segments": len(getattr(transcript, "segments", [])),
                "language": getattr(transcript, "language", ""),
            },
        )

    def analyze_stage():
        nonlocal analysis, analysis_reused
        analysis = (
            pipeline.load_current_analysis(cfg, project, intent)
            if resume and extract_reused
            else None
        )
        analysis_reused = analysis is not None
        if analysis is not None:
            return StageResult(
                value=analysis,
                status="skipped",
                outputs={"analysis": str(pp.analysis_json)},
                metrics={"intent": analysis.intent, "reused": True},
            )
        analysis = pipeline.run_analyze(cfg, project, intent)
        return StageResult(
            value=analysis,
            outputs={"analysis": str(pp.analysis_json)},
            metrics={"intent": analysis.intent},
        )

    def research_stage():
        nonlocal research_reused
        research_md = (
            pipeline.load_current_research(cfg, project, theme)
            if resume and analysis_reused
            else None
        )
        research_reused = research_md is not None
        if research_md is not None:
            return StageResult(
                value=research_md,
                status="skipped",
                outputs={"research": str(research_md)},
                metrics={"reused": True},
            )
        research_md = pipeline.run_research(cfg, project, theme=theme)
        return StageResult(value=research_md, outputs={"research": str(research_md)})

    def source_draft_stage():
        nonlocal draft
        draft = (
            pipeline.load_current_source_draft(
                cfg,
                project,
                intent,
                stance,
                use_research=research,
                research_theme=theme,
            )
            if resume and analysis_reused and (not research or research_reused)
            else None
        )
        if draft is not None:
            return StageResult(
                value=draft,
                status="skipped",
                outputs={"draft": str(draft)},
                metrics={"reused": True},
            )
        draft = pipeline.run_source_draft(
            cfg,
            project,
            intent=intent,
            stance=stance,
            use_research=research,
            research_theme=theme,
            allow_untracked_research=False,
        )
        return StageResult(value=draft, outputs={"draft": str(draft)})

    _execute(
        cfg,
        project,
        "source-draft",
        {
            "download": stage_execution("download", download_stage, {"url": url}),
            "extract": stage_execution(
                "extract",
                extract_stage,
                {"use_subtitles": str(use_subtitles)},
            ),
            "analyze": stage_execution(
                "analyze",
                analyze_stage,
                {"intent": intent.value},
            ),
            "research": stage_execution("research", research_stage, {"theme": theme}),
            "source-draft": stage_execution(
                "source-draft",
                source_draft_stage,
                {"stance": stance},
            ),
        },
        {"research": research},
    )
    if analysis is None or draft is None:
        raise RuntimeError("source-draft workflow did not produce required outputs")
    return {
        "workflow": workflow,
        "hook": analysis.hook,
        "draft": str(draft),
        "run_status": _run_status_path(cfg, project, workflow),
    }


@_project_locked("remix")
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
    workflow = REGISTRY.workflow("remix").status_key
    _begin(cfg, project, workflow)
    out = None

    def download_stage():
        return StageResult(value=pipeline.run_download(cfg, project, url))

    def extract_stage():
        return StageResult(
            value=pipeline.run_extract(cfg, project, use_subtitles=use_subtitles)
        )

    def analyze_stage():
        return StageResult(value=pipeline.run_analyze(cfg, project, intent))

    def research_stage():
        research_md = pipeline.run_research(cfg, project, theme=theme)
        return StageResult(value=research_md, outputs={"research": str(research_md)})

    def storyboard_stage():
        sb = pipeline.run_storyboard(
            cfg,
            project,
            theme,
            fmt=VideoFormat.remix,
            intent=intent,
            stance=stance,
            product=product,
            duration_sec=duration,
            n_shots=n_shots,
            use_research=research,
            allow_untracked_research=False,
        )
        return StageResult(
            value=sb,
            metrics={"shots": len(sb.shots), "format": sb.format.value},
        )

    def voiceover_stage():
        produced = pipeline.run_voiceover(cfg, project)
        return StageResult(value=produced, metrics={"voiceovers": len(produced)})

    def assemble_stage():
        nonlocal out
        out = pipeline.run_assemble(cfg, project)
        return StageResult(value=out, outputs={"output": str(out)})

    _execute(
        cfg,
        project,
        "remix",
        {
            "download": stage_execution("download", download_stage, {"url": url}),
            "extract": stage_execution(
                "extract",
                extract_stage,
                {"use_subtitles": str(use_subtitles)},
            ),
            "analyze": stage_execution(
                "analyze",
                analyze_stage,
                {"intent": intent.value},
            ),
            "research": stage_execution("research", research_stage, {"theme": theme}),
            "storyboard": stage_execution(
                "storyboard",
                storyboard_stage,
                {"theme": theme, "duration": str(duration), "shots": str(n_shots)},
            ),
            "voiceover": stage_execution("voiceover", voiceover_stage),
            "assemble": stage_execution("assemble", assemble_stage),
        },
        {"research": research},
    )
    if out is None:
        raise RuntimeError("remix workflow did not produce output")
    return {"workflow": workflow, "output": str(out), "run_status": _run_status_path(cfg, project, workflow)}


@_project_locked("original")
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

    workflow = REGISTRY.workflow("original").status_key
    _begin(cfg, project, workflow)
    pp = _paths(cfg, project)
    sb = None
    generated = 0
    missing = []
    out = None
    flags = {"research": research, "assets-ready": False}

    def research_stage():
        research_md = pipeline.run_topic_research(cfg, project, theme=theme)
        return StageResult(value=research_md, outputs={"research": str(research_md)})

    def storyboard_stage():
        nonlocal sb
        sb = pipeline.run_storyboard(
            cfg,
            project,
            theme,
            fmt=fmt,
            intent=intent,
            stance=stance,
            product=product,
            duration_sec=duration,
            n_shots=n_shots,
            use_research=research,
            allow_untracked_research=False,
            use_source_context=False,
            research_mode="topic",
        )
        return StageResult(
            value=sb,
            metrics={"shots": len(sb.shots), "format": sb.format.value},
        )

    def assets_stage():
        nonlocal generated, missing
        generated = pipeline.run_assets(cfg, project)
        if sb is None:
            raise RuntimeError("original workflow did not produce storyboard")
        missing = check_assets(sb, pp.assets_dir)
        flags["assets-ready"] = not missing
        return StageResult(
            value=generated,
            status="waiting" if missing else "succeeded",
            metrics={"generated": generated, **({"missing_assets": len(missing)} if missing else {})},
        )

    def voiceover_stage():
        produced = pipeline.run_voiceover(cfg, project)
        return StageResult(value=produced, metrics={"voiceovers": len(produced)})

    def assemble_stage():
        nonlocal missing, out
        if sb is None:
            raise RuntimeError("original workflow did not produce storyboard")
        missing = check_assets(sb, pp.assets_dir)
        if missing:
            return StageResult(
                status="skipped",
                metrics={"missing_assets": len(missing)},
            )
        out = pipeline.run_assemble(cfg, project)
        return StageResult(value=out, outputs={"output": str(out)})

    _execute(
        cfg,
        project,
        "original",
        {
            "topic-research": stage_execution(
                "topic-research",
                research_stage,
                {"theme": theme},
            ),
            "storyboard": stage_execution(
                "storyboard",
                storyboard_stage,
                {
                    "theme": theme,
                    "format": fmt.value,
                    "duration": str(duration),
                    "shots": str(n_shots),
                },
            ),
            "assets": stage_execution("assets", assets_stage),
            "voiceover": stage_execution("voiceover", voiceover_stage),
            "assemble": stage_execution("assemble", assemble_stage),
        },
        flags,
    )
    if missing:
        return {
            "workflow": workflow, "status": "needs_assets",
            "generated": generated, "missing": missing,
            "assets_dir": str(pp.assets_dir),
            "run_status": _run_status_path(cfg, project, workflow),
        }
    if out is None:
        raise RuntimeError("original workflow did not produce output")
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
        "verification": result.verification_path,
    }


def discover_top_url(
    cfg: Config, project: str, topic: str,
    platform: Platform = Platform.youtube, since_days: int = 7,
) -> str | None:
    """Helper: discover and return the single most-viral candidate URL."""
    result = pipeline.run_discover(cfg, project, topic, platform, since_days=since_days)
    return result.candidates[0].url if result.candidates else None
