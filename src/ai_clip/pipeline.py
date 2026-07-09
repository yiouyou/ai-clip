"""Stage orchestration. Each function reads/writes JSON artifacts under the
project dir so any stage can be run on its own or chained end to end."""

from __future__ import annotations

from pathlib import Path

from ai_clip.analyze import analyze as analyze_stage
from ai_clip.core import billing
from ai_clip.core.config import Config
from ai_clip.core.models import (
    CandidateList,
    Clip,
    Intent,
    Platform,
    ProductProfile,
    Storyboard,
    Transcript,
    ViralAnalysis,
    VideoFormat,
)
from ai_clip.pair.models import PairReviewReport
from ai_clip.radar.models import (
    RadarBackfillResult,
    RadarCandidates,
    RadarRunResult,
    ZackDraft,
    ZackSelection,
)
from ai_clip.source_research import SourceResearchReport
from ai_clip.core.paths import ProjectPaths, read_model, write_model
from ai_clip.core.artifacts import write_artifact_manifest, write_text_atomic
from ai_clip.discover import discover as discover_stage
from ai_clip.download import download as download_stage
from ai_clip.extract import extract as extract_stage
from ai_clip.extract.export import write_srt, write_txt
from ai_clip.produce import assemble as assemble_stage
from ai_clip.produce import generate_storyboard, write_storyboard_files
from ai_clip.produce.voiceover import build_mimo, generate_voiceover


def _paths(cfg: Config, project: str) -> ProjectPaths:
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    return pp


def _require_daily_radar(workflow: str) -> None:
    if workflow != "daily-radar":
        raise ValueError(f"unsupported workflow {workflow!r}; only daily-radar is supported")


def run_discover(
    cfg: Config,
    project: str,
    topic: str,
    platform: Platform = Platform.youtube,
    channel: str | None = None,
    since_days: int = 7,
    limit: int = 15,
    top_n: int = 5,
) -> CandidateList:
    pp = _paths(cfg, project)
    result = discover_stage(topic, platform, channel, since_days, limit, top_n)
    write_model(pp.candidates_json, result)
    return result


def run_download(cfg: Config, project: str, url: str) -> Clip:
    pp = _paths(cfg, project)
    clip = download_stage(url, pp.root, clip_id=project)
    write_model(pp.clip_json, clip)
    return clip


def run_extract(
    cfg: Config, project: str, use_subtitles: bool = False
) -> Transcript:
    pp = _paths(cfg, project)
    clip = read_model(pp.clip_json, Clip)
    transcript = extract_stage(clip, pp.root, cfg.whisper, use_subtitles=use_subtitles)
    write_model(pp.transcript_json, transcript)
    return transcript


def run_export(cfg: Config, project: str) -> tuple[object, object]:
    pp = _paths(cfg, project)
    transcript = read_model(pp.transcript_json, Transcript)
    return (
        write_srt(transcript, pp.transcript_srt),
        write_txt(transcript, pp.transcript_txt),
    )


def run_analyze(
    cfg: Config, project: str, intent: Intent = Intent.info
) -> ViralAnalysis:
    pp = _paths(cfg, project)
    transcript = read_model(pp.transcript_json, Transcript)
    with billing.account(pp.root, "analyze"):
        analysis = analyze_stage(transcript, cfg.llm, intent)
    write_model(pp.analysis_json, analysis)
    return analysis


def run_research(cfg: Config, project: str, theme: str = "") -> Path:
    from ai_clip.research import generate_project_research

    pp = _paths(cfg, project)
    transcript = read_model(pp.transcript_json, Transcript)
    analysis = read_model(pp.analysis_json, ViralAnalysis) if pp.analysis_json.exists() else None
    with billing.account(pp.root, "research"):
        report = generate_project_research(
            transcript=transcript,
            cfg=cfg,
            analysis=analysis,
            theme=theme,
        )
    write_model(pp.research_json, report)
    write_text_atomic(pp.research_md, report.markdown, encoding="utf-8")
    inputs = _existing_paths(pp.transcript_json, pp.analysis_json)
    params = {
        "theme": theme,
        "max_searches": str(cfg.source_research.max_searches),
    }
    write_artifact_manifest(
        pp.research_json,
        stage="research",
        inputs=inputs,
        params=params,
        model=cfg.llm.model,
    )
    write_artifact_manifest(
        pp.research_md,
        stage="research",
        inputs=inputs,
        params=params,
        model=cfg.llm.model,
    )
    return pp.research_md


def run_storyboard(
    cfg: Config,
    project: str,
    theme: str,
    fmt: VideoFormat = VideoFormat.talking_head,
    intent: Intent = Intent.info,
    stance: str = "",
    product: ProductProfile | None = None,
    duration_sec: float = 30.0,
    n_shots: int = 6,
) -> Storyboard:
    pp = _paths(cfg, project)
    analysis = (
        read_model(pp.analysis_json, ViralAnalysis)
        if pp.analysis_json.exists()
        else None
    )
    transcript = (
        read_model(pp.transcript_json, Transcript)
        if pp.transcript_json.exists()
        else None
    )
    research_markdown = pp.research_md.read_text(encoding="utf-8") if pp.research_md.exists() else ""
    with billing.account(pp.root, "storyboard"):
        sb = generate_storyboard(
            project=project,
            theme=theme,
            cfg=cfg.llm,
            fmt=fmt,
            analysis=analysis,
            transcript=transcript,
            intent=intent,
            stance=stance,
            product=product,
            research_markdown=research_markdown,
            duration_sec=duration_sec,
            aspect_ratio=cfg.aspect_ratio,
            n_shots=n_shots,
        )
    write_model(pp.storyboard_json, sb)
    write_storyboard_files(sb, pp.prompts_dir, pp.storyboard_md)
    inputs = _existing_paths(pp.analysis_json, pp.transcript_json, pp.research_md)
    params = {
        "theme": theme,
        "format": fmt.value,
        "intent": intent.value,
        "duration_sec": str(duration_sec),
        "n_shots": str(n_shots),
    }
    write_artifact_manifest(
        pp.storyboard_json,
        stage="storyboard",
        inputs=inputs,
        params=params,
        model=cfg.llm.model,
    )
    write_artifact_manifest(
        pp.storyboard_md,
        stage="storyboard",
        inputs=inputs,
        params=params,
        model=cfg.llm.model,
    )
    return sb


def _existing_paths(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]


def run_source_draft(
    cfg: Config,
    project: str,
    intent: Intent = Intent.info,
    stance: str = "",
) -> Path:
    from ai_clip.produce.source_draft import generate_source_draft

    pp = _paths(cfg, project)
    transcript = read_model(pp.transcript_json, Transcript)
    analysis = read_model(pp.analysis_json, ViralAnalysis) if pp.analysis_json.exists() else None
    research_markdown = pp.research_md.read_text(encoding="utf-8") if pp.research_md.exists() else ""
    with billing.account(pp.root, "source_draft"):
        markdown = generate_source_draft(
            transcript=transcript,
            analysis=analysis,
            cfg=cfg.llm,
            intent=intent,
            stance=stance,
            research_markdown=research_markdown,
        )
    write_text_atomic(pp.source_draft_md, markdown, encoding="utf-8")
    inputs = _existing_paths(pp.transcript_json, pp.analysis_json, pp.research_md)
    params = {
        "intent": intent.value,
        "stance": stance,
    }
    write_artifact_manifest(
        pp.source_draft_md,
        stage="source_draft",
        inputs=inputs,
        params=params,
        model=cfg.llm.model,
    )
    return pp.source_draft_md


def run_assets(cfg: Config, project: str) -> int:
    """Fill in image assets for shots that expect them, using the configured
    provider (ComfyUI when available, else prompt_only which is a no-op and
    leaves it to a human). Returns the number of assets generated."""
    from ai_clip.produce.assets.factory import resolve_image_provider

    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    generated = 0
    for shot in sb.shots:
        if not shot.image_file or not shot.image_prompt:
            continue
        if (pp.assets_dir / shot.image_file).exists():
            continue
        provider = resolve_image_provider(cfg.assets, engine=shot.asset_engine)
        if provider.generate(shot, pp.assets_dir) is not None:
            generated += 1
    return generated


def run_review_export(cfg: Config, project: str):
    """Write an editable script.md from storyboard.json."""
    from ai_clip.produce.review import to_script_md

    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    pp.script_md.write_text(to_script_md(sb), encoding="utf-8")
    return pp.script_md


def run_review_apply(cfg: Config, project: str) -> Storyboard:
    """Parse edited script.md back into storyboard.json."""
    from ai_clip.produce.review import apply_script_md
    from ai_clip.produce.storyboard import write_storyboard_files

    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    source_max = None
    if pp.clip_json.exists():
        source_max = read_model(pp.clip_json, Clip).duration_sec or None
    text = pp.script_md.read_text(encoding="utf-8")
    updated = apply_script_md(sb, text, source_max=source_max)
    write_model(pp.storyboard_json, updated)
    write_storyboard_files(updated, pp.prompts_dir, pp.storyboard_md)
    return updated


def run_voiceover(cfg: Config, project: str) -> dict[int, object]:
    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    source_audio = None
    if pp.transcript_json.exists():
        source_audio = read_model(pp.transcript_json, Transcript).audio_path
    tts = build_mimo(cfg.tts, source_audio, pp.reference_audio)
    with billing.account(pp.root, "voiceover"):
        return generate_voiceover(sb, tts, pp.voice_dir)


def run_assemble(cfg: Config, project: str):
    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    voice_dir = pp.voice_dir if any(pp.voice_dir.glob("shot_*.wav")) else None
    source_video = None
    if pp.clip_json.exists():
        source_video = Path(read_model(pp.clip_json, Clip).video_path)
    return assemble_stage(
        sb, pp.assets_dir, pp.output_mp4, voice_dir=voice_dir,
        source_video=source_video, burn_captions=cfg.burn_captions,
    )


def run_pair_review(
    cfg: Config,
    project: str,
    artifact: str,
    run_date: str | None = None,
) -> PairReviewReport:
    from ai_clip.pair import review_artifact

    return review_artifact(cfg, project, artifact, run_date=run_date)


def run_pair_rewrite(
    cfg: Config,
    project: str,
    artifact: str,
    report: PairReviewReport,
    run_date: str | None = None,
) -> Path:
    from ai_clip.pair.stage import rewrite_reviewed_artifact

    return rewrite_reviewed_artifact(
        cfg,
        project,
        artifact,
        report,
        run_date=run_date,
    )


def run_collect(
    cfg: Config,
    date: str | None = None,
    workflow: str = "daily-radar",
    force: bool = False,
) -> int:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_collect as collect_stage

    return collect_stage(cfg, date, force=force)


def run_zack_ranking(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
    workflow: str = "daily-radar",
) -> RadarCandidates:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_zack_ranking as zack_ranking_stage

    return zack_ranking_stage(cfg, date, top_n)


def run_source_content(
    cfg: Config,
    date: str | None = None,
    workflow: str = "daily-radar",
) -> RadarCandidates:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_source_content as source_content_stage

    return source_content_stage(cfg, date)


def run_zack_selection(
    cfg: Config,
    date: str | None = None,
    workflow: str = "daily-radar",
) -> ZackSelection:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_zack_selection as zack_selection_stage

    return zack_selection_stage(cfg, date)


def run_source_research(
    cfg: Config,
    date: str | None = None,
    workflow: str = "daily-radar",
) -> SourceResearchReport:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_source_research as source_research_stage

    return source_research_stage(cfg, date)


def run_zack_draft(
    cfg: Config,
    date: str | None = None,
    workflow: str = "daily-radar",
) -> ZackDraft:
    _require_daily_radar(workflow)
    from ai_clip.radar import run_zack_draft as zack_draft_stage

    return zack_draft_stage(cfg, date)


def run_daily_radar(
    cfg: Config,
    date: str | None = None,
    top_n: int | None = None,
    research: bool = False,
    force_collect: bool = False,
    review: bool = False,
    rewrite: bool = False,
) -> RadarRunResult:
    from ai_clip.radar import run_all

    return run_all(
        cfg,
        date,
        top_n,
        research=research,
        force_collect=force_collect,
        review=review,
        rewrite=rewrite,
    )


def run_daily_radar_backfill(
    cfg: Config,
    days: int = 7,
    end_date: str | None = None,
    top_n: int | None = None,
    channel_limit: int | None = None,
    channel_timeout: int = 30,
) -> RadarBackfillResult:
    from ai_clip.radar import run_backfill

    return run_backfill(
        cfg,
        days=days,
        end_date=end_date,
        top_n=top_n,
        channel_limit=channel_limit,
        channel_timeout=channel_timeout,
    )
