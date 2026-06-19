"""Stage orchestration. Each function reads/writes JSON artifacts under the
project dir so any stage can be run on its own or chained end to end."""

from __future__ import annotations

from pathlib import Path

from ai_clip.analyze import analyze as analyze_stage
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
from ai_clip.core.paths import ProjectPaths, read_model, write_model
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
    analysis = analyze_stage(transcript, cfg.llm, intent)
    write_model(pp.analysis_json, analysis)
    return analysis


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
        duration_sec=duration_sec,
        aspect_ratio=cfg.aspect_ratio,
        n_shots=n_shots,
    )
    write_model(pp.storyboard_json, sb)
    write_storyboard_files(sb, pp.prompts_dir, pp.storyboard_md)
    return sb


def run_assets(cfg: Config, project: str) -> int:
    """Fill in image assets for shots that expect them, using the configured
    provider (ComfyUI when available, else prompt_only which is a no-op and
    leaves it to a human). Returns the number of assets generated."""
    from ai_clip.produce.assets.factory import resolve_image_provider

    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    provider = resolve_image_provider(cfg.assets)
    generated = 0
    for shot in sb.shots:
        if not shot.image_file or not shot.image_prompt:
            continue
        if (pp.assets_dir / shot.image_file).exists():
            continue
        if provider.generate(shot, pp.assets_dir) is not None:
            generated += 1
    return generated


def run_voiceover(cfg: Config, project: str) -> dict[int, object]:
    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    source_audio = None
    if pp.transcript_json.exists():
        source_audio = read_model(pp.transcript_json, Transcript).audio_path
    tts = build_mimo(cfg.tts, source_audio, pp.reference_audio)
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
