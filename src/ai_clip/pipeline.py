"""Stage orchestration. Each function reads/writes JSON artifacts under the
project dir so any stage can be run on its own or chained end to end."""

from __future__ import annotations

from ai_clip.analyze import analyze as analyze_stage
from ai_clip.core.config import Config
from ai_clip.core.models import Clip, Storyboard, Transcript, ViralAnalysis
from ai_clip.core.paths import ProjectPaths, read_model, write_model
from ai_clip.download import download as download_stage
from ai_clip.extract import extract as extract_stage
from ai_clip.produce import assemble as assemble_stage
from ai_clip.produce import generate_storyboard, write_storyboard_files


def _paths(cfg: Config, project: str) -> ProjectPaths:
    pp = ProjectPaths(cfg.data_dir, project)
    pp.ensure()
    return pp


def run_download(cfg: Config, project: str, url: str) -> Clip:
    pp = _paths(cfg, project)
    clip = download_stage(url, pp.root, clip_id=project)
    write_model(pp.clip_json, clip)
    return clip


def run_extract(cfg: Config, project: str) -> Transcript:
    pp = _paths(cfg, project)
    clip = read_model(pp.clip_json, Clip)
    transcript = extract_stage(clip, pp.root, cfg.whisper)
    write_model(pp.transcript_json, transcript)
    return transcript


def run_analyze(cfg: Config, project: str) -> ViralAnalysis:
    pp = _paths(cfg, project)
    transcript = read_model(pp.transcript_json, Transcript)
    analysis = analyze_stage(transcript, cfg.llm)
    write_model(pp.analysis_json, analysis)
    return analysis


def run_storyboard(
    cfg: Config,
    project: str,
    theme: str,
    duration_sec: float = 30.0,
    n_shots: int = 6,
) -> Storyboard:
    pp = _paths(cfg, project)
    analysis = (
        read_model(pp.analysis_json, ViralAnalysis)
        if pp.analysis_json.exists()
        else None
    )
    sb = generate_storyboard(
        project=project,
        theme=theme,
        cfg=cfg.llm,
        analysis=analysis,
        duration_sec=duration_sec,
        aspect_ratio=cfg.aspect_ratio,
        n_shots=n_shots,
    )
    write_model(pp.storyboard_json, sb)
    write_storyboard_files(sb, pp.prompts_dir, pp.storyboard_md)
    return sb


def run_assemble(cfg: Config, project: str):
    pp = _paths(cfg, project)
    sb = read_model(pp.storyboard_json, Storyboard)
    return assemble_stage(sb, pp.assets_dir, pp.output_mp4)
