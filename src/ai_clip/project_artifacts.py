from __future__ import annotations

from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_path, artifact_matches, write_artifact_manifest
from ai_clip.core.config import Config
from ai_clip.core.models import Clip, Intent, Transcript, ViralAnalysis
from ai_clip.core.paths import ProjectPaths, read_model

PROMPT_VERSIONS = {
    "analyze": "1",
    "research": "1",
    "storyboard": "1",
    "source_draft": "1",
}


def project_paths(cfg: Config, project: str) -> ProjectPaths:
    paths = ProjectPaths(cfg.data_dir, project)
    paths.ensure()
    return paths


def existing_paths(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]


def download_params(url: str) -> dict[str, str]:
    return {"url": url}


def extract_params(cfg: Config, use_subtitles: bool) -> dict[str, str]:
    return {
        "use_subtitles": str(use_subtitles),
        "model_size": cfg.whisper.model_size,
        "compute_type": cfg.whisper.compute_type,
        "language": cfg.whisper.language or "",
    }


def analyze_params(intent: Intent) -> dict[str, str]:
    return {
        "intent": intent.value,
        "prompt_version": PROMPT_VERSIONS["analyze"],
    }


def research_params(cfg: Config, theme: str, mode: str = "source") -> dict[str, str]:
    return {
        "theme": theme,
        "mode": mode,
        "max_searches": str(cfg.source_research.max_searches),
        "max_results": str(cfg.source_research.max_results),
        "search_depth": cfg.source_research.search_depth,
        "prompt_version": PROMPT_VERSIONS["research"],
    }


def source_draft_params(
    intent: Intent,
    stance: str,
    research_used: bool,
) -> dict[str, str]:
    return {
        "intent": intent.value,
        "stance": stance,
        "research_used": str(research_used),
        "prompt_version": PROMPT_VERSIONS["source_draft"],
    }


def load_current_download(cfg: Config, project: str, url: str) -> Clip | None:
    paths = project_paths(cfg, project)
    if not paths.clip_json.exists():
        return None
    try:
        clip = read_model(paths.clip_json, Clip)
    except (OSError, ValueError):
        return None
    if clip.source_url != url or not Path(clip.video_path).exists():
        return None
    params = download_params(url)
    manifest = artifact_manifest_path(paths.clip_json)
    if manifest.exists() and not artifact_matches(paths.clip_json, params=params):
        return None
    if not manifest.exists():
        write_artifact_manifest(paths.clip_json, stage="download", params=params)
    return clip


def load_current_extract(
    cfg: Config,
    project: str,
    use_subtitles: bool = False,
) -> Transcript | None:
    paths = project_paths(cfg, project)
    if not paths.clip_json.exists() or not paths.transcript_json.exists():
        return None
    try:
        clip = read_model(paths.clip_json, Clip)
        transcript = read_model(paths.transcript_json, Transcript)
    except (OSError, ValueError):
        return None
    if transcript.clip_id != clip.clip_id:
        return None
    params = extract_params(cfg, use_subtitles)
    model = f"faster-whisper/{cfg.whisper.model_size}"
    manifest = artifact_manifest_path(paths.transcript_json)
    if manifest.exists() and not artifact_matches(
        paths.transcript_json,
        inputs=[paths.clip_json],
        params=params,
        model=model,
    ):
        return None
    if not manifest.exists():
        write_artifact_manifest(
            paths.transcript_json,
            stage="extract",
            inputs=[paths.clip_json],
            params=params,
            model=model,
        )
    return transcript


def load_current_analysis(
    cfg: Config,
    project: str,
    intent: Intent = Intent.info,
) -> ViralAnalysis | None:
    paths = project_paths(cfg, project)
    if not paths.transcript_json.exists() or not paths.analysis_json.exists():
        return None
    try:
        transcript = read_model(paths.transcript_json, Transcript)
        analysis = read_model(paths.analysis_json, ViralAnalysis)
    except (OSError, ValueError):
        return None
    if analysis.clip_id != transcript.clip_id or analysis.intent != intent:
        return None
    params = analyze_params(intent)
    manifest = artifact_manifest_path(paths.analysis_json)
    if manifest.exists() and not artifact_matches(
        paths.analysis_json,
        inputs=[paths.transcript_json],
        params=params,
        model=cfg.llm.model,
    ):
        return None
    if not manifest.exists():
        write_artifact_manifest(
            paths.analysis_json,
            stage="analyze",
            inputs=[paths.transcript_json],
            params=params,
            model=cfg.llm.model,
        )
    return analysis


def load_current_research(cfg: Config, project: str, theme: str = "") -> Path | None:
    return current_research_path(project_paths(cfg, project), cfg, theme)


def load_current_source_draft(
    cfg: Config,
    project: str,
    intent: Intent = Intent.info,
    stance: str = "",
    *,
    use_research: bool = True,
    research_theme: str = "",
) -> Path | None:
    paths = project_paths(cfg, project)
    research_path = (
        current_research_path(paths, cfg, research_theme)
        if use_research
        else None
    )
    inputs = existing_paths(
        paths.transcript_json,
        paths.analysis_json,
        *([research_path] if research_path else []),
    )
    if not artifact_matches(
        paths.source_draft_md,
        inputs=inputs,
        params=source_draft_params(intent, stance, bool(research_path)),
        model=cfg.llm.model,
    ):
        return None
    return paths.source_draft_md


def current_research_path(
    paths: ProjectPaths,
    cfg: Config,
    theme: str,
    *,
    mode: str = "source",
    allow_untracked: bool = False,
) -> Path | None:
    inputs = (
        existing_paths(paths.transcript_json, paths.analysis_json)
        if mode == "source"
        else []
    )
    if artifact_matches(
        paths.research_md,
        inputs=inputs,
        params=research_params(cfg, theme, mode),
        model=cfg.llm.model,
    ):
        return paths.research_md
    if allow_untracked and paths.research_md.exists() and not artifact_manifest_path(
        paths.research_md
    ).exists():
        return paths.research_md
    return None
