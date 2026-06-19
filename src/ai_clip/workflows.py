"""Composed workflows: thin orchestrations over the verified pipeline stages.

Each returns a small result dict describing what was produced and whether a
human step is still required (creating assets for generated formats).
"""

from __future__ import annotations

from ai_clip import pipeline
from ai_clip.core.config import Config
from ai_clip.core.models import Intent, Platform, ProductProfile, VideoFormat
from ai_clip.core.paths import ProjectPaths
from ai_clip.produce.assemble import check_assets


def transcribe(cfg: Config, project: str, url: str) -> dict:
    """W1 提文案: download -> extract -> export srt/txt."""
    pipeline.run_download(cfg, project, url)
    pipeline.run_extract(cfg, project)
    srt, txt = pipeline.run_export(cfg, project)
    return {"workflow": "transcribe", "srt": str(srt), "txt": str(txt)}


def teardown(
    cfg: Config, project: str, url: str, intent: Intent = Intent.info
) -> dict:
    """W2 爆款拆解: download -> extract -> analyze (intent-aware)."""
    pipeline.run_download(cfg, project, url)
    pipeline.run_extract(cfg, project)
    analysis = pipeline.run_analyze(cfg, project, intent)
    return {"workflow": "teardown", "hook": analysis.hook, "formula": analysis.formula}


def remix(
    cfg: Config, project: str, url: str, theme: str,
    intent: Intent = Intent.info, stance: str = "",
    product: ProductProfile | None = None,
    duration: float = 30.0, n_shots: int = 6,
) -> dict:
    """W3 二创(全自动): download -> extract -> analyze -> remix storyboard ->
    voiceover(clone) -> assemble. Needs no manual assets."""
    pipeline.run_download(cfg, project, url)
    pipeline.run_extract(cfg, project)
    pipeline.run_analyze(cfg, project, intent)
    pipeline.run_storyboard(
        cfg, project, theme, fmt=VideoFormat.remix, intent=intent,
        stance=stance, product=product, duration_sec=duration, n_shots=n_shots,
    )
    pipeline.run_voiceover(cfg, project)
    out = pipeline.run_assemble(cfg, project)
    return {"workflow": "remix", "output": str(out)}


def original(
    cfg: Config, project: str, theme: str,
    fmt: VideoFormat = VideoFormat.talking_head,
    intent: Intent = Intent.info, stance: str = "",
    product: ProductProfile | None = None,
    duration: float = 30.0, n_shots: int = 6,
) -> dict:
    """W4 原创 / W5 全自动本地: storyboard -> assets(ComfyUI if available) ->
    voiceover -> assemble. If assets are still missing (prompt_only), stop and
    ask the human to fill assets/ then run `ai-clip assemble`."""
    if fmt == VideoFormat.remix:
        raise ValueError("remix needs a source clip; use the remix workflow")

    sb = pipeline.run_storyboard(
        cfg, project, theme, fmt=fmt, intent=intent, stance=stance,
        product=product, duration_sec=duration, n_shots=n_shots,
    )
    generated = pipeline.run_assets(cfg, project)
    pipeline.run_voiceover(cfg, project)

    pp = ProjectPaths(cfg.data_dir, project)
    missing = check_assets(sb, pp.assets_dir)
    if missing:
        return {
            "workflow": "original", "status": "needs_assets",
            "generated": generated, "missing": missing,
            "assets_dir": str(pp.assets_dir),
        }
    out = pipeline.run_assemble(cfg, project)
    return {"workflow": "original", "status": "done", "output": str(out)}


def discover_top_url(
    cfg: Config, project: str, topic: str,
    platform: Platform = Platform.youtube, since_days: int = 7,
) -> str | None:
    """Helper: discover and return the single most-viral candidate URL."""
    result = pipeline.run_discover(cfg, project, topic, platform, since_days=since_days)
    return result.candidates[0].url if result.candidates else None
