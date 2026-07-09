"""Storyboard step: dispatch to a format-specific generator, then write prompt
files + a human-readable storyboard.md.

The filename contract: shot N expects assets/shot_NN.png / shot_NN.mp4 (for
generated formats), so assemble doesn't care whether an asset came from ComfyUI
or a human download. `remix` shots carry source spans instead of asset files.
"""

from __future__ import annotations

from pathlib import Path

from ai_clip.core.config import LLMConfig
from ai_clip.core.models import (
    Intent,
    ProductProfile,
    Storyboard,
    Transcript,
    ViralAnalysis,
    VideoFormat,
)
from ai_clip.produce.formats import get_generator
from ai_clip.produce.formats.base import GenerateArgs


def generate_storyboard(
    project: str,
    theme: str,
    cfg: LLMConfig,
    fmt: VideoFormat = VideoFormat.talking_head,
    analysis: ViralAnalysis | None = None,
    transcript: Transcript | None = None,
    intent: Intent = Intent.info,
    stance: str = "",
    product: ProductProfile | None = None,
    research_markdown: str = "",
    duration_sec: float = 30.0,
    aspect_ratio: str = "9:16",
    n_shots: int = 6,
) -> Storyboard:
    args = GenerateArgs(
        project=project, theme=theme, cfg=cfg, analysis=analysis,
        transcript=transcript, intent=intent, stance=stance, product=product,
        research_markdown=research_markdown,
        duration_sec=duration_sec, aspect_ratio=aspect_ratio, n_shots=n_shots,
    )
    return get_generator(fmt)(args)


def write_storyboard_files(sb: Storyboard, prompts_dir: Path, md_path: Path) -> None:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for shot in sb.shots:
        stem = f"shot_{shot.index:02d}"
        if shot.image_prompt:
            (prompts_dir / f"{stem}_image.txt").write_text(
                shot.image_prompt, encoding="utf-8"
            )
        if shot.video_prompt:
            (prompts_dir / f"{stem}_video.txt").write_text(
                shot.video_prompt, encoding="utf-8"
            )
    md_path.write_text(_render_markdown(sb), encoding="utf-8")


def _render_markdown(sb: Storyboard) -> str:
    lines = [
        f"# Storyboard: {sb.project}",
        "",
        f"- Format: {sb.format}",
        f"- Theme: {sb.theme}",
        f"- Aspect ratio: {sb.aspect_ratio}",
        f"- Shots: {len(sb.shots)}",
        "",
    ]
    if sb.format != VideoFormat.remix:
        lines += [
            "Generate each asset (即梦 / Gemini / ComfyUI), save into `assets/` with "
            "the exact filename shown, then run `ai-clip assemble`.",
            "",
        ]
    else:
        lines += [
            "Remix: shots are cut from the source clip automatically; just run "
            "`ai-clip voiceover` then `ai-clip assemble`.",
            "",
        ]
    for shot in sb.shots:
        lines.append(f"## Shot {shot.index:02d} — {shot.shot_type} ({shot.duration_sec:g}s)")
        lines.append("")
        if shot.voiceover:
            lines += [f"**Voiceover:** {shot.voiceover}", ""]
        if shot.caption:
            lines += [f"**Caption:** {shot.caption}", ""]
        if shot.asset_engine:
            lines += [f"**Asset engine:** {shot.asset_engine}", ""]
        if shot.is_source_segment:
            lines += [f"**Source span:** {shot.source_start:g}s – {shot.source_end:g}s", ""]
        if shot.image_prompt:
            lines += [f"**Image prompt:**\n\n> {shot.image_prompt}", ""]
        if shot.video_prompt:
            lines += [f"**Video prompt:**\n\n> {shot.video_prompt}", ""]
        if shot.image_file or shot.video_file:
            saves = " / ".join(f"`assets/{f}`" for f in shot.expected_files())
            lines += [f"**Save as:** {saves}", ""]
    return "\n".join(lines)
