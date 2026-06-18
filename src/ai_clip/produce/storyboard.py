"""Storyboard step: theme (+ optional viral formula) -> Storyboard + prompt files.

The filename contract: shot N writes prompts/shot_NN_image.txt / _video.txt and
expects assets/shot_NN.png / shot_NN.mp4. assemble() relies only on these names,
so assets may come from ComfyUI's API or a human downloading from a website.
"""

from __future__ import annotations

from pathlib import Path

from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Shot, Storyboard, ViralAnalysis
from ai_clip.produce.prompts import (
    STORYBOARD_SYSTEM,
    STORYBOARD_USER_TEMPLATE,
    formula_block,
)


def generate_storyboard(
    project: str,
    theme: str,
    cfg: LLMConfig,
    analysis: ViralAnalysis | None = None,
    duration_sec: float = 30.0,
    aspect_ratio: str = "9:16",
    n_shots: int = 6,
) -> Storyboard:
    reply = llm_mod.chat(
        cfg,
        system=STORYBOARD_SYSTEM,
        user=STORYBOARD_USER_TEMPLATE.format(
            theme=theme,
            duration=duration_sec,
            aspect=aspect_ratio,
            n_shots=n_shots,
            formula_block=formula_block(analysis.formula if analysis else ""),
        ),
    )
    data = llm_mod.extract_json(reply)
    shots: list[Shot] = []
    for i, raw in enumerate(data.get("shots", []), start=1):
        stem = f"shot_{i:02d}"
        shots.append(
            Shot(
                index=i,
                duration_sec=float(raw.get("duration_sec", 3.0)),
                shot_type=str(raw.get("shot_type", "")),
                image_prompt=str(raw.get("image_prompt", "")),
                video_prompt=str(raw.get("video_prompt", "")),
                voiceover=str(raw.get("voiceover", "")),
                image_file=f"{stem}.png",
                video_file=f"{stem}.mp4",
            )
        )
    return Storyboard(
        project=project,
        theme=theme,
        source_clip_id=analysis.clip_id if analysis else None,
        aspect_ratio=aspect_ratio,
        shots=shots,
    )


def write_storyboard_files(sb: Storyboard, prompts_dir: Path, md_path: Path) -> None:
    """Drop per-shot prompt .txt files and a human-readable storyboard.md."""
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for shot in sb.shots:
        stem = f"shot_{shot.index:02d}"
        (prompts_dir / f"{stem}_image.txt").write_text(
            shot.image_prompt, encoding="utf-8"
        )
        (prompts_dir / f"{stem}_video.txt").write_text(
            shot.video_prompt, encoding="utf-8"
        )
    md_path.write_text(_render_markdown(sb), encoding="utf-8")


def _render_markdown(sb: Storyboard) -> str:
    lines = [
        f"# Storyboard: {sb.project}",
        "",
        f"- Theme: {sb.theme}",
        f"- Aspect ratio: {sb.aspect_ratio}",
        f"- Shots: {len(sb.shots)}",
        "",
        "Generate each asset (即梦 / Gemini / ComfyUI), then save the download "
        "into `assets/` with the exact filename shown, and run `ai-clip assemble`.",
        "",
    ]
    for shot in sb.shots:
        lines += [
            f"## Shot {shot.index:02d} — {shot.shot_type} ({shot.duration_sec:g}s)",
            "",
            f"**Voiceover:** {shot.voiceover}",
            "",
            f"**Image prompt:**\n\n> {shot.image_prompt}",
            "",
            f"**Video prompt:**\n\n> {shot.video_prompt}",
            "",
            f"**Save as:** `assets/{shot.image_file}` (and/or `assets/{shot.video_file}`)",
            "",
        ]
    return "\n".join(lines)
