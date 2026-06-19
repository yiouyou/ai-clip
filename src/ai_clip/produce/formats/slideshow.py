"""Slideshow (图文): a sequence of image cards with on-screen captions and
narration — the 抖音图文 style."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs, asset_names
from ai_clip.produce.formats.prompts import (
    SLIDESHOW_SYSTEM,
    SLIDESHOW_USER,
    formula_block,
    intent_block,
)


def generate(args: GenerateArgs) -> Storyboard:
    reply = llm_mod.chat(
        args.cfg,
        system=SLIDESHOW_SYSTEM,
        user=SLIDESHOW_USER.format(
            theme=args.theme, duration=args.duration_sec,
            aspect=args.aspect_ratio, n_shots=args.n_shots,
            intent=intent_block(args), formula=formula_block(args.analysis),
        ),
    )
    data = llm_mod.extract_json(reply)
    shots = []
    for i, raw in enumerate(data.get("cards", []), start=1):
        png, _ = asset_names(i)
        shots.append(Shot(
            index=i,
            duration_sec=float(raw.get("duration_sec", 3.0)),
            shot_type="card",
            image_prompt=str(raw.get("image_prompt", "")),
            caption=str(raw.get("caption", "")),
            voiceover=str(raw.get("voiceover", "")),
            image_file=png,
        ))
    return Storyboard(
        project=args.project, format=VideoFormat.slideshow, theme=args.theme,
        source_clip_id=args.analysis.clip_id if args.analysis else None,
        aspect_ratio=args.aspect_ratio, shots=shots,
    )
