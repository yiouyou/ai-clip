"""Talking-head (口播): narration-driven. Each shot is one spoken line with an
optional b-roll still. No heavy AI video generation required."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs, asset_names
from ai_clip.produce.formats.prompts import (
    TALKING_HEAD_SYSTEM,
    TALKING_HEAD_USER,
    formula_block,
    intent_block,
)


def generate(args: GenerateArgs) -> Storyboard:
    reply = llm_mod.chat(
        args.cfg,
        system=TALKING_HEAD_SYSTEM,
        user=TALKING_HEAD_USER.format(
            theme=args.theme, duration=args.duration_sec,
            aspect=args.aspect_ratio, n_shots=args.n_shots,
            intent=intent_block(args), formula=formula_block(args.analysis),
        ),
    )
    data = llm_mod.extract_json(reply)
    shots = []
    for i, raw in enumerate(data.get("lines", []), start=1):
        png, _ = asset_names(i)
        broll = str(raw.get("broll_prompt", "")).strip()
        shots.append(Shot(
            index=i,
            duration_sec=float(raw.get("duration_sec", 4.0)),
            shot_type="broll",
            image_prompt=broll,
            voiceover=str(raw.get("voiceover", "")),
            image_file=png if broll else "",
        ))
    return Storyboard(
        project=args.project, format=VideoFormat.talking_head, theme=args.theme,
        source_clip_id=args.analysis.clip_id if args.analysis else None,
        aspect_ratio=args.aspect_ratio, shots=shots,
    )
