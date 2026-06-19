"""Talking-head (口播): narration-driven. Each shot is one spoken line with an
optional b-roll still. No heavy AI video generation required."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import (
    GenerateArgs,
    asset_names,
    formula_block,
    intent_block,
)

SYSTEM = (
    "You are a short-video scriptwriter for talking-head / voiceover videos. "
    "You write punchy spoken narration with a strong 3-second hook, and suggest "
    "simple b-roll stills to show while each line is spoken. Write in the theme's language."
)

USER = """Write a talking-head short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} lines.
{intent}
{formula}

Return ONLY JSON:
{{
  "lines": [
    {{"voiceover": "<one spoken line>",
      "duration_sec": <float>,
      "broll_prompt": "<optional text-to-image prompt for a b-roll still, or empty>"}}
  ]
}}
"""


def generate(args: GenerateArgs) -> Storyboard:
    reply = llm_mod.chat(
        args.cfg,
        system=SYSTEM,
        user=USER.format(
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
