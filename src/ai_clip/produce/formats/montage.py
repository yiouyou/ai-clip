"""Montage (AI 漫剧/短剧): fully generated multi-shot drama with per-shot image
and image-to-video prompts."""

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
    "You are a short-video director. You turn a topic into a concrete shot list "
    "for a vertical short video, with vivid self-contained image and image-to-video "
    "prompts. Write prompts in the theme's language."
)

USER = """Create a storyboard for a short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} shots.
{intent}
{formula}

Return ONLY JSON:
{{
  "shots": [
    {{"duration_sec": <float>, "shot_type": "<close-up|wide|...>",
      "image_prompt": "<text-to-image prompt for the key frame>",
      "video_prompt": "<image-to-video prompt referencing that frame>",
      "voiceover": "<one line of narration>"}}
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
    for i, raw in enumerate(data.get("shots", []), start=1):
        png, mp4 = asset_names(i)
        shots.append(Shot(
            index=i,
            duration_sec=float(raw.get("duration_sec", 3.0)),
            shot_type=str(raw.get("shot_type", "")),
            image_prompt=str(raw.get("image_prompt", "")),
            video_prompt=str(raw.get("video_prompt", "")),
            voiceover=str(raw.get("voiceover", "")),
            image_file=png, video_file=mp4,
        ))
    return Storyboard(
        project=args.project, format=VideoFormat.montage, theme=args.theme,
        source_clip_id=args.analysis.clip_id if args.analysis else None,
        aspect_ratio=args.aspect_ratio, shots=shots,
    )
