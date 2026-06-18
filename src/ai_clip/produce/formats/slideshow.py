"""Slideshow (图文): a sequence of image cards with on-screen captions and
narration — the 抖音图文 style."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs, asset_names, formula_block

SYSTEM = (
    "You design 图文 slideshow short videos: a few image cards, each with a short "
    "on-screen caption and a spoken line. The first card must hook hard. "
    "Write in the theme's language."
)

USER = """Design a slideshow short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} cards.
{formula}

Return ONLY JSON:
{{
  "cards": [
    {{"caption": "<short on-screen text>",
      "voiceover": "<spoken line>",
      "image_prompt": "<text-to-image prompt for the card>",
      "duration_sec": <float>}}
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
            formula=formula_block(args.analysis),
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
