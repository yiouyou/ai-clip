"""Montage (AI 漫剧/短剧): fully generated multi-shot drama with per-shot image
and image-to-video prompts."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs, asset_names, parse_asset_engine
from ai_clip.produce.formats.prompts import (
    MONTAGE_SYSTEM,
    MONTAGE_USER,
    asset_engine_block,
    formula_block,
    intent_block,
    research_block,
)


def generate(args: GenerateArgs) -> Storyboard:
    reply = llm_mod.chat(
        args.cfg,
        system=MONTAGE_SYSTEM,
        user=MONTAGE_USER.format(
            theme=args.theme, duration=args.duration_sec,
            aspect=args.aspect_ratio, n_shots=args.n_shots,
            intent=intent_block(args), formula=formula_block(args.analysis),
            research=research_block(args.research_markdown),
            asset_engine=asset_engine_block(),
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
            asset_engine=parse_asset_engine(raw.get("asset_engine")),
            voiceover=str(raw.get("voiceover", "")),
            image_file=png, video_file=mp4,
        ))
    return Storyboard(
        project=args.project, format=VideoFormat.montage, theme=args.theme,
        source_clip_id=args.analysis.clip_id if args.analysis else None,
        aspect_ratio=args.aspect_ratio, target_duration_sec=args.duration_sec, shots=shots,
    )
