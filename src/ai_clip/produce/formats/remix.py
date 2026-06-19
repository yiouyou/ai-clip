"""Remix / 解说 (二创): keep the strongest spans of the SOURCE clip and write new
narration over them. No image generation — shots point at [start, end] of the
source video, which the assemble stage cuts directly."""

from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs
from ai_clip.produce.formats.prompts import (
    REMIX_SYSTEM,
    REMIX_USER,
    formula_block,
    intent_block,
)


def _segments_text(transcript) -> str:
    return "\n".join(
        f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments
    )


def generate(args: GenerateArgs) -> Storyboard:
    if not args.transcript or not args.transcript.segments:
        raise ValueError("remix format requires a transcript with segments")

    max_end = max(s.end for s in args.transcript.segments)
    ratio = max(max_end / args.duration_sec, 1.0) if args.duration_sec else 1.0
    reply = llm_mod.chat(
        args.cfg,
        system=REMIX_SYSTEM,
        user=REMIX_USER.format(
            theme=args.theme, duration=args.duration_sec, n_shots=args.n_shots,
            source_min=max_end / 60.0, ratio=ratio,
            intent=intent_block(args), formula=formula_block(args.analysis),
            segments=_segments_text(args.transcript),
        ),
    )
    data = llm_mod.extract_json(reply)
    shots = []
    for i, raw in enumerate(data.get("spans", []), start=1):
        start = max(0.0, float(raw.get("source_start", 0.0)))
        end = min(max_end, float(raw.get("source_end", start)))
        if end <= start:
            continue
        shots.append(Shot(
            index=i,
            duration_sec=round(end - start, 3),
            shot_type="source",
            voiceover=str(raw.get("voiceover", "")),
            source_start=start,
            source_end=end,
        ))
    return Storyboard(
        project=args.project, format=VideoFormat.remix, theme=args.theme,
        source_clip_id=args.transcript.clip_id,
        aspect_ratio=args.aspect_ratio, shots=shots,
    )
