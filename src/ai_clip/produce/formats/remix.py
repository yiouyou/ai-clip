"""Remix / 解说 (二创): keep the strongest spans of the SOURCE clip and write new
narration over them. No image generation — shots point at [start, end] of the
source video, which the assemble stage cuts directly."""

from __future__ import annotations

import math

from ai_clip.core import llm as llm_mod
from ai_clip.core.models import Shot, Storyboard, VideoFormat
from ai_clip.produce.formats.base import GenerateArgs
from ai_clip.produce.formats.prompts import (
    REMIX_SYSTEM,
    REMIX_USER,
    formula_block,
    intent_block,
    research_block,
)


def _segments_text(transcript) -> str:
    return "\n".join(
        f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in transcript.segments
    )


def generate(args: GenerateArgs) -> Storyboard:
    if not args.transcript or not args.transcript.segments:
        raise ValueError("remix format requires a transcript with segments")
    if args.duration_sec <= 0:
        raise ValueError("remix duration must be greater than zero")

    max_end = max(s.end for s in args.transcript.segments)
    target_duration = min(args.duration_sec, max_end)
    ratio = max(max_end / target_duration, 1.0)
    reply = llm_mod.chat(
        args.cfg,
        system=REMIX_SYSTEM,
        user=REMIX_USER.format(
            theme=args.theme, duration=target_duration, n_shots=args.n_shots,
            source_min=max_end / 60.0, ratio=ratio,
            intent=intent_block(args), formula=formula_block(args.analysis),
            research=research_block(args.research_markdown),
            segments=_segments_text(args.transcript),
        ),
    )
    data = llm_mod.extract_json(reply)
    shots = []
    for i, raw in enumerate(data.get("spans", []), start=1):
        try:
            start = float(raw.get("source_start", 0.0))
            end = float(raw.get("source_end", start))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = min(max(0.0, start), max_end)
        end = min(max_end, end)
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
    shots = _cap_total_duration(shots, target_duration)
    return Storyboard(
        project=args.project, format=VideoFormat.remix, theme=args.theme,
        source_clip_id=args.transcript.clip_id,
        aspect_ratio=args.aspect_ratio, target_duration_sec=target_duration, shots=shots,
    )


def _cap_total_duration(shots: list[Shot], target_duration: float) -> list[Shot]:
    """Proportionally shorten over-budget spans while preserving every selected moment."""
    total = sum(shot.duration_sec for shot in shots)
    if total <= target_duration or total <= 0:
        return shots

    scale = target_duration / total
    remaining = target_duration
    fitted = []
    for shot in shots:
        duration = min(shot.duration_sec * scale, remaining)
        duration = min(round(duration, 3), round(remaining, 3))
        if duration <= 0:
            continue
        end = round(float(shot.source_start) + duration, 3)
        actual_duration = round(end - float(shot.source_start), 3)
        fitted.append(
            shot.model_copy(
                update={
                    "source_end": end,
                    "duration_sec": actual_duration,
                }
            )
        )
        remaining = max(0.0, remaining - actual_duration)
    return fitted
