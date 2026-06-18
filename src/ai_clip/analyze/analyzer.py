"""Analyze stage: turn a transcript into a reusable viral formula via an LLM."""

from __future__ import annotations

from ai_clip.analyze.prompts import ANALYZE_SYSTEM, ANALYZE_USER_TEMPLATE
from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Transcript, ViralAnalysis


def analyze(transcript: Transcript, cfg: LLMConfig) -> ViralAnalysis:
    if not transcript.text.strip():
        raise ValueError("transcript is empty; run extract first")

    reply = llm_mod.chat(
        cfg,
        system=ANALYZE_SYSTEM,
        user=ANALYZE_USER_TEMPLATE.format(transcript=transcript.text),
    )
    data = llm_mod.extract_json(reply)
    return ViralAnalysis(
        clip_id=transcript.clip_id,
        hook=str(data.get("hook", "")),
        structure=[str(x) for x in data.get("structure", [])],
        emotion_curve=[str(x) for x in data.get("emotion_curve", [])],
        formula=str(data.get("formula", "")),
        scores={k: float(v) for k, v in (data.get("scores") or {}).items()},
        notes=str(data.get("notes", "")),
    )
