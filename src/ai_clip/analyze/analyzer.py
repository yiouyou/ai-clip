"""Analyze stage: turn a transcript into a reusable viral formula via an LLM,
steered by intent (info / emotion / sales)."""

from __future__ import annotations

from ai_clip.analyze.prompts import ANALYZE_SYSTEM, build_user_prompt
from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Intent, Transcript, ViralAnalysis


def analyze(
    transcript: Transcript,
    cfg: LLMConfig,
    intent: Intent = Intent.info,
) -> ViralAnalysis:
    if not transcript.text.strip():
        raise ValueError("transcript is empty; run extract first")

    reply = llm_mod.chat(
        cfg,
        system=ANALYZE_SYSTEM,
        user=build_user_prompt(transcript.text, intent),
    )
    data = llm_mod.extract_json(reply)
    return ViralAnalysis(
        clip_id=transcript.clip_id,
        intent=intent,
        hook=str(data.get("hook", "")),
        structure=[str(x) for x in data.get("structure", [])],
        emotion_curve=[str(x) for x in data.get("emotion_curve", [])],
        formula=str(data.get("formula", "")),
        scores={k: float(v) for k, v in (data.get("scores") or {}).items()},
        notes=str(data.get("notes", "")),
        stance=str(data.get("stance", "")),
        pain_points=[str(x) for x in data.get("pain_points", [])],
        objections=[str(x) for x in data.get("objections", [])],
    )
