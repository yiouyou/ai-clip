"""Shared helpers for format generators."""

from __future__ import annotations

from dataclasses import dataclass

from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Intent, ProductProfile, Transcript, ViralAnalysis


@dataclass
class GenerateArgs:
    project: str
    theme: str
    cfg: LLMConfig
    analysis: ViralAnalysis | None = None
    transcript: Transcript | None = None  # used by remix
    intent: Intent = Intent.info
    stance: str = ""  # optional override for the emotion intent
    product: ProductProfile | None = None  # required by the sales intent
    duration_sec: float = 30.0
    aspect_ratio: str = "9:16"
    n_shots: int = 6


def formula_block(analysis: ViralAnalysis | None) -> str:
    if not analysis or not analysis.formula:
        return "No reference formula; design an original structure with a strong hook."
    return f"Apply this proven viral formula to the new theme:\n{analysis.formula}"


def intent_block(args: GenerateArgs) -> str:
    """Intent-specific direction injected into every format's prompt."""
    if args.intent == Intent.emotion:
        stance = args.stance or (args.analysis.stance if args.analysis else "")
        lead = f"Take this stance: {stance}." if stance else "Pick a strong, resonant stance."
        return (
            f"INTENT=emotion (opinionated take, NOT neutral reporting). {lead} "
            "Open by asserting an attitude/contrarian opinion; keep an emotional "
            "charge throughout; end on an emotional punch or rhetorical question, "
            "not a neutral summary."
        )
    if args.intent == Intent.sales:
        p = args.product
        if not p or not p.name:
            return (
                "INTENT=sales but no product profile provided; write a generic "
                "pain -> agitate -> product -> proof -> CTA structure."
            )
        sp = "; ".join(p.selling_points)
        return (
            f"INTENT=sales for product '{p.name}': {p.description}. "
            f"Audience: {p.audience}. Selling points: {sp}. "
            f"Structure: hook on the audience pain -> agitate -> reveal {p.name} -> "
            f"proof/selling points -> CTA: {p.cta}."
        )
    return "INTENT=info: lead with a counter-intuitive hook, deliver the key point, end with a takeaway."


def asset_names(index: int) -> tuple[str, str]:
    stem = f"shot_{index:02d}"
    return f"{stem}.png", f"{stem}.mp4"
