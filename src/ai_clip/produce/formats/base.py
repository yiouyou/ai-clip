"""Shared dataclass + helpers for format generators.

Prompt text (SYSTEM/USER per format) and the shared intent_block/formula_block
fragments live in prompts.py; this module holds only data/logic helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_clip.core.config import LLMConfig
from ai_clip.core.models import AssetEngine, Intent, ProductProfile, Transcript, ViralAnalysis


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
    research_markdown: str = ""
    duration_sec: float = 30.0
    aspect_ratio: str = "9:16"
    n_shots: int = 6


def asset_names(index: int) -> tuple[str, str]:
    stem = f"shot_{index:02d}"
    return f"{stem}.png", f"{stem}.mp4"


def parse_asset_engine(raw: object) -> AssetEngine | None:
    if not raw:
        return None
    try:
        return AssetEngine(str(raw))
    except ValueError:
        return None
