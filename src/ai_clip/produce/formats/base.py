"""Shared helpers for format generators."""

from __future__ import annotations

from dataclasses import dataclass

from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Transcript, ViralAnalysis


@dataclass
class GenerateArgs:
    project: str
    theme: str
    cfg: LLMConfig
    analysis: ViralAnalysis | None = None
    transcript: Transcript | None = None  # used by remix
    duration_sec: float = 30.0
    aspect_ratio: str = "9:16"
    n_shots: int = 6


def formula_block(analysis: ViralAnalysis | None) -> str:
    if not analysis or not analysis.formula:
        return "No reference formula; design an original structure with a strong hook."
    return f"Apply this proven viral formula to the new theme:\n{analysis.formula}"


def asset_names(index: int) -> tuple[str, str]:
    stem = f"shot_{index:02d}"
    return f"{stem}.png", f"{stem}.mp4"
