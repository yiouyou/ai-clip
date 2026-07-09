"""Tool registry: each pipeline stage exposed as a uniform, named tool.

This is the seam for a future agent layer. The deterministic workflow stays the
backbone (pipeline.py); an agent — if/when one is added — would select and call
these tools rather than us rewriting the flow. Every tool takes (cfg, **params)
and returns a JSON-serializable result, so the same registry drives the CLI,
composed workflows, and an eventual agent tool-calling loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_clip import pipeline
from ai_clip.core.config import Config


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    params: dict[str, str]  # param name -> human description (schema seed)

    def __call__(self, cfg: Config, **kwargs: Any) -> Any:
        return self.func(cfg, **kwargs)


_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    if tool.name in _TOOLS:
        raise ValueError(f"duplicate tool: {tool.name}")
    _TOOLS[tool.name] = tool
    return tool


def get(name: str) -> Tool:
    if name not in _TOOLS:
        raise KeyError(f"unknown tool: {name}; available: {sorted(_TOOLS)}")
    return _TOOLS[name]


def all_tools() -> list[Tool]:
    return list(_TOOLS.values())


def invoke(name: str, cfg: Config, **kwargs: Any) -> Any:
    return get(name)(cfg, **kwargs)


register(Tool(
    "download", "Download a source clip from a URL via yt-dlp.",
    pipeline.run_download, {"project": "project id", "url": "source video URL"},
))
register(Tool(
    "extract", "Split audio and transcribe the clip with faster-whisper.",
    pipeline.run_extract, {"project": "project id"},
))
register(Tool(
    "analyze", "Reverse-engineer the viral formula from the transcript via LLM.",
    pipeline.run_analyze, {"project": "project id"},
))
register(Tool(
    "research", "Research source facts/context before storyboard.",
    pipeline.run_research, {"project": "project id", "theme": "optional research theme"},
))
register(Tool(
    "storyboard", "Generate a shot list with image/video prompts for a theme.",
    pipeline.run_storyboard,
    {"project": "project id", "theme": "video theme",
     "duration_sec": "target length", "n_shots": "number of shots"},
))
register(Tool(
    "source_draft", "Generate an original talking-head draft from a source transcript.",
    pipeline.run_source_draft, {"project": "project id", "intent": "info|emotion|sales"},
))
register(Tool(
    "assets", "Generate missing image assets with the configured provider.",
    pipeline.run_assets, {"project": "project id"},
))
register(Tool(
    "voiceover", "Synthesize per-shot narration via MiMo TTS (clones source voice).",
    pipeline.run_voiceover, {"project": "project id"},
))
register(Tool(
    "assemble", "Stitch collected assets (and voiceover) into the final MP4.",
    pipeline.run_assemble, {"project": "project id"},
))
register(Tool(
    "pair_review", "Run two-model review over a text artifact.",
    pipeline.run_pair_review,
    {"project": "project id", "artifact": "analysis|research|script|storyboard|source_draft|zack_draft"},
))
register(Tool(
    "pair_rewrite", "Revise a source/zack draft using a pair-review report.",
    pipeline.run_pair_rewrite,
    {
        "project": "project id",
        "artifact": "research|script|source_draft|zack_draft",
        "report": "PairReviewReport",
    },
))
register(Tool(
    "collect", "Collect daily-radar channel snapshots.",
    pipeline.run_collect,
    {"workflow": "daily-radar", "date": "YYYY-MM-DD", "force": "fetch channels again"},
))
register(Tool(
    "zack_ranking", "Rank daily-radar snapshots using Zack's topic-selection policy.",
    pipeline.run_zack_ranking, {"workflow": "daily-radar", "date": "YYYY-MM-DD", "top_n": "number of candidates"},
))
register(Tool(
    "source_content", "Fetch scripts/subtitles/transcripts for daily-radar candidates.",
    pipeline.run_source_content, {"workflow": "daily-radar", "date": "YYYY-MM-DD"},
))
register(Tool(
    "zack_selection", "Select one daily-radar topic before research and drafting.",
    pipeline.run_zack_selection, {"workflow": "daily-radar", "date": "YYYY-MM-DD"},
))
register(Tool(
    "source_research", "Research event details and safe framing for the selected daily-radar topic.",
    pipeline.run_source_research, {"workflow": "daily-radar", "date": "YYYY-MM-DD"},
))
register(Tool(
    "zack_draft", "Generate Zack's daily-radar topic brief and draft, using source-research when present.",
    pipeline.run_zack_draft, {"workflow": "daily-radar", "date": "YYYY-MM-DD"},
))
