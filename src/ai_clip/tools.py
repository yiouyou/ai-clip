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

from ai_clip.core.config import Config
from ai_clip.registry import REGISTRY


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


for spec in REGISTRY.stages():
    if spec.tool_name is None:
        continue
    if spec.run is None:
        raise ValueError(f"tool stage {spec.name!r} has no runner")
    register(Tool(
        name=spec.tool_name,
        description=spec.description,
        func=spec.run,
        params=dict(spec.tool_params),
    ))
