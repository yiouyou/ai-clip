"""Per-project cost accounting.

Every LLM call (and TTS synthesis) appends one line to data/<project>/cost.jsonl.
`ai-clip cost -p P` sums it. A context manager tags each call with its stage so
the breakdown shows where the spend went. Local steps (whisper, ffmpeg, ComfyUI)
are free and not metered.

Prices are USD; EDIT the tables below to match your provider's plan.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

# USD per 1,000,000 tokens: model -> (input, output). Edit to match your plan.
LLM_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-v4-pro": (0.28, 0.42),
    "gpt-5.5": (1.25, 10.00),
}
_LLM_FALLBACK = (0.0, 0.0)

# USD per 1,000,000 characters for TTS providers.
TTS_PRICES: dict[str, float] = {"mimo": 0.0}

_ctx: ContextVar[tuple[Path, str] | None] = ContextVar("aiclip_billing", default=None)


@contextmanager
def account(project_dir: str | Path, stage: str):
    """Tag all metered calls inside this block with (project_dir, stage)."""
    token = _ctx.set((Path(project_dir), stage))
    try:
        yield
    finally:
        _ctx.reset(token)


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    cin, cout = LLM_PRICES.get(model, _LLM_FALLBACK)
    return input_tokens / 1e6 * cin + output_tokens / 1e6 * cout


def tts_cost(provider: str, chars: int) -> float:
    return chars / 1e6 * TTS_PRICES.get(provider, 0.0)


def _append(item: dict) -> None:
    ctx = _ctx.get()
    if ctx is None:
        return  # not inside an accounting block (e.g. unit tests) -> no-op
    project_dir, stage = ctx
    project_dir.mkdir(parents=True, exist_ok=True)
    item = {"ts": round(time.time(), 3), "stage": stage, **item}
    with (project_dir / "cost.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def record_llm(model: str, input_tokens: int, output_tokens: int) -> None:
    _append({
        "kind": "llm", "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost": round(llm_cost(model, input_tokens, output_tokens), 6),
    })


def record_tts(provider: str, chars: int) -> None:
    _append({
        "kind": "tts", "model": provider, "chars": chars,
        "cost": round(tts_cost(provider, chars), 6),
    })


def summarize(project_dir: str | Path) -> dict:
    """Read cost.jsonl and return totals grouped by stage and by model."""
    path = Path(project_dir) / "cost.jsonl"
    by_stage: dict[str, float] = {}
    by_model: dict[str, float] = {}
    totals = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "chars": 0, "calls": 0}
    if not path.exists():
        return {"total": totals, "by_stage": by_stage, "by_model": by_model, "items": []}
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        items.append(it)
        c = it.get("cost", 0.0)
        by_stage[it.get("stage", "")] = round(by_stage.get(it.get("stage", ""), 0.0) + c, 6)
        by_model[it.get("model", "")] = round(by_model.get(it.get("model", ""), 0.0) + c, 6)
        totals["cost"] = round(totals["cost"] + c, 6)
        totals["input_tokens"] += it.get("input_tokens", 0)
        totals["output_tokens"] += it.get("output_tokens", 0)
        totals["chars"] += it.get("chars", 0)
        totals["calls"] += 1
    return {"total": totals, "by_stage": by_stage, "by_model": by_model, "items": items}
