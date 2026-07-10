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
from datetime import datetime
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

# USD per 1,000,000 tokens: model -> (input, output). Edit to match your plan.
# Rates as of June 2026 (verify on the provider's pricing page):
#   deepseek-v4-pro: current official $0.435/$0.87 (standard rate $1.74/$3.48)
#   gpt-5.5:         standard $5.00/$30.00 (batch $2.50/$15.00, cached input $0.50)
LLM_PRICES: dict[str, tuple[float, float]] = {
    "deepseek-v4-pro": (0.435, 0.87),
    "gpt-5.5": (5.00, 30.00),
}
_LLM_FALLBACK = (0.0, 0.0)

# USD per 1,000,000 characters for TTS providers.
# NOTE: MiMo's TTS price is not publicly itemized (TTS was promo-free, then folded
# into the MiMo V2.5 series billing on 2026-06-18). This is an ESTIMATE based on the
# MiMo V2.5 ~$1 / 1M input rate; confirm and adjust for your plan.
TTS_PRICES: dict[str, float] = {"mimo": 1.0}

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


def record_search(provider: str, query: str, results: int) -> None:
    _append({
        "kind": "search",
        "model": provider,
        "query": query,
        "results": results,
        "cost": 0.0,
    })


def summarize(project_dir: str | Path, since: str | float | None = None) -> dict:
    """Read cost.jsonl and return totals grouped by stage and by model."""
    path = Path(project_dir) / "cost.jsonl"
    by_stage: dict[str, float] = {}
    by_model: dict[str, float] = {}
    by_kind: dict[str, int] = {}
    totals = {
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "chars": 0,
        "calls": 0,
        "searches": 0,
    }
    if not path.exists():
        return {
            "total": totals,
            "by_stage": by_stage,
            "by_model": by_model,
            "by_kind": by_kind,
            "items": [],
        }
    since_ts = _since_timestamp(since)
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if float(it.get("ts", 0.0)) < since_ts:
            continue
        items.append(it)
        c = it.get("cost", 0.0)
        by_stage[it.get("stage", "")] = round(by_stage.get(it.get("stage", ""), 0.0) + c, 6)
        by_model[it.get("model", "")] = round(by_model.get(it.get("model", ""), 0.0) + c, 6)
        totals["cost"] = round(totals["cost"] + c, 6)
        totals["input_tokens"] += it.get("input_tokens", 0)
        totals["output_tokens"] += it.get("output_tokens", 0)
        totals["chars"] += it.get("chars", 0)
        totals["calls"] += 1
        kind = str(it.get("kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if kind == "search":
            totals["searches"] += 1
    return {
        "total": totals,
        "by_stage": by_stage,
        "by_model": by_model,
        "by_kind": by_kind,
        "items": items,
    }


def _since_timestamp(value: str | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0.0
