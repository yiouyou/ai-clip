"""Minimal OpenAI-compatible chat client over httpx.

Using raw HTTP (not the openai SDK) keeps deps light and makes the call trivial
to mock in tests. Works with DeepSeek/Qwen/Moonshot/OpenAI endpoints.
"""

from __future__ import annotations

import json
import re

import httpx

from ai_clip.core import billing
from ai_clip.core.config import LLMConfig


class LLMError(RuntimeError):
    pass


def chat(cfg: LLMConfig, system: str, user: str, timeout: float = 120.0) -> str:
    if not cfg.api_key:
        raise LLMError("LLM api_key is empty. Set AICLIP_LLM_API_KEY in your .env.")
    payload: dict = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature
    resp = httpx.post(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage") or {}
    billing.record_llm(
        cfg.model,
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"no JSON object found in LLM reply: {text[:200]}")
    return json.loads(candidate[start : end + 1])
