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
from ai_clip.core.retry import (
    CallOutcome,
    ExternalCallError,
    FailureCategory,
    RetryPolicy,
    run_with_retry,
)


class LLMError(ExternalCallError):
    pass


def chat(cfg: LLMConfig, system: str, user: str, timeout: float = 120.0) -> str:
    if not cfg.api_key:
        raise LLMError(
            "LLM api_key is empty. Set AICLIP_LLM_API_KEY in your .env.",
            service="llm",
            operation="chat",
            category=FailureCategory.CONFIGURATION,
        )
    payload: dict = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if cfg.temperature is not None:
        payload["temperature"] = cfg.temperature
    outcome = _post_chat(cfg, payload, timeout)
    data = outcome.value
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    billing.record_llm(
        cfg.model,
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
        attempts=outcome.attempts,
    )
    return data["choices"][0]["message"]["content"]


def _post_chat(cfg: LLMConfig, payload: dict, timeout: float) -> CallOutcome[dict]:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    def request() -> dict:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        _validate_chat_response(data)
        return data

    return run_with_retry(
        request,
        service="llm",
        operation_name=f"chat model={cfg.model}",
        policy=RetryPolicy(
            max_attempts=cfg.max_attempts,
            retry_categories=frozenset({
                FailureCategory.RATE_LIMIT,
                FailureCategory.TIMEOUT,
                FailureCategory.TRANSIENT,
            }),
        ),
        error_type=LLMError,
    )


def _validate_chat_response(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("chat response is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("chat response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise ValueError("chat response has no message")
    if not isinstance(first["message"].get("content"), str):
        raise ValueError("chat response content is not text")


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM reply (handles ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(
            f"no JSON object found in LLM reply: {text[:200]}",
            service="llm",
            operation="parse-json",
            category=FailureCategory.INVALID_RESPONSE,
        )
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(
            "invalid JSON object in LLM reply",
            service="llm",
            operation="parse-json",
            category=FailureCategory.INVALID_RESPONSE,
        ) from exc
