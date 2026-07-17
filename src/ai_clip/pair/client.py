from __future__ import annotations

from dataclasses import dataclass

from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig, PairConfig


@dataclass(frozen=True)
class ReviewModel:
    model: str
    base_url: str
    api_key: str
    max_attempts: int = 3


def configured_models(cfg: PairConfig) -> list[ReviewModel]:
    models: list[ReviewModel] = []
    if cfg.base_url and cfg.api_key:
        models.extend(
            ReviewModel(
                model=m,
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                max_attempts=cfg.max_attempts,
            )
            for m in cfg.models
        )
    if cfg.deepseek_api_key:
        models.extend(
            ReviewModel(
                model=m,
                base_url=cfg.deepseek_base_url,
                api_key=cfg.deepseek_api_key,
                max_attempts=cfg.max_attempts,
            )
            for m in cfg.deepseek_models
        )

    seen: set[tuple[str, str]] = set()
    deduped: list[ReviewModel] = []
    for item in models:
        key = (item.base_url.rstrip("/"), item.model)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def chat(model: ReviewModel, system: str, user: str, timeout: float) -> str:
    return llm_mod.chat(
        LLMConfig(
            base_url=model.base_url,
            api_key=model.api_key,
            model=model.model,
            max_attempts=model.max_attempts,
        ),
        system=system,
        user=user,
        timeout=timeout,
    )
