from __future__ import annotations

from ai_clip.core.config import Config
from ai_clip.radar.models import ZackSelection

_RISK_LEVELS = {"low": 0, "medium": 1, "high": 2}


def automatic_research_searches(selection: ZackSelection, cfg: Config) -> int:
    if not cfg.radar.auto_research or not cfg.source_research.tavily_api_key:
        return 0
    risk = _RISK_LEVELS.get(selection.fact_risk, 1)
    threshold = _RISK_LEVELS.get(cfg.radar.auto_research_min_risk, 2)
    if risk < threshold:
        return 0
    risk_cap = 2 if risk >= _RISK_LEVELS["high"] else 1
    return max(
        0,
        min(
            cfg.source_research.max_searches,
            cfg.radar.auto_research_max_searches,
            risk_cap,
            3,
        ),
    )
