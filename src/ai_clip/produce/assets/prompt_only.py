"""Human-in-the-loop provider: generates nothing, the prompt files written by the
storyboard step are the deliverable. The user creates assets on a website
(已购 plan: 即梦 / Gemini) and drops them into assets/ using the contract names."""

from __future__ import annotations

from pathlib import Path

from ai_clip.core.models import Shot


class PromptOnlyProvider:
    name = "prompt_only"

    def cache_params(self) -> dict[str, str]:
        return {"provider": self.name}

    def generate(self, shot: Shot, assets_dir: Path) -> None:  # noqa: ARG002
        return None
