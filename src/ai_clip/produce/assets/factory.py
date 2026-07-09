"""Resolve which image provider to use.

auto -> ComfyUI when its service answers and a workflow template exists, else
fall back to prompt_only (human creates assets on a website).
"""

from __future__ import annotations

from pathlib import Path

from ai_clip.core.models import AssetEngine
from ai_clip.core.config import AssetsConfig
from ai_clip.produce.assets.comfyui import ComfyUIProvider
from ai_clip.produce.assets.prompt_only import PromptOnlyProvider
from ai_clip.produce.assets.smart_illustrator import SmartIllustratorProvider

_DEFAULT_WORKFLOW = (
    Path(__file__).resolve().parents[4] / "workflows" / "txt2img.json"
)


def resolve_image_provider(
    cfg: AssetsConfig,
    workflow_path: Path | None = None,
    engine: AssetEngine | None = None,
):
    workflow_path = workflow_path or _DEFAULT_WORKFLOW
    choice = cfg.image_provider
    hinted = False
    if engine in (AssetEngine.smart_illustrator, AssetEngine.gemini):
        choice = "smart_illustrator"
        hinted = cfg.image_provider == "auto"
    elif engine == AssetEngine.comfyui:
        choice = "comfyui"
        hinted = cfg.image_provider == "auto"
    elif engine == AssetEngine.manual:
        choice = "prompt_only"

    if choice == "prompt_only":
        return PromptOnlyProvider()

    if choice in ("comfyui", "auto"):
        usable = workflow_path.exists() and ComfyUIProvider.is_available(cfg.comfyui_url)
        if usable:
            return ComfyUIProvider.from_file(cfg.comfyui_url, workflow_path)
        if choice == "comfyui":
            if hinted:
                return PromptOnlyProvider()
            raise RuntimeError(
                f"comfyui provider requested but unavailable "
                f"(url={cfg.comfyui_url}, workflow={workflow_path})"
            )
        return PromptOnlyProvider()

    if choice == "smart_illustrator":
        if SmartIllustratorProvider.is_available(cfg):
            return SmartIllustratorProvider(cfg)
        if hinted:
            return PromptOnlyProvider()
        raise RuntimeError(
            "smart_illustrator provider requested but unavailable "
            "(set AICLIP_SMART_ILLUSTRATOR_DIR or AICLIP_SMART_ILLUSTRATOR_SCRIPT, "
            "and ensure npx/bun plus image API keys are configured)"
        )

    raise ValueError(f"unknown image_provider: {choice}")
