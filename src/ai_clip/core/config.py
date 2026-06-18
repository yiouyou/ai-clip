"""Configuration loading: YAML defaults overlaid with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "default.yaml"


class WhisperConfig(BaseModel):
    model_size: str = "medium"
    # "auto" lets device.py pick int8 (CPU) or float16 (GPU).
    compute_type: str = "auto"
    language: str | None = None


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"


class AssetsConfig(BaseModel):
    # auto -> comfyui when its service answers, else prompt_only.
    image_provider: str = "auto"
    video_provider: str = "prompt_only"
    comfyui_url: str = "http://127.0.0.1:8188"


class Config(BaseModel):
    data_dir: str = "./data"
    aspect_ratio: str = "9:16"
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)


def _apply_env(cfg: Config) -> Config:
    cfg.llm.base_url = os.getenv("AICLIP_LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.api_key = os.getenv("AICLIP_LLM_API_KEY", cfg.llm.api_key)
    cfg.llm.model = os.getenv("AICLIP_LLM_MODEL", cfg.llm.model)
    cfg.assets.comfyui_url = os.getenv("AICLIP_COMFYUI_URL", cfg.assets.comfyui_url)
    cfg.data_dir = os.getenv("AICLIP_DATA_DIR", cfg.data_dir)
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    source = Path(path) if path else _DEFAULT_CONFIG
    raw: dict = {}
    if source.exists():
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return _apply_env(Config.model_validate(raw))
