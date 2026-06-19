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
    # Full base incl. /v1 (OpenAI convention). DeepSeek: https://api.deepseek.com/v1
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    # DeepSeek: deepseek-v4-pro | OpenAI: gpt-5.5 (set to match base_url).
    model: str = "deepseek-v4-pro"


class AssetsConfig(BaseModel):
    # auto -> comfyui when its service answers, else prompt_only.
    image_provider: str = "auto"
    video_provider: str = "prompt_only"
    comfyui_url: str = "http://127.0.0.1:8188"


class TTSConfig(BaseModel):
    base_url: str = "https://api.xiaomimimo.com/v1"
    api_key: str = ""
    # mimo-v2.5-tts (preset) | mimo-v2.5-tts-voiceclone | mimo-v2.5-tts-voicedesign
    model: str = "mimo-v2.5-tts-voiceclone"
    voice: str = "Chloe"  # preset voice id, used when not cloning
    clone_from_source: bool = True  # clone the source clip's speaker by default
    reference_seconds: float = 10.0  # length of reference snippet for cloning


class ProduceConfig(BaseModel):
    # External produce backend endpoints (optional alternatives to self-built).
    moneyprinter_url: str = "http://127.0.0.1:8080"


class Config(BaseModel):
    data_dir: str = "./data"
    aspect_ratio: str = "9:16"
    burn_captions: bool = False  # burn caption/voiceover text into the video
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    produce: ProduceConfig = Field(default_factory=ProduceConfig)


def _resolve_llm_key(base_url: str) -> str:
    """Pick the provider-specific key from the env based on the endpoint.

    Lets a standard .env (DEEPSEEK_API_KEY / OPENAI_API_KEY / ...) work without
    an AICLIP_LLM_API_KEY override.
    """
    url = base_url.lower()
    if "deepseek" in url:
        return os.getenv("DEEPSEEK_API_KEY", "")
    if "openai" in url:
        return os.getenv("OPENAI_API_KEY", "")
    if "moonshot" in url:
        return os.getenv("MOONSHOT_API_KEY", "")
    if "dashscope" in url or "qwen" in url:
        return os.getenv("DASHSCOPE_API_KEY", "")
    return ""


def _apply_env(cfg: Config) -> Config:
    cfg.llm.base_url = os.getenv("AICLIP_LLM_BASE_URL", cfg.llm.base_url)
    cfg.llm.model = os.getenv("AICLIP_LLM_MODEL", cfg.llm.model)
    cfg.llm.api_key = (
        os.getenv("AICLIP_LLM_API_KEY")
        or _resolve_llm_key(cfg.llm.base_url)
        or cfg.llm.api_key
    )

    cfg.tts.base_url = os.getenv("AICLIP_TTS_BASE_URL", cfg.tts.base_url)
    cfg.tts.model = os.getenv("AICLIP_TTS_MODEL", cfg.tts.model)
    cfg.tts.api_key = os.getenv("MIMO_API_KEY", cfg.tts.api_key)

    cfg.assets.comfyui_url = os.getenv("AICLIP_COMFYUI_URL", cfg.assets.comfyui_url)
    cfg.produce.moneyprinter_url = os.getenv(
        "AICLIP_MPT_URL", cfg.produce.moneyprinter_url
    )
    cfg.data_dir = os.getenv("AICLIP_DATA_DIR", cfg.data_dir)
    return cfg


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, without overriding existing env."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_product(path: str | Path | None):
    """Load a reusable product profile (YAML) for the sales intent, or None."""
    from ai_clip.core.models import ProductProfile

    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"product profile not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ProductProfile.model_validate(data)


def load_config(path: str | Path | None = None) -> Config:
    load_dotenv()
    source = Path(path) if path else _DEFAULT_CONFIG
    raw: dict = {}
    if source.exists():
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return _apply_env(Config.model_validate(raw))
