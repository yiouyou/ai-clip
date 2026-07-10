"""Configuration loading: YAML defaults overlaid with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

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
    # Omitted from the request when None (GPT-5 line rejects non-default values).
    temperature: float | None = None


class AssetsConfig(BaseModel):
    # auto -> comfyui when its service answers, else prompt_only.
    image_provider: str = "auto"
    video_provider: str = "prompt_only"
    comfyui_url: str = "http://127.0.0.1:8188"
    smart_illustrator_dir: str = ""
    smart_illustrator_script: str = ""
    smart_illustrator_provider: str = ""
    smart_illustrator_model: str = ""
    smart_illustrator_candidates: int = 1


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


class PairConfig(BaseModel):
    # OpenAI-compatible review endpoint. NEWAPP_* is applied from .env.
    base_url: str = ""
    api_key: str = ""
    models: list[str] = Field(default_factory=lambda: [
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-opus-4.8",
    ])
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_models: list[str] = Field(default_factory=lambda: ["deepseek-4-pro"])
    timeout: float = 120.0


class SourceResearchConfig(BaseModel):
    tavily_api_key: str = ""
    max_searches: int = 2
    max_results: int = 5
    search_depth: str = "basic"
    timeout: float = 30.0


class RadarConfig(BaseModel):
    channels_path: str = "config/channels.yaml"
    feedback_path: str = "config/radar-feedback.yaml"
    top_n: int = 3
    channel_limit: int = 20
    channel_timeout_sec: int = 60
    channel_workers: int = 4
    since_days: int = 2
    max_duration_sec: float = 900.0
    timezone: str = "Asia/Shanghai"
    transcribe_missing: bool = True
    transcribe_model_size: str = "small"


class Config(BaseModel):
    data_dir: str = "./data"
    aspect_ratio: str = "9:16"
    burn_captions: bool = False  # burn caption/voiceover text into the video
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    produce: ProduceConfig = Field(default_factory=ProduceConfig)
    pair: PairConfig = Field(default_factory=PairConfig)
    source_research: SourceResearchConfig = Field(default_factory=SourceResearchConfig)
    radar: RadarConfig = Field(default_factory=RadarConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data):
        if isinstance(data, dict) and "radar" not in data and "scout" in data:
            data = dict(data)
            data["radar"] = data["scout"]
        return data


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
    cfg.assets.smart_illustrator_dir = os.getenv(
        "AICLIP_SMART_ILLUSTRATOR_DIR", cfg.assets.smart_illustrator_dir
    )
    cfg.assets.smart_illustrator_script = os.getenv(
        "AICLIP_SMART_ILLUSTRATOR_SCRIPT", cfg.assets.smart_illustrator_script
    )
    cfg.assets.smart_illustrator_provider = os.getenv(
        "AICLIP_SMART_ILLUSTRATOR_PROVIDER", cfg.assets.smart_illustrator_provider
    )
    cfg.assets.smart_illustrator_model = os.getenv(
        "AICLIP_SMART_ILLUSTRATOR_MODEL", cfg.assets.smart_illustrator_model
    )
    cfg.produce.moneyprinter_url = os.getenv(
        "AICLIP_MPT_URL", cfg.produce.moneyprinter_url
    )
    cfg.pair.base_url = os.getenv("NEWAPP_URL", cfg.pair.base_url)
    cfg.pair.api_key = os.getenv("NEWAPP_API_KEY", cfg.pair.api_key)
    cfg.pair.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", cfg.pair.deepseek_api_key)
    cfg.source_research.tavily_api_key = os.getenv(
        "TAVILY_API_KEY", cfg.source_research.tavily_api_key
    )
    if "AICLIP_SOURCE_RESEARCH_MAX_SEARCHES" in os.environ:
        cfg.source_research.max_searches = _env_int(
            os.getenv("AICLIP_SOURCE_RESEARCH_MAX_SEARCHES", ""),
            cfg.source_research.max_searches,
        )
    if "AICLIP_SOURCE_RESEARCH_MAX_RESULTS" in os.environ:
        cfg.source_research.max_results = _env_int(
            os.getenv("AICLIP_SOURCE_RESEARCH_MAX_RESULTS", ""),
            cfg.source_research.max_results,
        )
    cfg.radar.channels_path = (
        os.getenv("AICLIP_RADAR_CHANNELS")
        or os.getenv("AICLIP_SCOUT_CHANNELS")
        or cfg.radar.channels_path
    )
    transcribe_missing = os.getenv("AICLIP_RADAR_TRANSCRIBE_MISSING")
    if transcribe_missing is None:
        transcribe_missing = os.getenv("AICLIP_SCOUT_TRANSCRIBE_MISSING")
    if transcribe_missing is not None:
        cfg.radar.transcribe_missing = _env_bool(
            transcribe_missing,
            cfg.radar.transcribe_missing,
        )
    cfg.radar.transcribe_model_size = (
        os.getenv("AICLIP_RADAR_TRANSCRIBE_MODEL_SIZE")
        or os.getenv("AICLIP_SCOUT_TRANSCRIBE_MODEL_SIZE")
        or cfg.radar.transcribe_model_size
    )
    channel_timeout = os.getenv("AICLIP_RADAR_CHANNEL_TIMEOUT_SEC")
    if channel_timeout is None:
        channel_timeout = os.getenv("AICLIP_SCOUT_CHANNEL_TIMEOUT_SEC")
    if channel_timeout is not None:
        cfg.radar.channel_timeout_sec = _env_int(
            channel_timeout,
            cfg.radar.channel_timeout_sec,
        )
    if "AICLIP_RADAR_CHANNEL_WORKERS" in os.environ:
        cfg.radar.channel_workers = _env_int(
            os.getenv("AICLIP_RADAR_CHANNEL_WORKERS", ""),
            cfg.radar.channel_workers,
        )
    cfg.radar.feedback_path = os.getenv(
        "AICLIP_RADAR_FEEDBACK",
        cfg.radar.feedback_path,
    )
    cfg.data_dir = os.getenv("AICLIP_DATA_DIR", cfg.data_dir)
    return cfg


def _env_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(value: str, default: int) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return default


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
