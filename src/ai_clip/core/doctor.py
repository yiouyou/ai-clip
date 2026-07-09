from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile

from ai_clip.core.config import Config
from ai_clip.radar.collect import load_channels


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


def run_doctor(cfg: Config) -> list[DoctorCheck]:
    checks = [
        _check_data_dir(Path(cfg.data_dir)),
        _check_binary("ffmpeg"),
        _check_binary("ffprobe"),
        _check_import("yt-dlp", "yt_dlp"),
        _check_import("faster-whisper", "faster_whisper"),
        _check_llm_key(cfg),
        _check_pair_key(cfg),
        _check_tavily_key(cfg),
        _check_mimo_key(cfg),
        _check_channels(cfg),
    ]
    return checks


def doctor_exit_code(checks: list[DoctorCheck]) -> int:
    return 1 if any(check.status == "fail" for check in checks) else 0


def _check_data_dir(path: Path) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".aiclip-doctor-", delete=False) as f:
            temp = Path(f.name)
        temp.unlink(missing_ok=True)
    except OSError as exc:
        return DoctorCheck(
            name="data_dir",
            status="fail",
            detail=str(path),
            hint=f"cannot write data directory: {exc}",
        )
    return DoctorCheck(name="data_dir", status="pass", detail=str(path))


def _check_binary(binary: str) -> DoctorCheck:
    found = shutil.which(binary)
    if found:
        return DoctorCheck(name=binary, status="pass", detail=found)
    return DoctorCheck(
        name=binary,
        status="fail",
        hint=f"{binary} is required on PATH for video/audio processing",
    )


def _check_import(label: str, module: str) -> DoctorCheck:
    if importlib.util.find_spec(module) is not None:
        return DoctorCheck(name=label, status="pass")
    return DoctorCheck(
        name=label,
        status="warn",
        hint="install the matching extra if needed, e.g. uv pip install -e .[download,extract]",
    )


def _check_llm_key(cfg: Config) -> DoctorCheck:
    if cfg.llm.api_key:
        return DoctorCheck(name="llm_api_key", status="pass", detail=cfg.llm.model)
    return DoctorCheck(
        name="llm_api_key",
        status="fail",
        detail=cfg.llm.model,
        hint="set AICLIP_LLM_API_KEY or the provider-specific key for llm.base_url",
    )


def _check_pair_key(cfg: Config) -> DoctorCheck:
    if cfg.pair.api_key or cfg.pair.deepseek_api_key:
        return DoctorCheck(name="pair_review_keys", status="pass")
    return DoctorCheck(
        name="pair_review_keys",
        status="warn",
        hint="set NEWAPP_API_KEY/NEWAPP_URL or DEEPSEEK_API_KEY to use pair-review",
    )


def _check_tavily_key(cfg: Config) -> DoctorCheck:
    if cfg.source_research.tavily_api_key:
        return DoctorCheck(name="tavily_key", status="pass")
    return DoctorCheck(
        name="tavily_key",
        status="warn",
        hint="set TAVILY_API_KEY to use source-research",
    )


def _check_mimo_key(cfg: Config) -> DoctorCheck:
    if cfg.tts.api_key:
        return DoctorCheck(name="mimo_key", status="pass")
    return DoctorCheck(
        name="mimo_key",
        status="warn",
        hint="set MIMO_API_KEY to use voiceover",
    )


def _check_channels(cfg: Config) -> DoctorCheck:
    path = Path(cfg.radar.channels_path)
    if not path.exists():
        return DoctorCheck(
            name="radar_channels",
            status="warn",
            detail=str(path),
            hint="copy config/channels.example.yaml to config/channels.yaml",
        )
    try:
        channels = load_channels(path)
    except Exception as exc:
        return DoctorCheck(
            name="radar_channels",
            status="fail",
            detail=str(path),
            hint=f"cannot parse channels: {exc}",
        )
    missing_cookies = [
        channel.cookies
        for channel in channels
        if channel.cookies and not Path(os.path.expandvars(channel.cookies)).exists()
    ]
    if missing_cookies:
        return DoctorCheck(
            name="radar_channels",
            status="warn",
            detail=f"{len(channels)} channel(s)",
            hint=f"missing cookie file: {missing_cookies[0]}",
        )
    return DoctorCheck(name="radar_channels", status="pass", detail=f"{len(channels)} channel(s)")
