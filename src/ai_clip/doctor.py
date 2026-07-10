from __future__ import annotations

import os
from pathlib import Path

from ai_clip.core.config import Config
from ai_clip.core.doctor import DoctorCheck, doctor_exit_code, run_core_doctor
from ai_clip.radar.collect import load_channels

__all__ = ["DoctorCheck", "doctor_exit_code", "run_doctor"]


def run_doctor(cfg: Config) -> list[DoctorCheck]:
    return [*run_core_doctor(cfg), _check_channels(cfg)]


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
