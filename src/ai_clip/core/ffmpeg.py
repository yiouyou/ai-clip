"""Thin ffmpeg/ffprobe helpers. ffmpeg is a hard runtime requirement."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(
                f"{tool} not found on PATH. Install ffmpeg (winget install ffmpeg / "
                "apt install ffmpeg) and retry."
            )


def run(args: list[str]) -> None:
    proc = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise FFmpegError(f"command failed: {' '.join(args)}\n{proc.stderr.strip()}")


def probe_duration(path: str | Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def extract_audio(video_path: str | Path, audio_path: str | Path) -> None:
    """Pull a 16 kHz mono wav, the form ASR engines expect."""
    ensure_ffmpeg()
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ])
