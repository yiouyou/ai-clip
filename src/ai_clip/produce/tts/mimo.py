"""Xiaomi MiMo-V2.5-TTS provider (https://mimo.mi.com).

OpenAI-style /chat/completions endpoint where the text to speak goes in the
*assistant* message and style/voice description goes in the *user* message.
Voice cloning passes a base64 reference clip as the `audio.voice` data URI, so
we can clone the speaker extracted from the source video.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from ai_clip.core.config import TTSConfig
from ai_clip.core.ffmpeg import ensure_ffmpeg, run

_CLONE_MODEL = "mimo-v2.5-tts-voiceclone"
_MAX_B64_BYTES = 10 * 1024 * 1024


class TTSError(RuntimeError):
    pass


def make_reference_clip(audio_path: str | Path, out_path: str | Path, seconds: float) -> Path:
    """Cut a short, clean reference snippet from the extracted voice track for
    cloning. mp3 keeps the base64 well under MiMo's 10 MB limit."""
    ensure_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-i", str(audio_path), "-t", str(seconds),
        "-ac", "1", "-ar", "24000", "-codec:a", "libmp3lame", "-q:a", "4",
        str(out),
    ])
    return out


def _voice_data_uri(reference_path: str | Path) -> str:
    raw = Path(reference_path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    if len(b64) > _MAX_B64_BYTES:
        raise TTSError("reference audio base64 exceeds MiMo's 10 MB limit; shorten it")
    return f"data:audio/mpeg;base64,{b64}"


class MimoTTS:
    name = "mimo"

    def __init__(self, cfg: TTSConfig, reference_path: str | Path | None = None):
        if not cfg.api_key:
            raise TTSError("MIMO_API_KEY is empty; set it in your .env")
        self.cfg = cfg
        self.reference_path = reference_path

    def _voice(self) -> str:
        if self.cfg.model == _CLONE_MODEL:
            if not self.reference_path:
                raise TTSError("voiceclone model selected but no reference audio provided")
            return _voice_data_uri(self.reference_path)
        return self.cfg.voice

    def synthesize(self, text: str, out_path: str | Path, style: str = "") -> Path:
        if not text.strip():
            raise TTSError("empty text for TTS")
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "user", "content": style},
                {"role": "assistant", "content": text},
            ],
            "audio": {"format": "wav", "voice": self._voice()},
            "stream": False,
        }
        resp = httpx.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            headers={"api-key": self.cfg.api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=180.0,
        )
        resp.raise_for_status()
        data = resp.json()["choices"][0]["message"]["audio"]["data"]

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        return out
