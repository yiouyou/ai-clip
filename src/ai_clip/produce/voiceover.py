"""Voiceover step: synthesize each shot's narration, optionally in a voice cloned
from the source clip's speaker. Writes voice/shot_NN.wav for the assemble step."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ai_clip.core.config import TTSConfig
from ai_clip.core.models import Storyboard
from ai_clip.produce.tts.mimo import MimoTTS, make_reference_clip


class TTSProvider(Protocol):
    def synthesize(self, text: str, out_path: str | Path, style: str = "") -> Path: ...


def voice_filename(index: int) -> str:
    return f"shot_{index:02d}.wav"


def build_mimo(
    cfg: TTSConfig, source_audio: str | Path | None, reference_out: str | Path
) -> MimoTTS:
    """Construct a MiMo provider, cutting a clone reference from source audio when
    voiceclone is configured and a source voice track exists."""
    reference_path = None
    if cfg.clone_from_source and source_audio and Path(source_audio).exists():
        reference_path = make_reference_clip(source_audio, reference_out, cfg.reference_seconds)
    return MimoTTS(cfg, reference_path=reference_path)


def generate_voiceover(
    sb: Storyboard, tts: TTSProvider, voice_dir: str | Path
) -> dict[int, Path]:
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[int, Path] = {}
    for shot in sb.shots:
        if not shot.voiceover.strip():
            continue
        out = voice_dir / voice_filename(shot.index)
        tts.synthesize(shot.voiceover, out)
        produced[shot.index] = out
    return produced
