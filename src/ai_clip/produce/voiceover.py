"""Voiceover step: synthesize each shot's narration, optionally in a voice cloned
from the source clip's speaker. Writes voice/shot_NN.wav for the assemble step."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ai_clip.core.artifacts import (
    artifact_manifest_path,
    artifact_matches,
    read_artifact_manifest,
    write_artifact_manifest,
)
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
        source_path = Path(source_audio)
        reference_path = Path(reference_out)
        params = {"seconds": str(cfg.reference_seconds), "format": "mp3-mono-24khz"}
        if not artifact_matches(reference_path, inputs=[source_path], params=params):
            reference_path = make_reference_clip(source_path, reference_path, cfg.reference_seconds)
            write_artifact_manifest(
                reference_path,
                stage="voice-reference",
                inputs=[source_path],
                params=params,
            )
    effective_cfg = cfg
    if cfg.model.endswith("-voiceclone") and reference_path is None:
        effective_cfg = cfg.model_copy(update={
            "model": "mimo-v2.5-tts",
            "clone_from_source": False,
        })
    return MimoTTS(effective_cfg, reference_path=reference_path)


def generate_voiceover(
    sb: Storyboard,
    tts: TTSProvider,
    voice_dir: str | Path,
    *,
    invocation_params: dict[str, str] | None = None,
    inputs: list[Path] | tuple[Path, ...] = (),
) -> dict[int, Path]:
    voice_dir = Path(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[int, Path] = {}
    expected = {
        voice_filename(shot.index)
        for shot in sb.shots
        if shot.voiceover.strip()
    }
    _remove_orphaned_voice_files(voice_dir, expected)
    for shot in sb.shots:
        if not shot.voiceover.strip():
            continue
        out = voice_dir / voice_filename(shot.index)
        params = {
            "text": shot.voiceover,
            "style": "",
            **(invocation_params or {"provider": type(tts).__name__}),
        }
        if out.exists() and not artifact_manifest_path(out).exists():
            produced[shot.index] = out
            continue
        if artifact_matches(out, inputs=inputs, params=params):
            produced[shot.index] = out
            continue
        tts.synthesize(shot.voiceover, out)
        write_artifact_manifest(
            out,
            stage="voiceover",
            inputs=inputs,
            params=params,
        )
        produced[shot.index] = out
    return produced


def _remove_orphaned_voice_files(voice_dir: Path, expected: set[str]) -> None:
    for path in voice_dir.glob("shot_*.wav"):
        if path.name in expected:
            continue
        manifest_path = artifact_manifest_path(path)
        if not manifest_path.exists():
            continue
        try:
            manifest = read_artifact_manifest(path)
        except (OSError, ValueError):
            continue
        if manifest.stage != "voiceover":
            continue
        path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
