"""Extract stage: split audio with ffmpeg, transcribe with faster-whisper.

faster-whisper is imported lazily. Device/compute_type are resolved from the
config + hardware so CPU boxes use int8 and GPU boxes use float16 automatically.
"""

from __future__ import annotations

from pathlib import Path

from ai_clip.core.config import WhisperConfig
from ai_clip.core.device import whisper_runtime
from ai_clip.core.ffmpeg import extract_audio
from ai_clip.core.models import Clip, Transcript, TranscriptSegment


def extract(clip: Clip, out_dir: str | Path, whisper: WhisperConfig) -> Transcript:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audio_path = out / f"{clip.clip_id}.wav"
    extract_audio(clip.video_path, audio_path)

    segments, language, text = _transcribe(audio_path, whisper)
    return Transcript(
        clip_id=clip.clip_id,
        language=language,
        text=text,
        segments=segments,
        audio_path=str(audio_path),
    )


def _transcribe(
    audio_path: Path, whisper: WhisperConfig
) -> tuple[list[TranscriptSegment], str, str]:
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "faster-whisper not installed. Install with: pip install 'ai-clip[extract]'"
        ) from exc

    device, compute_type = whisper_runtime(whisper.model_size, whisper.compute_type)
    model = WhisperModel(whisper.model_size, device=device, compute_type=compute_type)

    raw_segments, info = model.transcribe(
        str(audio_path), language=whisper.language, vad_filter=True
    )
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in raw_segments
    ]
    text = " ".join(s.text for s in segments).strip()
    return segments, info.language, text
