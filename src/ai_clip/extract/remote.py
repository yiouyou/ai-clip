from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ai_clip.core.artifacts import (
    artifact_manifest_path,
    artifact_matches,
    read_artifact_manifest,
    write_artifact_manifest,
    write_text_atomic,
)
from ai_clip.core.config import WhisperConfig
from ai_clip.core.ffmpeg import FFmpegError
from ai_clip.core.models import TranscriptSegment
from ai_clip.extract.extractor import transcribe_audio
from ai_clip.extract.subtitles import parse_vtt

PREFERRED_LANGS = ["zh", "zh-Hans", "zh-CN", "zh-Hant", "en", "en-US", "en-GB"]
_SCRIPT_CACHE_VERSION = "1"


@dataclass(frozen=True)
class VideoScript:
    text: str
    language: str
    segments: list[TranscriptSegment]
    source: str


@dataclass(frozen=True)
class VideoScriptResult:
    script: VideoScript | None
    status: str
    error: str = ""
    attempts: tuple[str, ...] = ()
    cache_path: str = ""


def fetch_video_script(
    url: str,
    out_dir: Path,
    cookiefile: str = "",
    whisper: WhisperConfig | None = None,
    transcribe_missing: bool = True,
) -> VideoScript | None:
    return fetch_video_script_report(
        url,
        out_dir,
        cookiefile,
        whisper=whisper,
        transcribe_missing=transcribe_missing,
    ).script


def fetch_video_script_report(
    url: str,
    out_dir: Path,
    cookiefile: str = "",
    whisper: WhisperConfig | None = None,
    transcribe_missing: bool = True,
) -> VideoScriptResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = read_cached_script(out_dir)
    cache_path = _script_cache_path(out_dir)
    cache_inputs = _script_cache_inputs(cookiefile)
    cache_params = _script_cache_params(url, cached, whisper) if cached else {}
    manifest_path = artifact_manifest_path(cache_path)
    if cached is not None and artifact_matches(
        cache_path,
        inputs=cache_inputs,
        params=cache_params,
    ):
        return VideoScriptResult(
            script=cached,
            status="cached",
            attempts=("cache",),
            cache_path=str(cache_path),
        )
    if cached is not None and cached.source == "subtitles" and not manifest_path.exists():
        _write_script_manifest(cache_path, cache_inputs, cache_params)
        return VideoScriptResult(
            script=cached,
            status="cached",
            attempts=("cache",),
            cache_path=str(cache_path),
        )

    attempts: list[str] = ["subtitles"]
    script = fetch_video_subtitles(url, out_dir, cookiefile)
    if script is not None:
        write_cached_script(
            out_dir,
            script,
            url=url,
            cookiefile=cookiefile,
            whisper=whisper,
        )
        return VideoScriptResult(
            script=script,
            status="available",
            attempts=tuple(attempts),
            cache_path=str(_script_cache_path(out_dir)),
        )
    if cached is not None and _can_refresh_cached_manifest(
        cache_path,
        cache_params,
    ):
        attempts.append("cache")
        _write_script_manifest(cache_path, cache_inputs, cache_params)
        return VideoScriptResult(
            script=cached,
            status="cached",
            attempts=tuple(attempts),
            cache_path=str(cache_path),
        )
    if not transcribe_missing or whisper is None:
        error = "subtitles unavailable; transcription disabled"
        write_content_status(out_dir, "missing", error, attempts)
        return VideoScriptResult(script=None, status="missing", error=error, attempts=tuple(attempts))

    attempts.append("whisper")
    script = transcribe_video_audio(url, out_dir, whisper, cookiefile)
    if script is None:
        error = "subtitles and audio transcription unavailable"
        write_content_status(out_dir, "missing", error, attempts)
        return VideoScriptResult(script=None, status="missing", error=error, attempts=tuple(attempts))
    write_cached_script(
        out_dir,
        script,
        url=url,
        cookiefile=cookiefile,
        whisper=whisper,
    )
    return VideoScriptResult(
        script=script,
        status="available",
        attempts=tuple(attempts),
        cache_path=str(_script_cache_path(out_dir)),
    )


def fetch_video_subtitles(url: str, out_dir: Path, cookiefile: str = "") -> VideoScript | None:
    import yt_dlp  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": PREFERRED_LANGS,
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / "subs.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError:
        return None

    candidates = [
        *(candidate for lang in PREFERRED_LANGS for candidate in out_dir.glob(f"subs.{lang}.vtt")),
        *sorted(out_dir.glob("subs.*.vtt")),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        segments = parse_vtt(candidate.read_text(encoding="utf-8", errors="replace"))
        if segments:
            parts = candidate.name.split(".")
            language = parts[1] if len(parts) > 2 else ""
            return VideoScript(
                text=" ".join(segment.text for segment in segments).strip(),
                language=language,
                segments=segments,
                source="subtitles",
            )
    return None


def transcribe_video_audio(
    url: str,
    out_dir: Path,
    whisper: WhisperConfig,
    cookiefile: str = "",
) -> VideoScript | None:
    audio_path = download_video_audio(url, out_dir, cookiefile)
    if audio_path is None:
        return None
    try:
        segments, language, text = transcribe_audio(audio_path, whisper)
    except (RuntimeError, FFmpegError):
        return None
    if not text:
        return None
    return VideoScript(text=text, language=language, segments=segments, source="whisper")


def download_video_audio(url: str, out_dir: Path, cookiefile: str = "") -> Path | None:
    import yt_dlp  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared = Path(ydl.prepare_filename(info)) if info else None
    except yt_dlp.utils.DownloadError:
        return None
    if prepared and prepared.exists():
        return prepared
    candidates = [
        path for path in out_dir.glob("audio.*")
        if not path.name.endswith((".part", ".ytdl", ".tmp"))
    ]
    return sorted(candidates)[0] if candidates else None


def read_cached_script(out_dir: Path) -> VideoScript | None:
    path = _script_cache_path(out_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    text = str(data.get("text") or "")
    if not text.strip():
        return None
    return VideoScript(
        text=text,
        language=str(data.get("language") or ""),
        source=str(data.get("source") or "cache"),
        segments=[
            TranscriptSegment.model_validate(item)
            for item in data.get("segments", [])
            if isinstance(item, dict)
        ],
    )


def write_cached_script(
    out_dir: Path,
    script: VideoScript,
    *,
    url: str = "",
    cookiefile: str = "",
    whisper: WhisperConfig | None = None,
) -> None:
    payload = {
        "text": script.text,
        "language": script.language,
        "source": script.source,
        "segments": [segment.model_dump(mode="json") for segment in script.segments],
    }
    write_text_atomic(
        _script_cache_path(out_dir),
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if url:
        _write_script_manifest(
            _script_cache_path(out_dir),
            _script_cache_inputs(cookiefile),
            _script_cache_params(url, script, whisper),
        )
    write_content_status(out_dir, "available", "", [script.source])


def write_content_status(out_dir: Path, status: str, error: str, attempts: list[str]) -> None:
    write_text_atomic(
        _content_status_path(out_dir),
        json.dumps({"status": status, "error": error, "attempts": attempts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _script_cache_path(out_dir: Path) -> Path:
    return out_dir / "script.json"


def _content_status_path(out_dir: Path) -> Path:
    return out_dir / "status.json"


def _script_cache_inputs(cookiefile: str) -> list[Path]:
    if not cookiefile:
        return []
    path = Path(cookiefile)
    return [path] if path.exists() else []


def _script_cache_params(
    url: str,
    script: VideoScript,
    whisper: WhisperConfig | None,
) -> dict[str, str]:
    params = {
        "cache_version": _SCRIPT_CACHE_VERSION,
        "url": url,
        "source": script.source,
    }
    if script.source == "whisper" and whisper is not None:
        params.update({
            "whisper_model_size": whisper.model_size,
            "whisper_compute_type": whisper.compute_type,
            "whisper_language": whisper.language or "",
        })
    return params


def _write_script_manifest(
    cache_path: Path,
    inputs: list[Path],
    params: dict[str, str],
) -> None:
    write_artifact_manifest(
        cache_path,
        stage="source-content",
        inputs=inputs,
        params=params,
    )


def _can_refresh_cached_manifest(cache_path: Path, expected_params: dict[str, str]) -> bool:
    manifest_path = artifact_manifest_path(cache_path)
    if not manifest_path.exists():
        return True
    try:
        return read_artifact_manifest(cache_path).params == expected_params
    except (OSError, ValueError, json.JSONDecodeError):
        return False
