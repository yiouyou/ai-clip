from pathlib import Path

from ai_clip.core.artifacts import artifact_manifest_path
from ai_clip.core.config import WhisperConfig
from ai_clip.extract.remote import VideoScript, fetch_video_script_report, write_cached_script
from ai_clip.research_engine import align_queries, execute_searches, search_count
from ai_clip.source_research.models import ResearchQuery, SearchResult


def test_research_engine_caps_searches_and_aligns_focus():
    queries = [ResearchQuery(query="background", angle="structural_background")]
    focus = [
        ("event_facts", "verify facts"),
        ("structural_background", "find context"),
    ]
    aligned = align_queries(
        queries,
        focus,
        lambda angle, description: ResearchQuery(query=description, angle=angle),
    )
    assert search_count(0) == 1
    assert search_count(10) == 3
    assert [(item.angle, item.query) for item in aligned] == [
        ("event_facts", "verify facts"),
        ("structural_background", "background"),
    ]


def test_research_engine_labels_results_with_query_angle():
    results = execute_searches(
        [ResearchQuery(query="claim", angle="event_facts")],
        lambda query: [SearchResult(query=query, title="source")],
    )
    assert len(results) == 1
    assert results[0].angle == "event_facts"


def test_remote_video_engine_reuses_cached_script(tmp_path: Path):
    write_cached_script(
        tmp_path,
        VideoScript(text="cached", language="en", segments=[], source="subtitles"),
    )
    result = fetch_video_script_report(
        "https://example.com/video",
        tmp_path,
        transcribe_missing=False,
    )
    assert result.status == "cached"
    assert result.script is not None
    assert result.script.text == "cached"
    assert result.attempts == ("cache",)


def test_remote_video_cache_tracks_url_and_whisper_settings(monkeypatch, tmp_path: Path):
    url = "https://example.com/video"
    small = WhisperConfig(model_size="small")
    write_cached_script(
        tmp_path,
        VideoScript(text="old", language="en", segments=[], source="whisper"),
        url=url,
        whisper=small,
    )
    assert artifact_manifest_path(tmp_path / "script.json").exists()
    calls = []
    monkeypatch.setattr("ai_clip.extract.remote.fetch_video_subtitles", lambda *a: None)

    def transcribe(*args):
        calls.append(args[2].model_size)
        return VideoScript(text="new", language="en", segments=[], source="whisper")

    monkeypatch.setattr("ai_clip.extract.remote.transcribe_video_audio", transcribe)

    reused = fetch_video_script_report(url, tmp_path, whisper=small)
    refreshed = fetch_video_script_report(
        url,
        tmp_path,
        whisper=WhisperConfig(model_size="medium"),
    )

    assert reused.script and reused.script.text == "old"
    assert refreshed.script and refreshed.script.text == "new"
    assert calls == ["medium"]


def test_remote_video_cache_retries_subtitles_when_cookie_changes(monkeypatch, tmp_path: Path):
    url = "https://example.com/video"
    cookie = tmp_path / "cookies.txt"
    cookie.write_text("old", encoding="utf-8")
    whisper = WhisperConfig(model_size="small")
    write_cached_script(
        tmp_path,
        VideoScript(text="whisper", language="en", segments=[], source="whisper"),
        url=url,
        cookiefile=str(cookie),
        whisper=whisper,
    )
    cookie.write_text("updated-cookie", encoding="utf-8")
    monkeypatch.setattr(
        "ai_clip.extract.remote.fetch_video_subtitles",
        lambda *a: VideoScript(text="official", language="en", segments=[], source="subtitles"),
    )
    monkeypatch.setattr(
        "ai_clip.extract.remote.transcribe_video_audio",
        lambda *a: (_ for _ in ()).throw(AssertionError("Whisper should not run")),
    )

    result = fetch_video_script_report(url, tmp_path, str(cookie), whisper=whisper)

    assert result.script and result.script.text == "official"
    assert result.script.source == "subtitles"
