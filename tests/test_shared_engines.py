from pathlib import Path

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
