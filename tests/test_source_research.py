from ai_clip.core.config import Config
from ai_clip.core.models import Platform
from ai_clip.radar.models import RadarVideo, ZackSelection
from ai_clip.source_research.models import SearchResult
from ai_clip.source_research.stage import generate_source_research


def test_generate_source_research_clamps_searches(monkeypatch):
    selection = ZackSelection(
        date="2026-01-02",
        selected_video_id="youtube:1",
        selected_index=1,
        selected_video=RadarVideo(
            video_id="youtube:1",
            url="https://example.com/v",
            platform=Platform.youtube,
            title="事件标题",
            transcript_text="事件脚本",
        ),
        topic="事件标题",
        research_focus=[
            "event_facts: verify facts",
            "structural_background: find background",
            "counterclaims_risk: check title risk",
        ],
    )
    cfg = Config()
    cfg.llm.api_key = "llm"
    cfg.source_research.tavily_api_key = "tavily"
    cfg.source_research.max_searches = 9

    replies = [
        """
        {"queries": [
          {"angle": "event_facts", "query": "query 1", "rationale": "r1"},
          {"angle": "structural_background", "query": "query 2", "rationale": "r2"},
          {"angle": "counterclaims_risk", "query": "query 3", "rationale": "r3"},
          {"angle": "extra", "query": "query 4", "rationale": "r4"}
        ]}
        """,
        "# Source Research\n\n## Confirmed Facts\n- ok",
    ]

    def fake_chat(*args, **kwargs):
        return replies.pop(0)

    seen_queries = []

    def fake_search(query, cfg):
        seen_queries.append(query)
        return [SearchResult(query=query, title=f"title {query}", url=f"https://{query}.example")]

    monkeypatch.setattr("ai_clip.source_research.stage.chat", fake_chat)
    monkeypatch.setattr("ai_clip.source_research.stage.tavily_search", fake_search)

    report = generate_source_research(selection, cfg)

    assert seen_queries == ["query 1", "query 2", "query 3"]
    assert [query.angle for query in report.queries] == [
        "event_facts",
        "structural_background",
        "counterclaims_risk",
    ]
    assert report.search_calls == 3
    assert len(report.results) == 3
    assert "Confirmed Facts" in report.markdown


def test_generate_source_research_falls_back_to_top_title(monkeypatch):
    selection = ZackSelection(
        date="2026-01-02",
        selected_video_id="youtube:1",
        selected_index=1,
        selected_video=RadarVideo(
            video_id="youtube:1",
            url="https://example.com/v",
            platform=Platform.youtube,
            title="Top Event Title",
        ),
        topic="Top Event Title",
    )
    cfg = Config()
    cfg.source_research.max_searches = 1
    replies = [
        '{"queries": []}',
        "# Source Research\n\n## Confirmed Facts\n- ok",
    ]
    seen_queries = []

    def fake_chat(*args, **kwargs):
        return replies.pop(0)

    def fake_search(query, cfg):
        seen_queries.append(query)
        return [SearchResult(query=query)]

    monkeypatch.setattr("ai_clip.source_research.stage.chat", fake_chat)
    monkeypatch.setattr("ai_clip.source_research.stage.tavily_search", fake_search)

    report = generate_source_research(selection, cfg)

    assert seen_queries == [
        "Top Event Title Verify who, what, when, official statements, and mainstream reporting."
    ]
    assert report.search_calls == 1
