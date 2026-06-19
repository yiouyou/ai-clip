import pytest

from ai_clip.core.models import Candidate, Platform
from ai_clip.discover.discover import discover
from ai_clip.discover.base import age_days_from
from ai_clip.discover.ranking import rank, virality_score
from ai_clip.discover.youtube import YouTubeProvider, entry_to_candidate


def test_virality_recent_beats_old_with_same_views():
    recent = virality_score(100_000, 5000, 500, age_days=1)
    old = virality_score(100_000, 5000, 500, age_days=120)
    assert recent > old


def test_virality_engagement_matters():
    high_eng = virality_score(10_000, 2000, 500, age_days=5)
    low_eng = virality_score(10_000, 50, 5, age_days=5)
    assert high_eng > low_eng


def test_virality_zero_views_safe():
    assert virality_score(0, 0, 0, age_days=3) == 0.0


def test_rank_orders_desc_and_limits():
    cands = [Candidate(url=str(i), virality=v) for i, v in enumerate([1, 9, 5])]
    ranked = rank(cands, top_n=2)
    assert [c.virality for c in ranked] == [9, 5]


def test_age_days_from_timestamp_and_date():
    assert age_days_from(None, None) == 0.0
    assert age_days_from(None, "20200101") > 1000  # long ago
    assert age_days_from(None, "bad") == 0.0


def test_entry_to_candidate_maps_fields():
    entry = {
        "webpage_url": "https://www.youtube.com/watch?v=abc",
        "title": "AI 震撼演示",
        "uploader": "Some Channel",
        "view_count": 50000,
        "like_count": 3000,
        "comment_count": 200,
        "duration": 45,
        "upload_date": "20200101",
    }
    c = entry_to_candidate(entry)
    assert c.platform == Platform.youtube
    assert c.view_count == 50000
    assert c.duration_sec == 45
    assert c.age_days > 1000


def test_entry_to_candidate_bare_id_builds_url():
    c = entry_to_candidate({"url": "xyz", "title": "t"})
    assert c.url == "https://www.youtube.com/watch?v=xyz"


def test_youtube_search_filters_long_and_old(monkeypatch):
    entries = {
        "entries": [
            {"webpage_url": "u1", "title": "ok", "duration": 30, "upload_date": "20200101",
             "view_count": 100},  # too old
            {"webpage_url": "u2", "title": "long", "duration": 600,
             "view_count": 100},  # too long
            {"webpage_url": "u3", "title": "good", "duration": 40,
             "timestamp": None, "view_count": 100},  # age 0 -> kept
        ]
    }

    class FakeYDL:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): return entries

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    prov = YouTubeProvider(max_duration_sec=90)
    out = prov.search("AI", channel=None, since_days=7, limit=10)
    titles = [c.title for c in out]
    assert "good" in titles
    assert "long" not in titles  # duration filter
    assert "ok" not in titles    # age filter


def test_discover_unknown_platform_raises():
    # tiktok has no provider registered -> NotImplementedError
    with pytest.raises(NotImplementedError):
        discover("AI", platform=Platform.tiktok)


def test_bilibili_search_keyword(monkeypatch):
    from ai_clip.discover.bilibili import BilibiliProvider, entry_to_candidate

    c = entry_to_candidate({"url": "BV1xx", "title": "测评", "duration": 40,
                            "view_count": 8000, "like_count": 600})
    assert c.platform == Platform.bilibili
    assert c.url == "https://www.bilibili.com/video/BV1xx"

    entries = {"entries": [
        {"webpage_url": "https://www.bilibili.com/video/BV1", "title": "AI", "duration": 50,
         "view_count": 9000, "timestamp": None},
    ]}

    class FakeYDL:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, q, **k):
            assert q.startswith("bilisearch")
            return entries

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    out = BilibiliProvider().search("AI", channel=None, since_days=7, limit=10)
    assert len(out) == 1 and out[0].platform == Platform.bilibili


def test_douyin_keyword_without_channel_raises():
    from ai_clip.discover.social import DouyinProvider

    with pytest.raises(NotImplementedError):
        DouyinProvider().search("麻将", channel=None, since_days=7, limit=10)


def test_douyin_channel_listing(monkeypatch):
    from ai_clip.discover.social import DouyinProvider

    entries = {"entries": [
        {"webpage_url": "https://v.douyin.com/a", "title": "牌局", "duration": 30,
         "view_count": 5000, "like_count": 400, "timestamp": None},
    ]}

    class FakeYDL:
        def __init__(self, *a, **k): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): return entries

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    out = DouyinProvider().search("x", channel="https://www.douyin.com/user/X",
                                  since_days=7, limit=10)
    assert len(out) == 1
    assert out[0].platform == Platform.douyin
    assert out[0].view_count == 5000
