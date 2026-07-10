import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ai_clip.core.artifacts import read_artifact_manifest, write_artifact_manifest
from ai_clip.core.config import Config, RadarConfig
from ai_clip.core.models import Platform
from ai_clip.pair.models import PairReviewReport
from ai_clip.radar.collect import (
    collect_channels,
    collect_channels_with_diagnostics,
    entry_to_video,
    load_channels,
)
from ai_clip.radar.feedback import apply_feedback_events, read_feedback_events, record_feedback
from ai_clip.radar.models import (
    ChannelSpec,
    RadarCandidates,
    RadarCollectReport,
    RadarSnapshot,
    RadarVideo,
)
from ai_clip.radar.research_policy import automatic_research_searches
from ai_clip.source_content import add_source_content
from ai_clip.zack_ranking import RankingFeedback, rank_videos, rerank_by_content, score_video
from ai_clip.radar.stage import (
    run_collect,
    run_source_research,
    run_zack_ranking,
    run_zack_draft,
    run_zack_selection,
)
from ai_clip.radar.backfill import run_backfill
from ai_clip.radar.workflow import run_all
from ai_clip.radar.models import RadarRunResult, ZackSelection
from ai_clip.radar.artifact_status import radar_artifact_statuses
from ai_clip.radar.storage import RadarPaths
from ai_clip import workflows


def test_load_channels_yaml(tmp_path: Path):
    path = tmp_path / "channels.yaml"
    path.write_text(
        """
channels:
  - platform: youtube
    url: https://www.youtube.com/@x
    name: X
    pool: ai
    role: signal
    tags: [ai]
    priority: 2
    lens_fit: 1.5
    max_duration_sec: 1200
    cookies: E:/cookies/youtube.txt
""",
        encoding="utf-8",
    )
    channels = load_channels(path)
    assert len(channels) == 1
    assert channels[0].platform == Platform.youtube
    assert channels[0].pool == "ai"
    assert channels[0].role == "signal"
    assert channels[0].priority == 2
    assert channels[0].lens_fit == 1.5
    assert channels[0].max_duration_sec == 1200
    assert channels[0].cookies.endswith("youtube.txt")


def test_entry_to_video_maps_extra_metrics():
    channel = ChannelSpec(platform=Platform.bilibili, url="https://space.bilibili.com/1")
    video = entry_to_video(
        {
            "id": "BV1xx",
            "title": "热点",
            "duration": 80,
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 10,
            "favorite_count": 20,
            "coin_count": 5,
            "danmaku_count": 7,
            "upload_date": "20260101",
        },
        channel,
    )
    assert video.video_id == "bilibili:BV1xx"
    assert video.url == "https://www.bilibili.com/video/BV1xx"
    assert video.pool == "general"
    assert video.role == "signal"
    assert video.published_date == "2026-01-01"
    assert video.favorite_count == 20
    assert video.coin_count == 5


def test_score_uses_previous_deltas():
    current = RadarVideo(
        video_id="youtube:1",
        url="u",
        platform=Platform.youtube,
        view_count=2000,
        like_count=200,
        comment_count=20,
        age_days=1,
    )
    previous = current.model_copy(update={"view_count": 1000, "like_count": 50})
    scored = score_video(current, previous)
    assert scored.score > 0
    assert "pool=general" in scored.score_reasons
    assert "role=signal" in scored.score_reasons
    assert "view_delta=1,000" in scored.score_reasons
    assert scored.score_components["reach"] > 0
    assert scored.score_components["feedback_multiplier"] == 1.0


def test_score_prefers_biology_lens_signal_over_style_reference():
    signal = RadarVideo(
        video_id="signal",
        url="u",
        platform=Platform.youtube,
        role="signal",
        lens_fit=1.6,
        view_count=1000,
        like_count=100,
        comment_count=10,
        age_days=1,
    )
    style = signal.model_copy(update={"video_id": "style", "role": "style", "lens_fit": 0.8})
    assert score_video(signal).score > score_video(style).score


def test_score_downranks_weather_topics():
    baseline = RadarVideo(
        video_id="society",
        url="u",
        platform=Platform.youtube,
        title="东北人口外流的临界点",
        role="signal",
        lens_fit=1.3,
        view_count=10_000,
        like_count=1000,
        comment_count=100,
        age_days=1,
    )
    weather = baseline.model_copy(update={
        "video_id": "weather",
        "title": "追踪超强台风巴威：中央山脉削弱台风原理",
    })

    scored_baseline = score_video(baseline)
    scored_weather = score_video(weather)

    assert scored_weather.score < scored_baseline.score
    assert "topic_downrank=weather:0.72" in scored_weather.score_reasons


def test_rank_videos_limits_top():
    snapshots = [
        RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(video_id="a", url="a", platform=Platform.youtube, view_count=100),
        ),
        RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(video_id="b", url="b", platform=Platform.youtube, view_count=10_000),
        ),
    ]
    out = rank_videos(snapshots, previous={}, top_n=1)
    assert [v.video_id for v in out] == ["b"]


def test_rank_videos_prefers_pool_diversity_before_filling():
    snapshots = [
        RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(
                video_id="ai:1",
                url="u1",
                platform=Platform.youtube,
                pool="ai",
                view_count=50_000,
                like_count=5_000,
                age_days=1,
            ),
        ),
        RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(
                video_id="ai:2",
                url="u2",
                platform=Platform.youtube,
                pool="ai",
                view_count=40_000,
                like_count=4_000,
                age_days=1,
            ),
        ),
        RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(
                video_id="society:1",
                url="u3",
                platform=Platform.youtube,
                pool="society",
                view_count=10_000,
                like_count=1_000,
                age_days=1,
            ),
        ),
    ]

    out = rank_videos(snapshots, previous={}, top_n=2)

    assert {video.pool for video in out} == {"ai", "society"}


def test_rank_videos_records_channel_and_platform_normalization():
    def snapshot(video_id: str, channel: str, views: int) -> RadarSnapshot:
        return RadarSnapshot(
            collected_at="2026-01-01T00:00:00+00:00",
            video=RadarVideo(
                video_id=video_id,
                url=video_id,
                platform=Platform.youtube,
                channel_url=channel,
                view_count=views,
                age_days=1,
            ),
        )

    ranked = rank_videos(
        [
            snapshot("large:baseline", "large", 1_000_000),
            snapshot("large:new", "large", 1_100_000),
            snapshot("small:baseline", "small", 1_000),
            snapshot("small:breakout", "small", 100_000),
        ],
        previous={},
        top_n=4,
    )
    by_id = {video.video_id: video for video in ranked}

    assert by_id["small:breakout"].score_components["channel_reach_ratio"] > 1
    assert by_id["large:new"].score_components["channel_reach_ratio"] < 2
    assert "platform_velocity_ratio" in by_id["small:breakout"].score_components


def test_content_rerank_can_promote_a_deep_mechanism_source():
    videos = [
        RadarVideo(
            video_id=f"v{i}",
            url=f"u{i}",
            platform=Platform.youtube,
            pool=f"pool{i}",
            score=float(100 - i * 10),
            lens_fit=1.0,
        )
        for i in range(4)
    ]
    videos[-1] = videos[-1].model_copy(update={
        "transcript_text": "反馈循环 复杂系统 演化 生态 激励 机制 网络 涌现 " * 500,
        "lens_fit": 1.5,
    })

    reranked = rerank_by_content(videos, top_n=3)

    assert "v3" in {video.video_id for video in reranked}
    promoted = next(video for video in reranked if video.video_id == "v3")
    assert promoted.score_components["content_fit"] == 1.0


def test_feedback_events_are_upserted_and_calibrated(tmp_path: Path):
    paths = RadarPaths(tmp_path, "2026-01-02")
    paths.ensure()
    video = RadarVideo(
        video_id="youtube:1",
        url="u",
        platform=Platform.youtube,
        pool="ai",
        tags=["systems"],
    )
    paths.candidates_json.write_text(
        RadarCandidates(date=paths.date, top_n=3, videos=[video]).model_dump_json(),
        encoding="utf-8",
    )
    paths.selection_json.write_text(
        ZackSelection(
            date=paths.date,
            selected_video_id=video.video_id,
            selected_video=video,
            topic="topic",
        ).model_dump_json(),
        encoding="utf-8",
    )

    record_feedback(paths, "reject", reason="too generic")
    record_feedback(paths, "accept", reason="useful angle")
    events = read_feedback_events(paths.feedback_events_jsonl)
    calibrated = apply_feedback_events(RankingFeedback(), events)

    assert len(events) == 1
    assert events[0].decision == "accept"
    assert calibrated.learned_pool_multipliers["ai"] > 1.0
    assert calibrated.learned_tag_multipliers["systems"] > 1.0


def test_automatic_research_is_bounded_by_fact_risk():
    cfg = Config()
    cfg.source_research.tavily_api_key = "key"
    video = RadarVideo(video_id="v", url="u", platform=Platform.youtube)
    selection = ZackSelection(
        date="2026-01-02",
        selected_video_id="v",
        selected_video=video,
        fact_risk="high",
    )

    assert automatic_research_searches(selection, cfg) == 2
    assert automatic_research_searches(selection.model_copy(update={"fact_risk": "low"}), cfg) == 0
    cfg.source_research.tavily_api_key = ""
    assert automatic_research_searches(selection, cfg) == 0


def test_collect_channels_with_cookies(monkeypatch):
    seen_opts = []

    class FakeYDL:
        def __init__(self, opts):
            seen_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            return {"entries": [{"id": "abc", "title": "A", "view_count": 100, "duration": 60}]}

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    snapshots = collect_channels(
        [ChannelSpec(platform=Platform.youtube, url="https://www.youtube.com/@x", cookies="c.txt")],
        RadarConfig(channel_limit=5, since_days=0),
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert len(snapshots) == 1
    assert snapshots[0].video.title == "A"
    assert seen_opts[0]["cookiefile"] == "c.txt"


def test_collect_channels_records_channel_failures(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            raise RuntimeError("cookie=secret failed")

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    report = collect_channels_with_diagnostics(
        [ChannelSpec(platform=Platform.youtube, url="https://www.youtube.com/@x", name="X")],
        RadarConfig(channel_limit=5, since_days=0),
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert report.snapshots == []
    assert report.channels[0].status == "failed"
    assert "cookie=<redacted>" in report.channels[0].error


def test_collect_channel_uses_channel_duration_override(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            return {"entries": [{"id": "long", "title": "Long", "view_count": 100, "duration": 1200}]}

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    snapshots = collect_channels(
        [
            ChannelSpec(
                platform=Platform.youtube,
                url="https://www.youtube.com/@x",
                max_duration_sec=1800,
            )
        ],
        RadarConfig(channel_limit=5, since_days=0, max_duration_sec=900),
        collected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert len(snapshots) == 1


def test_radar_run_writes_candidates_brief_and_draft(monkeypatch, tmp_path: Path):
    channels_path = tmp_path / "channels.yaml"
    channels_path.write_text(
        """
channels:
  - platform: youtube
    url: https://www.youtube.com/@x
    name: X
    tags: [ai]
    priority: 1
""",
        encoding="utf-8",
    )

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            if self.opts.get("skip_download"):
                out = Path(self.opts["outtmpl"].replace("%(ext)s", "zh.vtt"))
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n这是脚本\n",
                    encoding="utf-8",
                )
                return {}
            return {
                "entries": [
                    {
                        "id": "abc",
                        "webpage_url": "https://www.youtube.com/watch?v=abc",
                        "title": "AI 热点",
                        "view_count": 1000,
                        "like_count": 100,
                        "comment_count": 20,
                        "duration": 60,
                    }
                ]
            }

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(
        "ai_clip.zack_selection.selector.chat",
        lambda *a, **k: '{"selected_index": 1, "topic": "AI 热点", "fact_risk": "low"}',
    )
    monkeypatch.setattr("ai_clip.zack_draft.draft.chat", lambda *a, **k: "# draft")

    cfg = Config(data_dir=str(tmp_path))
    cfg.radar.channels_path = str(channels_path)
    cfg.radar.top_n = 3
    cfg.radar.channel_timeout_sec = 0
    result = run_all(cfg, date="2026-01-01", top_n=3)
    assert result.collected == 1
    assert (tmp_path / "radar" / "shortlists" / "2026-01-01.json").exists()
    assert Path(result.candidates_path).exists()
    final_candidates = RadarCandidates.model_validate_json(
        Path(result.candidates_path).read_text(encoding="utf-8")
    )
    assert final_candidates.ranking_phase == "content-reranked"
    assert Path(result.selection_path).exists()
    assert Path(result.brief_path).exists()
    assert Path(result.draft_path).read_text(encoding="utf-8") == "# draft"
    assert Path(result.run_status_path).exists()
    status = json.loads(Path(result.run_status_path).read_text(encoding="utf-8"))
    assert status["status"] == "succeeded"
    assert [stage["name"] for stage in status["stages"]] == [
        "collect",
        "zack-ranking",
        "source-content",
        "content-rerank",
        "zack-selection",
        "zack-draft",
    ]
    assert Path(tmp_path / "radar" / "collect-reports" / "2026-01-01.json").exists()


def test_collect_reuses_existing_snapshots(monkeypatch, tmp_path: Path):
    snapshot_dir = tmp_path / "radar" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    existing = RadarSnapshot(
        collected_at="2026-01-02T00:00:00+00:00",
        video=RadarVideo(video_id="youtube:1", url="u", platform=Platform.youtube),
    )
    (snapshot_dir / "2026-01-02.jsonl").write_text(
        existing.model_dump_json() + "\n",
        encoding="utf-8",
    )
    called = False

    def fake_collect(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "ai_clip.radar.stage.collect_channels_with_diagnostics",
        fake_collect,
    )
    cfg = Config(data_dir=str(tmp_path))
    cfg.radar.channel_timeout_sec = 0

    count = run_collect(cfg, date="2026-01-02")

    assert count == 1
    assert called is False


def test_collect_force_overwrites_and_dedupes(monkeypatch, tmp_path: Path):
    channels_path = tmp_path / "channels.yaml"
    channels_path.write_text(
        """
channels:
  - platform: youtube
    url: https://www.youtube.com/@x
    name: X
""",
        encoding="utf-8",
    )
    snapshot_dir = tmp_path / "radar" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    stale = RadarSnapshot(
        collected_at="2026-01-02T00:00:00+00:00",
        video=RadarVideo(video_id="youtube:old", url="old", platform=Platform.youtube),
    )
    (snapshot_dir / "2026-01-02.jsonl").write_text(
        stale.model_dump_json() + "\n",
        encoding="utf-8",
    )
    fresh = RadarSnapshot(
        collected_at="2026-01-02T01:00:00+00:00",
        video=RadarVideo(video_id="youtube:new", url="new", platform=Platform.youtube),
    )

    monkeypatch.setattr(
        "ai_clip.radar.stage.collect_channels_with_diagnostics",
        lambda *a, **k: RadarCollectReport(
            collected_at="2026-01-02T01:00:00+00:00",
            snapshots=[fresh, fresh],
        ),
    )
    cfg = Config(data_dir=str(tmp_path))
    cfg.radar.channels_path = str(channels_path)
    cfg.radar.channel_timeout_sec = 0

    count = run_collect(cfg, date="2026-01-02", force=True)
    snapshots = (snapshot_dir / "2026-01-02.jsonl").read_text(encoding="utf-8").splitlines()

    assert count == 1
    assert len(snapshots) == 1
    assert "youtube:new" in snapshots[0]
    assert "youtube:old" not in snapshots[0]
    status = json.loads(
        (tmp_path / "radar" / "runs" / "2026-01-02.json").read_text(encoding="utf-8")
    )
    stale = {stage["name"] for stage in status["stages"] if stage["status"] == "stale"}
    assert {"zack-ranking", "source-content", "zack-selection", "zack-draft"} <= stale


def test_zack_ranking_dedupes_repeated_snapshots(tmp_path: Path):
    snapshot_dir = tmp_path / "radar" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    first = RadarSnapshot(
        collected_at="2026-01-02T00:00:00+00:00",
        video=RadarVideo(
            video_id="youtube:1",
            url="u1",
            platform=Platform.youtube,
            view_count=100,
        ),
    )
    latest = first.model_copy(update={
        "video": first.video.model_copy(update={"view_count": 1000}),
    })
    (snapshot_dir / "2026-01-02.jsonl").write_text(
        "\n".join([first.model_dump_json(), latest.model_dump_json(), ""]),
        encoding="utf-8",
    )
    candidates = run_zack_ranking(Config(data_dir=str(tmp_path)), date="2026-01-02", top_n=3)

    assert len(candidates.videos) == 1
    assert candidates.videos[0].view_count == 1000
    manifest = read_artifact_manifest(tmp_path / "radar" / "shortlists" / "2026-01-02.json")
    assert manifest.stage == "zack-ranking"
    assert manifest.params["top_n"] == "3"


def test_radar_artifact_statuses(tmp_path: Path):
    candidates = tmp_path / "radar" / "candidates" / "2026-01-02.json"
    snapshots = tmp_path / "radar" / "snapshots" / "2026-01-02.jsonl"
    candidates.parent.mkdir(parents=True)
    snapshots.parent.mkdir(parents=True)
    candidates.write_text("{}", encoding="utf-8")
    snapshots.write_text("{}", encoding="utf-8")
    write_artifact_manifest(candidates, stage="zack-ranking", inputs=[snapshots])

    paths = RadarPaths(tmp_path, "2026-01-02")
    statuses = {item.name: item.status for item in radar_artifact_statuses(paths)}

    assert statuses["candidates"] == "fresh"
    assert statuses["selection"] == "missing"

    snapshots.write_text("changed", encoding="utf-8")
    statuses = {item.name: item.status for item in radar_artifact_statuses(paths)}

    assert statuses["candidates"] == "stale"


def test_daily_radar_workflow_wraps_radar_pipeline(monkeypatch):
    def fake_run_daily_radar(
        cfg,
        date=None,
        top_n=None,
        research=False,
        force_collect=False,
        review=False,
        rewrite=False,
    ):
        assert date == "2026-01-02"
        assert top_n == 3
        assert research is True
        assert force_collect is True
        assert review is True
        assert rewrite is True
        return RadarRunResult(
            date="2026-01-02",
            collected=7,
            candidates_path="candidates.json",
            selection_path="selection.json",
            brief_path="brief.md",
            draft_path="draft.md",
            run_status_path="run.json",
            review_path="review.json",
            revised_draft_path="draft.revised.md",
        )

    monkeypatch.setattr("ai_clip.workflows.pipeline.run_daily_radar", fake_run_daily_radar)
    result = workflows.daily_radar(
        Config(),
        date="2026-01-02",
        top_n=3,
        research=True,
        force_collect=True,
        review=True,
        rewrite=True,
    )

    assert result == {
        "workflow": "daily_radar",
        "date": "2026-01-02",
        "collected": 7,
        "candidates": "candidates.json",
        "selection": "selection.json",
        "brief": "brief.md",
        "draft": "draft.md",
        "run_status": "run.json",
        "review": "review.json",
        "revised_draft": "draft.revised.md",
        "verification": "",
    }


def test_run_all_can_pair_review_and_rewrite(monkeypatch, tmp_path: Path):
    calls = []

    monkeypatch.setattr("ai_clip.radar.stage.run_collect", lambda *a, **k: 1)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_ranking", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_source_content", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_content_rerank", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_selection", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_draft", lambda *a, **k: None)

    def fake_review(cfg, project, artifact, run_date=None):
        calls.append(("review", project, artifact, run_date))
        return PairReviewReport(
            artifact=artifact,
            source_path="draft.md",
            producer_model="producer",
            status="passed",
            reviewers=[],
        )

    def fake_rewrite(cfg, project, artifact, report, run_date=None):
        calls.append(("rewrite", project, artifact, run_date))
        out = tmp_path / "radar" / "drafts" / "2026-01-02.revised.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# revised", encoding="utf-8")
        return out

    def fake_verify(cfg, project, artifact, run_date=None):
        calls.append(("verify", project, artifact, run_date))
        return PairReviewReport(
            artifact=artifact,
            source_path="draft.revised.md",
            kind="verify",
            status="passed",
            reviewers=[],
        )

    monkeypatch.setattr("ai_clip.pair.stage.review_artifact", fake_review)
    monkeypatch.setattr("ai_clip.pair.stage.rewrite_reviewed_artifact", fake_rewrite)
    monkeypatch.setattr("ai_clip.pair.stage.verify_rewritten_artifact", fake_verify)

    result = run_all(
        Config(data_dir=str(tmp_path)),
        date="2026-01-02",
        review=True,
        rewrite=True,
    )

    assert calls == [
        ("review", "radar", "zack_draft", "2026-01-02"),
        ("rewrite", "radar", "zack_draft", "2026-01-02"),
        ("verify", "radar", "zack_draft", "2026-01-02"),
    ]
    assert result.review_path.endswith("2026-01-02_zack_draft_review.json")
    assert result.revised_draft_path.endswith("2026-01-02.revised.md")
    assert result.verification_path.endswith("2026-01-02_zack_draft_verify.json")


def test_run_all_skips_pair_rewrite_when_review_blocked(monkeypatch, tmp_path: Path):
    calls = []

    monkeypatch.setattr("ai_clip.radar.stage.run_collect", lambda *a, **k: 1)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_ranking", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_source_content", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_content_rerank", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_selection", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_draft", lambda *a, **k: None)

    def fake_review(cfg, project, artifact, run_date=None):
        calls.append(("review", project, artifact, run_date))
        return PairReviewReport(
            artifact=artifact,
            source_path="draft.md",
            producer_model="producer",
            status="blocked",
            reviewers=[],
        )

    def fake_rewrite(*args, **kwargs):
        calls.append(("rewrite",))
        raise AssertionError("blocked review must not be rewritten")

    monkeypatch.setattr("ai_clip.pair.stage.review_artifact", fake_review)
    monkeypatch.setattr("ai_clip.pair.stage.rewrite_reviewed_artifact", fake_rewrite)

    result = run_all(
        Config(data_dir=str(tmp_path)),
        date="2026-01-02",
        review=True,
        rewrite=True,
    )

    assert calls == [("review", "radar", "zack_draft", "2026-01-02")]
    assert result.review_path.endswith("2026-01-02_zack_draft_review.json")
    assert result.revised_draft_path == ""

    status = json.loads((tmp_path / "radar" / "runs" / "2026-01-02.json").read_text())
    rewrite_stage = next(stage for stage in status["stages"] if stage["name"] == "pair-rewrite")
    assert rewrite_stage["status"] == "skipped"
    assert rewrite_stage["metrics"]["reason"] == "pair-review blocked"


def test_run_all_auto_researches_high_risk_selection(monkeypatch, tmp_path: Path):
    calls = []
    video = RadarVideo(video_id="v", url="u", platform=Platform.youtube)
    selection = ZackSelection(
        date="2026-01-02",
        selected_video_id=video.video_id,
        selected_video=video,
        fact_risk="high",
    )
    monkeypatch.setattr("ai_clip.radar.stage.run_collect", lambda *a, **k: 1)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_ranking", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_source_content", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_content_rerank", lambda *a, **k: None)
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_selection", lambda *a, **k: selection)
    monkeypatch.setattr(
        "ai_clip.radar.stage.run_source_research",
        lambda cfg, date: calls.append(("research", cfg.source_research.max_searches)),
    )
    monkeypatch.setattr("ai_clip.radar.stage.run_zack_draft", lambda *a, **k: None)
    cfg = Config(data_dir=str(tmp_path))
    cfg.source_research.tavily_api_key = "key"
    cfg.source_research.max_searches = 3

    run_all(cfg, date="2026-01-02")

    assert calls == [("research", 2)]


def test_run_all_rejects_active_lock(tmp_path: Path):
    lock_dir = tmp_path / "radar" / "locks"
    lock_dir.mkdir(parents=True)
    (lock_dir / "2026-01-02.lock").write_text(
        json.dumps({
            "pid": os.getpid(),
            "date": "2026-01-02",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(RuntimeError, match="already running"):
        run_all(Config(data_dir=str(tmp_path)), date="2026-01-02")


def test_zack_selection_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    candidates_dir = tmp_path / "radar" / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "2026-01-02.json").write_text(
        """
        {
          "date": "2026-01-02",
          "top_n": 1,
          "videos": [
            {
              "video_id": "youtube:1",
              "url": "https://example.com/v",
              "platform": "youtube",
              "title": "事件标题",
              "transcript_text": "事件脚本"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    cfg = Config(data_dir=str(tmp_path))

    monkeypatch.setattr(
        "ai_clip.radar.stage.generate_zack_selection",
        lambda candidates, cfg: ZackSelection(
            date=candidates.date,
            selected_video_id=candidates.videos[0].video_id,
            selected_index=1,
            selected_video=candidates.videos[0],
            topic="事件标题",
            fact_risk="low",
        ),
    )

    selection = run_zack_selection(cfg, date="2026-01-02")

    assert selection.topic == "事件标题"
    assert (tmp_path / "radar" / "selections" / "2026-01-02.json").exists()
    assert (tmp_path / "radar" / "selections" / "2026-01-02.md").exists()
    manifest = read_artifact_manifest(tmp_path / "radar" / "selections" / "2026-01-02.json")
    assert manifest.stage == "zack-selection"


def test_zack_selection_rejects_empty_candidates_before_llm(monkeypatch, tmp_path: Path):
    candidates_dir = tmp_path / "radar" / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "2026-01-02.json").write_text(
        '{"date": "2026-01-02", "top_n": 3, "videos": []}',
        encoding="utf-8",
    )
    called = False

    def fake_generate(candidates, cfg):
        nonlocal called
        called = True

    monkeypatch.setattr("ai_clip.radar.stage.generate_zack_selection", fake_generate)

    import pytest

    with pytest.raises(ValueError, match="no candidates available"):
        run_zack_selection(Config(data_dir=str(tmp_path)), date="2026-01-02")

    assert called is False


def test_source_research_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    selected = RadarVideo(
        video_id="youtube:1",
        url="https://example.com/v",
        platform=Platform.youtube,
        title="事件标题",
        transcript_text="事件脚本",
    )
    selection_dir = tmp_path / "radar" / "selections"
    selection_dir.mkdir(parents=True)
    (selection_dir / "2026-01-02.json").write_text(
        ZackSelection(
            date="2026-01-02",
            selected_video_id=selected.video_id,
            selected_index=1,
            selected_video=selected,
            topic="事件标题",
        ).model_dump_json(),
        encoding="utf-8",
    )
    cfg = Config(data_dir=str(tmp_path))

    def fake_research(selection, cfg):
        from ai_clip.source_research.models import SourceResearchReport

        return SourceResearchReport(
            date=selection.date,
            selected_video_id=selection.selected_video_id,
            search_calls=1,
            markdown="# Source Research\n\nok",
        )

    monkeypatch.setattr("ai_clip.radar.stage.generate_source_research", fake_research)

    report = run_source_research(cfg, date="2026-01-02")

    assert report.search_calls == 1
    assert (tmp_path / "radar" / "research" / "2026-01-02.json").exists()
    assert (tmp_path / "radar" / "research" / "2026-01-02.md").read_text(encoding="utf-8") == "# Source Research\n\nok"
    manifest = read_artifact_manifest(tmp_path / "radar" / "research" / "2026-01-02.md")
    assert manifest.stage == "source-research"
    assert manifest.params["max_searches"] == "2"


def test_zack_draft_injects_existing_source_research(monkeypatch, tmp_path: Path):
    candidates_dir = tmp_path / "radar" / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "2026-01-02.json").write_text(
        """
        {
          "date": "2026-01-02",
          "top_n": 1,
          "videos": [
            {
              "video_id": "youtube:1",
              "url": "https://example.com/v",
              "platform": "youtube",
              "title": "事件标题"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    research_dir = tmp_path / "radar" / "research"
    research_dir.mkdir(parents=True)
    research_path = research_dir / "2026-01-02.md"
    research_path.write_text("confirmed detail", encoding="utf-8")
    selection_dir = tmp_path / "radar" / "selections"
    selection_dir.mkdir(parents=True)
    selected = RadarVideo(
        video_id="youtube:1",
        url="https://example.com/v",
        platform=Platform.youtube,
        title="事件标题",
    )
    selection_path = selection_dir / "2026-01-02.json"
    selection_path.write_text(
        ZackSelection(
            date="2026-01-02",
            selected_video_id=selected.video_id,
            selected_index=1,
            selected_video=selected,
            topic="事件标题",
        ).model_dump_json(),
        encoding="utf-8",
    )
    cfg = Config(data_dir=str(tmp_path))
    write_artifact_manifest(
        research_path,
        stage="source-research",
        inputs=[selection_path],
        params={"max_searches": "2"},
        model=cfg.llm.model,
    )
    seen = {}

    def fake_generate(candidates, llm_cfg, selection=None, research_markdown=""):
        seen["selection"] = selection.topic
        seen["research"] = research_markdown
        return "# draft"

    monkeypatch.setattr("ai_clip.radar.stage.generate_zack_draft", fake_generate)

    run_zack_draft(cfg, date="2026-01-02")

    assert seen["selection"] == "事件标题"
    assert seen["research"] == "confirmed detail"
    manifest = read_artifact_manifest(tmp_path / "radar" / "drafts" / "2026-01-02.md")
    assert manifest.stage == "zack-draft"
    assert manifest.params["research_used"] == "True"


def test_zack_draft_ignores_stale_source_research(monkeypatch, tmp_path: Path):
    candidates_dir = tmp_path / "radar" / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "2026-01-02.json").write_text(
        """
        {
          "date": "2026-01-02",
          "top_n": 1,
          "videos": [
            {
              "video_id": "youtube:1",
              "url": "https://example.com/v",
              "platform": "youtube",
              "title": "事件标题"
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    research_dir = tmp_path / "radar" / "research"
    research_dir.mkdir(parents=True)
    research_path = research_dir / "2026-01-02.md"
    research_path.write_text("stale detail", encoding="utf-8")
    selection_dir = tmp_path / "radar" / "selections"
    selection_dir.mkdir(parents=True)
    selected = RadarVideo(
        video_id="youtube:1",
        url="https://example.com/v",
        platform=Platform.youtube,
        title="事件标题",
    )
    selection_path = selection_dir / "2026-01-02.json"
    selection = ZackSelection(
        date="2026-01-02",
        selected_video_id=selected.video_id,
        selected_index=1,
        selected_video=selected,
        topic="事件标题",
    )
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")
    cfg = Config(data_dir=str(tmp_path))
    write_artifact_manifest(
        research_path,
        stage="source-research",
        inputs=[selection_path],
        params={"max_searches": "2"},
        model=cfg.llm.model,
    )
    selection_path.write_text(
        selection.model_copy(update={"topic": "已经变更的选题"}).model_dump_json(),
        encoding="utf-8",
    )
    seen = {}

    def fake_generate(candidates, llm_cfg, selection=None, research_markdown=""):
        seen["research"] = research_markdown
        return "# draft"

    monkeypatch.setattr("ai_clip.radar.stage.generate_zack_draft", fake_generate)

    run_zack_draft(cfg, date="2026-01-02")

    assert seen["research"] == ""


def test_source_content_transcribes_audio_when_subtitles_missing(monkeypatch, tmp_path: Path):
    seen_opts = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            seen_opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            if self.opts.get("skip_download"):
                return {}
            out = Path(self.opts["outtmpl"].replace("%(ext)s", "m4a"))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("audio", encoding="utf-8")
            return {"id": "abc", "ext": "m4a"}

        def prepare_filename(self, info):
            return self.opts["outtmpl"].replace("%(ext)s", info["ext"])

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(
        "ai_clip.extract.remote.transcribe_audio",
        lambda audio, whisper: ([], "zh", "转写脚本"),
    )

    video = RadarVideo(
        video_id="youtube:abc",
        url="https://www.youtube.com/watch?v=abc",
        platform=Platform.youtube,
        channel_url="https://www.youtube.com/@x",
    )
    channel = ChannelSpec(
        platform=Platform.youtube,
        url="https://www.youtube.com/@x",
        cookies="cookies.txt",
    )
    enriched = add_source_content([video], [channel], tmp_path, whisper=Config().whisper)

    assert enriched[0].transcript_text == "转写脚本"
    assert enriched[0].transcript_language == "zh"
    assert enriched[0].transcript_source == "whisper"
    assert enriched[0].content_status == "available"
    assert Path(enriched[0].content_cache_path).exists()
    assert seen_opts[0]["cookiefile"] == "cookies.txt"
    assert seen_opts[1]["format"] == "bestaudio/best"


def test_source_content_skips_subtitle_download_errors(monkeypatch, tmp_path: Path):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            import yt_dlp

            raise yt_dlp.utils.DownloadError("HTTP Error 429")

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    video = RadarVideo(
        video_id="youtube:abc",
        url="https://www.youtube.com/watch?v=abc",
        platform=Platform.youtube,
    )
    enriched = add_source_content(
        [video],
        [],
        tmp_path,
        whisper=Config().whisper,
        transcribe_missing=False,
    )

    assert enriched[0].transcript_text == ""
    assert enriched[0].content_status == "missing"
    assert "transcription disabled" in enriched[0].content_error


def test_source_content_reuses_cached_script(tmp_path: Path):
    cache_dir = tmp_path / "youtube_abc"
    cache_dir.mkdir()
    (cache_dir / "script.json").write_text(
        json.dumps({
            "text": "cached script",
            "language": "en",
            "source": "subtitles",
            "segments": [],
        }),
        encoding="utf-8",
    )
    video = RadarVideo(
        video_id="youtube:abc",
        url="https://www.youtube.com/watch?v=abc",
        platform=Platform.youtube,
    )

    enriched = add_source_content(
        [video],
        [],
        tmp_path,
        whisper=Config().whisper,
        transcribe_missing=False,
    )

    assert enriched[0].transcript_text == "cached script"
    assert enriched[0].content_status == "cached"


def test_radar_backfill_writes_daily_top_files(monkeypatch, tmp_path: Path):
    channels_path = tmp_path / "channels.yaml"
    channels_path.write_text(
        """
channels:
  - platform: youtube
    url: https://www.youtube.com/@x
    name: X
""",
        encoding="utf-8",
    )

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, *args, **kwargs):
            return {
                "entries": [
                    {
                        "id": "a",
                        "title": "A",
                        "view_count": 1000,
                        "like_count": 100,
                        "duration": 60,
                        "upload_date": "20260101",
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "view_count": 2000,
                        "like_count": 100,
                        "duration": 60,
                        "upload_date": "20260102",
                    },
                ]
            }

    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    cfg = Config(data_dir=str(tmp_path))
    cfg.radar.channels_path = str(channels_path)
    result = run_backfill(cfg, days=2, end_date="2026-01-02", top_n=3, channel_timeout=0)
    assert result.collected == 2
    out_dir = Path(result.output_dir)
    assert (out_dir / "2026-01-01_top3.json").exists()
    assert (out_dir / "2026-01-02_top3.md").exists()
    assert (out_dir / "2026-01-02_summary.md").exists()



