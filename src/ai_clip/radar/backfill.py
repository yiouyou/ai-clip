from __future__ import annotations

from datetime import datetime, timedelta

from ai_clip.core.config import Config
from ai_clip.radar.collect import collect_channels, collect_channels_with_timeout, load_channels
from ai_clip.radar.models import RadarBackfillResult, RadarCandidates, RadarSnapshot
from ai_clip.radar.storage import RadarPaths, write_json_model
from ai_clip.radar.time import today_in_tz
from ai_clip.zack_draft import render_brief
from ai_clip.zack_ranking import rank_videos


def run_backfill(
    cfg: Config,
    days: int = 7,
    end_date: str | None = None,
    top_n: int | None = None,
    channel_limit: int | None = None,
    channel_timeout: int = 30,
) -> RadarBackfillResult:
    end_date = end_date or today_in_tz(cfg.radar.timezone)
    top_n = top_n or cfg.radar.top_n
    days = max(days, 1)
    end = datetime.fromisoformat(end_date).date()
    wanted_dates = [(end - timedelta(days=offset)).isoformat() for offset in range(days)]
    wanted = set(wanted_dates)

    effective_channel_limit = channel_limit or cfg.radar.channel_limit
    radar_cfg = cfg.radar.model_copy(update={
        "since_days": 0,
        "channel_limit": effective_channel_limit,
        "bilibili_detail_limit": effective_channel_limit,
    })
    channels = load_channels(cfg.radar.channels_path)
    if channel_timeout <= 0:
        collected_snapshots = collect_channels(channels, radar_cfg)
    else:
        collected_snapshots = collect_channels_with_timeout(
            channels,
            radar_cfg,
            channel_timeout,
        ).snapshots
    snapshots = [
        snapshot for snapshot in collected_snapshots if snapshot.video.published_date in wanted
    ]

    paths = RadarPaths(cfg.data_dir, end_date)
    paths.ensure()
    out_dir = paths.backfill_run_dir(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    by_date: dict[str, list[RadarSnapshot]] = {date: [] for date in reversed(wanted_dates)}
    for snapshot in snapshots:
        by_date.setdefault(snapshot.video.published_date, []).append(snapshot)

    summary_lines = [f"# Radar Backfill Top {top_n}: {end_date}", ""]
    for date, day_snapshots in by_date.items():
        ranked = rank_videos(day_snapshots, previous={}, top_n=top_n)
        candidates = RadarCandidates(date=date, top_n=top_n, videos=ranked)
        json_path = out_dir / f"{date}_top{top_n}.json"
        md_path = out_dir / f"{date}_top{top_n}.md"
        write_json_model(json_path, candidates)
        md_path.write_text(render_brief(candidates), encoding="utf-8")
        files.extend([str(json_path), str(md_path)])
        summary_lines += [f"## {date}", ""]
        if not ranked:
            summary_lines += ["- No candidates collected.", ""]
            continue
        for index, video in enumerate(ranked, start=1):
            summary_lines.append(
                f"{index}. {video.title} | {video.platform} | score={video.score} | {video.url}"
            )
        summary_lines.append("")

    summary_path = out_dir / f"{end_date}_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    files.append(str(summary_path))
    return RadarBackfillResult(
        end_date=end_date,
        days=days,
        collected=len(snapshots),
        output_dir=str(out_dir),
        files=files,
    )
