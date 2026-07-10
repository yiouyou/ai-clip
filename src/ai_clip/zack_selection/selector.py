from __future__ import annotations

import json

from ai_clip.core.config import LLMConfig
from ai_clip.core.llm import chat, extract_json
from ai_clip.radar.models import RadarCandidates, RadarVideo, ZackSelection
from ai_clip.zack_selection.prompts import ZACK_SELECTION_SYSTEM, ZACK_SELECTION_USER


def generate_zack_selection(candidates: RadarCandidates, cfg: LLMConfig) -> ZackSelection:
    text = chat(
        cfg,
        system=ZACK_SELECTION_SYSTEM,
        user=ZACK_SELECTION_USER.format(
            date=candidates.date,
            videos=_videos_for_prompt(candidates.videos),
        ),
    )
    data = extract_json(text)
    return _selection_from_data(candidates, data)


def fallback_selection(candidates: RadarCandidates) -> ZackSelection:
    return _selection_from_data(candidates, {})


def render_selection(selection: ZackSelection) -> str:
    focus = selection.research_focus or []
    lines = [
        f"# Zack Selection {selection.date}",
        "",
        f"- Selected index: {selection.selected_index}",
        f"- Selected video id: {selection.selected_video_id}",
        f"- Topic: {selection.topic}",
        f"- Angle: {selection.angle}",
        f"- Fact risk: {selection.fact_risk}",
        f"- URL: {selection.selected_video.url}",
        "",
        "## Why Selected",
        "",
        selection.why_selected or "(not provided)",
        "",
        "## Research Focus",
        "",
    ]
    if focus:
        lines.extend(f"- {item}" for item in focus)
    else:
        lines.append("- event_facts: verify the selected topic with reliable sources")
    if selection.backup_video_ids:
        lines.extend(["", "## Backup Video IDs", ""])
        lines.extend(f"- {video_id}" for video_id in selection.backup_video_ids)
    return "\n".join(lines).strip() + "\n"


def selection_for_prompt(selection: ZackSelection) -> str:
    return json.dumps({
        "selected_video_id": selection.selected_video_id,
        "selected_index": selection.selected_index,
        "topic": selection.topic,
        "angle": selection.angle,
        "why_selected": selection.why_selected,
        "fact_risk": selection.fact_risk,
        "research_focus": selection.research_focus,
        "selected_title": selection.selected_video.title,
        "selected_url": selection.selected_video.url,
    }, ensure_ascii=False, indent=2)


def _selection_from_data(candidates: RadarCandidates, data: dict) -> ZackSelection:
    if not candidates.videos:
        raise ValueError("no candidates available for zack-selection")
    index = _selected_index(data, candidates.videos)
    selected = candidates.videos[index - 1]
    selected_id = selected.video_id
    topic = str(data.get("topic") or selected.title).strip()
    focus = _focus_list(data.get("research_focus"))
    return ZackSelection(
        date=candidates.date,
        selected_video_id=selected_id,
        selected_index=index,
        selected_video=selected,
        topic=topic,
        angle=str(data.get("angle") or "").strip(),
        why_selected=str(data.get("why_selected") or "").strip(),
        fact_risk=_fact_risk(data.get("fact_risk")),
        research_focus=focus,
        backup_video_ids=[video.video_id for video in candidates.videos if video.video_id != selected_id],
    )


def _selected_index(data: dict, videos: list[RadarVideo]) -> int:
    selected_id = str(data.get("selected_video_id") or "").strip()
    if selected_id:
        for index, video in enumerate(videos, start=1):
            if video.video_id == selected_id:
                return index
    try:
        index = int(data.get("selected_index") or 1)
    except (TypeError, ValueError):
        index = 1
    return min(max(index, 1), len(videos))


def _fact_risk(value: object) -> str:
    risk = str(value or "medium").strip().lower()
    if risk in {"low", "medium", "high"}:
        return risk
    return "medium"


def _focus_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _videos_for_prompt(videos: list[RadarVideo]) -> str:
    blocks = []
    for i, video in enumerate(videos, start=1):
        transcript = video.transcript_text[:3500] if video.transcript_text else "(no script available)"
        blocks.append("\n".join([
            f"## Candidate {i}",
            f"video_id: {video.video_id}",
            f"title: {video.title}",
            f"url: {video.url}",
            f"platform: {video.platform}",
            f"channel: {video.channel_name or video.uploader}",
            f"pool: {video.pool}",
            f"role: {video.role}",
            f"lens_fit: {video.lens_fit}",
            f"tags: {', '.join(video.tags)}",
            f"score: {video.score}",
            f"metrics: views={video.view_count}, likes={video.like_count}, comments={video.comment_count}, "
            f"shares={video.share_count}, favorites={video.favorite_count}, coins={video.coin_count}, "
            f"danmaku={video.danmaku_count}, age_days={video.age_days:.2f}",
            f"score_reasons: {'; '.join(video.score_reasons)}",
            "script:",
            transcript,
        ]))
    return "\n\n".join(blocks)
