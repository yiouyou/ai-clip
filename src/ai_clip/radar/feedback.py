from __future__ import annotations

import json
from pathlib import Path

from ai_clip.core.artifacts import write_text_atomic
from ai_clip.radar.models import RadarCandidates, RadarFeedbackEvent, RadarVideo, ZackSelection
from ai_clip.radar.storage import RadarPaths
from ai_clip.zack_ranking import RankingFeedback


def read_feedback_events(path: Path) -> list[RadarFeedbackEvent]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(RadarFeedbackEvent.model_validate_json(line))
        except (ValueError, json.JSONDecodeError):
            continue
    return events


def record_feedback(
    paths: RadarPaths,
    decision: str,
    video_id: str = "",
    reason: str = "",
) -> RadarFeedbackEvent:
    candidates = _read_candidates(paths.candidates_json)
    selection = _read_selection(paths.selection_json)
    selected_id = video_id or (selection.selected_video_id if selection else "")
    video = _find_video(candidates, selection, selected_id)
    if video is None:
        raise ValueError(f"video not found for feedback: {selected_id or '(selected topic)'}")
    event = RadarFeedbackEvent(
        date=paths.date,
        video_id=video.video_id,
        decision=decision,
        reason=reason.strip(),
        title=video.title,
        topic=selection.topic if selection and selection.selected_video_id == video.video_id else "",
        pool=video.pool,
        platform=video.platform,
        tags=video.tags,
    )
    events = [
        item for item in read_feedback_events(paths.feedback_events_jsonl)
        if (item.date, item.video_id) != (event.date, event.video_id)
    ]
    events.append(event)
    content = "".join(item.model_dump_json() + "\n" for item in events)
    write_text_atomic(paths.feedback_events_jsonl, content, encoding="utf-8")
    return event


def apply_feedback_events(
    feedback: RankingFeedback,
    events: list[RadarFeedbackEvent],
) -> RankingFeedback:
    pool_counts: dict[str, list[int]] = {}
    tag_counts: dict[str, list[int]] = {}
    rejected_ids = set(feedback.avoid_video_ids)
    for event in events:
        value = 1 if event.decision == "accept" else -1
        pool_counts.setdefault(event.pool, []).append(value)
        for tag in set(event.tags):
            tag_counts.setdefault(tag, []).append(value)
        if event.decision == "reject":
            rejected_ids.add(event.video_id)
    return feedback.model_copy(update={
        "avoid_video_ids": sorted(rejected_ids),
        "learned_pool_multipliers": _learned_multipliers(pool_counts),
        "learned_tag_multipliers": _learned_multipliers(tag_counts),
    })


def _learned_multipliers(counts: dict[str, list[int]]) -> dict[str, float]:
    return {
        key: round(1.0 + 0.25 * sum(values) / (len(values) + 3), 3)
        for key, values in counts.items()
    }


def _read_candidates(path: Path) -> RadarCandidates | None:
    if not path.exists():
        return None
    return RadarCandidates.model_validate_json(path.read_text(encoding="utf-8"))


def _read_selection(path: Path) -> ZackSelection | None:
    if not path.exists():
        return None
    return ZackSelection.model_validate_json(path.read_text(encoding="utf-8"))


def _find_video(
    candidates: RadarCandidates | None,
    selection: ZackSelection | None,
    video_id: str,
) -> RadarVideo | None:
    if candidates:
        for video in candidates.videos:
            if video.video_id == video_id:
                return video
    if selection and selection.selected_video.video_id == video_id:
        return selection.selected_video
    return None
