from __future__ import annotations

from ai_clip.core.config import LLMConfig
from ai_clip.core.llm import chat
from ai_clip.radar.models import RadarCandidates, RadarVideo, ZackSelection
from ai_clip.zack_draft.prompts import ZACK_DRAFT_SYSTEM, ZACK_DRAFT_USER
from ai_clip.zack_selection.selector import selection_for_prompt


def generate_zack_draft(
    candidates: RadarCandidates,
    cfg: LLMConfig,
    selection: ZackSelection | None = None,
    research_markdown: str = "",
) -> str:
    videos_text = _videos_for_prompt(candidates.videos)
    selection_text = (
        selection_for_prompt(selection)
        if selection is not None
        else "(no zack-selection available; choose one main topic from candidates)"
    )
    research = research_markdown.strip() or "(no source research available)"
    return chat(
        cfg,
        system=ZACK_DRAFT_SYSTEM,
        user=ZACK_DRAFT_USER.format(
            date=candidates.date,
            videos=videos_text,
            selection=selection_text,
            research=research,
        ),
    )


def render_brief(candidates: RadarCandidates) -> str:
    lines = [f"# 今日选题候选 {candidates.date}", ""]
    for i, video in enumerate(candidates.videos, start=1):
        lines += [
            f"## {i}. {video.title}",
            "",
            f"- Platform: {video.platform}",
            f"- URL: {video.url}",
            f"- Channel: {video.channel_name or video.uploader}",
            f"- Pool/role/lens: {video.pool} / {video.role} / {video.lens_fit:g}",
            f"- Score: {video.score}",
            f"- Metrics: views={video.view_count:,}, likes={video.like_count:,}, comments={video.comment_count:,}",
            f"- Reasons: {'; '.join(video.score_reasons)}",
            f"- Script: {_script_status(video)}",
            "",
        ]
    return "\n".join(lines)


def _videos_for_prompt(videos: list[RadarVideo]) -> str:
    blocks = []
    for i, video in enumerate(videos, start=1):
        transcript = video.transcript_text[:4000] if video.transcript_text else "(no script available)"
        blocks.append(
            "\n".join([
                f"## Candidate {i}",
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
            ])
        )
    return "\n\n".join(blocks)


def _script_status(video: RadarVideo) -> str:
    if not video.transcript_text:
        return "missing"
    source = f" via {video.transcript_source}" if video.transcript_source else ""
    language = f", lang={video.transcript_language}" if video.transcript_language else ""
    return f"available{source}{language}"
