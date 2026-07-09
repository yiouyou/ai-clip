from __future__ import annotations

from ai_clip.core import llm as llm_mod
from ai_clip.core.config import LLMConfig
from ai_clip.core.models import Intent, Transcript, ViralAnalysis

SOURCE_DRAFT_SYSTEM = (
    "You are an editorial producer for original Chinese talking-head videos. "
    "The creator has a biology background and likes explaining politics, economics, "
    "technology, AI, and society through biology, ecology, evolution, immune systems, "
    "complex systems, feedback loops, group behavior, and resource metabolism. "
    "Use the source video as signal and source material, but do not copy its structure "
    "or wording. Turn it into an original point of view."
)

SOURCE_DRAFT_USER = """Create an original talking-head draft from this source video.

Intent: {intent}
Optional stance: {stance}

Research brief:
{research}

Source analysis:
- hook: {hook}
- formula: {formula}
- stance: {analysis_stance}
- structure: {structure}
- notes: {notes}

Transcript excerpt:
```text
{transcript}
```

Requirements:
- Write in Chinese.
- The source video is input material, not a script to copy.
- Prefer the creator's biology/complex-systems lens when it is natural.
- If the topic is political/economic/social/technology news, separate title framing
  from verified facts and avoid repeating unsupported claims.
- Avoid weather/natural-disaster topics unless they reveal a broader social,
  economic, technological, or systems principle.
- Recommend one clear original angle.
- Include factual-check notes and safe wording boundaries.
- Make the口播稿 speakable, not an essay.

Return Markdown:

# 单视频原创口播稿

## 原创角度

## 事实核查与安全边界
- 可采用事实
- 待核查断言
- 不建议复述的说法

## 口播稿
- 标题
- 3秒开头
- 主体结构
- 正文

## 可视化/素材建议
"""


def generate_source_draft(
    transcript: Transcript,
    analysis: ViralAnalysis | None,
    cfg: LLMConfig,
    intent: Intent = Intent.info,
    stance: str = "",
    research_markdown: str = "",
) -> str:
    excerpt = transcript.text[:8000]
    research = research_markdown.strip() or "(no research brief available)"
    reply = llm_mod.chat(
        cfg,
        system=SOURCE_DRAFT_SYSTEM,
        user=SOURCE_DRAFT_USER.format(
            intent=intent.value,
            stance=stance,
            research=research[:5000],
            hook=analysis.hook if analysis else "",
            formula=analysis.formula if analysis else "",
            analysis_stance=analysis.stance if analysis else "",
            structure="; ".join(analysis.structure) if analysis else "",
            notes=analysis.notes if analysis else "",
            transcript=excerpt,
        ),
    )
    return reply.strip()
