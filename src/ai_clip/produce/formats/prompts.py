"""All storyboard-generation prompts in one place (one module per stage, mirroring
analyze/prompts.py). Format modules keep only the LLM-output→Shot mapping logic;
the prompt *text* lives here so it can be reviewed and tuned without touching code.

Includes the shared fragments injected into every format: `intent_block`
(info/emotion/sales steering) and `formula_block` (the analyzed viral formula).
"""

from __future__ import annotations

from ai_clip.core.models import Intent, ViralAnalysis
from ai_clip.produce.formats.base import GenerateArgs

# ---------------------------------------------------------------- talking_head
TALKING_HEAD_SYSTEM = (
    "You are a short-video scriptwriter for talking-head / voiceover videos. "
    "You write punchy spoken narration with a strong 3-second hook, and suggest "
    "simple b-roll stills to show while each line is spoken. Write in the theme's language."
)

TALKING_HEAD_USER = """Write a talking-head short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} lines.
{intent}
{formula}
{research}
{asset_engine}

Return ONLY JSON:
{{
  "lines": [
    {{"voiceover": "<one spoken line>",
      "duration_sec": <float>,
      "broll_prompt": "<optional text-to-image prompt for a b-roll still, or empty>",
      "asset_engine": "<optional: smart_illustrator|gemini|comfyui|manual>"}}
  ]
}}
"""

# -------------------------------------------------------------------- slideshow
SLIDESHOW_SYSTEM = (
    "You design 图文 slideshow short videos: a few image cards, each with a short "
    "on-screen caption and a spoken line. The first card must hook hard. "
    "Write in the theme's language."
)

SLIDESHOW_USER = """Design a slideshow short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} cards.
{intent}
{formula}
{research}
{asset_engine}

Return ONLY JSON:
{{
  "cards": [
    {{"caption": "<short on-screen text>",
      "voiceover": "<spoken line>",
      "image_prompt": "<text-to-image prompt for the card>",
      "asset_engine": "<optional: smart_illustrator|gemini|comfyui|manual>",
      "duration_sec": <float>}}
  ]
}}
"""

# ------------------------------------------------------------------------ remix
REMIX_SYSTEM = (
    "You are a 解说/二创 editor who condenses long videos into tight viral shorts. "
    "Given a timestamped transcript, you pick the highest-impact moments and write "
    "fresh narration for each. Spans must stay within the transcript's time range. "
    "Write in the source language."
)

REMIX_USER = """Condense this {source_min:.0f}-minute source video into a ~{duration}s short \
({ratio:.0f}:1 compression) with new narration.

Theme/angle: {theme}
Pick about {n_shots} of the BEST moments — the strongest hook, most quotable lines,
sharpest opinions, conflicts, surprises or emotional peaks — spread across the WHOLE
video (not just the opening). Keep each span SHORT (about 3-8 seconds) and
non-contiguous. The first span must be a scroll-stopping hook. The kept spans should
together sum to roughly {duration} seconds and MUST NOT exceed {duration} seconds.
Each new narration replaces the span's audio and MUST fit inside that span at a natural
speaking pace. Use at most roughly 3 Chinese characters or 2 English words per second;
prefer one sharp sentence over a dense explanation.
{intent}
{formula}
{research}

Timestamped transcript (seconds):
{segments}

Return ONLY JSON:
{{
  "spans": [
    {{"source_start": <float>, "source_end": <float>, "voiceover": "<new narration>"}}
  ]
}}
"""

# ---------------------------------------------------------------------- montage
MONTAGE_SYSTEM = (
    "You are a short-video director. You turn a topic into a concrete shot list "
    "for a vertical short video, with vivid self-contained image and image-to-video "
    "prompts. Write prompts in the theme's language."
)

MONTAGE_USER = """Create a storyboard for a short video.

Theme: {theme}
Target length: ~{duration} seconds, aspect ratio {aspect}, about {n_shots} shots.
{intent}
{formula}
{research}
{asset_engine}

Return ONLY JSON:
{{
  "shots": [
    {{"duration_sec": <float>, "shot_type": "<close-up|wide|...>",
      "image_prompt": "<text-to-image prompt for the key frame>",
      "video_prompt": "<image-to-video prompt referencing that frame>",
      "asset_engine": "<optional: smart_illustrator|gemini|comfyui|manual>",
      "voiceover": "<one line of narration>"}}
  ]
}}
"""


# ----------------------------------------------------- shared prompt fragments
def formula_block(analysis: ViralAnalysis | None) -> str:
    if not analysis or not analysis.formula:
        return "No reference formula; design an original structure with a strong hook."
    return f"Apply this proven viral formula to the new theme:\n{analysis.formula}"


def research_block(markdown: str) -> str:
    text = markdown.strip()
    if not text:
        return "No research brief is available."
    return (
        "Use this research brief for facts, safer framing, and original angles. "
        "Do not copy it verbatim; turn it into concise short-video narration.\n"
        f"{text[:5000]}"
    )


def asset_engine_block() -> str:
    return (
        "For each generated still image, optionally set asset_engine. Use "
        "'smart_illustrator' or 'gemini' for polished information graphics, "
        "concept cards, metaphor images, and thumbnail-like visuals; use 'comfyui' "
        "for photoreal/illustrative local image generation; use 'manual' only when "
        "the shot should be created by a human. Omit asset_engine when unsure."
    )


def intent_block(args: GenerateArgs) -> str:
    """Intent-specific direction injected into every format's prompt."""
    if args.intent == Intent.emotion:
        stance = args.stance or (args.analysis.stance if args.analysis else "")
        lead = f"Take this stance: {stance}." if stance else "Pick a strong, resonant stance."
        return (
            f"INTENT=emotion (opinionated take, NOT neutral reporting). {lead} "
            "Open by asserting an attitude/contrarian opinion; keep an emotional "
            "charge throughout; end on an emotional punch or rhetorical question, "
            "not a neutral summary."
        )
    if args.intent == Intent.sales:
        p = args.product
        if not p or not p.name:
            return (
                "INTENT=sales but no product profile provided; write a generic "
                "pain -> agitate -> product -> proof -> CTA structure."
            )
        sp = "; ".join(p.selling_points)
        return (
            f"INTENT=sales for product '{p.name}': {p.description}. "
            f"Audience: {p.audience}. Selling points: {sp}. "
            f"Structure: hook on the audience pain -> agitate -> reveal {p.name} -> "
            f"proof/selling points -> CTA: {p.cta}."
        )
    return "INTENT=info: lead with a counter-intuitive hook, deliver the key point, end with a takeaway."
