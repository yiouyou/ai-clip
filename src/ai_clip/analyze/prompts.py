"""Prompt templates for viral teardown. Inspired by the tiktok-viral-hooks
approach (3-second hook, retention move, copy-able formula) and the
AI-Youtube-Shorts-Generator idea of scoring why a clip performs — implemented as
a single structured LLM call rather than a dependency.

Intent steers what we extract: `info` = knowledge structure; `emotion` = the
stance + emotional charge to express an opinionated take (not neutral reporting);
`sales` = audience pain points and objections to convert against.
"""

from ai_clip.core.models import Intent

ANALYZE_SYSTEM = (
    "You are a short-video viral-content analyst. Given a transcript of a short "
    "video, you reverse-engineer why it holds attention. Be concrete and concise. "
    "Always answer in the same language as the transcript."
)

_BASE_KEYS = """- "hook": the opening hook (first ~3 seconds) and why it grabs attention
- "structure": array of strings, the beat-by-beat structure
- "emotion_curve": array of strings, how emotion/tension evolves
- "formula": one reusable, copy-able formula another creator could apply to a new topic
- "scores": object with float 0-1 fields: hook_strength, retention, emotional_peak, replicability
- "notes": any extra observations"""

_INTENT_KEYS = {
    Intent.info: "",
    Intent.emotion: (
        '\n- "stance": the underlying opinion/attitude (not neutral facts) that '
        "could be sharpened into an opinionated take on this news/event"
    ),
    Intent.sales: (
        '\n- "pain_points": array of the audience pain points this could sell against'
        '\n- "objections": array of likely buyer objections to overcome'
    ),
}

_INTENT_FOCUS = {
    Intent.info: "Focus on the knowledge structure.",
    Intent.emotion: (
        "Focus on the STANCE and emotional charge, not neutral reporting: what "
        "attitude/opinion would resonate or provoke if expressed about this."
    ),
    Intent.sales: (
        "Focus on what makes an audience buy: pain points, desire, and objections."
    ),
}

ANALYZE_USER_TEMPLATE = """Analyze this short-video transcript ({intent} intent). {focus}
Return ONLY a JSON object with these keys:
{keys}

Transcript:
\"\"\"
{transcript}
\"\"\"
"""


def build_user_prompt(transcript: str, intent: Intent) -> str:
    return ANALYZE_USER_TEMPLATE.format(
        intent=intent.value,
        focus=_INTENT_FOCUS[intent],
        keys=_BASE_KEYS + _INTENT_KEYS[intent],
        transcript=transcript,
    )
