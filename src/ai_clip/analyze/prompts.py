"""Prompt templates for viral teardown. Inspired by the tiktok-viral-hooks
approach: identify the 3-second hook, the retention move, and a reusable formula."""

ANALYZE_SYSTEM = (
    "You are a short-video viral-content analyst. Given a transcript of a short "
    "video, you reverse-engineer why it holds attention. Be concrete and concise. "
    "Always answer in the same language as the transcript."
)

ANALYZE_USER_TEMPLATE = """Analyze this short-video transcript and return ONLY a JSON object with these keys:
- "hook": the opening hook (first ~3 seconds) and why it grabs attention
- "structure": array of strings, the beat-by-beat structure
- "emotion_curve": array of strings, how emotion/tension evolves
- "formula": one reusable, copy-able formula another creator could apply to a new topic
- "scores": object with float 0-1 fields: hook_strength, retention, emotional_peak, replicability
- "notes": any extra observations

Transcript:
\"\"\"
{transcript}
\"\"\"
"""
