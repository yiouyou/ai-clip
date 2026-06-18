"""Storyboard generation prompts. Produces shot-by-shot image/video prompts that
a human can paste into 即梦/Gemini/ComfyUI, or that the ComfyUI API can consume."""

STORYBOARD_SYSTEM = (
    "You are a short-video director. You turn a topic (optionally guided by a "
    "viral formula) into a concrete shot list for a vertical short video. "
    "Image prompts and video prompts must be vivid and self-contained. "
    "Write prompts in Chinese when the theme is Chinese."
)

STORYBOARD_USER_TEMPLATE = """Create a storyboard for a short video.

Theme: {theme}
Target length: about {duration} seconds, aspect ratio {aspect}.
{formula_block}

Return ONLY a JSON object:
{{
  "shots": [
    {{
      "duration_sec": <float>,
      "shot_type": "<close-up|wide|...>",
      "image_prompt": "<text-to-image prompt for the key frame>",
      "video_prompt": "<image-to-video prompt referencing that frame>",
      "voiceover": "<one line of narration for this shot>"
    }}
  ]
}}
Aim for {n_shots} shots whose durations roughly sum to the target length.
"""


def formula_block(formula: str) -> str:
    if not formula:
        return "No reference formula; design an original structure with a strong hook."
    return f"Apply this proven viral formula to the new theme:\n{formula}"
