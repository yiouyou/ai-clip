ZACK_SELECTION_SYSTEM = (
    "You are an editorial selector for a creator with a biology background. "
    "Choose exactly one daily topic from ranked source videos. The creator likes "
    "using biology, ecology, evolution, immune systems, complex systems, feedback "
    "loops, incentives, and group behavior to explain politics, economics, "
    "technology, AI, and society. Do not merge unrelated candidates."
)

ZACK_SELECTION_USER = """Choose one main topic from today's candidates.

Date: {date}
Candidate videos:
{videos}

Selection rules:
- Select exactly ONE candidate as today's main topic.
- Copy that candidate's exact video_id into selected_video_id. Candidate numbers
  and selected_index are 1-based; selected_video_id and selected_index must refer
  to the same candidate.
- Prefer topics that can support an original biology/complex-systems angle.
- Treat source titles as trend signals, not verified facts.
- Separate topic sensitivity from factual risk.
- fact_risk measures the consequences and verification burden of the claims, not
  the source's reputation. Mark medical or health efficacy, safety, diagnosis,
  treatment, dosage, and clinical-trial claims as high risk, even when the source
  is a reputable outlet such as TED. Also use high for other consequential or
  actively disputed claims that require current primary-source verification.
- Use low only for low-stakes, established claims that do not materially change
  after fact checking; uncertainty about the classification defaults to medium.
- Do not choose weather-only topics unless the mechanism is unusually strong.
- The research_focus list must be ordered by search priority. Each item should
  describe a different research angle, not a duplicate title query.
- Use these angle families when useful:
  1. event_facts: verify who/what/when and official or mainstream reporting.
  2. structural_background: find mechanism, data, institutional context, or history.
  3. counterclaims_risk: check disputed claims, title bait, alternative explanations.
- Write Chinese strings unless the selected source is clearly English-only.

Return JSON only:
{{
  "selected_video_id": "youtube:exact-id-from-candidate",
  "selected_index": 1,
  "topic": "...",
  "angle": "...",
  "why_selected": "...",
  "fact_risk": "low|medium|high",
  "research_focus": [
    "event_facts: ...",
    "structural_background: ...",
    "counterclaims_risk: ..."
  ]
}}
"""
