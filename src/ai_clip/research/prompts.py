QUERY_SYSTEM = (
    "You plan small web-research batches for short-video production. "
    "Create one search per requested angle. Prefer official, institutional, "
    "mainstream media, primary data, or high-quality explanatory sources."
)

QUERY_USER = """Create web search queries for a project-level research brief.

Theme:
{theme}

Viral analysis:
{analysis}

Source transcript:
{transcript}

Research budget: {max_searches}
Required research angles:
{focus}

Rules:
- Create at most one query for each required angle.
- Search for factual claims, named entities, dates, mechanisms, data, and context.
- Prefer Chinese queries for Chinese topics and English queries for English/global topics.
- Keep each query broad enough to find reliable sources, not only the source title.
- Do not create more than {max_searches} queries.

Return JSON only:
{{
  "queries": [
    {{"angle": "event_facts", "query": "...", "rationale": "..."}}
  ]
}}
"""

SYNTHESIS_SYSTEM = (
    "You synthesize research for original short-video storyboards and scripts. "
    "Separate confirmed facts from uncertain claims. Do not invent facts beyond "
    "the search results. Look for original angles that connect the topic to "
    "biology, ecology, complex systems, incentives, feedback loops, or group behavior."
)

SYNTHESIS_USER = """Synthesize a project-level research brief.

Theme:
{theme}

Viral analysis:
{analysis}

Source transcript:
{transcript}

Search results:
{results}

Return Markdown with this structure:

# Research Brief

## Confirmed Facts
- Facts directly supported by reliable search results.

## Uncertain Claims
- Claims that need caution, better sourcing, or safer wording.

## Original Angles
- Short-video angles that go beyond summarizing the source.
- Favor biology/ecology/complex-systems metaphors only when they clarify the topic.

## Useful Context
- Background, dates, named entities, numbers, or mechanisms that can improve the script.

## Storyboard Guidance
- What the storyboard should emphasize or avoid.

## Sources
- Title | URL | why it matters
"""
