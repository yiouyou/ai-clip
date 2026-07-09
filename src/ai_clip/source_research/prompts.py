QUERY_SYSTEM = (
    "You plan very small web-research batches for one selected short-video topic. "
    "Create one search per requested research angle. Do not duplicate angles. "
    "Prefer official, institutional, mainstream media, or primary data sources."
)

QUERY_USER = """Create web search queries for today's selected topic.

Date: {date}
Search budget: {max_searches}

Selection:
{selection}

Selected source material:
{source}

Required research angles:
{focus}

Rules:
- Create at most one query for each required angle.
- Keep searches broad enough to find reliable sources, not just the source video title.
- Prefer Chinese queries for Chinese topics and English queries for English/global topics.
- Focus on factual claims, dates, named entities, institutions, data, and event context.
- If there is a high-risk or sensational title claim, one query should explicitly test
  whether that framing is verified, exaggerated, or disputed.
- Do not create more than {max_searches} queries.

Return JSON only:
{{
  "queries": [
    {{"angle": "event_facts", "query": "...", "rationale": "..."}}
  ]
}}
"""

SYNTHESIS_SYSTEM = (
    "You synthesize source research for original Chinese talking-head videos. "
    "Separate verified facts from uncertain claims, and rewrite risky title framing "
    "into safe, source-grounded wording. Do not invent facts beyond the search results."
)

SYNTHESIS_USER = """Synthesize source research for zack-draft.

Date: {date}
Selection:
{selection}

Selected source material:
{source}

Search results:
{results}

Return Markdown with this structure:

# Source Research {date}

## Confirmed Facts
- Facts that are directly supported by reliable search results.

## Uncertain Claims
- Claims that still need caution, better sourcing, or safer wording.

## Safe Framing
- How zack-draft should phrase the topic without repeating title bait.

## Useful Context
- Background, dates, named entities, numbers, or mechanisms useful for original commentary.

## Angle Opportunities
- Original angles that connect the event to biology, ecology, complex systems, incentives, feedback loops, or group behavior.

## Sources
- Title | URL | why it matters
"""
