from ai_clip.source_research.models import ResearchQuery, SearchResult, SourceResearchReport


def generate_source_research(*args, **kwargs):
    from ai_clip.source_research.stage import generate_source_research as generate

    return generate(*args, **kwargs)

__all__ = [
    "ResearchQuery",
    "SearchResult",
    "SourceResearchReport",
    "generate_source_research",
]
