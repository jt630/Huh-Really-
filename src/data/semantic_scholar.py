"""
Semantic Scholar Academic Graph API - Paper search and citation data.

API: https://api.semanticscholar.org/graph/v1/
Rate limits: 100 req/5min (no key); higher with SEMANTIC_SCHOLAR_API_KEY

Fields: paper_id, title, abstract, authors, year, venue,
        citation_count, influential_citation_count,
        external_ids, fields_of_study, tldr
"""
from pydantic import BaseModel


class SemanticScholarPaper(BaseModel):
    paper_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    citation_count: int = 0
    influential_citation_count: int = 0
    external_ids: dict[str, str] = {}
    fields_of_study: list[str] = []
    tldr: str | None = None
