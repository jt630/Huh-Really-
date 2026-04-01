"""
Semantic Scholar Academic Graph API - Paper search and citation data.

API: https://api.semanticscholar.org/graph/v1/
Rate limits: 100 req/5min (no key); higher with SEMANTIC_SCHOLAR_API_KEY

Fields: paper_id, title, abstract, authors, year, venue,
        citation_count, influential_citation_count,
        external_ids, fields_of_study, tldr
"""
from __future__ import annotations

import logging
import time

import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = (
    "paperId,title,abstract,authors,year,venue,citationCount,"
    "influentialCitationCount,externalIds,fieldsOfStudy,tldr"
)


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


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(4),
)
def _search(query: str, limit: int, api_key: str | None) -> dict:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    params = {"query": query, "fields": FIELDS, "limit": limit}
    resp = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_semantic_scholar(
    query: str,
    limit: int = 50,
    api_key: str | None = None,
) -> list[SemanticScholarPaper]:
    """
    Search Semantic Scholar for papers matching the query.

    Parameters
    ----------
    query : str
        Free-text search query.
    limit : int
        Maximum number of results.
    api_key : str | None
        Optional Semantic Scholar API key.

    Returns
    -------
    list[SemanticScholarPaper]
    """
    try:
        from src.config import get_settings
        if api_key is None:
            settings = get_settings()
            api_key = getattr(settings, "semantic_scholar_api_key", None)
    except Exception:
        pass

    try:
        data = _search(query, limit, api_key)
        time.sleep(1.0)  # be polite
    except Exception as exc:
        logger.error("Semantic Scholar: search failed: %s", exc)
        return []

    papers = []
    for item in data.get("data", []):
        try:
            authors = [
                a.get("name", "") for a in (item.get("authors") or [])
            ]
            tldr_obj = item.get("tldr")
            tldr = tldr_obj.get("text") if isinstance(tldr_obj, dict) else None
            ext_ids = {k: str(v) for k, v in (item.get("externalIds") or {}).items()}
            papers.append(SemanticScholarPaper(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                abstract=item.get("abstract"),
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                citation_count=item.get("citationCount", 0) or 0,
                influential_citation_count=item.get("influentialCitationCount", 0) or 0,
                external_ids=ext_ids,
                fields_of_study=item.get("fieldsOfStudy") or [],
                tldr=tldr,
            ))
        except Exception as exc:
            logger.debug("Semantic Scholar: skipping record: %s", exc)

    return papers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = search_semantic_scholar("paraquat Parkinson's disease epidemiology", limit=5)
    for p in papers:
        print(p.paper_id, p.title[:80])
