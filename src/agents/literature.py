"""
Literature Agent - Prompt 4.

Capabilities:
- Claude-driven PubMed MeSH query generation
- PubMed search via NCBI E-utilities
- Claude synthesis: supporting/contradicting evidence, gaps
- Iterative deepening with depth parameter
- Facts-vs-analysis separation throughout

[CO-DESIGN] Requires domain expert review before production use.
"""
from __future__ import annotations

import hashlib
import json
import time
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import anthropic
import requests
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

from src.config import get_settings

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LiteratureRequest(BaseModel):
    hypothesis: str
    max_results: int = 50
    depth: int = 1


class Paper(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_date: str
    year: int
    mesh_terms: list[str] = []
    doi: str | None = None


class LiteratureSynthesis(BaseModel):
    supporting_count: int
    contradicting_count: int
    evidence_weight: str  # "Strong"|"Moderate"|"Weak"|"Insufficient"
    key_findings: list[str]       # verbatim: "Author (Year): finding"
    contradictions: list[str]
    dose_response_evidence: str
    evidence_gaps: list[str]
    suggested_queries: list[str]
    confidence: str


class LiteratureResult(BaseModel):
    request: LiteratureRequest
    papers: list[Paper]
    synthesis: LiteratureSynthesis
    queries_used: list[str]


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------


def generate_pubmed_queries(hypothesis: str, client: anthropic.Anthropic) -> list[str]:
    """Convert a hypothesis string into 3-5 PubMed search strings.

    Uses Claude to generate MeSH-term-enriched PubMed queries.

    Parameters
    ----------
    hypothesis:
        Free-text research hypothesis.
    client:
        An initialised Anthropic client.

    Returns
    -------
    list[str]
        Between 3 and 5 PubMed search strings ready for use with ESearch.
    """
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="Return ONLY a JSON array of PubMed search strings. Include MeSH terms where applicable.",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Convert this research hypothesis into 3-5 PubMed search strings "
                    f"with relevant MeSH terms:\n\n{hypothesis}"
                ),
            }
        ],
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# PubMed search helpers
# ---------------------------------------------------------------------------

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _cache_path(query: str) -> Path:
    """Return the filesystem path for caching XML of *query*."""
    settings = get_settings()
    cache_dir = settings.cache_dir / "pubmed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    query_hash = hashlib.md5(query.encode()).hexdigest()
    return cache_dir / f"{query_hash}.xml"


def _esearch(query: str, max_results: int, api_key: str | None) -> list[str]:
    """Run ESearch and return a list of PMIDs."""
    params: dict = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(_ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch_xml(pmids: list[str], api_key: str | None) -> str:
    """Run EFetch for *pmids* and return raw XML string."""
    params: dict = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(_EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def _parse_xml(xml_text: str) -> list[Paper]:
    """Parse PubMed EFetch XML into a list of Paper objects."""
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for article_elem in root.findall(".//PubmedArticle"):
        medline = article_elem.find("MedlineCitation")
        if medline is None:
            continue

        # PMID
        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else ""
        if not pmid:
            continue

        article = medline.find("Article")
        if article is None:
            continue

        # Title
        title_elem = article.find("ArticleTitle")
        title = (title_elem.text or "").strip() if title_elem is not None else ""

        # Abstract
        abstract_parts: list[str] = []
        abstract_elem = article.find("Abstract")
        if abstract_elem is not None:
            for at in abstract_elem.findall("AbstractText"):
                text = (at.text or "").strip()
                label = at.get("Label")
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors: list[str] = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName", default="")
                fore = author.findtext("ForeName", default="")
                if last:
                    authors.append(f"{last} {fore}".strip())
                else:
                    collective = author.findtext("CollectiveName", default="")
                    if collective:
                        authors.append(collective)

        # Journal
        journal_elem = article.find("Journal")
        journal = ""
        if journal_elem is not None:
            journal_title = journal_elem.findtext("Title", default="")
            journal = journal_title.strip()

        # Publication date
        pub_date_elem = None
        if journal_elem is not None:
            journal_issue = journal_elem.find("JournalIssue")
            if journal_issue is not None:
                pub_date_elem = journal_issue.find("PubDate")

        pub_date = ""
        year = 0
        if pub_date_elem is not None:
            year_text = pub_date_elem.findtext("Year", default="")
            month_text = pub_date_elem.findtext("Month", default="")
            day_text = pub_date_elem.findtext("Day", default="")
            medline_date = pub_date_elem.findtext("MedlineDate", default="")
            if year_text:
                pub_date = " ".join(filter(None, [year_text, month_text, day_text]))
                try:
                    year = int(year_text)
                except ValueError:
                    year = 0
            elif medline_date:
                pub_date = medline_date
                try:
                    year = int(medline_date[:4])
                except ValueError:
                    year = 0

        # MeSH terms
        mesh_terms: list[str] = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall("MeshHeading"):
                descriptor = mesh.find("DescriptorName")
                if descriptor is not None and descriptor.text:
                    mesh_terms.append(descriptor.text.strip())

        # DOI
        doi: str | None = None
        pubmed_data = article_elem.find("PubmedData")
        if pubmed_data is not None:
            article_id_list = pubmed_data.find("ArticleIdList")
            if article_id_list is not None:
                for aid in article_id_list.findall("ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text.strip()
                        break

        papers.append(Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            year=year,
            mesh_terms=mesh_terms,
            doi=doi,
        ))

    return papers


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicate papers by PMID, preserving order."""
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        if paper.pmid not in seen:
            seen.add(paper.pmid)
            result.append(paper)
    return result


def search_pubmed(
    queries: list[str],
    max_results: int,
    ncbi_api_key: str | None = None,
) -> list[Paper]:
    """Search PubMed for each query, returning a deduplicated list of Papers.

    Results are cached per-query as XML files under ``data/cache/pubmed/``.
    Rate limiting: 0.34 s between requests without an API key; 0.11 s with one.

    Parameters
    ----------
    queries:
        List of PubMed search strings.
    max_results:
        Maximum number of results to retrieve per query.
    ncbi_api_key:
        Optional NCBI API key to increase rate limits.

    Returns
    -------
    list[Paper]
        Deduplicated list of Papers across all queries.
    """
    sleep_time = 0.11 if ncbi_api_key else 0.34
    all_papers: list[Paper] = []

    for query in queries:
        cache_file = _cache_path(query)

        if cache_file.exists():
            xml_text = cache_file.read_text(encoding="utf-8")
        else:
            # ESearch to get PMIDs
            pmids = _esearch(query, max_results, ncbi_api_key)
            time.sleep(sleep_time)

            if not pmids:
                continue

            # EFetch to get full records
            xml_text = _efetch_xml(pmids, ncbi_api_key)
            time.sleep(sleep_time)

            # Cache the response
            cache_file.write_text(xml_text, encoding="utf-8")

        papers = _parse_xml(xml_text)
        all_papers.extend(papers)

    return _deduplicate(all_papers)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SCHEMA = json.dumps({
    "supporting_count": "<int>",
    "contradicting_count": "<int>",
    "evidence_weight": "Strong|Moderate|Weak|Insufficient",
    "key_findings": ["Author (Year): verbatim statistic or finding"],
    "contradictions": ["Explanation of why studies differ"],
    "dose_response_evidence": "<string describing dose-response evidence or lack thereof>",
    "evidence_gaps": ["<gap description>"],
    "suggested_queries": ["<PubMed query string>"],
    "confidence": "<string e.g. High|Moderate|Low>",
}, indent=2)

_SYNTHESIS_SYSTEM = textwrap.dedent(f"""\
    You are a biomedical research analyst synthesising PubMed literature.
    Separate EVIDENCE from ANALYSIS.
    In key_findings: quote verbatim statistics with Author (Year) citation.
    In contradictions: explain why studies differ.
    Label all interpretations as AI analysis.
    Return ONLY valid JSON matching this schema: {_SYNTHESIS_SCHEMA}
""")


def synthesize_with_claude(
    hypothesis: str,
    papers: list[Paper],
    client: anthropic.Anthropic,
) -> LiteratureSynthesis:
    """Synthesise a list of papers relative to a hypothesis using Claude.

    Parameters
    ----------
    hypothesis:
        The research hypothesis being evaluated.
    papers:
        List of Paper objects retrieved from PubMed.
    client:
        An initialised Anthropic client.

    Returns
    -------
    LiteratureSynthesis
        Structured synthesis of the literature.
    """
    # Build paper summaries (up to 30, truncate long abstracts)
    paper_summaries: list[str] = []
    for i, paper in enumerate(papers[:30]):
        first_author = paper.authors[0] if paper.authors else "Unknown"
        abstract_preview = paper.abstract[:300] if len(paper.abstract) > 300 else paper.abstract
        summary = (
            f"[{i + 1}] {first_author} ({paper.year}). "
            f"{paper.title}. {paper.journal}.\n"
            f"Abstract: {abstract_preview}"
        )
        paper_summaries.append(summary)

    papers_text = "\n\n".join(paper_summaries) if paper_summaries else "No papers retrieved."

    user_prompt = textwrap.dedent(f"""\
        Hypothesis: {hypothesis}

        === Retrieved Papers ({len(papers)} total, showing up to 30) ===

        {papers_text}

        Synthesise the above literature with respect to the hypothesis.
        Count how many papers support vs. contradict the hypothesis.
        Identify key findings with verbatim statistics and Author (Year) citations.
        Explain any contradictions between studies.
        Describe dose-response evidence if present.
        List evidence gaps and suggested follow-up PubMed queries.
        Provide an overall confidence rating.
    """)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=_SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return LiteratureSynthesis(**data)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class LiteratureAgent:
    """Orchestrates PubMed query generation, retrieval, and synthesis.

    Reads ``ncbi_api_key`` from application settings via :func:`src.config.get_settings`.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._ncbi_api_key: str | None = settings.ncbi_api_key

    def run(
        self,
        request: LiteratureRequest,
        client: anthropic.Anthropic,
    ) -> LiteratureResult:
        """Run the full literature-review pipeline.

        Parameters
        ----------
        request:
            A fully populated ``LiteratureRequest``.
        client:
            An initialised Anthropic client.

        Returns
        -------
        LiteratureResult
            Papers retrieved, synthesis, and all queries used.
        """
        ncbi_api_key = self._ncbi_api_key

        # Step 1 – Generate PubMed queries from hypothesis
        queries = generate_pubmed_queries(request.hypothesis, client)

        # Step 2 – Search PubMed
        papers = search_pubmed(queries, request.max_results, ncbi_api_key)

        # Step 3 – Synthesise with Claude
        synthesis = synthesize_with_claude(request.hypothesis, papers, client)

        # Step 4 – Iterative deepening if depth > 1 and gaps were found
        if request.depth > 1 and synthesis.evidence_gaps:
            gap_text = " ".join(synthesis.evidence_gaps[:3])
            follow_up_queries = generate_pubmed_queries(gap_text, client)
            more_papers = search_pubmed(follow_up_queries, request.max_results, ncbi_api_key)
            papers = _deduplicate(papers + more_papers)
            queries = queries + follow_up_queries

        return LiteratureResult(
            request=request,
            papers=papers,
            synthesis=synthesis,
            queries_used=queries,
        )
