"""
PubMed via NCBI E-utilities - Biomedical literature search.

ESearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
EFetch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
Rate limits: 3 req/s (no key), 10 req/s (with NCBI_API_KEY)

Fields: pmid, title, abstract, authors, journal, pub_date,
        year, mesh_terms, doi
"""
from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


class PubMedPaper(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_date: str
    year: int
    mesh_terms: list[str] = []
    doi: str | None = None


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
)
def _esearch(query: str, max_results: int, api_key: str | None) -> list[str]:
    params: dict = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
)
def _efetch(pmids: list[str], api_key: str | None) -> str:
    params: dict = {"db": "pubmed", "id": ",".join(pmids),
                    "rettype": "abstract", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def _parse_papers(xml_text: str) -> list[PubMedPaper]:
    root = ET.fromstring(xml_text)
    papers = []
    for article_elem in root.findall(".//PubmedArticle"):
        medline = article_elem.find("MedlineCitation")
        if medline is None:
            continue
        pmid_elem = medline.find("PMID")
        pmid = (pmid_elem.text or "").strip() if pmid_elem is not None else ""
        if not pmid:
            continue
        article = medline.find("Article")
        if article is None:
            continue
        title = (article.findtext("ArticleTitle") or "").strip()
        abstract_parts = []
        for at in (article.find("Abstract") or ET.Element("_")).findall("AbstractText"):
            text = (at.text or "").strip()
            label = at.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)
        authors = []
        for auth in (article.find("AuthorList") or ET.Element("_")).findall("Author"):
            last = auth.findtext("LastName", "")
            fore = auth.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())
            else:
                coll = auth.findtext("CollectiveName", "")
                if coll:
                    authors.append(coll)
        journal_elem = article.find("Journal")
        journal = (journal_elem.findtext("Title") or "") if journal_elem is not None else ""
        pub_date_elem = None
        if journal_elem is not None:
            ji = journal_elem.find("JournalIssue")
            if ji is not None:
                pub_date_elem = ji.find("PubDate")
        year_str = ""
        pub_date = ""
        if pub_date_elem is not None:
            year_str = pub_date_elem.findtext("Year") or pub_date_elem.findtext("MedlineDate", "")[:4]
            month = pub_date_elem.findtext("Month", "")
            pub_date = f"{year_str} {month}".strip()
        mesh_terms = [
            mh.findtext("DescriptorName", "") or ""
            for mh in medline.findall(".//MeshHeading")
        ]
        doi = None
        for aid in article_elem.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break
        try:
            year_int = int(year_str[:4]) if year_str else 0
        except ValueError:
            year_int = 0
        papers.append(PubMedPaper(
            pmid=pmid, title=title, abstract=abstract, authors=authors,
            journal=journal, pub_date=pub_date, year=year_int,
            mesh_terms=[m for m in mesh_terms if m], doi=doi,
        ))
    return papers


def search_pubmed(
    query: str,
    max_results: int = 50,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[PubMedPaper]:
    """
    Search PubMed and return Paper objects.

    Parameters
    ----------
    query : str
        PubMed search string (MeSH terms supported).
    max_results : int
        Maximum number of records to return.
    api_key : str | None
        NCBI API key (raises rate limit to 10/s).

    Returns
    -------
    list[PubMedPaper]
    """
    from src.config import get_settings
    if api_key is None:
        api_key = get_settings().ncbi_api_key

    cache_dir = get_settings().cache_dir / "pubmed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{hashlib.md5(query.encode()).hexdigest()}.xml"

    if use_cache and cache_file.exists():
        xml_text = cache_file.read_text(encoding="utf-8")
    else:
        rate_limit = 0.11 if api_key else 0.34
        pmids = _esearch(query, max_results, api_key)
        if not pmids:
            return []
        time.sleep(rate_limit)
        xml_text = _efetch(pmids, api_key)
        cache_file.write_text(xml_text, encoding="utf-8")
        time.sleep(rate_limit)

    return _parse_papers(xml_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    papers = search_pubmed("paraquat[MeSH] AND Parkinson disease[MeSH]", max_results=5)
    for p in papers:
        print(p.pmid, p.title[:80])
