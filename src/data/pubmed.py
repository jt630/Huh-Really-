"""
PubMed via NCBI E-utilities - Biomedical literature search.

ESearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
EFetch:  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
Rate limits: 3 req/s (no key), 10 req/s (with NCBI_API_KEY)

Fields: pmid, title, abstract, authors, journal, pub_date,
        year, mesh_terms, doi
"""
from pydantic import BaseModel


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
