"""
FDA FAERS via openFDA - Adverse event reports (state-level only; no county FIPS).

API: https://api.fda.gov/drug/event.json

Fields: state, drug_name, reaction_term, serious, report_count, year
"""
from pydantic import BaseModel


class FAERSRecord(BaseModel):
    state: str
    drug_name: str
    reaction_term: str
    serious: bool
    report_count: int
    year: int
