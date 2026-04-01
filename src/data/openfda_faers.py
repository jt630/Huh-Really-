"""
FDA FAERS via openFDA - Adverse event reports (state-level only; no county FIPS).

API: https://api.fda.gov/drug/event.json

Fields: state, drug_name, reaction_term, serious, report_count, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

FAERS_URL = "https://api.fda.gov/drug/event.json"


class FAERSRecord(BaseModel):
    state: str
    drug_name: str
    reaction_term: str
    serious: bool
    report_count: int
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_counts(drug: str, year: int, limit: int = 100) -> list[dict]:
    params = {
        "search": (
            f'patient.drug.openfda.generic_name:"{drug}"'
            f'+AND+receivedate:[{year}0101+TO+{year}1231]'
        ),
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    resp = requests.get(FAERS_URL, params=params, timeout=60)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_faers(
    drugs: list[str] | None = None,
    year: int = 2020,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch FDA FAERS adverse event counts for specified drugs.

    Returns DataFrame with columns: state, drug_name, reaction_term,
    serious, report_count, year.
    Note: FAERS data is at national level; state column = 'US'.
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"openfda_faers_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("openFDA FAERS: loading from cache")
        return pd.read_parquet(cache_file)

    if drugs is None:
        drugs = ["paraquat", "rotenone", "chlorpyrifos"]

    records = []
    for drug in drugs:
        try:
            results = _fetch_counts(drug, year)
            for item in results:
                records.append(FAERSRecord(
                    state="US",
                    drug_name=drug,
                    reaction_term=str(item.get("term", "")),
                    serious=False,
                    report_count=int(item.get("count", 0)),
                    year=year,
                ))
        except Exception as exc:
            logger.warning("openFDA FAERS: failed drug=%s: %s", drug, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records])
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_faers(drugs=["paraquat"], year=2020)
    print(df.head(10))
