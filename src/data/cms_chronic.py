"""
CMS Chronic Conditions - Medicare beneficiary prevalence by county.

API: data.cms.gov Socrata endpoint
Endpoint: https://data.cms.gov/resource/pq84-egyk.json
Filter: Bene_Geo_Lvl = "County"

Fields: county_fips, county_name, state, condition,
        prevalence_pct, beneficiary_count, year
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ENDPOINT = "https://data.cms.gov/resource/pq84-egyk.json"
PAGE_SIZE = 50_000


class ChronicConditionRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    condition: str
    prevalence_pct: float | None
    beneficiary_count: int | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_page(offset: int, year: int) -> list[dict]:
    params = {
        "$where": f"Bene_Geo_Lvl='County' AND Year={year}",
        "$limit": PAGE_SIZE,
        "$offset": offset,
    }
    resp = requests.get(ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_cms_chronic(
    year: int = 2020,
    conditions: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch CMS Chronic Conditions prevalence by county.

    Returns DataFrame indexed by county_fips with columns:
        county_name, state, condition, prevalence_pct, beneficiary_count, year
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"cms_chronic_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("CMS Chronic: loading from cache %s", cache_file)
        df = pd.read_parquet(cache_file)
        if conditions:
            df = df[df["condition"].isin(conditions)]
        return df

    logger.info("CMS Chronic: fetching year=%d", year)
    records: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(offset, year)
        if not page:
            break
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not records:
        return pd.DataFrame()

    parsed = []
    for raw in records:
        try:
            fips = str(raw.get("bene_geo_cd", "")).strip().zfill(5)
            if len(fips) != 5:
                continue
            parsed.append(ChronicConditionRecord(
                county_fips=fips,
                county_name=str(raw.get("bene_geo_desc", "")).strip(),
                state=str(raw.get("bene_geo_desc", "")).strip(),
                condition=str(raw.get("bene_cond", "")).strip(),
                prevalence_pct=_safe_float(raw.get("prvlnc")),
                beneficiary_count=_safe_int(raw.get("tot_benes")),
                year=int(raw.get("year", year)),
            ))
        except Exception as exc:
            logger.debug("CMS Chronic: skipping record: %s", exc)

    df = pd.DataFrame([r.model_dump() for r in parsed]).set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

    if conditions:
        df = df[df["condition"].isin(conditions)]
    return df


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_int(v: object) -> int | None:
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_cms_chronic(year=2020)
    print(df.head(10))
    print(f"Shape: {df.shape}")
