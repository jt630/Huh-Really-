"""
CDC PLACES - Local chronic disease estimates via Socrata.

Endpoint: https://data.cdc.gov/resource/cwsq-ngmh.json

Fields: county_fips, county_name, state_abbr, measure, measure_id,
        data_value (age-adj prevalence %), confidence limits, year
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

ENDPOINT = "https://data.cdc.gov/resource/cwsq-ngmh.json"
PAGE_SIZE = 50_000


class PlacesRecord(BaseModel):
    county_fips: str
    county_name: str
    state_abbr: str
    measure: str
    measure_id: str
    data_value: float | None
    low_confidence_limit: float | None
    high_confidence_limit: float | None
    year: int

    @field_validator("county_fips")
    @classmethod
    def pad_fips(cls, v: str) -> str:
        return v.zfill(5)


def _cache_path(year: int, measure_ids: list[str] | None) -> Path:
    from src.config import get_settings

    tag = "_".join(sorted(measure_ids)) if measure_ids else "all"
    return get_settings().cache_dir / f"cdc_places_{year}_{tag}.parquet"


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_page(offset: int, year: int, measure_ids: list[str] | None) -> list[dict]:
    """Fetch a single page from the Socrata endpoint."""
    where_clauses = [f"year={year}"]
    if measure_ids:
        quoted = ", ".join(f"'{m}'" for m in measure_ids)
        where_clauses.append(f"measureid in ({quoted})")

    params: dict[str, str | int] = {
        "$where": " AND ".join(where_clauses),
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "locationid,measureid",
    }
    logger.debug("CDC PLACES fetch offset=%d year=%d", offset, year)
    resp = requests.get(ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


def fetch_places(
    year: int = 2022,
    measure_ids: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch CDC PLACES data for the given year, optionally filtering by measure IDs.

    Returns DataFrame with columns:
        county_fips, measure, measure_id, data_value, year
    """
    cache_file = _cache_path(year, measure_ids)
    if use_cache and cache_file.exists():
        logger.info("CDC PLACES: loading from cache %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("CDC PLACES: fetching year=%d measures=%s", year, measure_ids)
    records: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(offset, year, measure_ids)
        if not page:
            break
        records.extend(page)
        logger.debug("CDC PLACES: fetched %d records so far", len(records))
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not records:
        logger.warning("CDC PLACES: no records returned for year=%d", year)
        return pd.DataFrame(
            columns=["county_fips", "measure", "measure_id", "data_value", "year"]
        )

    parsed: list[PlacesRecord] = []
    for raw in records:
        try:
            rec = PlacesRecord(
                county_fips=raw.get("locationid", ""),
                county_name=raw.get("locationname", ""),
                state_abbr=raw.get("stateabbr", ""),
                measure=raw.get("measure", ""),
                measure_id=raw.get("measureid", ""),
                data_value=_safe_float(raw.get("data_value")),
                low_confidence_limit=_safe_float(raw.get("low_confidence_limit")),
                high_confidence_limit=_safe_float(raw.get("high_confidence_limit")),
                year=int(raw.get("year", year)),
            )
            parsed.append(rec)
        except Exception as exc:
            logger.debug("CDC PLACES: skipping bad record %s: %s", raw, exc)

    df = pd.DataFrame([r.model_dump() for r in parsed])
    df = df[["county_fips", "measure", "measure_id", "data_value", "year"]]
    df = df.set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        logger.info("CDC PLACES: cached %d rows to %s", len(df), cache_file)

    return df


def _safe_float(v: object) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_places(year=2022, measure_ids=["CANCER", "DIABETES", "BPHIGH"])
    print(df.head(10))
    print(f"Shape: {df.shape}")
    print(f"Measures: {df['measure_id'].unique()}")
