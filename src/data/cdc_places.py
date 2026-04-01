"""
CDC PLACES - Local chronic disease estimates via Socrata.

Endpoint: https://data.cdc.gov/resource/swc5-untb.json

Fields: county_fips, county_name, state_abbr, measure, measure_id,
        data_value (age-adj prevalence %), confidence limits, year
"""
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MEASURES = ["CSMOKING", "OBESITY", "DIABETES", "CHD", "STROKE", "CANCER"]
SOCRATA_URL = "https://data.cdc.gov/resource/swc5-untb.json"


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


def _fetch_with_backoff(url: str, params: dict, max_retries: int = 4) -> list[dict]:
    """HTTP GET with exponential backoff on errors."""
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if attempt == max_retries:
                raise
            logger.warning("HTTP error %s on attempt %d; retrying in %ds", exc, attempt + 1, delay)
            time.sleep(delay)
            delay *= 2
    return []  # unreachable


def fetch_places(
    measures: list[str] | None = None,
    year: int = 2022,
    state_abbr: list[str] | None = None,
    cache_dir: str = "data/cache/cdc_places",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch CDC PLACES county-level chronic disease estimates.

    Parameters
    ----------
    measures : list of measure IDs (e.g. CSMOKING, OBESITY). Defaults to 6 key measures.
    year : data release year.
    state_abbr : optional list of 2-letter state abbreviations to filter.
    cache_dir : directory for cached responses.
    force_refresh : if True, ignore cache and re-fetch.

    Returns
    -------
    pd.DataFrame with columns: county_fips, county_name, state, measure, data_value, year
    """
    measures = measures or DEFAULT_MEASURES
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    state_key = "_".join(sorted(state_abbr)) if state_abbr else "all"
    measure_key = "_".join(sorted(measures))
    cache_file = cache_path / f"places_{year}_{state_key}_{measure_key}.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading CDC PLACES from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching CDC PLACES data for year=%d measures=%s", year, measures)

    all_records: list[dict] = []
    page_size = 1000
    offset = 0

    while True:
        params: dict = {
            "geographiclevel": "County",
            "year": str(year),
            "$limit": page_size,
            "$offset": offset,
        }
        if state_abbr:
            # Socrata doesn't support IN natively via simple params; filter client-side
            pass

        rows = _fetch_with_backoff(SOCRATA_URL, params)
        logger.debug("Fetched %d rows at offset %d", len(rows), offset)

        if not rows:
            break

        for row in rows:
            measure_id = row.get("measureid", row.get("measure_id", ""))
            if measure_id not in measures:
                continue
            state = row.get("stateabbr", row.get("state_abbr", ""))
            if state_abbr and state not in state_abbr:
                continue

            loc_id = row.get("locationid", "")
            # Ensure 5-digit FIPS with leading zeros
            county_fips = str(loc_id).zfill(5) if loc_id else ""

            try:
                dv = float(row["data_value"]) if row.get("data_value") not in (None, "") else None
            except (ValueError, TypeError):
                dv = None
            try:
                lcl = float(row["low_confidence_limit"]) if row.get("low_confidence_limit") not in (None, "") else None
            except (ValueError, TypeError):
                lcl = None
            try:
                hcl = float(row["high_confidence_limit"]) if row.get("high_confidence_limit") not in (None, "") else None
            except (ValueError, TypeError):
                hcl = None

            all_records.append({
                "county_fips": county_fips,
                "county_name": row.get("locationname", ""),
                "state": state,
                "measure": row.get("measure", ""),
                "measure_id": measure_id,
                "data_value": dv,
                "low_confidence_limit": lcl,
                "high_confidence_limit": hcl,
                "year": int(row.get("year", year)),
            })

        if len(rows) < page_size:
            break
        offset += page_size

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No CDC PLACES records returned")
    else:
        logger.info("Fetched %d CDC PLACES records", len(df))
        df.to_parquet(cache_file, index=False)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_places(measures=["CSMOKING", "OBESITY"], year=2022, state_abbr=["CA", "TX"])
    print(df.head())
