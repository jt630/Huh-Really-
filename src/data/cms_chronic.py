"""
CMS Chronic Conditions - Medicare beneficiary prevalence by county.

API: data.cms.gov Socrata/SODA endpoint
Dataset ID: hytk-nfam (County-level chronic conditions)
Base URL: https://data.cms.gov/resource/hytk-nfam.json
Filter: bene_geo_lvl=County

Documentation:
  https://data.cms.gov/medicare-chronic-conditions/multiple-chronic-conditions

Fields: county_fips, county_name, state, condition,
        prevalence_pct, beneficiary_count, year
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ChronicConditionRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    condition: str
    prevalence_pct: Optional[float]
    beneficiary_count: Optional[int]
    year: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://data.cms.gov/resource/hytk-nfam.json"
PAGE_LIMIT = 50_000


def _exponential_backoff(attempt: int, base: float = 1.0, max_wait: float = 30.0) -> None:
    wait = min(base * (2 ** attempt), max_wait)
    logger.debug("Backing off %.1fs (attempt %d)", wait, attempt + 1)
    time.sleep(wait)


def _ensure_cache_dir(cache_dir: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(conditions: list[str] | None, years: list[int] | None) -> str:
    cond_str = "_".join(sorted(conditions)) if conditions else "all"
    years_str = "_".join(str(y) for y in sorted(years)) if years else "all"
    return f"cms_chronic_{cond_str}_{years_str}"


def _fetch_page(session: requests.Session, params: dict, offset: int, max_retries: int = 4) -> list[dict]:
    """Fetch a single page from the Socrata API with retry logic."""
    page_params = {**params, "$limit": PAGE_LIMIT, "$offset": offset}

    for attempt in range(max_retries):
        try:
            resp = session.get(BASE_URL, params=page_params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if attempt < max_retries - 1:
                logger.warning("HTTP %s on page offset=%d, retrying...", exc, offset)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"CMS Chronic Conditions API failed after {max_retries} attempts: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Request error at offset=%d: %s, retrying...", offset, exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(f"CMS API error: {exc}") from exc

    return []


def _build_county_fips(state_fips: str, county_fips: str) -> str:
    """Zero-pad state FIPS to 2 digits and county FIPS to 3 digits."""
    try:
        state_part = str(int(state_fips)).zfill(2)
    except (ValueError, TypeError):
        state_part = str(state_fips).zfill(2)
    try:
        county_part = str(int(county_fips)).zfill(3)
    except (ValueError, TypeError):
        county_part = str(county_fips).zfill(3)
    return state_part + county_part


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_chronic_conditions(
    conditions: list[str] | None = None,
    years: list[int] | None = None,
    cache_dir: str = "data/cache/cms_chronic",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch CMS county-level chronic condition prevalence data.

    Parameters
    ----------
    conditions : list[str] | None
        List of condition names to filter (e.g. ["Diabetes", "Alzheimer's Disease"]).
        If None, all conditions are returned.
    years : list[int] | None
        Calendar years to retrieve. If None, all available years.
    cache_dir : str
        Directory for caching results.
    force_refresh : bool
        If True, bypass cache and re-fetch from API.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, condition,
                 prevalence_pct, beneficiary_count, year
    """
    cache_dir_path = _ensure_cache_dir(cache_dir)
    cache_key = _cache_key(conditions, years)
    pkl_path = cache_dir_path / f"{cache_key}.pkl"

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached CMS chronic conditions data from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    # Build Socrata SODA query parameters
    where_clauses: list[str] = ["bene_geo_lvl='County'"]

    if years:
        year_list = ", ".join(f"'{y}'" for y in years)
        where_clauses.append(f"year in({year_list})")

    if conditions:
        # Case-insensitive match using LIKE for each condition
        cond_clauses = " OR ".join(
            f"upper(bene_cond) like '{c.upper()}'" for c in conditions
        )
        where_clauses.append(f"({cond_clauses})")

    base_params: dict = {
        "$where": " AND ".join(where_clauses),
        "$order": "year ASC",
    }

    session = requests.Session()
    # Add app token if available
    app_token = os.environ.get("SOCRATA_APP_TOKEN") or os.environ.get("CMS_APP_TOKEN")
    if app_token:
        session.headers["X-App-Token"] = app_token
    else:
        logger.debug("No SOCRATA_APP_TOKEN found; using unauthenticated access (rate limited)")

    all_rows: list[dict] = []
    offset = 0

    while True:
        logger.info("Fetching CMS chronic conditions page offset=%d", offset)
        page = _fetch_page(session, base_params, offset)
        if not page:
            break
        all_rows.extend(page)
        logger.info("  -> received %d records (total so far: %d)", len(page), len(all_rows))
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT

    if not all_rows:
        logger.warning("No CMS chronic condition records returned")
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "condition",
                     "prevalence_pct", "beneficiary_count", "year"]
        )

    # Build DataFrame and normalize columns
    raw_df = pd.DataFrame(all_rows)
    logger.debug("Raw CMS columns: %s", list(raw_df.columns))

    # Column mappings (Socrata field names may vary; handle common variants)
    col_map = {
        "bene_geo_desc": "county_name",
        "bene_state_abv": "state",
        "bene_cond": "condition",
        "prevalence": "prevalence_pct",
        "bene_geo_cd": "bene_geo_cd",
        "bene_state_cd": "bene_state_cd",
        "tot_benes": "beneficiary_count",
        "year": "year",
    }

    df = raw_df.rename(columns={k: v for k, v in col_map.items() if k in raw_df.columns})

    # Build county_fips from state + county FIPS components
    if "bene_state_cd" in raw_df.columns and "bene_geo_cd" in raw_df.columns:
        df["county_fips"] = df.apply(
            lambda r: _build_county_fips(
                r.get("bene_state_cd", ""),
                r.get("bene_geo_cd", ""),
            ),
            axis=1,
        )
    elif "county_fips" not in df.columns:
        df["county_fips"] = ""

    # Coerce numeric fields
    if "prevalence_pct" in df.columns:
        df["prevalence_pct"] = pd.to_numeric(df["prevalence_pct"], errors="coerce")
    else:
        df["prevalence_pct"] = None

    if "beneficiary_count" in df.columns:
        df["beneficiary_count"] = pd.to_numeric(df["beneficiary_count"], errors="coerce").astype("Int64")
    else:
        df["beneficiary_count"] = None

    df["year"] = pd.to_numeric(df.get("year", None), errors="coerce").astype("Int64")

    for col in ["county_name", "state", "condition"]:
        if col not in df.columns:
            df[col] = ""

    out_cols = ["county_fips", "county_name", "state", "condition",
                "prevalence_pct", "beneficiary_count", "year"]
    df = df[out_cols]

    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached CMS chronic conditions DataFrame (%d rows) to %s", len(df), pkl_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_chronic_conditions(
        conditions=["Diabetes"],
        years=[2019],
        cache_dir="data/cache/cms_chronic",
    )
    print(f"Shape: {df.shape}")
    print(df.head(5))
