"""
Census ACS 5-year estimates - Demographics and socioeconomics by county.

API: https://api.census.gov/data/{year}/acs/acs5
Documentation: https://www.census.gov/data/developers/data-sets/acs-5year.html

Default variables fetched:
  B01002_001E → median_age
  B19013_001E → median_household_income
  B01003_001E → total_population
  B17001_002E → poverty_count (below poverty)
  B23025_005E → unemployed_count

Authentication:
  Set CENSUS_API_KEY env var for higher rate limits.
  The Census API works without a key at lower rate limits; the key is
  appended only when present.

Fields: county_fips, county_name, state, variable_name,
        variable_code, value, year
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


class ACSRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    variable_name: str
    variable_code: str
    value: Optional[float]
    year: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

DEFAULT_VARIABLES: dict[str, str] = {
    "B01002_001E": "median_age",
    "B19013_001E": "median_household_income",
    "B01003_001E": "total_population",
    "B17001_002E": "poverty_count",
    "B23025_005E": "unemployed_count",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exponential_backoff(attempt: int, base: float = 1.0, max_wait: float = 30.0) -> None:
    wait = min(base * (2 ** attempt), max_wait)
    logger.debug("Backing off %.1fs (attempt %d)", wait, attempt + 1)
    time.sleep(wait)


def _ensure_cache_dir(cache_dir: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(cache_dir: Path, variables: list[str], year: int, state_fips: str) -> Path:
    vars_str = "_".join(sorted(variables))[:80]
    return cache_dir / f"acs_{year}_{state_fips}_{vars_str}.pkl"


def _fetch_acs_page(
    url: str,
    params: dict,
    max_retries: int = 4,
) -> list[list]:
    """Fetch a Census API response with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 404:
                raise RuntimeError(
                    f"Census ACS endpoint not found: {url}. "
                    "Check that the year is available (data lags ~1 year)."
                ) from exc
            if attempt < max_retries - 1:
                logger.warning("HTTP %s querying Census API, retrying...", exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"Census ACS API failed after {max_retries} attempts: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Census API error: %s, retrying...", exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(f"Census ACS API error: {exc}") from exc
    return []


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_acs(
    variables: list[str] | None = None,
    year: int = 2019,
    state_fips: str = "*",
    cache_dir: str = "data/cache/census_acs",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch Census ACS 5-year county-level estimates.

    Parameters
    ----------
    variables : list[str] | None
        ACS variable codes to retrieve. If None, uses the default set:
        B01002_001E, B19013_001E, B01003_001E, B17001_002E, B23025_005E.
    year : int
        ACS 5-year estimate year (the ending year, e.g. 2019 = 2015-2019).
    state_fips : str
        Two-digit state FIPS code or "*" for all states.
    cache_dir : str
        Directory for caching results.
    force_refresh : bool
        If True, bypass cache and re-fetch from API.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, variable_name,
                 variable_code, value, year
        One row per (county, variable) combination.
    """
    cache_dir_path = _ensure_cache_dir(cache_dir)

    if variables is None:
        variables = list(DEFAULT_VARIABLES.keys())

    # Validate variable codes
    for v in variables:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Invalid ACS variable code: {v!r}")

    pkl_path = _cache_path(cache_dir_path, variables, year, state_fips)

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached ACS data from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        logger.info(
            "CENSUS_API_KEY not set; using unauthenticated Census API access (rate-limited). "
            "Set CENSUS_API_KEY for higher throughput."
        )

    url = ACS_BASE_URL.format(year=year)

    # The Census API accepts up to ~50 variables per request; batch if needed
    BATCH_SIZE = 49
    get_vars = ["NAME"] + variables

    all_rows: list[list] = []
    header: list[str] = []

    for batch_start in range(0, len(variables), BATCH_SIZE):
        batch_vars = variables[batch_start: batch_start + BATCH_SIZE]
        get_str = "NAME," + ",".join(batch_vars)

        params: dict = {
            "get": get_str,
            "for": "county:*",
        }
        if state_fips != "*":
            params["in"] = f"state:{state_fips}"
        else:
            params["for"] = "county:*"
            params["in"] = "state:*"

        if api_key:
            params["key"] = api_key

        logger.info(
            "Fetching ACS %d variables (batch %d) for year %d, state=%s",
            len(batch_vars), batch_start // BATCH_SIZE + 1, year, state_fips,
        )

        raw = _fetch_acs_page(url, params)
        if not raw:
            logger.warning("Empty response from Census API for batch starting at %d", batch_start)
            continue

        batch_header = raw[0]
        batch_rows = raw[1:]
        logger.info("  -> %d county records", len(batch_rows))

        if not header:
            header = batch_header
            all_rows = batch_rows
        else:
            # Merge additional variable columns into existing rows
            # Key on (state, county) which are the last two columns
            existing_by_key = {
                (row[-2], row[-1]): row for row in all_rows
            }
            # Find new variable indices (skip NAME, state, county at end)
            new_var_indices = [
                i for i, col in enumerate(batch_header)
                if col not in ("NAME", "state", "county")
            ]
            for row in batch_rows:
                key = (row[-2], row[-1])
                if key in existing_by_key:
                    existing_by_key[key].extend(row[i] for i in new_var_indices)
            header = header + [batch_header[i] for i in new_var_indices]
            all_rows = list(existing_by_key.values())

    if not all_rows or not header:
        logger.warning("No ACS data returned for year %d", year)
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "variable_name",
                     "variable_code", "value", "year"]
        )

    raw_df = pd.DataFrame(all_rows, columns=header)

    # Build county_fips from state + county columns
    raw_df["county_fips"] = raw_df["state"].str.zfill(2) + raw_df["county"].str.zfill(3)

    # Extract county_name and state abbreviation from NAME column
    # NAME format: "County Name, State Name"
    raw_df["county_name"] = raw_df["NAME"].str.split(",").str[0].str.strip()
    raw_df["state_name"] = raw_df["NAME"].str.split(",").str[-1].str.strip()

    # Melt to long format: one row per (county, variable)
    id_vars = ["county_fips", "county_name", "state_name", "NAME", "state", "county"]
    value_vars = [v for v in variables if v in raw_df.columns]

    melted = raw_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="variable_code",
        value_name="value_raw",
    )

    melted["variable_name"] = melted["variable_code"].map(
        lambda c: DEFAULT_VARIABLES.get(c, c)
    )
    melted["value"] = pd.to_numeric(melted["value_raw"], errors="coerce")
    # Census API uses -666666666 and similar sentinels for missing
    melted.loc[melted["value"] < -999999, "value"] = None
    melted["year"] = year

    out_cols = [
        "county_fips", "county_name", "state_name", "variable_name",
        "variable_code", "value", "year",
    ]
    df = melted[out_cols].rename(columns={"state_name": "state"}).reset_index(drop=True)

    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached ACS DataFrame (%d rows) to %s", len(df), pkl_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_acs(
        variables=["B01002_001E", "B19013_001E"],
        year=2019,
        state_fips="06",  # California
        cache_dir="data/cache/census_acs",
    )
    print(f"Shape: {df.shape}")
    print(df.head(5))
