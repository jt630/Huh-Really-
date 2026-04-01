"""
USGS Pesticide National Synthesis Project - County-level pesticide use estimates.

Download URL pattern:
  https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/PesticideUseEstimates/EPest.county.{year}.txt

File format: tab-delimited with columns:
  COMPOUND, YEAR, STATE_FIPS_CODE, COUNTY_FIPS_CODE, EPEST_LOW_KG, EPEST_HIGH_KG

The HIGH estimate (EPEST_HIGH_KG) is used as kg_applied.

Compounds of interest (uppercase in source): PARAQUAT DICHLORIDE, ROTENONE,
CHLORPYRIFOS, MANEB — matching is case-insensitive.

Fields: county_fips, county_name, state, compound, kg_applied, year
"""

from __future__ import annotations

import io
import logging
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


class PesticideUseRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    compound: str
    kg_applied: float
    year: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/"
    "PesticideUseEstimates/EPest.county.{year}.txt"
)

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


def _raw_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"EPest.county.{year}.txt"


def _build_county_fips(state_fips: int | str, county_fips: int | str) -> str:
    """Zero-pad state FIPS (2 digits) + county FIPS (3 digits) = 5-digit string."""
    try:
        state_part = str(int(state_fips)).zfill(2)
    except (ValueError, TypeError):
        state_part = str(state_fips).zfill(2)
    try:
        county_part = str(int(county_fips)).zfill(3)
    except (ValueError, TypeError):
        county_part = str(county_fips).zfill(3)
    return state_part + county_part


def _download_year(year: int, cache_dir: Path, max_retries: int = 4) -> Path:
    """Download USGS EPest file for a given year; returns local path."""
    local_path = _raw_cache_path(cache_dir, year)
    if local_path.exists():
        logger.info("Using cached USGS file: %s", local_path)
        return local_path

    url = BASE_URL.format(year=year)
    logger.info("Downloading USGS pesticide data for year %d from %s", year, url)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            logger.info("Saved %s", local_path)
            return local_path
        except requests.HTTPError as exc:
            if attempt < max_retries - 1:
                logger.warning("HTTP error downloading year %d: %s, retrying...", year, exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"Failed to download USGS pesticide data for year {year} after "
                    f"{max_retries} attempts: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Error downloading year %d: %s, retrying...", year, exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"USGS download error for year {year}: {exc}"
                ) from exc

    raise RuntimeError(f"Unexpected failure downloading USGS data for year {year}")


def _parse_usgs_file(filepath: Path, compounds_upper: set[str]) -> pd.DataFrame:
    """Parse a USGS EPest tab-delimited file, filtering to requested compounds."""
    try:
        df = pd.read_csv(filepath, sep="\t", dtype=str, on_bad_lines="warn")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse USGS file {filepath}: {exc}") from exc

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    logger.debug("USGS file columns: %s", list(df.columns))

    # Expected columns: COMPOUND, YEAR, STATE_FIPS_CODE, COUNTY_FIPS_CODE,
    # EPEST_LOW_KG, EPEST_HIGH_KG
    required = {"COMPOUND", "YEAR", "STATE_FIPS_CODE", "COUNTY_FIPS_CODE", "EPEST_HIGH_KG"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"USGS file {filepath} missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter to requested compounds (case-insensitive)
    df["COMPOUND_UPPER"] = df["COMPOUND"].str.strip().str.upper()
    if compounds_upper:
        df = df[df["COMPOUND_UPPER"].isin(compounds_upper)].copy()

    if df.empty:
        return pd.DataFrame()

    # Build county_fips
    df["county_fips"] = df.apply(
        lambda r: _build_county_fips(r["STATE_FIPS_CODE"], r["COUNTY_FIPS_CODE"]),
        axis=1,
    )

    # Use high estimate as kg_applied
    df["kg_applied"] = pd.to_numeric(df["EPEST_HIGH_KG"], errors="coerce").fillna(0.0)
    df["year"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["compound"] = df["COMPOUND"].str.strip()

    # county_name and state are not in the USGS file; leave as empty strings
    # (callers can join with a FIPS lookup table if needed)
    df["county_name"] = ""
    df["state"] = ""

    out_cols = ["county_fips", "county_name", "state", "compound", "kg_applied", "year"]
    return df[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_pesticide_use(
    compounds: list[str],
    years: list[int],
    cache_dir: str = "data/cache/usgs_pesticide",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch county-level pesticide use estimates from the USGS PNSP.

    Parameters
    ----------
    compounds : list[str]
        Compound names to filter (case-insensitive, e.g.
        ["CHLORPYRIFOS", "PARAQUAT DICHLORIDE", "ROTENONE", "MANEB"]).
    years : list[int]
        Calendar years to retrieve.
    cache_dir : str
        Directory for caching downloaded files and results.
    force_refresh : bool
        If True, bypass cached pickle results (raw txt files are still
        reused if present unless deleted manually).

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, compound, kg_applied, year
    """
    cache_dir_path = _ensure_cache_dir(cache_dir)

    compounds_upper = {c.strip().upper() for c in compounds} if compounds else set()
    ckey = "_".join(sorted(compounds_upper))[:80].replace(" ", "_")
    years_str = "_".join(str(y) for y in sorted(years))
    pkl_path = cache_dir_path / f"pesticide_{ckey}_{years_str}.pkl"

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached pesticide data from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    all_frames: list[pd.DataFrame] = []
    for year in sorted(years):
        try:
            filepath = _download_year(year, cache_dir_path)
            year_df = _parse_usgs_file(filepath, compounds_upper)
            if not year_df.empty:
                logger.info("Year %d: %d records after filtering", year, len(year_df))
                all_frames.append(year_df)
            else:
                logger.warning(
                    "Year %d: no records matched compounds %s", year, compounds_upper
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process year %d: %s (skipping)", year, exc)

    if not all_frames:
        logger.warning("No pesticide records loaded for years=%s compounds=%s", years, compounds)
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "compound", "kg_applied", "year"]
        )

    df = pd.concat(all_frames, ignore_index=True)

    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached pesticide DataFrame (%d rows) to %s", len(df), pkl_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_pesticide_use(
        compounds=["CHLORPYRIFOS", "PARAQUAT DICHLORIDE"],
        years=[2019],
        cache_dir="data/cache/usgs_pesticide",
    )
    print(f"Shape: {df.shape}")
    print(df.head(5))
