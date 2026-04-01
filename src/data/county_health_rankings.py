"""
County Health Rankings - Annual health outcomes by county.

Download URL:
  https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{year}.csv

The CHR national CSV is a wide-format file. This module:
  1. Downloads the CSV (or reads from cache).
  2. Extracts the requested measure columns (v001_rawvalue, v002_rawvalue, etc.).
  3. Pivots to long format with columns: county_fips, county_name, state, measure, value,
     ci_low, ci_high, year.

Key measures (column name → friendly measure name):
  v001_rawvalue  → premature_death_rate
  v002_rawvalue  → poor_or_fair_health_pct
  v070_rawvalue  → primary_care_physicians_rate
  v003_rawvalue  → poor_physical_health_days
  v042_rawvalue  → uninsured_pct

FIPS column: "fipscode" (5-digit integer in CSV; zero-padded to str).

Fields: county_fips, county_name, state, measure, value,
        confidence_interval_low, confidence_interval_high, year
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


class HealthRankingRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    measure: str
    value: Optional[float]
    confidence_interval_low: Optional[float]
    confidence_interval_high: Optional[float]
    year: int


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHR_URL = (
    "https://www.countyhealthrankings.org/sites/default/files/media/document/"
    "analytic_data{year}.csv"
)

# Map raw column prefixes to friendly measure names.
# Each measure may have: _rawvalue, _cilow, _cihigh columns.
DEFAULT_MEASURE_MAP: dict[str, str] = {
    "v001": "premature_death_rate",
    "v002": "poor_or_fair_health_pct",
    "v070": "primary_care_physicians_rate",
    "v003": "poor_physical_health_days",
    "v042": "uninsured_pct",
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


def _raw_csv_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"analytic_data{year}.csv"


def _pkl_path(cache_dir: Path, year: int, measures: list[str] | None) -> Path:
    mkey = "_".join(sorted(measures))[:80] if measures else "default"
    return cache_dir / f"chr_{year}_{mkey}.pkl"


def _download_chr_csv(year: int, cache_dir: Path, max_retries: int = 4) -> Path:
    """Download CHR analytic CSV for the given year; return local path."""
    local_path = _raw_csv_path(cache_dir, year)
    if local_path.exists():
        logger.info("Using cached CHR CSV: %s", local_path)
        return local_path

    url = CHR_URL.format(year=year)
    logger.info("Downloading County Health Rankings CSV for year %d: %s", year, url)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            logger.info("Saved CHR CSV to %s", local_path)
            return local_path
        except requests.HTTPError as exc:
            if attempt < max_retries - 1:
                logger.warning("HTTP error downloading CHR %d: %s, retrying...", year, exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(
                    f"Failed to download County Health Rankings CSV for year {year} "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Error downloading CHR %d: %s, retrying...", year, exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(f"CHR download error: {exc}") from exc

    raise RuntimeError(f"Unexpected failure downloading CHR for year {year}")


def _parse_chr_csv(
    filepath: Path,
    year: int,
    measure_map: dict[str, str],
) -> pd.DataFrame:
    """
    Parse a CHR wide-format CSV and pivot to long format.

    Parameters
    ----------
    filepath : Path
        Local path to the CSV file.
    year : int
        Year tag added to every row.
    measure_map : dict[str, str]
        Mapping from CHR column prefix (e.g. "v001") to friendly name.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: county_fips, county_name, state, measure,
        value, confidence_interval_low, confidence_interval_high, year.
    """
    # CHR CSVs sometimes have two header rows; the first row is sometimes a
    # "release" description row. We detect and skip it.
    try:
        raw = pd.read_csv(filepath, dtype=str, on_bad_lines="warn", header=0)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CHR CSV {filepath}: {exc}") from exc

    # Normalize column names to lowercase
    raw.columns = [c.strip().lower() for c in raw.columns]
    logger.debug("CHR CSV columns (first 20): %s", list(raw.columns)[:20])

    # Some years have a "release" first data row — detect by checking if
    # fipscode is non-numeric in first row and skip
    fips_col = None
    for candidate in ("fipscode", "fips", "5-digit fips code"):
        if candidate in raw.columns:
            fips_col = candidate
            break
    if fips_col is None:
        raise RuntimeError(
            f"Could not find FIPS column in CHR CSV. "
            f"Available columns: {list(raw.columns)[:30]}"
        )

    # Drop rows where fipscode is not numeric (e.g. header repetition or summary)
    numeric_mask = raw[fips_col].str.strip().str.match(r"^\d+$")
    raw = raw[numeric_mask].copy()

    # Zero-pad FIPS to 5 chars
    raw["county_fips"] = raw[fips_col].str.strip().str.zfill(5)

    # Determine county_name and state columns
    name_col = None
    for candidate in ("county", "name", "county name", "county_name"):
        if candidate in raw.columns:
            name_col = candidate
            break
    state_col = None
    for candidate in ("state", "state abbreviation", "statecode"):
        if candidate in raw.columns:
            state_col = candidate
            break

    raw["county_name"] = raw[name_col].str.strip() if name_col else ""
    raw["state"] = raw[state_col].str.strip() if state_col else ""

    # Build long-format rows
    records: list[dict] = []
    for prefix, measure_name in measure_map.items():
        val_col = f"{prefix}_rawvalue"
        low_col = f"{prefix}_cilow"
        high_col = f"{prefix}_cihigh"
        # Some CHR files use _low/_high without "ci"
        if low_col not in raw.columns:
            low_col = f"{prefix}_low"
        if high_col not in raw.columns:
            high_col = f"{prefix}_high"

        if val_col not in raw.columns:
            logger.warning("CHR CSV does not have column %s (skipping measure %s)", val_col, measure_name)
            continue

        sub = raw[["county_fips", "county_name", "state"]].copy()
        sub["measure"] = measure_name
        sub["value"] = pd.to_numeric(raw[val_col], errors="coerce")
        sub["confidence_interval_low"] = (
            pd.to_numeric(raw[low_col], errors="coerce") if low_col in raw.columns else None
        )
        sub["confidence_interval_high"] = (
            pd.to_numeric(raw[high_col], errors="coerce") if high_col in raw.columns else None
        )
        sub["year"] = year
        records.append(sub)

    if not records:
        logger.warning("No matching measures found in CHR CSV %s", filepath)
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "measure",
                     "value", "confidence_interval_low", "confidence_interval_high", "year"]
        )

    df = pd.concat(records, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_health_rankings(
    year: int = 2019,
    measures: list[str] | None = None,
    cache_dir: str = "data/cache/county_health_rankings",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch County Health Rankings data for a given year.

    Parameters
    ----------
    year : int
        CHR release year (e.g. 2019). Data availability: 2010–present.
    measures : list[str] | None
        List of friendly measure names to return (e.g. ["premature_death_rate",
        "uninsured_pct"]). If None, returns all default measures.
        Accepted values: premature_death_rate, poor_or_fair_health_pct,
        primary_care_physicians_rate, poor_physical_health_days, uninsured_pct.
    cache_dir : str
        Directory for caching results.
    force_refresh : bool
        If True, bypass cached pickle (raw CSV is still reused if present).

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, measure, value,
                 confidence_interval_low, confidence_interval_high, year
    """
    cache_dir_path = _ensure_cache_dir(cache_dir)
    pkl = _pkl_path(cache_dir_path, year, measures)

    if not force_refresh and pkl.exists():
        logger.info("Loading cached CHR data from %s", pkl)
        with open(pkl, "rb") as fh:
            return pickle.load(fh)

    # Build measure map from requested measures (or all defaults)
    if measures is None:
        measure_map = DEFAULT_MEASURE_MAP
    else:
        # Allow either friendly names or raw prefixes
        reverse_map = {v: k for k, v in DEFAULT_MEASURE_MAP.items()}
        measure_map = {}
        for m in measures:
            if m in DEFAULT_MEASURE_MAP:
                measure_map[m] = DEFAULT_MEASURE_MAP[m]
            elif m in reverse_map:
                measure_map[reverse_map[m]] = m
            else:
                logger.warning("Unknown measure %r (skipping); valid names: %s", m, list(DEFAULT_MEASURE_MAP.values()))

        if not measure_map:
            raise ValueError(
                f"None of the requested measures {measures} are recognized. "
                f"Valid measures: {list(DEFAULT_MEASURE_MAP.values())}"
            )

    csv_path = _download_chr_csv(year, cache_dir_path)
    df = _parse_chr_csv(csv_path, year, measure_map)

    if df.empty:
        logger.warning("CHR data is empty for year %d", year)
        return df

    with open(pkl, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached CHR DataFrame (%d rows) to %s", len(df), pkl)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_health_rankings(
        year=2019,
        measures=["premature_death_rate", "uninsured_pct"],
        cache_dir="data/cache/county_health_rankings",
    )
    print(f"Shape: {df.shape}")
    print(df.head(5))
