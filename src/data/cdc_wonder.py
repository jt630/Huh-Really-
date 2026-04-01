"""
CDC WONDER - County-level mortality data (ICD-10).

Website: https://wonder.cdc.gov/
Dataset: Underlying Cause of Death (D76), POST XML API
Primary approach: read pre-downloaded tab-delimited export files.

HOW TO DOWNLOAD DATA MANUALLY
==============================
1. Go to https://wonder.cdc.gov/ucd-icd10.html
2. Accept the data use agreement.
3. Under "Group Results By": select County, then Year.
4. Under "ICD-10 Codes": select/enter your ICD-10 code (e.g. G35 for MS).
5. Under "Other Options": select "Export Results" as tab-delimited.
6. Save the file as:  data/cache/cdc_wonder/<ICD10_CODE>_<YEAR>.txt
   e.g.  data/cache/cdc_wonder/G35_2019.txt
7. The file will have a header block (lines starting with "Notes"),
   then tab-delimited data, then a footer block. This module strips both.

API fallback (fetch_mortality_api) attempts the POST XML endpoint but will
gracefully degrade because CDC WONDER requires a browser session/cookie to
accept terms before responding to programmatic requests.

Fields: county_fips, county_name, state, deaths, population,
        age_adjusted_rate (per 100k), year
"""

from __future__ import annotations

import logging
import os
import pickle
import re
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


class MortalityRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    deaths: Optional[int]
    population: Optional[int]
    age_adjusted_rate: Optional[float]
    year: int


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


def _cache_path(cache_dir: Path, icd10_code: str, years: list[int]) -> Path:
    years_str = "_".join(str(y) for y in sorted(years))
    return cache_dir / f"{icd10_code}_{years_str}.pkl"


# ---------------------------------------------------------------------------
# File-based parsing (primary approach)
# ---------------------------------------------------------------------------


def _parse_wonder_txt(filepath: Path, year: int) -> list[MortalityRecord]:
    """Parse a CDC WONDER tab-delimited export file."""
    records: list[MortalityRecord] = []
    in_data = False

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # Skip empty lines and footer/notes sections
            if not line.strip():
                continue
            if line.startswith("---"):
                in_data = False
                continue
            if line.startswith("Notes"):
                in_data = False
                continue

            # The header row contains "County" as first token
            if not in_data:
                if line.lower().startswith("county\t") or line.lower().startswith('"county"'):
                    in_data = True
                    # peek at header to locate columns
                    headers = [h.strip().strip('"').lower() for h in line.split("\t")]
                    logger.debug("CDC WONDER file headers: %s", headers)

                    def _col(name: str) -> int:
                        for i, h in enumerate(headers):
                            if name in h:
                                return i
                        return -1

                    county_idx = _col("county")
                    code_idx = _col("county code")
                    deaths_idx = _col("deaths")
                    pop_idx = _col("population")
                    aar_idx = _col("age-adjusted")
                continue

            if not in_data:
                continue

            parts = [p.strip().strip('"') for p in line.split("\t")]
            if len(parts) < 3:
                continue

            # County name and code
            county_raw = parts[county_idx] if county_idx >= 0 and county_idx < len(parts) else ""
            county_code = parts[code_idx] if code_idx >= 0 and code_idx < len(parts) else ""

            # Some export files merge "County, State" in county column
            state = ""
            county_name = county_raw
            if "," in county_raw:
                segments = county_raw.rsplit(",", 1)
                county_name = segments[0].strip()
                state = segments[1].strip()

            # FIPS from county code (5-digit)
            fips = county_code.replace(".", "").strip().zfill(5) if county_code else ""

            def _safe_int(val: str) -> Optional[int]:
                val = val.replace(",", "").strip()
                return int(val) if val.lstrip("-").isdigit() else None

            def _safe_float(val: str) -> Optional[float]:
                val = val.replace(",", "").strip()
                try:
                    return float(val)
                except ValueError:
                    return None

            deaths = _safe_int(parts[deaths_idx]) if deaths_idx >= 0 and deaths_idx < len(parts) else None
            population = _safe_int(parts[pop_idx]) if pop_idx >= 0 and pop_idx < len(parts) else None
            aar = _safe_float(parts[aar_idx]) if aar_idx >= 0 and aar_idx < len(parts) else None

            if not fips:
                continue

            records.append(
                MortalityRecord(
                    county_fips=fips,
                    county_name=county_name,
                    state=state,
                    deaths=deaths,
                    population=population,
                    age_adjusted_rate=aar,
                    year=year,
                )
            )

    return records


def _find_predownloaded_files(cache_dir: Path, icd10_code: str, years: list[int]) -> dict[int, Path]:
    """Locate pre-downloaded files matching pattern <ICD10>_<YEAR>.txt or <ICD10>_<YEAR>*.txt."""
    found: dict[int, Path] = {}
    code_upper = icd10_code.upper()
    for year in years:
        # Try exact match first, then glob
        candidates = list(cache_dir.glob(f"{code_upper}_{year}*.txt"))
        if not candidates:
            candidates = list(cache_dir.glob(f"{code_upper.replace('.', '_')}_{year}*.txt"))
        if candidates:
            found[year] = candidates[0]
            logger.info("Found pre-downloaded file for %s/%d: %s", icd10_code, year, candidates[0])
    return found


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_mortality(
    icd10_code: str,
    years: list[int],
    cache_dir: str = "data/cache/cdc_wonder",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch county-level mortality data for a given ICD-10 code and year range.

    Primary approach: reads pre-downloaded CDC WONDER tab-delimited export files.
    Falls back to the WONDER POST API (fetch_mortality_api) if no local files exist.

    Parameters
    ----------
    icd10_code : str
        ICD-10 cause-of-death code (e.g. "G35" for multiple sclerosis).
    years : list[int]
        Calendar years to retrieve.
    cache_dir : str
        Directory for caching results.
    force_refresh : bool
        If True, bypass cached pickle and re-read source files.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, deaths, population,
                 age_adjusted_rate, year
    """
    cache_path_obj = _ensure_cache_dir(Path(cache_dir))
    pkl_path = _cache_path(cache_path_obj, icd10_code, years)

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached mortality data from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    # Check for pre-downloaded files
    file_map = _find_predownloaded_files(cache_path_obj, icd10_code, years)
    records: list[MortalityRecord] = []

    if file_map:
        for year, filepath in file_map.items():
            logger.info("Parsing CDC WONDER file %s for year %d", filepath, year)
            recs = _parse_wonder_txt(filepath, year)
            logger.info("  -> %d records parsed", len(recs))
            records.extend(recs)

        missing_years = [y for y in years if y not in file_map]
        if missing_years:
            logger.warning(
                "No pre-downloaded files for years %s. "
                "Attempting API fallback for those years.",
                missing_years,
            )
            try:
                api_df = fetch_mortality_api(icd10_code, missing_years)
                if not api_df.empty:
                    for row in api_df.itertuples(index=False):
                        records.append(MortalityRecord(**row._asdict()))
            except Exception as exc:  # noqa: BLE001
                logger.warning("API fallback failed: %s", exc)
    else:
        logger.info(
            "No pre-downloaded files found for %s. Trying WONDER POST API.", icd10_code
        )
        try:
            df = fetch_mortality_api(icd10_code, years)
            with open(pkl_path, "wb") as fh:
                pickle.dump(df, fh)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.error("WONDER API failed and no local files found: %s", exc)
            raise RuntimeError(
                f"No pre-downloaded CDC WONDER files found for ICD-10 code '{icd10_code}' "
                f"and the API request failed ({exc}). "
                f"Please download data manually to {cache_dir}/<CODE>_<YEAR>.txt "
                "following the instructions in the module docstring."
            ) from exc

    if not records:
        logger.warning("No records loaded for %s / %s", icd10_code, years)
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "deaths", "population",
                     "age_adjusted_rate", "year"]
        )

    df = pd.DataFrame([r.model_dump() for r in records])

    with open(pkl_path, "wb") as fh:
        pickle.dump(df, fh)
    logger.info("Cached mortality DataFrame (%d rows) to %s", len(df), pkl_path)

    return df


# ---------------------------------------------------------------------------
# API fallback
# ---------------------------------------------------------------------------


def fetch_mortality_api(
    icd10_code: str,
    years: list[int],
    cache_dir: str = "data/cache/cdc_wonder",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Attempt to retrieve CDC WONDER data via POST XML API (D76).

    NOTE: The WONDER API requires the user to accept a data use agreement
    interactively (via a browser session/cookie). Programmatic access without
    a valid session will typically receive an HTML error response. This
    function will raise a RuntimeError with a descriptive message if the
    API request fails or returns an error page.

    Parameters
    ----------
    icd10_code : str
        ICD-10 code (e.g. "G35").
    years : list[int]
        Calendar years.
    cache_dir : str
        Cache directory.
    force_refresh : bool
        Bypass cache if True.

    Returns
    -------
    pd.DataFrame
        Same schema as fetch_mortality.
    """
    cache_path_obj = _ensure_cache_dir(Path(cache_dir))
    pkl_path = cache_path_obj / f"{icd10_code}_api_{'_'.join(str(y) for y in sorted(years))}.pkl"

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached API mortality data from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    url = "https://wonder.cdc.gov/controller/datarequest/D76"

    # Build the POST parameters for D76 (Underlying Cause of Death).
    # Group by county (V_D76.V9-level1) and year (V_D76.V7).
    year_codes = " ".join(str(y) for y in sorted(years))
    params = {
        "accept_datause_restrictions": "true",
        "B_1": "D76.V9-level1",  # group by county
        "B_2": "D76.V7",         # group by year
        "M_1": "D76.M1",         # Deaths
        "M_2": "D76.M2",         # Population
        "M_3": "D76.M3",         # Crude Rate
        "M_4": "D76.M4",         # Age-adjusted Rate
        "O_title": f"Mortality for ICD-10 {icd10_code}",
        "O_location": "D76",
        "O_ucd": icd10_code,
        "V_D76.V2": icd10_code,  # ICD-10 codes
        "V_D76.V7": year_codes,  # years
        "action-Send": "Send",
        "finder-stage-D76.V9": "codeset",
        "O_age": "D76.V5",
        "O_javascript": "on",
        "O_show_totals": "false",
        "O_timeout": "300",
        "VM_D76.M4_D76.V1_S": "*All*",
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (compatible; unlikely-correlations/0.1)",
    }

    session = requests.Session()
    max_retries = 4

    for attempt in range(max_retries):
        try:
            logger.info(
                "Attempting WONDER API for %s, years=%s (attempt %d/%d)",
                icd10_code, years, attempt + 1, max_retries,
            )
            resp = session.post(url, data=params, headers=headers, timeout=120)
            resp.raise_for_status()

            # CDC WONDER returns XML or tab-delimited; detect by content-type
            content_type = resp.headers.get("Content-Type", "")
            content = resp.text

            if "error" in content.lower()[:500] or "html" in content_type.lower():
                raise RuntimeError(
                    "CDC WONDER API returned an HTML error page. "
                    "The API requires interactive acceptance of data use terms. "
                    "Please download the data manually from https://wonder.cdc.gov/ucd-icd10.html "
                    f"and save as {cache_dir}/<CODE>_<YEAR>.txt"
                )

            # Try to parse tab-delimited response
            from io import StringIO
            df = pd.read_csv(StringIO(content), sep="\t", comment="N")

            # Normalize columns
            df.columns = [c.strip().lower() for c in df.columns]
            if "county code" in df.columns:
                df["county_fips"] = df["county code"].astype(str).str.zfill(5)
            if "county" in df.columns:
                df["county_name"] = df["county"].str.split(",").str[0].str.strip()
                df["state"] = df["county"].str.split(",").str[-1].str.strip()
            if "deaths" not in df.columns:
                df["deaths"] = None
            if "population" not in df.columns:
                df["population"] = None
            if "age-adjusted rate" in df.columns:
                df["age_adjusted_rate"] = pd.to_numeric(df["age-adjusted rate"], errors="coerce")
            elif "age_adjusted_rate" not in df.columns:
                df["age_adjusted_rate"] = None

            df["year"] = df.get("year", years[0] if len(years) == 1 else None)

            out_cols = ["county_fips", "county_name", "state", "deaths",
                        "population", "age_adjusted_rate", "year"]
            for col in out_cols:
                if col not in df.columns:
                    df[col] = None

            df = df[out_cols]
            df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").astype("Int64")
            df["population"] = pd.to_numeric(df["population"], errors="coerce").astype("Int64")
            df["age_adjusted_rate"] = pd.to_numeric(df["age_adjusted_rate"], errors="coerce")

            with open(pkl_path, "wb") as fh:
                pickle.dump(df, fh)
            logger.info("WONDER API success: %d rows cached to %s", len(df), pkl_path)
            return df

        except RuntimeError:
            raise
        except requests.HTTPError as exc:
            if attempt < max_retries - 1:
                logger.warning("HTTP error %s, retrying...", exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(f"CDC WONDER API failed after {max_retries} attempts: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Request error: %s, retrying...", exc)
                _exponential_backoff(attempt)
            else:
                raise RuntimeError(f"CDC WONDER API error: {exc}") from exc

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_mortality(icd10_code="G35", years=[2019], cache_dir="data/cache/cdc_wonder")
    print(f"Shape: {df.shape}")
    print(df.head(5))
