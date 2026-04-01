"""
CDC Wonder - County-level mortality data (ICD-10).

Website: https://wonder.cdc.gov/
Dataset: Underlying Cause of Death, POST XML API (dataset code D76)

NOTE: CDC Wonder blocks automated POST requests. This module implements
the file-based fallback only. Download tab-delimited .txt files manually:
  1. Go to https://wonder.cdc.gov/ucd-icd10.html
  2. Group by County, select ICD-10 codes, export as tab-delimited text
  3. Place files in data/cache/cdc_wonder/<year>.txt

Fields: county_fips, county_name, state, deaths, population,
        age_adjusted_rate (per 100k), year
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MortalityRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    deaths: int | None
    population: int | None
    age_adjusted_rate: float | None
    year: int


def _safe_int(v: str) -> int | None:
    try:
        return int(v.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_float(v: str) -> float | None:
    try:
        return float(v.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def fetch_wonder(
    icd10_code: str = "G20",
    years: list[int] | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load CDC Wonder mortality data from pre-downloaded tab-delimited files.

    Parameters
    ----------
    icd10_code:
        ICD-10 cause of death code (informational only; files are pre-filtered).
    years:
        List of years to load. If None, loads all files in cache_dir.
    cache_dir:
        Directory containing <year>.txt files. Defaults to data/cache/cdc_wonder/.

    Returns
    -------
    pd.DataFrame indexed by county_fips with columns:
        county_name, state, deaths, population, age_adjusted_rate, year
    """
    from src.config import get_settings
    if cache_dir is None:
        cache_dir = get_settings().cache_dir / "cdc_wonder"

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.warning("CDC Wonder cache dir not found: %s", cache_dir)
        return pd.DataFrame()

    txt_files = sorted(cache_dir.glob("*.txt"))
    if years:
        txt_files = [f for f in txt_files if any(str(y) in f.name for y in years)]

    records: list[MortalityRecord] = []
    for txt_file in txt_files:
        try:
            year = int("".join(filter(str.isdigit, txt_file.stem))[:4])
        except ValueError:
            year = 0

        try:
            df_raw = pd.read_csv(txt_file, sep="\t", dtype=str, encoding="latin-1")
            df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]

            # CDC Wonder uses "notes" lines as separators — drop them
            if "notes" in df_raw.columns:
                df_raw = df_raw[df_raw["notes"].isna()]

            for _, row in df_raw.iterrows():
                fips = str(row.get("county_code", row.get("fips", ""))).strip().zfill(5)
                if len(fips) != 5 or not fips.isdigit():
                    continue
                records.append(MortalityRecord(
                    county_fips=fips,
                    county_name=str(row.get("county", "")).strip(),
                    state=str(row.get("state", "")).strip(),
                    deaths=_safe_int(str(row.get("deaths", ""))),
                    population=_safe_int(str(row.get("population", ""))),
                    age_adjusted_rate=_safe_float(
                        str(row.get("age-adjusted_rate", row.get("age_adjusted_rate", "")))
                    ),
                    year=year,
                ))
        except Exception as exc:
            logger.warning("CDC Wonder: failed to parse %s: %s", txt_file, exc)

    if not records:
        logger.warning("CDC Wonder: no records loaded from %s", cache_dir)
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records])
    return df.set_index("county_fips")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_wonder()
    print(df.head(10))
    print(f"Shape: {df.shape}")
