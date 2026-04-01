"""
USGS Pesticide National Synthesis Project - County-level pesticide use estimates.

Download: https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/
Files: EPest.county.<YEAR>.txt (tab-delimited)

Columns: COMPOUND, YEAR, STATE_FIPS_CODE, COUNTY_FIPS_CODE,
         EPEST_LOW_KG, EPEST_HIGH_KG

Compounds of interest: PARAQUAT DICHLORIDE, ROTENONE, CHLORPYRIFOS, MANEB

Fields: county_fips, county_name, state, compound, kg_applied, year
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/"
    "PesticideUseEstimates/EPest.county.{year}.txt"
)

DEFAULT_COMPOUNDS = [
    "PARAQUAT DICHLORIDE",
    "ROTENONE",
    "CHLORPYRIFOS",
    "MANEB",
    "ZIRAM",
]


class PesticideUseRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    compound: str
    kg_applied: float
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _download_year(year: int, cache_path: Path) -> Path:
    url = BASE_URL.format(year=year)
    logger.info("USGS Pesticide: downloading %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return cache_path


def fetch_pesticide_use(
    years: list[int] | None = None,
    compounds: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch USGS county-level pesticide use estimates.

    Parameters
    ----------
    years:
        List of years to fetch. Defaults to [2015, 2016, 2017, 2018, 2019].
    compounds:
        List of compound names to filter (uppercase). Defaults to DEFAULT_COMPOUNDS.

    Returns
    -------
    pd.DataFrame indexed by county_fips with columns:
        compound, kg_applied, year
    """
    from src.config import get_settings
    if years is None:
        years = [2015, 2016, 2017, 2018, 2019]
    if compounds is None:
        compounds = DEFAULT_COMPOUNDS

    compounds_upper = [c.upper() for c in compounds]
    cache_base = get_settings().cache_dir / "usgs_pesticide"
    cache_base.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for year in years:
        cache_path = cache_base / f"EPest.county.{year}.txt"
        if not (use_cache and cache_path.exists()):
            try:
                _download_year(year, cache_path)
            except Exception as exc:
                logger.warning("USGS Pesticide: could not download year %d: %s", year, exc)
                continue

        try:
            df_raw = pd.read_csv(cache_path, sep="\t", dtype=str)
            df_raw.columns = [c.strip().upper() for c in df_raw.columns]

            # Build 5-digit FIPS
            df_raw["county_fips"] = (
                df_raw["STATE_FIPS_CODE"].str.zfill(2)
                + df_raw["COUNTY_FIPS_CODE"].str.zfill(3)
            )

            # Filter compounds
            df_raw["COMPOUND"] = df_raw["COMPOUND"].str.upper().str.strip()
            df_filtered = df_raw[df_raw["COMPOUND"].isin(compounds_upper)].copy()

            df_filtered["kg_applied"] = pd.to_numeric(
                df_filtered.get("EPEST_HIGH_KG", df_filtered.get("EPEST_LOW_KG", 0)),
                errors="coerce",
            ).fillna(0.0)

            df_filtered["year"] = year
            df_filtered = df_filtered.rename(columns={"COMPOUND": "compound"})
            df_filtered["county_name"] = ""
            df_filtered["state"] = ""

            all_frames.append(
                df_filtered[["county_fips", "county_name", "state", "compound", "kg_applied", "year"]]
            )
        except Exception as exc:
            logger.warning("USGS Pesticide: parse error year %d: %s", year, exc)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True).set_index("county_fips")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_pesticide_use(years=[2019], compounds=["PARAQUAT DICHLORIDE", "CHLORPYRIFOS"])
    print(df.head(10))
    print(f"Shape: {df.shape}")
