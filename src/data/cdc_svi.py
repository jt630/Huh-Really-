"""
CDC Social Vulnerability Index - Social vulnerability percentile ranks by county.

Download: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html
Direct CSV: https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US_county.csv

Fields: county_fips, county_name, state, theme, percentile_rank, year
Themes: RPL_THEME1-4 (socioeconomic, household, minority, housing), RPL_THEMES (overall)
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SVI_URLS = {
    2020: "https://svi.cdc.gov/Documents/Data/2020_SVI_Data/CSV/SVI2020_US_county.csv",
    2018: "https://svi.cdc.gov/Documents/Data/2018_SVI_Data/CSV/SVI2018_US_county.csv",
    2016: "https://svi.cdc.gov/Documents/Data/2016_SVI_Data/CSV/SVI2016_US.csv",
}

THEME_COLS = {
    "RPL_THEME1": "socioeconomic",
    "RPL_THEME2": "household_characteristics",
    "RPL_THEME3": "minority_status",
    "RPL_THEME4": "housing_transportation",
    "RPL_THEMES": "overall",
}


class SVIRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    theme: str
    percentile_rank: float | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _download(url: str) -> bytes:
    logger.info("CDC SVI: downloading from %s", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def fetch_svi(year: int = 2020, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch CDC Social Vulnerability Index by county.

    Returns long-format DataFrame indexed by county_fips with columns:
        county_name, state, theme, percentile_rank, year
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"cdc_svi_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("CDC SVI: loading from cache")
        return pd.read_parquet(cache_file)

    url = SVI_URLS.get(year, SVI_URLS[2020])
    try:
        content = _download(url)
    except Exception as exc:
        logger.error("CDC SVI: download failed: %s", exc)
        return pd.DataFrame()

    from io import BytesIO
    try:
        df_raw = pd.read_csv(BytesIO(content), dtype=str, low_memory=False)
        df_raw.columns = [c.strip().upper() for c in df_raw.columns]
    except Exception as exc:
        logger.error("CDC SVI: parse error: %s", exc)
        return pd.DataFrame()

    fips_col = "FIPS" if "FIPS" in df_raw.columns else df_raw.columns[0]
    name_col = "COUNTY" if "COUNTY" in df_raw.columns else "LOCATION"
    state_col = "STATE" if "STATE" in df_raw.columns else "ST_ABBR"

    records = []
    for _, row in df_raw.iterrows():
        fips = str(row.get(fips_col, "")).strip().zfill(5)
        if len(fips) != 5:
            continue
        for col, theme_name in THEME_COLS.items():
            if col not in row.index:
                continue
            val_str = str(row[col]).strip()
            try:
                val = float(val_str) if val_str not in ("", "-999", "-9") else None
            except ValueError:
                val = None
            records.append(SVIRecord(
                county_fips=fips,
                county_name=str(row.get(name_col, "")).strip(),
                state=str(row.get(state_col, "")).strip(),
                theme=theme_name,
                percentile_rank=val,
                year=year,
            ))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_svi(year=2020)
    print(df.head(10))
    print(f"Shape: {df.shape}")
