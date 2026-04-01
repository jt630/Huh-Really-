"""
EPA EJScreen - Environmental justice indicators by census block group.

Download: https://www.epa.gov/ejscreen/download-ejscreen-data
File: EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv.zip (~300 MB)

Fields: county_fips, census_block_group, percentile_pm25, percentile_ozone,
        percentile_diesel_pm, percentile_cancer_risk, percentile_resp_hazard,
        percentile_traffic, percentile_npdes, percentile_rmp,
        percentile_superfund, percentile_hazwaste, percentile_ust,
        ej_index_pm25, year
"""
from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

EJSCREEN_URL = (
    "https://gaftp.epa.gov/EJSCREEN/2023/"
    "EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.csv.zip"
)

PERCENTILE_COLS = {
    "P_PM25": "percentile_pm25",
    "P_OZONE": "percentile_ozone",
    "P_DSLPM": "percentile_diesel_pm",
    "P_CANCER": "percentile_cancer_risk",
    "P_RESP": "percentile_resp_hazard",
    "P_PTRAF": "percentile_traffic",
    "P_PWDIS": "percentile_npdes",
    "P_PRMP": "percentile_rmp",
    "P_PNPL": "percentile_superfund",
    "P_PHWDIS": "percentile_hazwaste",
    "P_UST": "percentile_ust",
    "P_D5_PM25": "ej_index_pm25",
}


class EJScreenRecord(BaseModel):
    county_fips: str
    census_block_group: str
    percentile_pm25: float | None
    percentile_ozone: float | None
    percentile_diesel_pm: float | None
    percentile_cancer_risk: float | None
    percentile_resp_hazard: float | None
    percentile_traffic: float | None
    percentile_npdes: float | None
    percentile_rmp: float | None
    percentile_superfund: float | None
    percentile_hazwaste: float | None
    percentile_ust: float | None
    ej_index_pm25: float | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=4, max=120),
    stop=stop_after_attempt(3),
)
def _download_zip() -> bytes:
    logger.info("EPA EJScreen: downloading ~300 MB zip...")
    resp = requests.get(EJSCREEN_URL, timeout=600, stream=True)
    resp.raise_for_status()
    return resp.content


def fetch_ejscreen(year: int = 2023, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch EPA EJScreen data, aggregated to county means.

    Returns DataFrame indexed by county_fips.
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"epa_ejscreen_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("EPA EJScreen: loading from cache")
        return pd.read_parquet(cache_file)

    try:
        zip_bytes = _download_zip()
    except Exception as exc:
        logger.error("EPA EJScreen: download failed: %s", exc)
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                df_raw = pd.read_csv(f, dtype=str, low_memory=False)
    except Exception as exc:
        logger.error("EPA EJScreen: parse error: %s", exc)
        return pd.DataFrame()

    df_raw.columns = [c.strip().upper() for c in df_raw.columns]

    # Block group GEOID is in "ID" column (12 digits)
    id_col = "ID" if "ID" in df_raw.columns else df_raw.columns[0]
    df_raw["census_block_group"] = df_raw[id_col].str.strip().str.zfill(12)
    df_raw["county_fips"] = df_raw["census_block_group"].str[:5]

    rename_map = {k: v for k, v in PERCENTILE_COLS.items() if k in df_raw.columns}
    df_raw = df_raw.rename(columns=rename_map)

    pct_cols = list(rename_map.values())
    for col in pct_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_county = df_raw.groupby("county_fips")[pct_cols].mean()
    df_county["year"] = year

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df_county.to_parquet(cache_file)

    return df_county


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_ejscreen()
    print(df.head(5))
    print(f"Shape: {df.shape}")
