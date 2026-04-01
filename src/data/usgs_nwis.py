"""
USGS NWIS - Water quality measurements by site.

API: https://waterservices.usgs.gov/rest/
Portal: https://www.waterqualitydata.us/

Fields: county_fips, site_number, site_name, latitude, longitude,
        parameter_code, parameter_name, mean_value, unit,
        observation_count, start_date, end_date
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

WQP_URL = "https://www.waterqualitydata.us/data/Result/search"


class NWISRecord(BaseModel):
    county_fips: str
    site_number: str
    site_name: str
    latitude: float | None
    longitude: float | None
    parameter_code: str
    parameter_name: str
    mean_value: float | None
    unit: str
    observation_count: int
    start_date: str
    end_date: str


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_wqp(county_fips: str, characteristic: str) -> pd.DataFrame:
    state_code = county_fips[:2]
    county_code = county_fips[2:]
    params = {
        "statecode": f"US:{state_code}",
        "countycode": f"US:{state_code}:{county_code}",
        "characteristicName": characteristic,
        "mimeType": "csv",
        "zip": "no",
        "dataProfile": "resultPhysChem",
    }
    resp = requests.get(WQP_URL, params=params, timeout=120)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text), dtype=str, low_memory=False)


def fetch_nwis(
    county_fips_list: list[str] | None = None,
    characteristics: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch USGS water quality data via Water Quality Portal.

    Returns DataFrame indexed by county_fips with columns:
        parameter_code, parameter_name, mean_value, unit, observation_count
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / "usgs_nwis.parquet"

    if use_cache and cache_file.exists():
        logger.info("USGS NWIS: loading from cache")
        return pd.read_parquet(cache_file)

    if characteristics is None:
        characteristics = ["Nitrate", "Phosphorus", "pH", "Arsenic"]
    if county_fips_list is None:
        county_fips_list = ["06037", "17031"]  # default sample

    records = []
    for fips in county_fips_list:
        for char in characteristics:
            try:
                df_raw = _fetch_wqp(fips, char)
                if df_raw.empty:
                    continue
                val_col = next((c for c in df_raw.columns if "ResultMeasureValue" in c), None)
                unit_col = next((c for c in df_raw.columns if "ResultMeasure/MeasureUnitCode" in c), None)
                if val_col is None:
                    continue
                vals = pd.to_numeric(df_raw[val_col], errors="coerce").dropna()
                records.append(NWISRecord(
                    county_fips=fips,
                    site_number="",
                    site_name="",
                    latitude=None,
                    longitude=None,
                    parameter_code=char,
                    parameter_name=char,
                    mean_value=float(vals.mean()) if len(vals) else None,
                    unit=str(df_raw[unit_col].iloc[0]) if unit_col and len(df_raw) else "",
                    observation_count=len(vals),
                    start_date="",
                    end_date="",
                ))
            except Exception as exc:
                logger.warning("USGS NWIS: failed fips=%s char=%s: %s", fips, char, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_nwis(county_fips_list=["06037"], characteristics=["Nitrate"])
    print(df.head())
