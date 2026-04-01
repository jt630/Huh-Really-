"""
USDA CropScape / Cropland Data Layer - Crop acreage by county.

API: https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat
Params: fips=<5digit>&year=<YYYY>

Fields: county_fips, county_name, state, crop_name, crop_code,
        area_acres, area_pct, year
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CDL_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat"


class CropScapeRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    crop_name: str
    crop_code: int
    area_acres: float
    area_pct: float
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
)
def _fetch_county(fips: str, year: int) -> str:
    resp = requests.get(CDL_URL, params={"year": year, "fips": fips}, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_cropscape(
    county_fips_list: list[str] | None = None,
    year: int = 2020,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch USDA CropScape crop acreage by county.

    Returns DataFrame indexed by county_fips with columns:
        crop_name, crop_code, area_acres, area_pct, year
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"usda_cropscape_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("USDA CropScape: loading from cache")
        return pd.read_parquet(cache_file)

    if county_fips_list is None:
        # Default to a small sample
        county_fips_list = ["06037", "17031", "48113"]

    records = []
    for fips in county_fips_list:
        try:
            xml_text = _fetch_county(fips, year)
            root = ET.fromstring(xml_text)
            # Response format: <Return><Category><Number>code</Number><Name>name</Name><Acreage>val</Acreage></Category>...
            categories = root.findall(".//Category")
            total_acres = sum(
                float(c.findtext("Acreage", "0") or "0") for c in categories
            )
            for cat in categories:
                code_str = cat.findtext("Number", "0") or "0"
                name = cat.findtext("Name", "") or ""
                acres_str = cat.findtext("Acreage", "0") or "0"
                try:
                    acres = float(acres_str)
                    code = int(code_str)
                except ValueError:
                    continue
                records.append(CropScapeRecord(
                    county_fips=fips,
                    county_name="",
                    state=fips[:2],
                    crop_name=name,
                    crop_code=code,
                    area_acres=acres,
                    area_pct=round(acres / total_acres * 100, 2) if total_acres > 0 else 0.0,
                    year=year,
                ))
        except Exception as exc:
            logger.warning("USDA CropScape: failed fips=%s: %s", fips, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_cropscape(county_fips_list=["06037"], year=2020)
    print(df.head(10))
