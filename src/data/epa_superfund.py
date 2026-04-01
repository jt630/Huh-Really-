"""
EPA Superfund - NPL hazardous waste sites by county.

API: Envirofacts SEMS - https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/
GeoJSON: https://catalog.data.gov/dataset/superfund-national-priorities-list-npl-sites

Fields: county_fips, county_name, state, site_count, sites[]
        (site_name, site_id, lat, lon, npl_status, contaminants)
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

NPL_GEOJSON_URL = (
    "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
    "Superfund_National_Priorities_List_Sites/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&f=geojson&resultRecordCount=10000"
)


class SuperfundSite(BaseModel):
    site_name: str
    site_id: str
    latitude: float | None
    longitude: float | None
    npl_status: str | None
    contaminants: list[str] | None = None


class SuperfundCountyRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    site_count: int
    sites: list[SuperfundSite] = []


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_geojson() -> dict:
    resp = requests.get(NPL_GEOJSON_URL, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_superfund(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch EPA Superfund NPL sites, spatially joined to county FIPS.

    Returns DataFrame indexed by county_fips with columns:
        county_name, state, site_count, site_names
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / "epa_superfund.parquet"

    if use_cache and cache_file.exists():
        logger.info("EPA Superfund: loading from cache")
        return pd.read_parquet(cache_file)

    logger.info("EPA Superfund: fetching NPL sites")
    try:
        data = _fetch_geojson()
    except Exception as exc:
        logger.error("EPA Superfund: fetch failed: %s", exc)
        return pd.DataFrame()

    features = data.get("features", [])
    rows = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None]) if geom else [None, None]
        rows.append({
            "site_name": props.get("SITE_NAME", ""),
            "site_id": props.get("SITE_ID", ""),
            "latitude": coords[1] if coords and len(coords) > 1 else None,
            "longitude": coords[0] if coords else None,
            "npl_status": props.get("NPL_STATUS", ""),
            "state": props.get("STATE_CODE", ""),
            "county_fips": str(props.get("COUNTY_FIPS", "")).zfill(5),
            "county_name": props.get("COUNTY_NAME", ""),
        })

    if not rows:
        return pd.DataFrame()

    df_sites = pd.DataFrame(rows)
    df = (df_sites.groupby("county_fips")
          .agg(
              county_name=("county_name", "first"),
              state=("state", "first"),
              site_count=("site_id", "count"),
              site_names=("site_name", list),
          )
          .reset_index()
          .set_index("county_fips"))

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df[["county_name", "state", "site_count"]].to_parquet(cache_file)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_superfund()
    print(df.head(10))
    print(f"Shape: {df.shape}")
