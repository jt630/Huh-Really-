"""
EPA Superfund - NPL hazardous waste sites by county.

API: Envirofacts SEMS - https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/
GeoJSON: https://catalog.data.gov/dataset/superfund-national-priorities-list-npl-sites

Fields: county_fips, county_name, state, site_count, sites[]
        (site_name, site_id, lat, lon, npl_status, contaminants)
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Primary: Envirofacts SEMS active sites
SEMS_URL = "https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/json"
# Fallback: EPA ECHO facility search (NPL sites)
ECHO_URL = "https://echo.epa.gov/tools/web-services/facility-search-downloads?output=JSON&p_act=Y&p_st=&p_ct=&p_npl=Y"
# Alternative: static GeoJSON from data.gov
GEOJSON_URL = "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/SuperfundNPL_Sites/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"


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


def _fetch_with_backoff(url: str, params: dict | None = None, max_retries: int = 4) -> requests.Response:
    """HTTP GET with exponential backoff."""
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            if attempt == max_retries:
                raise
            logger.warning("HTTP %s on attempt %d; retrying in %ds", exc, attempt + 1, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Max retries exceeded")


def _fetch_sems_paged() -> list[dict]:
    """Fetch all SEMS active sites using pagination."""
    all_rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        url = f"https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/json/rows/{offset}:{offset + page_size - 1}"
        logger.info("Fetching SEMS rows %d-%d", offset, offset + page_size - 1)
        try:
            resp = _fetch_with_backoff(url)
            rows = resp.json()
        except Exception as exc:
            logger.warning("SEMS fetch error at offset %d: %s", offset, exc)
            break
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def _parse_sems_rows(rows: list[dict]) -> pd.DataFrame:
    """Parse SEMS rows into a county-aggregated DataFrame."""
    records = []
    for row in rows:
        # Field names vary; try multiple possible names
        site_name = row.get("site_name", row.get("SITE_NAME", row.get("facility_name", "")))
        site_id = str(row.get("site_id", row.get("SITE_ID", row.get("epa_id", ""))))
        state = row.get("state_code", row.get("STATE_CODE", row.get("state", "")))
        county = row.get("county_name", row.get("COUNTY_NAME", row.get("county", "")))
        npl_status = row.get("npl_status", row.get("NPL_STATUS", row.get("site_status", "")))

        # FIPS
        state_fips = str(row.get("state_fips", row.get("STATE_FIPS", ""))).zfill(2)
        county_fips_raw = str(row.get("county_fips", row.get("COUNTY_FIPS", row.get("fips_code", "")))).zfill(3)
        if state_fips and county_fips_raw:
            county_fips = (state_fips + county_fips_raw).zfill(5)
        else:
            county_fips = ""

        try:
            lat = float(row.get("latitude", row.get("LATITUDE", 0) or 0)) or None
        except (TypeError, ValueError):
            lat = None
        try:
            lon = float(row.get("longitude", row.get("LONGITUDE", 0) or 0)) or None
        except (TypeError, ValueError):
            lon = None

        records.append({
            "county_fips": county_fips,
            "county_name": county,
            "state": state,
            "site_name": site_name,
            "site_id": site_id,
            "latitude": lat,
            "longitude": lon,
            "npl_status": npl_status,
            "site_status": npl_status,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate to county level
    county_df = (
        df.groupby(["county_fips", "county_name", "state"])
        .agg(
            site_count=("site_id", "count"),
            site_name=("site_name", lambda x: list(x)),
            npl_status=("npl_status", lambda x: list(x)),
        )
        .reset_index()
    )
    return county_df


def fetch_superfund_sites(
    cache_dir: str = "data/cache/epa_superfund",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch EPA Superfund NPL sites aggregated by county.

    Parameters
    ----------
    cache_dir : directory for cached responses.
    force_refresh : if True, ignore cache.

    Returns
    -------
    pd.DataFrame with columns: county_fips, state, site_name, site_status,
                                npl_status, count (sites per county)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "superfund_npl_sites.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading EPA Superfund from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching EPA Superfund NPL sites")
    rows = _fetch_sems_paged()

    if not rows:
        # Fallback: try ArcGIS GeoJSON
        logger.info("SEMS returned no data; trying ArcGIS GeoJSON fallback")
        try:
            resp = _fetch_with_backoff(GEOJSON_URL)
            data = resp.json()
            features = data.get("features", [])
            rows = [f.get("attributes", {}) for f in features]
        except Exception as exc:
            logger.warning("ArcGIS fallback failed: %s", exc)

    df = _parse_sems_rows(rows)

    if df.empty:
        logger.warning("No EPA Superfund records returned; returning empty DataFrame")
        df = pd.DataFrame(columns=["county_fips", "county_name", "state", "site_count", "site_name", "npl_status"])
    else:
        logger.info("Fetched %d county superfund records from %d sites", len(df), len(rows))
        df.to_parquet(cache_file, index=False)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_superfund_sites()
    print(df.head())
