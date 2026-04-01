"""
OpenStreetMap via Overpass API - Geographic feature counts by county.

Endpoint: https://overpass-api.de/api/interpreter
Use case: count golf courses [leisure=golf_course], industrial sites, etc.
Rate limit: ~1 req/2s; batch by state bounding box.

Fields: county_fips, county_name, state, feature_type, count, density_per_sq_km
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Approximate bounding boxes for US states (south, west, north, east)
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AL": (30.14, -88.47, 35.01, -84.89), "AK": (54.56, -179.15, 71.35, -129.99),
    "AZ": (31.33, -114.82, 37.00, -109.05), "AR": (33.00, -94.62, 36.50, -89.64),
    "CA": (32.53, -124.41, 42.01, -114.13), "CO": (36.99, -109.06, 41.00, -102.04),
    "CT": (40.99, -73.73, 42.05, -71.79), "DE": (38.45, -75.79, 39.84, -75.05),
    "FL": (24.52, -87.63, 31.00, -80.03), "GA": (30.36, -85.61, 35.00, -80.84),
    "HI": (18.91, -160.25, 22.24, -154.81), "ID": (41.99, -117.24, 49.00, -111.04),
    "IL": (36.97, -91.51, 42.51, -87.02), "IN": (37.77, -88.10, 41.76, -84.79),
    "IA": (40.38, -96.64, 43.50, -90.14), "KS": (36.99, -102.05, 40.00, -94.59),
    "KY": (36.50, -89.57, 39.15, -81.96), "LA": (28.93, -94.04, 33.02, -89.00),
    "ME": (42.98, -71.08, 47.46, -66.95), "MD": (37.89, -79.49, 39.72, -75.05),
    "MA": (41.24, -73.50, 42.89, -69.93), "MI": (41.70, -90.42, 48.26, -82.42),
    "MN": (43.50, -97.24, 49.38, -89.49), "MS": (30.17, -91.66, 35.01, -88.10),
    "MO": (35.99, -95.77, 40.61, -89.10), "MT": (44.36, -116.05, 49.00, -104.04),
    "NE": (39.99, -104.05, 43.00, -95.31), "NV": (35.00, -120.01, 42.00, -114.04),
    "NH": (42.70, -72.56, 45.31, -70.61), "NJ": (38.93, -75.56, 41.36, -73.89),
    "NM": (31.33, -109.05, 37.00, -103.00), "NY": (40.50, -79.76, 45.01, -71.86),
    "NC": (33.84, -84.32, 36.59, -75.46), "ND": (45.94, -104.05, 49.00, -96.55),
    "OH": (38.40, -84.82, 41.98, -80.52), "OK": (33.62, -103.00, 37.00, -94.43),
    "OR": (41.99, -124.57, 46.24, -116.46), "PA": (39.72, -80.52, 42.27, -74.70),
    "RI": (41.15, -71.86, 42.02, -71.12), "SC": (32.03, -83.35, 35.22, -78.54),
    "SD": (42.48, -104.06, 45.94, -96.44), "TN": (34.98, -90.31, 36.68, -81.65),
    "TX": (25.84, -106.65, 36.50, -93.51), "UT": (36.99, -114.05, 42.00, -109.04),
    "VT": (42.73, -73.44, 45.02, -71.50), "VA": (36.54, -83.68, 39.47, -75.24),
    "WA": (45.54, -124.74, 49.00, -116.92), "WV": (37.20, -82.64, 40.64, -77.72),
    "WI": (42.49, -92.89, 47.08, -86.25), "WY": (40.99, -111.06, 45.01, -104.05),
}


class OSMFeatureRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    feature_type: str
    count: int
    density_per_sq_km: float


def _parse_tag(tag_str: str) -> tuple[str, str]:
    """Parse 'key=value' into (key, value)."""
    if "=" in tag_str:
        k, v = tag_str.split("=", 1)
        return k.strip(), v.strip()
    return tag_str.strip(), ""


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(4),
)
def _query_overpass(query: str) -> dict:
    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _load_counties_gdf() -> "pd.DataFrame":
    """Load county boundaries GeoDataFrame, downloading if needed."""
    import geopandas as gpd

    from src.config import get_settings
    geojson_path = get_settings().cache_dir / "counties.geojson"

    if not geojson_path.exists():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        logger.info("OSM Features: downloading county GeoJSON")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        geojson_path.write_bytes(resp.content)

    gdf = gpd.read_file(str(geojson_path))
    gdf = gdf.rename(columns={"id": "county_fips"})
    gdf["area_sq_km"] = gdf.geometry.to_crs("EPSG:5070").area / 1e6
    return gdf


def fetch_osm_features(
    osm_tag: str = "leisure=golf_course",
    states: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Count OSM features by county via the Overpass API.

    Parameters
    ----------
    osm_tag:
        OSM tag string, e.g. "leisure=golf_course" or "landuse=industrial".
    states:
        List of 2-letter state codes to query. Defaults to all 50 states.

    Returns
    -------
    pd.DataFrame indexed by county_fips with columns:
        county_name, state, feature_type, count, density_per_sq_km
    """
    from src.config import get_settings
    tag_safe = osm_tag.replace("=", "_").replace("/", "_")
    cache_file = get_settings().cache_dir / f"osm_{tag_safe}.parquet"

    if use_cache and cache_file.exists():
        logger.info("OSM Features: loading from cache %s", cache_file)
        return pd.read_parquet(cache_file)

    tag_key, tag_val = _parse_tag(osm_tag)
    if states is None:
        states = list(STATE_BBOXES.keys())

    # Collect lat/lon points for all matching features
    features: list[tuple[float, float]] = []
    for state in states:
        if state not in STATE_BBOXES:
            continue
        s, w, n, e = STATE_BBOXES[state]
        bbox_str = f"{s},{w},{n},{e}"
        query = (
            f'[out:json][timeout:60];'
            f'('
            f'  way["{tag_key}"="{tag_val}"]({bbox_str});'
            f'  relation["{tag_key}"="{tag_val}"]({bbox_str});'
            f');'
            f'out center;'
        )
        try:
            data = _query_overpass(query)
            for elem in data.get("elements", []):
                if "center" in elem:
                    features.append((elem["center"]["lat"], elem["center"]["lon"]))
                elif "lat" in elem:
                    features.append((elem["lat"], elem["lon"]))
            logger.info("OSM Features: %s found %d in %s", osm_tag, len(features), state)
        except Exception as exc:
            logger.warning("OSM Features: query failed for %s: %s", state, exc)
        time.sleep(2)

    if not features:
        return pd.DataFrame()

    # Spatial join to counties
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        gdf_counties = _load_counties_gdf()
        pts_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lat, lon in features]},
            crs="EPSG:4326",
        )
        if gdf_counties.crs is None:
            gdf_counties = gdf_counties.set_crs("EPSG:4326")

        joined = gpd.sjoin(pts_gdf, gdf_counties[["county_fips", "geometry", "area_sq_km"]],
                           how="left", predicate="within")
        counts = joined.groupby("county_fips").size().rename("count").reset_index()

        result = counts.merge(
            gdf_counties[["county_fips", "area_sq_km"]].drop_duplicates(),
            on="county_fips", how="left"
        )
        result["density_per_sq_km"] = result["count"] / result["area_sq_km"].clip(lower=0.01)
        result["feature_type"] = osm_tag
        result["county_name"] = ""
        result["state"] = ""
        df = result[["county_fips", "county_name", "state", "feature_type",
                      "count", "density_per_sq_km"]].set_index("county_fips")

    except ImportError:
        logger.warning("OSM Features: geopandas not available; returning raw counts without spatial join")
        df = pd.DataFrame({"lat": [f[0] for f in features], "lon": [f[1] for f in features]})
        df["county_fips"] = "00000"
        df["feature_type"] = osm_tag
        df = df.groupby("county_fips").size().rename("count").reset_index()
        df["density_per_sq_km"] = 0.0
        df["county_name"] = ""
        df["state"] = ""
        df = df.set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_osm_features("leisure=golf_course", states=["CA", "FL"])
    print(df.head(10))
    print(f"Shape: {df.shape}")
