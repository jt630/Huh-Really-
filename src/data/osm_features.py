"""
OpenStreetMap via Overpass API - Geographic feature counts by county.

Endpoint: https://overpass-api.de/api/interpreter
Use case: count features by OSM tag (e.g. leisure=golf_course, shop=supermarket).
Strategy: query by state bounding boxes, then spatially join to county polygons.

County boundaries: US Census TIGER 2019 shapefile
  https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip

Rate limit: 2-second sleep between state queries.

Fields: county_fips, county_name, state, feature_type,
        count, density_per_sq_km
"""

from __future__ import annotations

import io
import logging
import pickle
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class OSMFeatureRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    feature_type: str
    count: int
    density_per_sq_km: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
TIGER_COUNTY_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip"
)

# Bounding boxes [S, W, N, E] for each US state FIPS code
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "01": (30.14, -88.47, 35.01, -84.89),   # AL
    "02": (51.17, -179.15, 71.46, -129.97), # AK
    "04": (31.33, -114.82, 37.00, -109.05), # AZ
    "05": (33.00, -94.62, 36.50, -89.64),   # AR
    "06": (32.53, -124.48, 42.01, -114.13), # CA
    "08": (36.99, -109.06, 41.00, -102.04), # CO
    "09": (40.95, -73.73, 42.05, -71.79),   # CT
    "10": (38.45, -75.79, 39.84, -75.05),   # DE
    "11": (38.79, -77.12, 38.99, -76.91),   # DC
    "12": (24.39, -87.63, 31.00, -79.97),   # FL
    "13": (30.36, -85.61, 35.00, -80.84),   # GA
    "15": (18.86, -160.25, 22.24, -154.79), # HI
    "16": (41.99, -117.24, 49.00, -111.04), # ID
    "17": (36.97, -91.51, 42.51, -87.01),   # IL
    "18": (37.77, -88.10, 41.77, -84.78),   # IN
    "19": (40.38, -96.64, 43.50, -90.14),   # IA
    "20": (36.99, -102.05, 40.00, -94.59),  # KS
    "21": (36.49, -89.57, 39.15, -81.96),   # KY
    "22": (28.93, -94.04, 33.02, -88.82),   # LA
    "23": (43.06, -71.08, 47.46, -66.95),   # ME
    "24": (37.91, -79.49, 39.72, -74.99),   # MD
    "25": (41.24, -73.50, 42.89, -69.93),   # MA
    "26": (41.70, -90.42, 48.19, -82.41),   # MI
    "27": (43.50, -97.24, 49.38, -89.48),   # MN
    "28": (30.17, -91.65, 35.00, -88.10),   # MS
    "29": (35.99, -95.77, 40.61, -89.10),   # MO
    "30": (44.36, -116.05, 49.00, -104.04), # MT
    "31": (39.99, -104.05, 43.00, -95.31),  # NE
    "32": (35.00, -120.00, 42.00, -114.03), # NV
    "33": (42.70, -72.56, 45.31, -70.61),   # NH
    "34": (38.92, -75.56, 41.36, -73.89),   # NJ
    "35": (31.33, -109.05, 37.00, -103.00), # NM
    "36": (40.50, -79.76, 45.01, -71.86),   # NY
    "37": (33.84, -84.32, 36.59, -75.46),   # NC
    "38": (45.93, -104.05, 49.00, -96.55),  # ND
    "39": (38.40, -84.82, 42.00, -80.52),   # OH
    "40": (33.62, -103.00, 37.00, -94.43),  # OK
    "41": (41.99, -124.56, 46.26, -116.46), # OR
    "42": (39.72, -80.52, 42.27, -74.69),   # PA
    "44": (41.15, -71.90, 42.02, -71.12),   # RI
    "45": (32.05, -83.35, 35.21, -78.54),   # SC
    "46": (42.48, -104.06, 45.94, -96.44),  # SD
    "47": (34.98, -90.31, 36.68, -81.65),   # TN
    "48": (25.84, -106.65, 36.50, -93.51),  # TX
    "49": (36.99, -114.05, 42.00, -109.04), # UT
    "50": (42.73, -73.44, 45.02, -71.46),   # VT
    "51": (36.54, -83.68, 39.47, -75.24),   # VA
    "53": (45.54, -124.73, 49.00, -116.92), # WA
    "54": (37.20, -82.64, 40.64, -77.72),   # WV
    "55": (42.49, -92.89, 47.08, -86.25),   # WI
    "56": (40.99, -111.05, 45.01, -104.05), # WY
    "72": (17.88, -67.96, 18.53, -65.22),   # PR
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exponential_backoff(attempt: int, base: float = 1.0, max_wait: float = 60.0) -> None:
    wait = min(base * (2 ** attempt), max_wait)
    logger.debug("Backing off %.1fs (attempt %d)", wait, attempt + 1)
    time.sleep(wait)


def _ensure_cache_dir(cache_dir: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_county_shapefile(cache_dir: Path):
    """Load US county shapefile from Census TIGER (download if not cached)."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required for osm_features: pip install geopandas") from exc

    shp_dir = cache_dir / "tiger_county"
    shp_path = shp_dir / "tl_2019_us_county.shp"

    if not shp_path.exists():
        logger.info("Downloading Census TIGER county shapefile...")
        shp_dir.mkdir(parents=True, exist_ok=True)
        max_retries = 4
        for attempt in range(max_retries):
            try:
                resp = requests.get(TIGER_COUNTY_URL, timeout=180, stream=True)
                resp.raise_for_status()
                zip_data = io.BytesIO(resp.content)
                with zipfile.ZipFile(zip_data) as zf:
                    zf.extractall(shp_dir)
                logger.info("TIGER shapefile extracted to %s", shp_dir)
                break
            except requests.HTTPError as exc:
                if attempt < max_retries - 1:
                    logger.warning("HTTP error downloading TIGER: %s, retrying...", exc)
                    _exponential_backoff(attempt)
                else:
                    raise RuntimeError(
                        f"Failed to download TIGER county shapefile: {exc}"
                    ) from exc
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries - 1:
                    logger.warning("Error downloading TIGER: %s, retrying...", exc)
                    _exponential_backoff(attempt)
                else:
                    raise RuntimeError(f"TIGER download error: {exc}") from exc

    logger.info("Loading county shapefile from %s", shp_path)
    gdf = gpd.read_file(shp_path)
    # STATEFP, COUNTYFP, GEOID, NAME, NAMELSAD
    gdf["county_fips"] = gdf["GEOID"].str.zfill(5)
    # Area in km² (project to equal-area CRS)
    gdf_proj = gdf.to_crs("ESRI:102003")  # Albers Equal Area
    gdf["area_sq_km"] = gdf_proj.geometry.area / 1e6
    return gdf


def _build_overpass_query(
    osm_tag: str,
    osm_value: str,
    bbox: tuple[float, float, float, float],
) -> str:
    """Build Overpass QL query for a given tag and bounding box."""
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    return (
        f"[out:json][timeout:90];\n"
        f"(\n"
        f'  node["{osm_tag}"="{osm_value}"]({bbox_str});\n'
        f'  way["{osm_tag}"="{osm_value}"]({bbox_str});\n'
        f'  relation["{osm_tag}"="{osm_value}"]({bbox_str});\n'
        f");\n"
        f"out center;"
    )


def _query_overpass(query: str, max_retries: int = 4) -> list[dict]:
    """POST a query to the Overpass API and return the list of elements."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("elements", [])
        except requests.HTTPError as exc:
            if attempt < max_retries - 1:
                logger.warning("Overpass HTTP error: %s, retrying...", exc)
                _exponential_backoff(attempt, base=2.0, max_wait=60.0)
            else:
                raise RuntimeError(
                    f"Overpass API failed after {max_retries} attempts: {exc}"
                ) from exc
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                logger.warning("Overpass error: %s, retrying...", exc)
                _exponential_backoff(attempt, base=2.0, max_wait=60.0)
            else:
                raise RuntimeError(f"Overpass API error: {exc}") from exc
    return []


# ---------------------------------------------------------------------------
# Public fetch function
# ---------------------------------------------------------------------------


def fetch_osm_features(
    osm_tag: str,
    osm_value: str,
    state_fips_list: list[str] | None = None,
    cache_dir: str = "data/cache/osm_features",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OSM feature counts by US county via Overpass API + Census TIGER spatial join.

    Queries Overpass by state bounding box (2-second rate-limit sleep between states),
    then spatially joins results to county polygons to compute per-county counts and
    density (features per km²).

    Parameters
    ----------
    osm_tag : str
        OSM tag key (e.g. "leisure", "shop", "amenity").
    osm_value : str
        OSM tag value (e.g. "golf_course", "supermarket", "hospital").
    state_fips_list : list[str] | None
        State FIPS codes (2-digit strings) to query. If None, query all 50 states + DC.
    cache_dir : str
        Directory for caching results and downloaded shapefiles.
    force_refresh : bool
        If True, bypass cached pickle results.

    Returns
    -------
    pd.DataFrame
        Columns: county_fips, county_name, state, feature_type,
                 count, density_per_sq_km
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError(
            "geopandas and shapely are required: pip install geopandas shapely"
        ) from exc

    cache_dir_path = _ensure_cache_dir(cache_dir)
    feature_type = f"{osm_tag}={osm_value}"
    states_str = "_".join(sorted(state_fips_list)) if state_fips_list else "all"
    pkl_path = cache_dir_path / f"osm_{osm_tag}_{osm_value}_{states_str}.pkl"

    if not force_refresh and pkl_path.exists():
        logger.info("Loading cached OSM features from %s", pkl_path)
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)

    # Determine which states to query
    if state_fips_list is None:
        states_to_query = {
            k: v for k, v in STATE_BBOXES.items() if int(k) <= 56
        }  # continental + HI/AK
    else:
        states_to_query = {
            fips: STATE_BBOXES[fips]
            for fips in state_fips_list
            if fips in STATE_BBOXES
        }
        unknown = [f for f in state_fips_list if f not in STATE_BBOXES]
        if unknown:
            logger.warning("Unknown state FIPS codes (skipping): %s", unknown)

    # Load county shapefile
    county_gdf = _load_county_shapefile(cache_dir_path)

    # Collect all OSM elements across states
    all_elements: list[dict] = []

    for i, (state_fips, bbox) in enumerate(sorted(states_to_query.items())):
        logger.info(
            "Querying Overpass for state %s (%s=%s) [%d/%d]",
            state_fips, osm_tag, osm_value, i + 1, len(states_to_query),
        )
        query = _build_overpass_query(osm_tag, osm_value, bbox)
        try:
            elements = _query_overpass(query)
            logger.info("  -> %d elements returned for state %s", len(elements), state_fips)
            all_elements.extend(elements)
        except RuntimeError as exc:
            logger.warning("Overpass query failed for state %s: %s (skipping)", state_fips, exc)

        # Rate-limit: 2 seconds between state queries
        if i < len(states_to_query) - 1:
            time.sleep(2)

    if not all_elements:
        logger.warning("No OSM elements found for %s=%s", osm_tag, osm_value)
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "feature_type",
                     "count", "density_per_sq_km"]
        )

    # Extract centroid coordinates from elements
    points: list[dict] = []
    for elem in all_elements:
        lat = elem.get("lat") or elem.get("center", {}).get("lat")
        lon = elem.get("lon") or elem.get("center", {}).get("lon")
        if lat is not None and lon is not None:
            points.append({"osm_id": elem.get("id"), "lat": float(lat), "lon": float(lon)})

    if not points:
        logger.warning("No valid coordinates extracted from OSM elements")
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state", "feature_type",
                     "count", "density_per_sq_km"]
        )

    points_df = pd.DataFrame(points)
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df["lon"], points_df["lat"]),
        crs="EPSG:4326",
    )

    # Ensure county_gdf has same CRS
    county_gdf_4326 = county_gdf.to_crs("EPSG:4326")

    # Spatial join: assign each point to a county
    logger.info("Performing spatial join of %d OSM points to county polygons...", len(points_gdf))
    joined = gpd.sjoin(
        points_gdf,
        county_gdf_4326[["county_fips", "NAME", "STUSPS" if "STUSPS" in county_gdf_4326.columns else "STATEFP", "area_sq_km", "geometry"]],
        how="left",
        predicate="within",
    )

    # Count features per county
    # Determine state column name
    state_col = "STUSPS" if "STUSPS" in joined.columns else "STATEFP"
    county_counts = (
        joined.groupby("county_fips")
        .agg(
            count=("osm_id", "count"),
            county_name=("NAME", "first"),
            state=(state_col, "first"),
            area_sq_km=("area_sq_km", "first"),
        )
        .reset_index()
    )

    county_counts["feature_type"] = feature_type
    county_counts["density_per_sq_km"] = (
        county_counts["count"] / county_counts["area_sq_km"].replace(0, float("nan"))
    ).fillna(0.0)

    # Ensure all counties in the queried states appear (even with count=0)
    queried_fips_prefixes = set(states_to_query.keys())
    all_counties = county_gdf_4326[
        county_gdf_4326["STATEFP"].isin(queried_fips_prefixes)
    ][["county_fips", "NAME", state_col if state_col in county_gdf_4326.columns else "STATEFP", "area_sq_km"]].copy()
    all_counties = all_counties.rename(
        columns={"NAME": "county_name", state_col if state_col in county_gdf_4326.columns else "STATEFP": "state"}
    )

    result = all_counties.merge(
        county_counts[["county_fips", "count", "density_per_sq_km", "feature_type"]],
        on="county_fips",
        how="left",
    )
    result["count"] = result["count"].fillna(0).astype(int)
    result["density_per_sq_km"] = result["density_per_sq_km"].fillna(0.0)
    result["feature_type"] = result["feature_type"].fillna(feature_type)

    out_cols = ["county_fips", "county_name", "state", "feature_type", "count", "density_per_sq_km"]
    result = result[out_cols].reset_index(drop=True)

    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    logger.info("Cached OSM features DataFrame (%d rows) to %s", len(result), pkl_path)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Query golf courses in California only as a quick test
    df = fetch_osm_features(
        osm_tag="leisure",
        osm_value="golf_course",
        state_fips_list=["06"],  # California only
        cache_dir="data/cache/osm_features",
    )
    print(f"Shape: {df.shape}")
    print(df.head(5))
