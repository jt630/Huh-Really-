"""
EPA Toxics Release Inventory - Facility toxic releases aggregated to county.

API: Envirofacts REST - https://data.epa.gov/efservice/tri_facility/
Bulk CSV: https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present

Fields: county_fips, county_name, state, chemical, cas_number,
        total_releases_lbs, air/water/land_releases_lbs, year
"""
import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# TRI basic data file column indices (1b file - releases)
# https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present
TRI_BULK_URL = "https://www3.epa.gov/tri/current/US_{year}_v{version}.zip"
TRI_BULK_FALLBACK = "https://www.epa.gov/system/files/other-files/{year}-09/{year}_us.zip"
ENVIROFACTS_URL = "https://data.epa.gov/efservice/tri_facility/state_abbr/{state}/json"

# Key column names in TRI CSV (1b file)
TRI_COL_YEAR = "REPORTING YEAR"
TRI_COL_FACILITY = "FACILITY NAME"
TRI_COL_COUNTY = "COUNTY"
TRI_COL_ST = "ST"
TRI_COL_FIPS = "COUNTY/FIPS CODE"
TRI_COL_CHEMICAL = "CHEMICAL"
TRI_COL_CAS = "CAS #/COMPOUND ID"
TRI_COL_TOTAL = "13 - TOTAL RELEASES"
TRI_COL_AIR = "5.1 - FUGITIVE AIR"
TRI_COL_WATER = "6.1 - STREAMS/WATER BODIES"
TRI_COL_LAND = "7.1 - UNDERGROUND INJECTION"


class TRIRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    chemical: str
    cas_number: str
    total_releases_lbs: float
    air_releases_lbs: float
    water_releases_lbs: float
    land_releases_lbs: float
    year: int


def _fetch_with_backoff(url: str, params: dict | None = None, max_retries: int = 4, stream: bool = False):
    """HTTP GET with exponential backoff."""
    delay = 1
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=120, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            if attempt == max_retries:
                raise
            logger.warning("HTTP %s on attempt %d; retrying in %ds", exc, attempt + 1, delay)
            time.sleep(delay)
            delay *= 2


def _try_bulk_download(year: int, cache_dir: Path) -> pd.DataFrame | None:
    """Try to download TRI bulk data file for the given year."""
    versions = ["10", "9", "8", "7"]
    for version in versions:
        url = TRI_BULK_URL.format(year=year, version=version)
        logger.info("Trying TRI bulk URL: %s", url)
        try:
            resp = _fetch_with_backoff(url, stream=True, max_retries=1)
            content = resp.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Find the 1b file (releases)
                names = zf.namelist()
                target = next((n for n in names if "1b" in n.lower() and n.endswith(".csv")), None)
                if target is None:
                    target = next((n for n in names if n.endswith(".csv")), None)
                if target is None:
                    continue
                logger.info("Reading TRI file: %s", target)
                with zf.open(target) as f:
                    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
            return df
        except Exception as exc:
            logger.debug("Bulk download failed for version %s: %s", version, exc)
            continue

    # Try fallback URL
    for yr_str in [str(year), str(year - 1)]:
        url = TRI_BULK_FALLBACK.format(year=yr_str)
        logger.info("Trying TRI fallback URL: %s", url)
        try:
            resp = _fetch_with_backoff(url, stream=True, max_retries=1)
            content = resp.content
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                target = next((n for n in names if n.endswith(".csv")), None)
                if target:
                    with zf.open(target) as f:
                        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
                    return df
        except Exception as exc:
            logger.debug("Fallback download failed: %s", exc)

    return None


def _parse_tri_bulk(df: pd.DataFrame, year: int, chemicals: list[str] | None) -> pd.DataFrame:
    """Parse and clean a TRI bulk DataFrame."""
    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Try to identify key columns flexibly
    col_map = {}
    for col in df.columns:
        upper = col.upper()
        if "COUNTY" in upper and "FIPS" in upper:
            col_map["fips"] = col
        elif upper == "COUNTY":
            col_map["county"] = col
        elif upper in ("ST", "STATE"):
            col_map["state"] = col
        elif "CHEMICAL" in upper and "COMPOUND" not in upper:
            col_map["chemical"] = col
        elif "CAS" in upper:
            col_map["cas"] = col
        elif "TOTAL" in upper and "RELEASE" in upper:
            col_map["total"] = col
        elif "FUGITIVE" in upper and "AIR" in upper:
            col_map["air"] = col
        elif "STREAM" in upper or ("WATER" in upper and "RELEASE" in upper):
            col_map["water"] = col
        elif "LAND" in upper and "RELEASE" in upper:
            col_map["land"] = col

    def safe_col(df, key, default=""):
        col = col_map.get(key)
        if col and col in df.columns:
            return df[col]
        return pd.Series([default] * len(df))

    result = pd.DataFrame({
        "county_fips": safe_col(df, "fips").astype(str).str.zfill(5),
        "county_name": safe_col(df, "county"),
        "state": safe_col(df, "state"),
        "chemical": safe_col(df, "chemical"),
        "cas_number": safe_col(df, "cas"),
        "total_releases_lbs": pd.to_numeric(safe_col(df, "total"), errors="coerce").fillna(0),
        "air_releases_lbs": pd.to_numeric(safe_col(df, "air"), errors="coerce").fillna(0),
        "water_releases_lbs": pd.to_numeric(safe_col(df, "water"), errors="coerce").fillna(0),
        "land_releases_lbs": pd.to_numeric(safe_col(df, "land"), errors="coerce").fillna(0),
        "year": year,
    })

    if chemicals:
        chems_upper = [c.upper() for c in chemicals]
        result = result[result["chemical"].str.upper().isin(chems_upper)]

    # Group by county + chemical
    grouped = (
        result.groupby(["county_fips", "county_name", "state", "chemical", "cas_number", "year"], dropna=False)
        .agg(
            total_releases_lbs=("total_releases_lbs", "sum"),
            air_releases_lbs=("air_releases_lbs", "sum"),
            water_releases_lbs=("water_releases_lbs", "sum"),
            land_releases_lbs=("land_releases_lbs", "sum"),
        )
        .reset_index()
    )
    # Convert to kg
    grouped["total_releases_kg"] = grouped["total_releases_lbs"] * 0.453592
    return grouped


def _fetch_via_envirofacts(year: int, chemicals: list[str] | None) -> pd.DataFrame:
    """Fallback: fetch from Envirofacts state by state."""
    import string
    states = list("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
    all_rows = []
    for state in states:
        url = ENVIROFACTS_URL.format(state=state)
        logger.info("Fetching TRI Envirofacts for state %s", state)
        try:
            rows = _fetch_with_backoff(url, max_retries=2).json()
            all_rows.extend(rows)
        except Exception as exc:
            logger.warning("Envirofacts failed for %s: %s", state, exc)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    return df


def fetch_tri(
    year: int = 2019,
    chemicals: list[str] | None = None,
    cache_dir: str = "data/cache/epa_tri",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch EPA Toxics Release Inventory data aggregated to county level.

    Parameters
    ----------
    year : reporting year.
    chemicals : optional list of chemical names to filter.
    cache_dir : directory for cached responses.
    force_refresh : if True, ignore cache.

    Returns
    -------
    pd.DataFrame with columns: county_fips, state, facility_name, chemical,
                                total_releases_kg, year
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    chem_key = "_".join(sorted(chemicals)) if chemicals else "all"
    cache_file = cache_path / f"tri_{year}_{chem_key}.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading EPA TRI from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info("Fetching EPA TRI data for year=%d", year)

    raw_df = _try_bulk_download(year, cache_path)
    if raw_df is not None:
        df = _parse_tri_bulk(raw_df, year, chemicals)
    else:
        logger.warning("Bulk download failed; trying Envirofacts API (slow)")
        df = _fetch_via_envirofacts(year, chemicals)

    if df.empty:
        logger.warning("No EPA TRI records returned")
    else:
        logger.info("Fetched %d EPA TRI county-chemical records", len(df))
        df.to_parquet(cache_file, index=False)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_tri(year=2019, chemicals=["LEAD", "MERCURY"])
    print(df.head())
