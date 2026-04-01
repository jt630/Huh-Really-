"""
EPA Toxics Release Inventory - Facility toxic releases aggregated to county.

API: Envirofacts REST - https://data.epa.gov/efservice/tri_facility/
Bulk CSV: https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present

Fields: county_fips, county_name, state, chemical, cas_number,
        total_releases_lbs, air/water/land_releases_lbs, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://data.epa.gov/efservice/TRI_FACILITY/PSTAT/{state}/JSON"
PAGE_SIZE = 10_000

STATE_CODES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
]


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


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_state(state: str) -> list[dict]:
    url = BASE_URL.format(state=state)
    resp = requests.get(url, params={"count": PAGE_SIZE}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_tri(
    year: int = 2020,
    states: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch EPA TRI facility data aggregated to county.

    Returns DataFrame indexed by county_fips with columns:
        county_name, state, chemical, cas_number, total_releases_lbs, year
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"epa_tri_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("EPA TRI: loading from cache")
        return pd.read_parquet(cache_file)

    if states is None:
        states = STATE_CODES

    records = []
    for state in states:
        try:
            data = _fetch_state(state)
            for row in data:
                try:
                    fips = (str(row.get("ST_FIPS", "")).zfill(2)
                            + str(row.get("COUNTY_FIPS", "")).zfill(3))
                    if len(fips) != 5:
                        continue
                    records.append(TRIRecord(
                        county_fips=fips,
                        county_name=str(row.get("COUNTY", "")).strip(),
                        state=state,
                        chemical=str(row.get("CHEMICAL_NAME_TEXT", "")).strip(),
                        cas_number=str(row.get("CAS_NUMBER", "")).strip(),
                        total_releases_lbs=_safe_float(row.get("TOTAL_RELEASES")) or 0.0,
                        air_releases_lbs=_safe_float(row.get("AIR_RELEASES")) or 0.0,
                        water_releases_lbs=_safe_float(row.get("WATER_RELEASES")) or 0.0,
                        land_releases_lbs=_safe_float(row.get("LAND_RELEASES")) or 0.0,
                        year=year,
                    ))
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("EPA TRI: failed for state %s: %s", state, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records])
    df = df.groupby(["county_fips", "county_name", "state", "chemical",
                     "cas_number", "year"], as_index=False).sum(numeric_only=True)
    df = df.set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_tri(year=2020, states=["CA"])
    print(df.head(10))
    print(f"Shape: {df.shape}")
