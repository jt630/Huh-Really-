"""
Census County Business Patterns - Establishment counts by industry and county.

API: https://api.census.gov/data/<YEAR>/cbp
Credentials: CENSUS_API_KEY in .env (optional)

Fields: county_fips, county_name, state, naics_code, naics_description,
        establishment_count, employment, annual_payroll_1k, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class CBPRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    naics_code: str
    naics_description: str
    establishment_count: int
    employment: int | None
    annual_payroll_1k: int | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_cbp(year: int, api_key: str | None) -> list[list]:
    url = f"https://api.census.gov/data/{year}/cbp"
    params: dict = {
        "get": "GEO_ID,NAICS2017,NAICS2017_LABEL,ESTAB,EMP,PAYANN",
        "for": "county:*",
        "in": "state:*",
    }
    if api_key:
        params["key"] = api_key
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_cbp(
    year: int = 2020,
    naics_codes: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch Census County Business Patterns by county.

    Returns DataFrame indexed by county_fips with columns:
        naics_code, naics_description, establishment_count, employment, year
    """
    from src.config import get_settings
    settings = get_settings()
    cache_file = settings.cache_dir / f"census_cbp_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("Census CBP: loading from cache")
        df = pd.read_parquet(cache_file)
        if naics_codes:
            df = df[df["naics_code"].isin(naics_codes)]
        return df

    logger.info("Census CBP: fetching year=%d", year)
    try:
        data = _fetch_cbp(year, settings.census_api_key)
    except Exception as exc:
        logger.error("Census CBP: fetch failed: %s", exc)
        return pd.DataFrame()

    headers = data[0]
    rows = data[1:]
    records = []
    for row in rows:
        d = dict(zip(headers, row))
        state_code = d.get("state", "").zfill(2)
        county_code = d.get("county", "").zfill(3)
        fips = state_code + county_code
        try:
            records.append(CBPRecord(
                county_fips=fips,
                county_name=d.get("GEO_ID", ""),
                state=state_code,
                naics_code=str(d.get("NAICS2017", "")),
                naics_description=str(d.get("NAICS2017_LABEL", "")),
                establishment_count=int(d.get("ESTAB", 0) or 0),
                employment=_si(d.get("EMP")),
                annual_payroll_1k=_si(d.get("PAYANN")),
                year=year,
            ))
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

    if naics_codes:
        df = df[df["naics_code"].isin(naics_codes)]
    return df


def _si(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_cbp(year=2020, naics_codes=["1110", "3250"])
    print(df.head(10))
