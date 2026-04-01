"""
EPA Air Quality System - Annual ambient air monitoring data by county.

API: https://aqs.epa.gov/data/api/ (requires free registration)
Endpoint: GET /annualData/byCounty
Credentials: EPA_AQS_EMAIL + EPA_AQS_KEY in .env

Fields: county_fips, county_name, state, parameter, parameter_code,
        arithmetic_mean, median, first_max_value, aqi, unit, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

AQS_BASE = "https://aqs.epa.gov/data/api"

DEFAULT_PARAMS = {
    "88101": "PM2.5",
    "44201": "Ozone",
    "42602": "NO2",
    "42101": "CO",
}

STATE_CODES = [
    "01","02","04","05","06","08","09","10","12","13","15","16","17","18","19",
    "20","21","22","23","24","25","26","27","28","29","30","31","32","33","34",
    "35","36","37","38","39","40","41","42","44","45","46","47","48","49","50",
    "51","53","54","55","56",
]


class AQSRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    parameter: str
    parameter_code: str
    arithmetic_mean: float | None
    median: float | None
    first_max_value: float | None
    aqi: int | None
    unit: str
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_annual(email: str, key: str, param: str, year: int, state: str) -> list[dict]:
    url = f"{AQS_BASE}/annualData/byCounty"
    params = {
        "email": email,
        "key": key,
        "param": param,
        "bdate": f"{year}0101",
        "edate": f"{year}1231",
        "state": state,
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("Data", [])


def fetch_aqs(
    year: int = 2020,
    parameter_codes: list[str] | None = None,
    states: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch EPA AQS annual air quality data by county.
    Requires EPA_AQS_EMAIL and EPA_AQS_KEY in .env.

    Returns DataFrame indexed by county_fips.
    """
    from src.config import get_settings
    settings = get_settings()

    if not settings.epa_aqs_email or not settings.epa_aqs_key:
        logger.warning("EPA AQS: EPA_AQS_EMAIL and EPA_AQS_KEY not set; skipping")
        return pd.DataFrame()

    cache_file = settings.cache_dir / f"epa_aqs_{year}.parquet"
    if use_cache and cache_file.exists():
        logger.info("EPA AQS: loading from cache")
        return pd.read_parquet(cache_file)

    if parameter_codes is None:
        parameter_codes = list(DEFAULT_PARAMS.keys())
    if states is None:
        states = STATE_CODES[:10]  # limit default to avoid rate limits

    records = []
    for state in states:
        for param in parameter_codes:
            try:
                rows = _fetch_annual(settings.epa_aqs_email, settings.epa_aqs_key,
                                     param, year, state)
                for row in rows:
                    fips = (str(row.get("state_code", "")).zfill(2)
                            + str(row.get("county_code", "")).zfill(3))
                    records.append(AQSRecord(
                        county_fips=fips,
                        county_name=str(row.get("county", "")).strip(),
                        state=state,
                        parameter=str(row.get("parameter", DEFAULT_PARAMS.get(param, param))),
                        parameter_code=param,
                        arithmetic_mean=_sf(row.get("arithmetic_mean")),
                        median=_sf(row.get("fifty_percentile")),
                        first_max_value=_sf(row.get("first_max_value")),
                        aqi=_si(row.get("aqi")),
                        unit=str(row.get("units_of_measure", "")),
                        year=year,
                    ))
            except Exception as exc:
                logger.warning("EPA AQS: failed state=%s param=%s: %s", state, param, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")
    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
    return df


def _sf(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _si(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_aqs(year=2020, states=["06"])
    print(df.head(10))
