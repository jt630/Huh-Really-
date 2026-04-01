"""
Census ACS 5-year estimates - Demographics and socioeconomics by county.

API: https://api.census.gov/data/<YEAR>/acs/acs5
Key variables: B01002_001E (median age), B19013_001E (median income),
               B01003_001E (population), B17001_002E (poverty)
Credentials: CENSUS_API_KEY in .env (optional but recommended)

Fields: county_fips, county_name, state, variable_name, variable_code, value, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ACS_VARIABLES = {
    "B01002_001E": "median_age",
    "B19013_001E": "median_income",
    "B01003_001E": "total_population",
    "B17001_002E": "poverty_population",
    "B23025_005E": "unemployed_civilian",
}


class ACSRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    variable_name: str
    variable_code: str
    value: float | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_acs(year: int, variables: list[str], api_key: str | None) -> list[list]:
    var_str = "NAME," + ",".join(variables)
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params: dict = {"get": var_str, "for": "county:*", "in": "state:*"}
    if api_key:
        params["key"] = api_key
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_acs(
    year: int = 2020,
    variable_codes: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch Census ACS 5-year estimates by county.

    Returns long-format DataFrame indexed by county_fips with columns:
        county_name, state, variable_name, variable_code, value, year
    """
    from src.config import get_settings
    settings = get_settings()
    if variable_codes is None:
        variable_codes = list(ACS_VARIABLES.keys())

    cache_file = settings.cache_dir / f"census_acs_{year}.parquet"
    if use_cache and cache_file.exists():
        logger.info("Census ACS: loading from cache %s", cache_file)
        df = pd.read_parquet(cache_file)
        if variable_codes:
            df = df[df["variable_code"].isin(variable_codes)]
        return df

    logger.info("Census ACS: fetching year=%d", year)
    try:
        data = _fetch_acs(year, variable_codes, settings.census_api_key)
    except Exception as exc:
        logger.error("Census ACS: fetch failed: %s", exc)
        return pd.DataFrame()

    headers = data[0]
    rows = data[1:]

    records = []
    for row in rows:
        row_dict = dict(zip(headers, row))
        state_code = row_dict.get("state", "").zfill(2)
        county_code = row_dict.get("county", "").zfill(3)
        fips = state_code + county_code
        county_name = row_dict.get("NAME", "")

        for code in variable_codes:
            if code not in row_dict:
                continue
            val_str = row_dict[code]
            try:
                val = float(val_str) if val_str not in (None, "", "-1", "-666666666") else None
            except (ValueError, TypeError):
                val = None

            records.append(ACSRecord(
                county_fips=fips,
                county_name=county_name,
                state=state_code,
                variable_name=ACS_VARIABLES.get(code, code),
                variable_code=code,
                value=val,
                year=year,
            ))

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_acs(year=2020)
    print(df.head(10))
    print(f"Shape: {df.shape}")
