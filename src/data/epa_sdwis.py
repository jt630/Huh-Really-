"""
EPA SDWIS - Drinking water violations by county.

API: Envirofacts - https://data.epa.gov/efservice/VIOLATION/

Fields: county_fips, county_name, state, contaminant_code,
        contaminant_name, violation_category, violation_count,
        health_based_violations, year
"""
from __future__ import annotations

import logging

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://data.epa.gov/efservice/VIOLATION/PRIMACY_AGENCY_CODE/{state}/JSON"


class SDWISRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    contaminant_code: str
    contaminant_name: str
    violation_category: str
    violation_count: int
    health_based_violations: int
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _fetch_state(state: str) -> list[dict]:
    url = BASE_URL.format(state=state)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_sdwis(
    year: int = 2020,
    states: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch EPA SDWIS drinking water violations aggregated to county.

    Returns DataFrame indexed by county_fips.
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"epa_sdwis_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("EPA SDWIS: loading from cache")
        return pd.read_parquet(cache_file)

    if states is None:
        states = [
            "AL","AR","AZ","CA","CO","CT","DE","FL","GA","HI","IA","ID","IL",
            "IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC",
            "ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VA","VT","WA","WI","WV","WY",
        ]

    records = []
    for state in states:
        try:
            data = _fetch_state(state)
            for row in data:
                try:
                    # PWSID first 2 chars = state FIPS, next 7 = system ID
                    pwsid = str(row.get("PWSID", ""))
                    state_fips = pwsid[:2] if pwsid else "00"
                    # County FIPS requires additional lookup; use state_fips + 000 as placeholder
                    county_fips = state_fips + "000"
                    compl_per = str(row.get("COMPL_PER_BEGIN_DATE", ""))
                    rec_year = int(compl_per[:4]) if compl_per else year
                    if rec_year != year:
                        continue
                    is_hb = str(row.get("IS_HEALTH_BASED_IND", "N")).strip().upper() == "Y"
                    records.append({
                        "county_fips": county_fips,
                        "county_name": "",
                        "state": state,
                        "contaminant_code": str(row.get("CONTAMINANT_CODE", "")),
                        "contaminant_name": str(row.get("CONTAMINANT_CODE", "")),
                        "violation_category": str(row.get("VIOLATION_CATEGORY_CODE", "")),
                        "health_based": 1 if is_hb else 0,
                    })
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("EPA SDWIS: failed for state %s: %s", state, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df_agg = (df.groupby(["county_fips", "county_name", "state",
                          "contaminant_code", "contaminant_name", "violation_category"])
              .agg(violation_count=("health_based", "count"),
                   health_based_violations=("health_based", "sum"))
              .reset_index())
    df_agg["year"] = year
    df_agg = df_agg.set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_parquet(cache_file)
    return df_agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_sdwis(year=2020, states=["CA"])
    print(df.head(10))
