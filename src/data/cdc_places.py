"""
CDC PLACES - Local chronic disease estimates via Socrata.

Endpoint: https://data.cdc.gov/resource/cwsq-ngmh.json

Fields: county_fips, county_name, state_abbr, measure, measure_id,
        data_value (age-adj prevalence %), confidence limits, year
"""
from pydantic import BaseModel


class PlacesRecord(BaseModel):
    county_fips: str
    county_name: str
    state_abbr: str
    measure: str
    measure_id: str
    data_value: float | None
    low_confidence_limit: float | None
    high_confidence_limit: float | None
    year: int
