"""
Census ACS 5-year estimates - Demographics and socioeconomics by county.

API: https://api.census.gov/data/<YEAR>/acs/acs5
Key variables: B01002_001E (median age), B19013_001E (median income),
               B01003_001E (population), B17001_002E (poverty)
Credentials: CENSUS_API_KEY in .env

Fields: county_fips, county_name, state, variable_name,
        variable_code, value, year
"""
from pydantic import BaseModel


class ACSRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    variable_name: str
    variable_code: str
    value: float | None
    year: int
