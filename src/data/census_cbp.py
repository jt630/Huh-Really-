"""
Census County Business Patterns - Establishment counts by industry and county.

API: https://api.census.gov/data/<YEAR>/cbp
Credentials: CENSUS_API_KEY in .env

Fields: county_fips, county_name, state, naics_code, naics_description,
        establishment_count, employment, annual_payroll_1k, year
"""
from pydantic import BaseModel


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
