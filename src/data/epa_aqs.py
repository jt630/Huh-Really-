"""
EPA Air Quality System - Annual ambient air monitoring data by county.

API: https://aqs.epa.gov/data/api/ (requires free registration)
Endpoint: GET /annualData/byCounty
Credentials: EPA_AQS_EMAIL + EPA_AQS_KEY in .env

Fields: county_fips, county_name, state, parameter, parameter_code,
        arithmetic_mean, median, first_max_value, aqi, unit, year
"""
from pydantic import BaseModel


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
