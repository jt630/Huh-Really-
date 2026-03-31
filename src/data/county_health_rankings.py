"""
County Health Rankings - Annual health outcomes by county.

Download: https://www.countyhealthrankings.org/ (national CSV)

Fields: county_fips, county_name, state, measure, value,
        confidence_interval_low, confidence_interval_high, year
"""
from pydantic import BaseModel


class HealthRankingRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    measure: str
    value: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    year: int
