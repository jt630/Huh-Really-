"""
EPA SDWIS - Drinking water violations by county.

API: Envirofacts - https://data.epa.gov/efservice/VIOLATION/

Fields: county_fips, county_name, state, contaminant_code,
        contaminant_name, violation_category, violation_count,
        health_based_violations, year
"""
from pydantic import BaseModel


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
