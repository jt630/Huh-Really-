"""
CMS Chronic Conditions - Medicare beneficiary prevalence by county.

API: data.cms.gov Socrata endpoint
Filter: Bene_Geo_Lvl = "County"

Fields: county_fips, county_name, state, condition,
        prevalence_pct, beneficiary_count, year
"""
from pydantic import BaseModel


class ChronicConditionRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    condition: str
    prevalence_pct: float | None
    beneficiary_count: int | None
    year: int
