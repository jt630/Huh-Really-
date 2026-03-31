"""
EPA Toxics Release Inventory - Facility toxic releases aggregated to county.

API: Envirofacts REST - https://data.epa.gov/efservice/tri_facility/
Bulk CSV: https://www.epa.gov/toxics-release-inventory-tri-program/tri-basic-data-files-calendar-years-1987-present

Fields: county_fips, county_name, state, chemical, cas_number,
        total_releases_lbs, air/water/land_releases_lbs, year
"""
from pydantic import BaseModel


class TRIRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    chemical: str
    cas_number: str
    total_releases_lbs: float
    air_releases_lbs: float
    water_releases_lbs: float
    land_releases_lbs: float
    year: int
