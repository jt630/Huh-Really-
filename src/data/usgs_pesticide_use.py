"""
USGS Pesticide National Synthesis Project - County-level pesticide use estimates.

Download: https://water.usgs.gov/nawqa/pnsp/usage/maps/county-level/
Files: EPest.county.<YEAR>.txt (tab-delimited)

Compounds of interest: PARAQUAT DICHLORIDE, ROTENONE, CHLORPYRIFOS, MANEB

Fields: county_fips, county_name, state, compound, kg_applied, year
"""
from pydantic import BaseModel


class PesticideUseRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    compound: str
    kg_applied: float
    year: int
