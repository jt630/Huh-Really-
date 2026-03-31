"""
CDC Social Vulnerability Index - Social vulnerability percentile ranks by county.

Download: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html

Fields: county_fips, county_name, state, theme, percentile_rank, year
Themes: RPL_THEME1-4 (socioeconomic, household, minority, housing), RPL_THEMES (overall)
"""
from pydantic import BaseModel


class SVIRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    theme: str
    percentile_rank: float | None
    year: int
