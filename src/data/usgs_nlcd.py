"""
USGS NLCD - National Land Cover Database (30m raster, zonal stats to county).

Download: https://www.mrlc.gov/data
Vintages: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021

Fields: county_fips, county_name, state, land_cover_class,
        land_cover_code, area_sq_km, area_pct, year
"""
from pydantic import BaseModel


class NLCDRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    land_cover_class: str
    land_cover_code: int
    area_sq_km: float
    area_pct: float
    year: int
