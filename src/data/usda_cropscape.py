"""
USDA CropScape / Cropland Data Layer - Crop acreage by county.

API: https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat
Params: fips=<5digit>&year=<YYYY>

Fields: county_fips, county_name, state, crop_name, crop_code,
        area_acres, area_pct, year
"""
from pydantic import BaseModel


class CropScapeRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    crop_name: str
    crop_code: int
    area_acres: float
    area_pct: float
    year: int
