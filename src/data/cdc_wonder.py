"""
CDC Wonder - County-level mortality data (ICD-10).

Website: https://wonder.cdc.gov/
Dataset: Underlying Cause of Death, POST XML API (dataset code D76)
Fallback: pre-downloaded .txt files in data/cache/cdc_wonder/

Fields: county_fips, county_name, state, deaths, population,
        age_adjusted_rate (per 100k), year
"""
from pydantic import BaseModel


class MortalityRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    deaths: int | None
    population: int | None
    age_adjusted_rate: float | None
    year: int
