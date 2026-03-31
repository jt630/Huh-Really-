"""
EPA Pesticide Registration - Compound metadata from EPA OPP.

Website: https://www.epa.gov/pesticides
Note: For county-level quantities use src/data/usgs_pesticide_use.py

Fields: active_ingredient, cas_number, registration_number,
        use_pattern, registration_status, year_registered, year_cancelled
"""
from pydantic import BaseModel


class PesticideRecord(BaseModel):
    active_ingredient: str
    cas_number: str | None
    registration_number: str
    use_pattern: str | None
    registration_status: str | None
    year_registered: int | None
    year_cancelled: int | None
