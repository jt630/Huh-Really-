"""
OpenStreetMap via Overpass API - Geographic feature counts by county.

Endpoint: https://overpass-api.de/api/interpreter
Use case: count golf courses [leisure=golf_course], industrial sites, etc.
Rate limit: ~1 req/2s; batch by state bounding box.

Fields: county_fips, county_name, state, feature_type,
        count, density_per_sq_km
"""
from pydantic import BaseModel


class OSMFeatureRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    feature_type: str
    count: int
    density_per_sq_km: float
