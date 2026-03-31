"""
EPA Superfund - NPL hazardous waste sites by county.

API: Envirofacts SEMS - https://data.epa.gov/efservice/SEMS_ACTIVE_SITES/
GeoJSON: https://catalog.data.gov/dataset/superfund-national-priorities-list-npl-sites

Fields: county_fips, county_name, state, site_count, sites[]
        (site_name, site_id, lat, lon, npl_status, contaminants)
"""
from pydantic import BaseModel


class SuperfundSite(BaseModel):
    site_name: str
    site_id: str
    latitude: float | None
    longitude: float | None
    npl_status: str | None
    contaminants: list[str] | None = None


class SuperfundCountyRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    site_count: int
    sites: list[SuperfundSite] = []
