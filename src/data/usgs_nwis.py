"""
USGS NWIS - Water quality measurements by site.

API: https://waterservices.usgs.gov/rest/
Portal: https://www.waterqualitydata.us/

Fields: county_fips, site_number, site_name, latitude, longitude,
        parameter_code, parameter_name, mean_value, unit,
        observation_count, start_date, end_date
"""
from pydantic import BaseModel


class NWISRecord(BaseModel):
    county_fips: str
    site_number: str
    site_name: str
    latitude: float | None
    longitude: float | None
    parameter_code: str
    parameter_name: str
    mean_value: float | None
    unit: str
    observation_count: int
    start_date: str
    end_date: str
