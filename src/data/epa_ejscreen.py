"""
EPA EJScreen - Environmental justice indicators by census block group.

Download: https://www.epa.gov/ejscreen/download-ejscreen-data

Fields: county_fips, census_block_group, percentile_pm25, percentile_ozone,
        percentile_diesel_pm, percentile_cancer_risk, percentile_resp_hazard,
        percentile_traffic, percentile_npdes, percentile_rmp,
        percentile_superfund, percentile_hazwaste, percentile_ust,
        ej_index_pm25, year
"""
from pydantic import BaseModel


class EJScreenRecord(BaseModel):
    county_fips: str
    census_block_group: str
    percentile_pm25: float | None
    percentile_ozone: float | None
    percentile_diesel_pm: float | None
    percentile_cancer_risk: float | None
    percentile_resp_hazard: float | None
    percentile_traffic: float | None
    percentile_npdes: float | None
    percentile_rmp: float | None
    percentile_superfund: float | None
    percentile_hazwaste: float | None
    percentile_ust: float | None
    ej_index_pm25: float | None
    year: int
