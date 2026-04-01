"""
County Health Rankings - Annual health outcomes by county.

Download: https://www.countyhealthrankings.org/ (national CSV)
URL pattern: https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{year}.csv

Key measures:
    v001_rawvalue -> premature_death_rate
    v002_rawvalue -> poor_health_pct
    v004_rawvalue -> uninsured_pct
    v005_rawvalue -> primary_care_physicians_ratio

Fields: county_fips, county_name, state, measure, value,
        confidence_interval_low, confidence_interval_high, year
"""
from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

MEASURES = {
    "v001_rawvalue": "premature_death_rate",
    "v002_rawvalue": "poor_health_pct",
    "v004_rawvalue": "uninsured_pct",
    "v005_rawvalue": "primary_care_physicians_ratio",
    "v009_rawvalue": "adult_smoking_pct",
    "v011_rawvalue": "adult_obesity_pct",
}

URL_PATTERN = (
    "https://www.countyhealthrankings.org/sites/default/files/"
    "media/document/analytic_data{year}.csv"
)


class HealthRankingRecord(BaseModel):
    county_fips: str
    county_name: str
    state: str
    measure: str
    value: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    year: int


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _download(year: int) -> str:
    url = URL_PATTERN.format(year=year)
    logger.info("County Health Rankings: downloading year=%d", year)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.text


def fetch_health_rankings(
    year: int = 2023,
    measures: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch County Health Rankings data.

    Returns long-format DataFrame indexed by county_fips with columns:
        county_name, state, measure, value, year
    """
    from src.config import get_settings
    cache_file = get_settings().cache_dir / f"county_health_rankings_{year}.parquet"

    if use_cache and cache_file.exists():
        logger.info("County Health Rankings: loading from cache")
        df = pd.read_parquet(cache_file)
        if measures:
            df = df[df["measure"].isin(measures)]
        return df

    try:
        csv_text = _download(year)
    except Exception as exc:
        logger.error("County Health Rankings: download failed: %s", exc)
        return pd.DataFrame()

    try:
        # CHR CSVs have a 2-row header; row 0 = column names, row 1 = descriptions
        df_raw = pd.read_csv(StringIO(csv_text), dtype=str, skiprows=1)
        df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
    except Exception as exc:
        logger.error("County Health Rankings: parse error: %s", exc)
        return pd.DataFrame()

    fips_col = "fipscode" if "fipscode" in df_raw.columns else "fips"
    name_col = "county" if "county" in df_raw.columns else "name"
    state_col = "state" if "state" in df_raw.columns else "statecode"

    measure_keys = list(MEASURES.keys()) if not measures else [
        k for k, v in MEASURES.items() if v in measures
    ]

    records = []
    for _, row in df_raw.iterrows():
        fips = str(row.get(fips_col, "")).strip().zfill(5)
        if len(fips) != 5 or not fips.isdigit():
            continue

        for col_key in measure_keys:
            if col_key not in row.index:
                continue
            val = _safe_float(str(row[col_key]))
            # Try CI columns (pattern: v001_cilow, v001_cihigh)
            base = col_key.replace("_rawvalue", "")
            ci_low = _safe_float(str(row.get(f"{base}_cilow", "")))
            ci_high = _safe_float(str(row.get(f"{base}_cihigh", "")))

            records.append(HealthRankingRecord(
                county_fips=fips,
                county_name=str(row.get(name_col, "")).strip(),
                state=str(row.get(state_col, "")).strip(),
                measure=MEASURES[col_key],
                value=val,
                confidence_interval_low=ci_low,
                confidence_interval_high=ci_high,
                year=year,
            ))

    df = pd.DataFrame([r.model_dump() for r in records]).set_index("county_fips")

    if use_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

    return df


def _safe_float(v: str) -> float | None:
    try:
        return float(v.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = fetch_health_rankings(year=2023)
    print(df.head(10))
    print(f"Shape: {df.shape}")
