"""
DuckDB interface — query the pre-built correlations database.

Usage:
    from src.db import get_db, query

    df = query("SELECT * FROM usgs_pesticide_use WHERE compound = 'PARAQUAT DICHLORIDE'")
    df = query("SELECT * FROM census_acs WHERE variable_name = 'median_age'")

The database is built by scripts/etl.py. Falls back gracefully if not present.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/correlations.duckdb")


def get_db_path() -> Path:
    try:
        from src.config import get_settings
        return Path(get_settings().cache_dir).parent / "correlations.duckdb"
    except Exception:
        return _DB_PATH


def available() -> bool:
    """Return True if the DuckDB file exists and duckdb is installed."""
    try:
        import duckdb  # noqa: F401
        return get_db_path().exists()
    except ImportError:
        return False


def tables() -> list[str]:
    """List tables currently in the database."""
    if not available():
        return []
    import duckdb
    con = duckdb.connect(str(get_db_path()), read_only=True)
    result = con.execute("SHOW TABLES").fetchall()
    con.close()
    return [r[0] for r in result]


def query(sql: str) -> pd.DataFrame:
    """Run a SQL query against the correlations database and return a DataFrame."""
    import duckdb
    con = duckdb.connect(str(get_db_path()), read_only=True)
    try:
        df = con.execute(sql).df()
    finally:
        con.close()
    return df


def load_source(source: str, **filters) -> pd.DataFrame | None:
    """
    Load a source table from DuckDB, optionally filtered.

    Examples
    --------
    load_source("usgs_pesticide_use", compound="PARAQUAT DICHLORIDE", year=2018)
    load_source("census_acs", variable_name="median_age")

    Returns None if the source table doesn't exist in the database.
    """
    if not available():
        return None
    if source not in tables():
        logger.debug("db.load_source: table '%s' not in database", source)
        return None

    import duckdb
    where_clauses = [f"{col} = ?" for col in filters]
    where_vals = list(filters.values())
    sql = f"SELECT * FROM {source}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    con = duckdb.connect(str(get_db_path()), read_only=True)
    try:
        df = con.execute(sql, where_vals).df()
    finally:
        con.close()

    if "county_fips" in df.columns:
        df = df.set_index("county_fips")
    return df
