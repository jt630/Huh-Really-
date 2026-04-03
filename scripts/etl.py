"""
ETL runner — pre-fetches all data sources into parquet cache and DuckDB.

Usage:
    python scripts/etl.py                          # full refresh
    python scripts/etl.py --preset demo            # Parkinson's demo sources only
    python scripts/etl.py --sources usgs_pesticide_use census_acs
    python scripts/etl.py --status                 # show cache age/size, no fetch
    python scripts/etl.py --force                  # re-fetch even if cached
    python scripts/etl.py --stale-after 30         # only refresh if older than 30 days
    python scripts/etl.py --compounds "PARAQUAT DICHLORIDE" ROTENONE
    python scripts/etl.py --measures CANCER STROKE DIABETES
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("etl")

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, list[str]] = {
    "demo": [
        "usgs_pesticide_use",
        "census_acs",
        "osm_features",
        "county_health_rankings",
        "cdc_svi",
    ],
    "health": [
        "cdc_places",
        "cms_chronic",
        "county_health_rankings",
        "cdc_svi",
    ],
    "environment": [
        "epa_tri",
        "epa_superfund",
        "epa_ejscreen",
        "epa_sdwis",
        "usgs_pesticide_use",
        "usgs_nwis",
    ],
    "all": [
        "cdc_places",
        "cdc_svi",
        "census_acs",
        "census_cbp",
        "cms_chronic",
        "county_health_rankings",
        "epa_ejscreen",
        "epa_sdwis",
        "epa_superfund",
        "epa_tri",
        "osm_features",
        "usda_cropscape",
        "usgs_nwis",
        "usgs_pesticide_use",
        # epa_aqs added automatically if keys present
        # openfda_faers included but state-level only
        "openfda_faers",
    ],
}

# Sources that are manual-only or stub-only — skip with a notice
SKIP_SOURCES = {
    "cdc_wonder": "manual download required — see https://wonder.cdc.gov/ucd-icd10.html",
    "pubmed": "literature search — not a county dataset",
    "semantic_scholar": "literature search — not a county dataset",
    "epa_pesticides": "stub only — use usgs_pesticide_use for county-level data",
    "usgs_nlcd": "stub only — raster processing not implemented",
}

# ---------------------------------------------------------------------------
# Per-source fetch config
# ---------------------------------------------------------------------------

def _build_fetch_kwargs(source: str, args: argparse.Namespace) -> dict:
    """Return kwargs to pass to registry.load() for this source."""
    kwargs: dict = {}
    if source == "usgs_pesticide_use" and args.compounds:
        kwargs["compounds"] = args.compounds
    if source == "cdc_places" and args.measures:
        kwargs["measure_ids"] = args.measures
    return kwargs


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def _cache_files(cache_dir: Path) -> dict[str, Path]:
    """Return {source_name: parquet_path} for cached files."""
    found: dict[str, Path] = {}
    for p in sorted(cache_dir.glob("*.parquet")):
        # strip year suffix — e.g. usgs_pesticide_use_2020.parquet -> usgs_pesticide_use
        name = p.stem.rsplit("_", 1)[0] if p.stem[-4:].isdigit() else p.stem
        found.setdefault(name, p)  # keep first (most recent by glob order)
    return found


def show_status(sources: list[str], cache_dir: Path, db_path: Path) -> None:
    cached = _cache_files(cache_dir)
    now = datetime.now(timezone.utc).timestamp()

    print(f"\n{'Source':<28} {'Cached':<8} {'Age':<12} {'Size'}")
    print("─" * 62)
    for src in sources:
        if src in SKIP_SOURCES:
            print(f"{src:<28} {'⚠ skip':<8} {'—':<12} {SKIP_SOURCES[src]}")
            continue
        p = cached.get(src)
        if p and p.exists():
            age_days = (now - p.stat().st_mtime) / 86400
            age_str = f"{age_days:.0f}d ago" if age_days >= 1 else f"{age_days*24:.0f}h ago"
            size_mb = p.stat().st_size / 1_048_576
            size_str = f"{size_mb:.1f} MB" if size_mb >= 0.1 else f"{p.stat().st_size / 1024:.0f} KB"
            print(f"{src:<28} {'✓':<8} {age_str:<12} {size_str}")
        else:
            print(f"{src:<28} {'✗':<8} {'—':<12} —")

    print()
    if db_path.exists():
        size_mb = db_path.stat().st_size / 1_048_576
        print(f"DuckDB: {db_path}  ({size_mb:.1f} MB)")
    else:
        print(f"DuckDB: {db_path}  (not built)")
    print()


# ---------------------------------------------------------------------------
# DuckDB loader
# ---------------------------------------------------------------------------

def load_into_duckdb(db_path: Path, source: str, df) -> None:
    try:
        import duckdb
        import pandas as pd
        con = duckdb.connect(str(db_path))
        # Reset index so county_fips becomes a column
        if df.index.name:
            df = df.reset_index()
        con.execute(f"DROP TABLE IF EXISTS {source}")
        con.execute(f"CREATE TABLE {source} AS SELECT * FROM df")
        row_count = con.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
        con.close()
        logger.info("DuckDB: %s loaded (%d rows)", source, row_count)
    except ImportError:
        logger.warning("duckdb not installed — skipping DB load for %s", source)
    except Exception as exc:
        logger.warning("DuckDB load failed for %s: %s", source, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fetch all data sources into parquet + DuckDB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", choices=list(PRESETS), default="all",
                       help="Named source preset (default: all)")
    group.add_argument("--sources", nargs="+", metavar="SOURCE",
                       help="Explicit list of sources to fetch")
    parser.add_argument("--status", action="store_true",
                        help="Show cache status without fetching")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if parquet cache is fresh")
    parser.add_argument("--stale-after", type=int, default=None, metavar="DAYS",
                        help="Only re-fetch sources older than N days")
    parser.add_argument("--compounds", nargs="+", metavar="COMPOUND",
                        help="Filter usgs_pesticide_use to these compounds")
    parser.add_argument("--measures", nargs="+", metavar="MEASURE_ID",
                        help="Filter cdc_places to these measure IDs")
    parser.add_argument("--db", default="data/correlations.duckdb", metavar="PATH",
                        help="DuckDB output path (default: data/correlations.duckdb)")
    args = parser.parse_args()

    # Load settings
    try:
        from src.config import get_settings
        settings = get_settings()
    except Exception as exc:
        print(f"Config error: {exc}\nMake sure ANTHROPIC_API_KEY is set in .env")
        sys.exit(1)

    cache_dir = settings.cache_dir
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine source list
    if args.sources:
        sources = args.sources
    else:
        sources = PRESETS[args.preset]
        # Auto-add epa_aqs if keys present
        if args.preset == "all":
            if settings.epa_aqs_email and settings.epa_aqs_key:
                sources = sources + ["epa_aqs"]

    # Status-only mode
    if args.status:
        show_status(sources, cache_dir, db_path)
        return

    # Stale threshold
    stale_cutoff: float | None = None
    if args.stale_after is not None:
        stale_cutoff = time.time() - args.stale_after * 86400

    from src.data.registry import DataSourceRegistry
    reg = DataSourceRegistry()

    cached_files = _cache_files(cache_dir)
    results: list[tuple[str, str, str]] = []  # (source, status, detail)

    for source in sources:
        if source in SKIP_SOURCES:
            reason = SKIP_SOURCES[source]
            print(f"  skip  {source:<28}  {reason}")
            results.append((source, "skip", reason))
            continue

        # Check if fetch needed
        cached_path = cached_files.get(source)
        if not args.force and cached_path and cached_path.exists():
            if stale_cutoff is None or cached_path.stat().st_mtime >= stale_cutoff:
                size_mb = cached_path.stat().st_size / 1_048_576
                print(f"  cache {source:<28}  {size_mb:.1f} MB (skipping)")
                # Still load into DuckDB if not already there
                try:
                    import pandas as pd
                    df = pd.read_parquet(cached_path)
                    load_into_duckdb(db_path, source, df)
                except Exception:
                    pass
                results.append((source, "cached", f"{size_mb:.1f} MB"))
                continue

        print(f"  fetch {source:<28} ", end="", flush=True)
        t0 = time.time()
        try:
            kwargs = _build_fetch_kwargs(source, args)
            df = reg.load(source, use_cache=False, **kwargs)
            elapsed = time.time() - t0
            # Estimate size from memory (rough)
            size_mb = df.memory_usage(deep=True).sum() / 1_048_576
            print(f"{len(df):>8,} rows  {size_mb:.1f} MB  ({elapsed:.1f}s)")
            load_into_duckdb(db_path, source, df)
            results.append((source, "ok", f"{len(df):,} rows"))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {exc}")
            results.append((source, "error", str(exc)))

    # Summary
    ok = sum(1 for _, s, _ in results if s in ("ok", "cached"))
    errors = [(src, d) for src, s, d in results if s == "error"]
    print(f"\n{'─'*50}")
    print(f"  {ok}/{len(sources)} sources loaded into {db_path}")
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for src, detail in errors:
            print(f"    {src}: {detail}")

    if not errors:
        print("\n  CDC Wonder: manual download required for mortality outcomes")
        print("  See: https://wonder.cdc.gov/ucd-icd10.html")
        print("  Place .txt exports in data/cache/cdc_wonder/")
    print()


if __name__ == "__main__":
    main()
