"""
DataSourceRegistry - Auto-discovers and standardizes all data source adapters.

Usage:
    from src.data.registry import DataSourceRegistry
    registry = DataSourceRegistry()
    df = registry.load("cdc_places", year=2022, measure_ids=["CANCER"])
    print(registry.available_sources())
    print(registry.list_variables())
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

_SOURCE_CATALOG: dict[str, dict[str, Any]] = {
    "cdc_places": {
        "module": "src.data.cdc_places", "function": "fetch_places", "required_keys": [],
        "description": "CDC PLACES chronic disease estimates by county",
        "output_columns": ["measure", "measure_id", "data_value", "year"], "index": "county_fips",
    },
    "epa_tri": {
        "module": "src.data.epa_tri", "function": "fetch_tri", "required_keys": [],
        "description": "EPA Toxics Release Inventory by county",
        "output_columns": ["chemical", "cas_number", "total_releases_lbs", "year"], "index": "county_fips",
    },
    "epa_superfund": {
        "module": "src.data.epa_superfund", "function": "fetch_superfund", "required_keys": [],
        "description": "EPA Superfund NPL sites by county",
        "output_columns": ["site_count", "site_names"], "index": "county_fips",
    },
    "epa_ejscreen": {
        "module": "src.data.epa_ejscreen", "function": "fetch_ejscreen", "required_keys": [],
        "description": "EPA EJScreen environmental justice indicators by county",
        "output_columns": ["percentile_pm25", "percentile_cancer_risk", "ej_index_pm25", "year"], "index": "county_fips",
    },
    "epa_aqs": {
        "module": "src.data.epa_aqs", "function": "fetch_aqs", "required_keys": ["epa_aqs_email", "epa_aqs_key"],
        "description": "EPA Air Quality System annual data by county",
        "output_columns": ["parameter", "arithmetic_mean", "aqi", "unit", "year"], "index": "county_fips",
    },
    "epa_sdwis": {
        "module": "src.data.epa_sdwis", "function": "fetch_sdwis", "required_keys": [],
        "description": "EPA SDWIS drinking water violations by county",
        "output_columns": ["contaminant_name", "violation_category", "violation_count", "year"], "index": "county_fips",
    },
    "usgs_nwis": {
        "module": "src.data.usgs_nwis", "function": "fetch_nwis", "required_keys": [],
        "description": "USGS Water Quality Portal measurements by county",
        "output_columns": ["parameter_code", "parameter_name", "mean_value", "unit"], "index": "county_fips",
    },
    "usda_cropscape": {
        "module": "src.data.usda_cropscape", "function": "fetch_cropscape", "required_keys": [],
        "description": "USDA CropScape crop acreage by county",
        "output_columns": ["crop_name", "crop_code", "area_acres", "year"], "index": "county_fips",
    },
    "census_cbp": {
        "module": "src.data.census_cbp", "function": "fetch_cbp", "required_keys": [],
        "description": "Census County Business Patterns by county",
        "output_columns": ["naics_code", "establishment_count", "employment", "year"], "index": "county_fips",
    },
    "cdc_svi": {
        "module": "src.data.cdc_svi", "function": "fetch_svi", "required_keys": [],
        "description": "CDC Social Vulnerability Index by county",
        "output_columns": ["theme", "percentile_rank", "year"], "index": "county_fips",
    },
    "openfda_faers": {
        "module": "src.data.openfda_faers", "function": "fetch_faers", "required_keys": [],
        "description": "FDA FAERS adverse events (state-level)",
        "output_columns": ["drug_name", "reaction_term", "report_count", "year"], "index": "state",
    },
    "pubmed": {
        "module": "src.data.pubmed", "function": "search_pubmed", "required_keys": [],
        "description": "PubMed biomedical literature search",
        "output_columns": ["pmid", "title", "abstract", "authors", "year"], "index": None,
    },
    "semantic_scholar": {
        "module": "src.data.semantic_scholar", "function": "search_semantic_scholar", "required_keys": [],
        "description": "Semantic Scholar academic paper search",
        "output_columns": ["paper_id", "title", "abstract", "year", "citation_count"], "index": None,
    },
}


class DataSourceRegistry:
    """
    Registry for all data source adapters in src/data/.

    Provides a uniform interface for loading data with caching,
    checking availability, and enumerating variables.
    """

    def __init__(self) -> None:
        self._catalog = _SOURCE_CATALOG.copy()
        self._loaded_modules: dict[str, Any] = {}
        self._settings = None

    def _get_settings(self) -> Any:
        if self._settings is None:
            try:
                from src.config import get_settings
                self._settings = get_settings()
            except Exception as exc:
                logger.warning("Registry: cannot load settings: %s", exc)
        return self._settings

    def _load_module(self, module_name: str) -> Any:
        if module_name not in self._loaded_modules:
            try:
                mod = importlib.import_module(module_name)
                self._loaded_modules[module_name] = mod
            except ImportError as exc:
                logger.error("Registry: cannot import %s: %s", module_name, exc)
                raise
        return self._loaded_modules[module_name]

    def _check_required_keys(self, source_name: str) -> list[str]:
        spec = self._catalog.get(source_name, {})
        required = spec.get("required_keys", [])
        if not required:
            return []
        settings = self._get_settings()
        if settings is None:
            return required
        return [key for key in required if not getattr(settings, key, None)]

    def available_sources(self) -> list[dict[str, Any]]:
        results = []
        for name, spec in self._catalog.items():
            missing = self._check_required_keys(name)
            results.append({
                "name": name, "description": spec.get("description", ""),
                "index": spec.get("index"), "output_columns": spec.get("output_columns", []),
                "is_available": len(missing) == 0, "missing_keys": missing,
            })
        return sorted(results, key=lambda x: (not x["is_available"], x["name"]))

    def list_variables(self) -> dict[str, list[str]]:
        return {name: spec.get("output_columns", []) for name, spec in self._catalog.items() if spec.get("output_columns")}

    def load(self, source_name: str, use_cache: bool = True, **params: Any) -> pd.DataFrame:
        if source_name not in self._catalog:
            raise KeyError(f"Unknown source '{source_name}'. Available: {list(self._catalog.keys())}")
        missing_keys = self._check_required_keys(source_name)
        if missing_keys:
            raise RuntimeError(f"Source '{source_name}' requires config keys: {missing_keys}. Add them to .env")
        spec = self._catalog[source_name]
        mod = self._load_module(spec["module"])
        func: Callable = getattr(mod, spec["function"])
        logger.info("Registry: loading source='%s' params=%s", source_name, params)
        try:
            result = func(use_cache=use_cache, **params)
        except TypeError:
            result = func(**params)
        if isinstance(result, pd.DataFrame):
            df = result
        elif isinstance(result, list):
            if result:
                df = pd.DataFrame([item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in result])
            else:
                df = pd.DataFrame(columns=spec.get("output_columns", []))
        else:
            raise RuntimeError(f"Source '{source_name}' returned unexpected type: {type(result)}")
        index_col = spec.get("index")
        if index_col and df.index.name != index_col and index_col in df.columns:
            df = df.set_index(index_col)
        return df

    def load_many(self, sources: list[str], use_cache: bool = True, **shared_params: Any) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for name in sources:
            try:
                results[name] = self.load(name, use_cache=use_cache, **shared_params)
            except Exception as exc:
                logger.warning("Registry: failed to load '%s': %s", name, exc)
        return results

    def describe(self) -> None:
        sources = self.available_sources()
        print(f"\nDataSourceRegistry: {len(sources)} sources\n")
        print(f"{'Source':<25} {'Available':<12} {'Index':<15} Variables")
        print("-" * 80)
        for s in sources:
            status = "YES" if s["is_available"] else f"NO ({', '.join(s['missing_keys'])})"
            cols = ", ".join(s["output_columns"][:4])
            if len(s["output_columns"]) > 4:
                cols += f" ... (+{len(s['output_columns']) - 4})"
            print(f"{s['name']:<25} {status:<12} {str(s['index']):<15} {cols}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    registry = DataSourceRegistry()
    registry.describe()
