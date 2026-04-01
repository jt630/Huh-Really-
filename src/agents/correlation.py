"""
Correlation Agent - Prompt 3.

Capabilities:
- Pairwise Pearson/Spearman correlation + p-values
- Partial correlation controlling for confounders
- Correlation sweep with Bonferroni correction
- Outlier/hot-spot county detection
- Choropleth and LISA cluster maps (folium)
- Global and Local Moran's I (PySAL esda)
- Bivariate Local Moran's I
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CorrelationRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    outcome_col: str
    exposure_col: str
    confounder_cols: list[str] = []
    data: pd.DataFrame  # indexed by county_fips


class CorrelationResult(BaseModel):
    outcome_col: str
    exposure_col: str
    r_pearson: float | None
    p_pearson: float | None
    r_spearman: float | None
    p_spearman: float | None
    r_partial: float | None
    p_partial: float | None
    n_counties: int
    significant: bool
    bonferroni_significant: bool = False


class SweepReport(BaseModel):
    outcome: str
    results: list[CorrelationResult]
    bonferroni_alpha: float
    top_n: int


class SpatialResult(BaseModel):
    variable: str
    moran_i: float
    moran_p: float
    lisa_clusters: dict[str, str]  # fips -> "HH"|"LL"|"HL"|"LH"|"NS"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("data/cache")
_COUNTIES_GEOJSON = _CACHE_DIR / "counties.geojson"
_COUNTIES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)
_MIN_OBS_FOR_CORRELATION = 3

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _partial_correlation_residuals(
    outcome: np.ndarray,
    exposure: np.ndarray,
    confounders: np.ndarray,
) -> tuple[float, float]:
    """Compute partial Pearson r and p-value by residual regression.

    Regresses both *outcome* and *exposure* on *confounders* via OLS, then
    correlates the residuals.

    Parameters
    ----------
    outcome, exposure:
        1-D arrays of equal length (already dropna'd).
    confounders:
        2-D array (n, k); may have k == 1.

    Returns
    -------
    (r, p)
    """
    # Add intercept column
    n = len(outcome)
    X = np.column_stack([np.ones(n), confounders])

    def _residualize(y: np.ndarray) -> np.ndarray:
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        except np.linalg.LinAlgError:
            return y - y.mean()

    res_y = _residualize(outcome)
    res_x = _residualize(exposure)

    if np.std(res_y) == 0 or np.std(res_x) == 0:
        return float("nan"), float("nan")

    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def _ensure_counties_geojson() -> Path:
    """Download and cache the county GeoJSON if not already present."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _COUNTIES_GEOJSON.exists():
        import urllib.request
        urllib.request.urlretrieve(_COUNTIES_GEOJSON_URL, _COUNTIES_GEOJSON)
    return _COUNTIES_GEOJSON


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def pairwise_correlation(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    confounder_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> CorrelationResult:
    """Compute Pearson, Spearman, and (optionally) partial correlation.

    Parameters
    ----------
    df:
        DataFrame indexed by county_fips.
    outcome_col, exposure_col:
        Column names.
    confounder_cols:
        Optional list of confounder column names for partial correlation.
    alpha:
        Significance threshold for the ``significant`` flag.

    Returns
    -------
    CorrelationResult
    """
    if confounder_cols is None:
        confounder_cols = []

    cols = [outcome_col, exposure_col] + list(confounder_cols)
    subset = df[cols].dropna()
    n = len(subset)

    nan_result = CorrelationResult(
        outcome_col=outcome_col,
        exposure_col=exposure_col,
        r_pearson=float("nan"),
        p_pearson=float("nan"),
        r_spearman=float("nan"),
        p_spearman=float("nan"),
        r_partial=None,
        p_partial=None,
        n_counties=n,
        significant=False,
    )

    if n < _MIN_OBS_FOR_CORRELATION:
        return nan_result

    y = subset[outcome_col].to_numpy(dtype=float)
    x = subset[exposure_col].to_numpy(dtype=float)

    # Pearson
    try:
        r_p, p_p = stats.pearsonr(x, y)
    except Exception:
        r_p, p_p = float("nan"), float("nan")

    # Spearman
    try:
        r_s, p_s = stats.spearmanr(x, y)
    except Exception:
        r_s, p_s = float("nan"), float("nan")

    # Partial
    r_partial: float | None = None
    p_partial: float | None = None
    if confounder_cols:
        conf_data = subset[list(confounder_cols)].to_numpy(dtype=float)
        r_partial, p_partial = _partial_correlation_residuals(y, x, conf_data)
        if math.isnan(r_partial):
            r_partial = None
            p_partial = None

    significant = bool(
        not math.isnan(p_p) and p_p < alpha
    )

    return CorrelationResult(
        outcome_col=outcome_col,
        exposure_col=exposure_col,
        r_pearson=float(r_p),
        p_pearson=float(p_p),
        r_spearman=float(r_s),
        p_spearman=float(p_s),
        r_partial=r_partial,
        p_partial=p_partial,
        n_counties=n,
        significant=significant,
    )


def correlation_sweep(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_cols: list[str],
    confounder_cols: list[str] | None = None,
    alpha: float = 0.05,
    top_n: int = 20,
) -> SweepReport:
    """Run pairwise_correlation for each exposure and return ranked results.

    Parameters
    ----------
    df:
        DataFrame indexed by county_fips.
    outcome_col:
        Name of the outcome column.
    exposure_cols:
        Names of exposure columns to test.
    confounder_cols:
        Optional confounders passed to each pairwise_correlation call.
    alpha:
        Family-wise alpha before Bonferroni correction.
    top_n:
        Maximum number of results to return (ranked by p_pearson).

    Returns
    -------
    SweepReport
    """
    if confounder_cols is None:
        confounder_cols = []

    # Filter to exposure cols that actually exist
    valid_exposures = [c for c in exposure_cols if c in df.columns]

    n_tests = len(valid_exposures)
    bonferroni_alpha = alpha / n_tests if n_tests > 0 else alpha

    results: list[CorrelationResult] = []
    for exp_col in valid_exposures:
        res = pairwise_correlation(df, outcome_col, exp_col, confounder_cols, alpha=alpha)
        # Mark Bonferroni significance
        if res.p_pearson is not None and not math.isnan(res.p_pearson):
            bonf_sig = bool(res.p_pearson < bonferroni_alpha)
        else:
            bonf_sig = False
        results.append(res.model_copy(update={"bonferroni_significant": bonf_sig}))

    # Sort by p_pearson ascending (NaN goes last)
    def _sort_key(r: CorrelationResult) -> float:
        if r.p_pearson is None or math.isnan(r.p_pearson):
            return float("inf")
        return r.p_pearson

    results.sort(key=_sort_key)
    results = results[:top_n]

    return SweepReport(
        outcome=outcome_col,
        results=results,
        bonferroni_alpha=bonferroni_alpha,
        top_n=top_n,
    )


def find_hotspot_counties(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    n_sd: float = 2.0,
) -> list[str]:
    """Return FIPS codes where both outcome and exposure exceed mean + n_sd*std.

    Parameters
    ----------
    df:
        DataFrame indexed by county_fips.
    outcome_col, exposure_col:
        Column names to assess.
    n_sd:
        Number of standard deviations above the mean to use as threshold.

    Returns
    -------
    list[str]
        Sorted list of FIPS strings that qualify as hotspots.
    """
    subset = df[[outcome_col, exposure_col]].dropna()
    if subset.empty:
        return []

    outcome_vals = subset[outcome_col]
    exposure_vals = subset[exposure_col]

    outcome_thresh = outcome_vals.mean() + n_sd * outcome_vals.std()
    exposure_thresh = exposure_vals.mean() + n_sd * exposure_vals.std()

    mask = (outcome_vals > outcome_thresh) & (exposure_vals > exposure_thresh)
    return sorted(subset.index[mask].tolist())


# ---------------------------------------------------------------------------
# Spatial weights
# ---------------------------------------------------------------------------

_weights_cache: dict[str, object] = {}


def build_county_weights(fips_list: list[str]) -> "libpysal.weights.W":  # type: ignore[name-defined]
    """Build Queen contiguity weights for the given FIPS list.

    Downloads and caches the county shapefile if needed.

    Parameters
    ----------
    fips_list:
        List of county FIPS strings.

    Returns
    -------
    libpysal.weights.W
    """
    import geopandas as gpd
    import libpysal.weights

    cache_key = ",".join(sorted(fips_list))
    if cache_key in _weights_cache:
        return _weights_cache[cache_key]  # type: ignore[return-value]

    geojson_path = _ensure_counties_geojson()
    gdf = gpd.read_file(str(geojson_path))

    # Normalise FIPS column
    fips_col = None
    for candidate in ("GEOID", "fips", "FIPS", "id", "GEO_ID"):
        if candidate in gdf.columns:
            fips_col = candidate
            break
    if fips_col is None:
        raise ValueError("Cannot identify FIPS column in county GeoJSON.")

    gdf[fips_col] = gdf[fips_col].astype(str).str.zfill(5)
    fips_set = set(f.zfill(5) for f in fips_list)
    gdf_filtered = gdf[gdf[fips_col].isin(fips_set)].copy().reset_index(drop=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = libpysal.weights.Queen.from_dataframe(gdf_filtered, silence_warnings=True)

    _weights_cache[cache_key] = w
    return w


# ---------------------------------------------------------------------------
# Moran's I
# ---------------------------------------------------------------------------


def global_morans_i(
    series: pd.Series,
    weights: "libpysal.weights.W",  # type: ignore[name-defined]
) -> tuple[float, float]:
    """Compute Global Moran's I.

    Parameters
    ----------
    series:
        Pandas Series of values (aligned to weights).
    weights:
        libpysal spatial weights object.

    Returns
    -------
    (I, p_sim)
    """
    import esda

    moran = esda.Moran(series.to_numpy(dtype=float), weights)
    return float(moran.I), float(moran.p_sim)


def local_morans_i(
    series: pd.Series,
    weights: "libpysal.weights.W",  # type: ignore[name-defined]
) -> pd.DataFrame:
    """Compute Local Moran's I (LISA) and classify clusters.

    Parameters
    ----------
    series:
        Pandas Series of values (aligned to weights).
    weights:
        libpysal spatial weights object.

    Returns
    -------
    pd.DataFrame
        Columns: fips, cluster  (HH/LL/HL/LH/NS)
    """
    import esda

    values = series.to_numpy(dtype=float)
    moran_loc = esda.Moran_Local(values, weights)

    fips_list = list(series.index)
    clusters = []
    for i, fips in enumerate(fips_list):
        sig = moran_loc.p_sim[i] < 0.05
        if not sig:
            label = "NS"
        else:
            q = int(moran_loc.q[i])
            label = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}.get(q, "NS")
        clusters.append({"fips": fips, "cluster": label})

    return pd.DataFrame(clusters)


def bivariate_local_morans(
    series1: pd.Series,
    series2: pd.Series,
    weights: "libpysal.weights.W",  # type: ignore[name-defined]
) -> pd.DataFrame:
    """Compute Bivariate Local Moran's I.

    Parameters
    ----------
    series1, series2:
        Pandas Series (aligned to weights).
    weights:
        libpysal spatial weights object.

    Returns
    -------
    pd.DataFrame
        Columns: fips, cluster  (HH/LL/HL/LH/NS)
    """
    import esda

    v1 = series1.to_numpy(dtype=float)
    v2 = series2.to_numpy(dtype=float)
    moran_bv = esda.Moran_Local_BV(v1, v2, weights)

    fips_list = list(series1.index)
    clusters = []
    for i, fips in enumerate(fips_list):
        sig = moran_bv.p_sim[i] < 0.05
        if not sig:
            label = "NS"
        else:
            q = int(moran_bv.q[i])
            label = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}.get(q, "NS")
        clusters.append({"fips": fips, "cluster": label})

    return pd.DataFrame(clusters)


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def choropleth_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    counties_geojson_path: str | Path,
) -> "folium.Map":  # type: ignore[name-defined]
    """Create a choropleth folium map.

    Parameters
    ----------
    df:
        DataFrame indexed by county_fips.
    value_col:
        Column to visualise.
    title:
        Map title (shown as a layer name).
    counties_geojson_path:
        Path to county GeoJSON file.

    Returns
    -------
    folium.Map
    """
    import folium
    import json as _json

    m = folium.Map(location=[37.8, -96], zoom_start=4)

    with open(counties_geojson_path) as f:
        geo_data = _json.load(f)

    data_df = df[[value_col]].reset_index()
    # Ensure the index column is named county_fips
    if data_df.columns[0] != "county_fips":
        data_df = data_df.rename(columns={data_df.columns[0]: "county_fips"})

    folium.Choropleth(
        geo_data=geo_data,
        name=title,
        data=data_df,
        columns=["county_fips", value_col],
        key_on="feature.id",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="white",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def lisa_cluster_map(
    lisa_df: pd.DataFrame,
    title: str,
    counties_geojson_path: str | Path,
) -> "folium.Map":  # type: ignore[name-defined]
    """Create a LISA cluster map coloured by HH/LL/HL/LH/NS.

    Parameters
    ----------
    lisa_df:
        DataFrame with columns ``fips`` and ``cluster``.
    title:
        Map title.
    counties_geojson_path:
        Path to county GeoJSON file.

    Returns
    -------
    folium.Map
    """
    import folium
    import json as _json

    _CLUSTER_COLORS: dict[str, str] = {
        "HH": "red",
        "LL": "blue",
        "HL": "orange",
        "LH": "lightblue",
        "NS": "white",
    }

    m = folium.Map(location=[37.8, -96], zoom_start=4)

    with open(counties_geojson_path) as f:
        geo_data = _json.load(f)

    cluster_lookup = dict(zip(lisa_df["fips"].astype(str), lisa_df["cluster"]))

    def _style(feature: dict) -> dict:
        fips = str(feature.get("id", ""))
        cluster = cluster_lookup.get(fips, "NS")
        color = _CLUSTER_COLORS.get(cluster, "white")
        return {
            "fillColor": color,
            "color": "grey",
            "weight": 0.5,
            "fillOpacity": 0.7,
        }

    folium.GeoJson(
        geo_data,
        name=title,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(fields=["NAME"] if "NAME" in (geo_data.get("features") or [{}])[0].get("properties", {}) else []),
    ).add_to(m)

    # Add a simple legend
    legend_html = (
        "<div style='position:fixed;bottom:30px;left:30px;z-index:1000;"
        "background:white;padding:10px;border:1px solid grey'>"
        f"<b>{title}</b><br>"
        + "".join(
            f"<i style='background:{c};width:12px;height:12px;display:inline-block;"
            f"margin-right:4px'></i>{label}<br>"
            for label, c in _CLUSTER_COLORS.items()
        )
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CorrelationAgent:
    """Orchestrates correlation analysis and optional spatial statistics."""

    def run(self, request: CorrelationRequest) -> dict:
        """Run pairwise correlation, hotspot detection, and spatial analysis.

        Parameters
        ----------
        request:
            A populated CorrelationRequest.

        Returns
        -------
        dict
            Keys: correlation, hotspot_fips, spatial (if available).
        """
        # Core correlation
        corr_result = pairwise_correlation(
            request.data,
            request.outcome_col,
            request.exposure_col,
            request.confounder_cols,
        )

        # Hotspot detection
        hotspot_fips = find_hotspot_counties(
            request.data,
            request.outcome_col,
            request.exposure_col,
        )

        output: dict = {
            "correlation": corr_result.model_dump(),
            "hotspot_fips": hotspot_fips,
        }

        # Attempt spatial analysis
        try:
            import libpysal.weights  # noqa: F401
            import esda  # noqa: F401

            fips_list = list(request.data.index)
            weights = build_county_weights(fips_list)

            series = request.data[request.outcome_col].dropna()
            common_fips = [f for f in fips_list if f in series.index]
            series = series.loc[common_fips]

            moran_i, moran_p = global_morans_i(series, weights)
            lisa_df = local_morans_i(series, weights)
            lisa_clusters = dict(zip(lisa_df["fips"], lisa_df["cluster"]))

            spatial_result = SpatialResult(
                variable=request.outcome_col,
                moran_i=moran_i,
                moran_p=moran_p,
                lisa_clusters=lisa_clusters,
            )
            output["spatial"] = spatial_result.model_dump()

        except ImportError:
            output["spatial"] = None
        except Exception as exc:
            output["spatial"] = {"error": str(exc)}

        return output
