"""
Correlation Agent (Prompt 3).

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

import logging
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats

from src.config import get_settings

if TYPE_CHECKING:
    import folium

logger = logging.getLogger(__name__)

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2019/COUNTY/tl_2019_us_county.zip"
TIGER_FILENAME = "tl_2019_us_county.zip"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CorrelationRequest(BaseModel):
    outcome_col: str
    exposure_col: str
    confounder_cols: list[str] = []
    alpha: float = 0.05


class CorrelationResult(BaseModel):
    outcome_col: str
    exposure_col: str
    n_counties: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    partial_r: float | None
    partial_p: float | None
    is_significant: bool
    hotspot_counties: list[str]


class SweepReport(BaseModel):
    outcome_col: str
    n_exposures_tested: int
    alpha_bonferroni: float
    results: list[CorrelationResult]
    significant_results: list[CorrelationResult]


class SpatialResult(BaseModel):
    variable: str
    global_moran_i: float
    global_moran_p: float
    global_moran_z: float
    lisa_hh_counties: list[str]
    lisa_ll_counties: list[str]
    lisa_hl_counties: list[str]
    lisa_lh_counties: list[str]


class BivariateResult(BaseModel):
    outcome_col: str
    exposure_col: str
    bivariate_moran_i: float
    bivariate_moran_p: float
    co_cluster_hh: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_fips_column(df: pd.DataFrame, fips_col: str = "county_fips") -> pd.DataFrame:
    """Return df with county_fips as a plain column (reset index if needed)."""
    if fips_col not in df.columns:
        if df.index.name == fips_col:
            return df.reset_index()
        raise ValueError(f"DataFrame must have '{fips_col}' as a column or index")
    return df.copy()


def _partial_correlation(
    outcome: np.ndarray,
    exposure: np.ndarray,
    confounders: np.ndarray,
) -> tuple[float, float]:
    """
    Compute partial correlation between outcome and exposure controlling for
    confounders via OLS residualisation.

    Falls back to pingouin when available.
    """
    try:
        import pingouin as pg  # type: ignore

        n = len(outcome)
        col_names = [f"c{i}" for i in range(confounders.shape[1])] if confounders.ndim == 2 else ["c0"]
        df_tmp = pd.DataFrame(confounders, columns=col_names)
        df_tmp["outcome"] = outcome
        df_tmp["exposure"] = exposure
        result = pg.partial_corr(data=df_tmp, x="exposure", y="outcome", covar=col_names)
        r = float(result["r"].iloc[0])
        p = float(result["p-val"].iloc[0])
        return r, p
    except ImportError:
        pass

    # Manual OLS residualisation
    if confounders.ndim == 1:
        confounders = confounders[:, np.newaxis]
    X = np.column_stack([np.ones(len(outcome)), confounders])
    try:
        beta_out = np.linalg.lstsq(X, outcome, rcond=None)[0]
        beta_exp = np.linalg.lstsq(X, exposure, rcond=None)[0]
        resid_out = outcome - X @ beta_out
        resid_exp = exposure - X @ beta_exp
        r, p = stats.pearsonr(resid_out, resid_exp)
        return float(r), float(p)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class CorrelationAgent:
    def __init__(self, weights_cache_dir: str = "data/cache/spatial_weights") -> None:
        self._weights = None  # lazy-loaded spatial weights matrix
        self._weights_cache_dir = Path(weights_cache_dir)
        self._weights_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core correlation
    # ------------------------------------------------------------------

    def correlate(
        self,
        outcome_df: pd.DataFrame,
        exposure_df: pd.DataFrame,
        request: CorrelationRequest,
    ) -> CorrelationResult:
        """
        Compute pairwise Pearson + Spearman correlation between outcome and
        exposure.  Both DataFrames must have county_fips as a column (or index).
        Merge on county_fips (inner join).
        Compute partial correlation if confounder_cols provided.
        Flag hotspot counties (>2 SD on BOTH variables).
        """
        outcome_df = _ensure_fips_column(outcome_df)
        exposure_df = _ensure_fips_column(exposure_df)

        merged = outcome_df.merge(exposure_df, on="county_fips", how="inner")

        y = merged[request.outcome_col].to_numpy(dtype=float)
        x = merged[request.exposure_col].to_numpy(dtype=float)

        # Drop rows with NaN in either variable
        mask = np.isfinite(y) & np.isfinite(x)
        fips_arr = merged["county_fips"].to_numpy()

        # Also require confounders to be finite
        if request.confounder_cols:
            for col in request.confounder_cols:
                mask &= np.isfinite(merged[col].to_numpy(dtype=float))

        y = y[mask]
        x = x[mask]
        fips_clean = fips_arr[mask]
        n = int(mask.sum())

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        partial_r: float | None = None
        partial_p: float | None = None
        if request.confounder_cols:
            conf_matrix = merged.loc[mask, request.confounder_cols].to_numpy(dtype=float)
            partial_r, partial_p = _partial_correlation(y, x, conf_matrix)

        # Hotspot detection: >2 SD above mean on BOTH variables
        y_z = (y - y.mean()) / (y.std() + 1e-12)
        x_z = (x - x.mean()) / (x.std() + 1e-12)
        hotspot_mask = (y_z > 2) & (x_z > 2)
        hotspot_counties = list(fips_clean[hotspot_mask])

        is_significant = float(pearson_p) < request.alpha

        return CorrelationResult(
            outcome_col=request.outcome_col,
            exposure_col=request.exposure_col,
            n_counties=n,
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_r=float(spearman_r),
            spearman_p=float(spearman_p),
            partial_r=float(partial_r) if partial_r is not None else None,
            partial_p=float(partial_p) if partial_p is not None else None,
            is_significant=is_significant,
            hotspot_counties=hotspot_counties,
        )

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        outcome_df: pd.DataFrame,
        exposure_dfs: dict[str, pd.DataFrame],
        outcome_col: str,
        confounder_cols: list[str] = [],
        alpha: float = 0.05,
    ) -> SweepReport:
        """
        Test outcome against all provided exposure DataFrames.
        Apply Bonferroni correction: alpha_bonferroni = alpha / n_exposures.
        Return SweepReport with all results sorted by p-value.
        """
        n = len(exposure_dfs)
        alpha_bonferroni = alpha / n if n > 0 else alpha

        all_results: list[CorrelationResult] = []
        for exposure_col, exp_df in exposure_dfs.items():
            req = CorrelationRequest(
                outcome_col=outcome_col,
                exposure_col=exposure_col,
                confounder_cols=confounder_cols,
                alpha=alpha_bonferroni,
            )
            result = self.correlate(outcome_df, exp_df, req)
            all_results.append(result)

        all_results.sort(key=lambda r: r.pearson_p)
        significant_results = [r for r in all_results if r.is_significant]

        return SweepReport(
            outcome_col=outcome_col,
            n_exposures_tested=n,
            alpha_bonferroni=alpha_bonferroni,
            results=all_results,
            significant_results=significant_results,
        )

    # ------------------------------------------------------------------
    # Spatial weights
    # ------------------------------------------------------------------

    def _get_county_gdf(self):
        """Download and cache the TIGER county shapefile, return GeoDataFrame."""
        import geopandas as gpd  # type: ignore

        zip_path = self._weights_cache_dir / TIGER_FILENAME
        shp_dir = self._weights_cache_dir / "tl_2019_us_county"

        if not shp_dir.exists():
            if not zip_path.exists():
                import urllib.request

                logger.info("Downloading TIGER county shapefile...")
                urllib.request.urlretrieve(TIGER_URL, zip_path)

            logger.info("Extracting TIGER county shapefile...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(shp_dir)

        shp_files = list(shp_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No shapefile found in {shp_dir}")
        return gpd.read_file(shp_files[0])

    def _get_spatial_weights(self, gdf=None):
        """Build (or load cached) Queen contiguity spatial weights."""
        import libpysal  # type: ignore

        cache_file = self._weights_cache_dir / "queen_weights.gwt"

        if cache_file.exists() and self._weights is None:
            try:
                self._weights = libpysal.weights.W.from_file(str(cache_file))
                return self._weights
            except Exception:
                logger.warning("Failed to load cached weights, rebuilding...")

        if self._weights is None:
            if gdf is None:
                gdf = self._get_county_gdf()
            logger.info("Building Queen contiguity spatial weights matrix...")
            self._weights = libpysal.weights.Queen.from_dataframe(gdf)
            try:
                self._weights.to_file(str(cache_file))
            except Exception as exc:
                logger.warning("Could not cache spatial weights: %s", exc)

        return self._weights

    # ------------------------------------------------------------------
    # Spatial analysis
    # ------------------------------------------------------------------

    def spatial_analysis(
        self,
        df: pd.DataFrame,
        variable_col: str,
        county_fips_col: str = "county_fips",
    ) -> SpatialResult:
        """
        Compute Global Moran's I and Local Moran's I (LISA) for a variable.
        Downloads and caches Census TIGER/Line county shapefile if not present.
        Builds Queen contiguity spatial weights matrix.
        """
        import geopandas as gpd  # type: ignore
        from esda.moran import Moran, Moran_Local  # type: ignore

        df = _ensure_fips_column(df, county_fips_col)
        gdf = self._get_county_gdf()

        # GEOID in TIGER is zero-padded 5-char string
        gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
        df[county_fips_col] = df[county_fips_col].astype(str).str.zfill(5)

        merged = gdf.merge(df[[county_fips_col, variable_col]], left_on="GEOID", right_on=county_fips_col, how="inner")
        merged = merged.dropna(subset=[variable_col])

        w = self._get_spatial_weights(merged)
        w.transform = "r"

        y = merged[variable_col].to_numpy(dtype=float)
        fips = merged["GEOID"].to_numpy()

        # Global Moran's I
        mi = Moran(y, w)

        # Local Moran's I (LISA)
        lm = Moran_Local(y, w)
        # lm.q: 1=HH, 2=LH, 3=LL, 4=HL
        sig = lm.p_sim < 0.05

        hh = list(fips[(lm.q == 1) & sig])
        lh = list(fips[(lm.q == 2) & sig])
        ll = list(fips[(lm.q == 3) & sig])
        hl = list(fips[(lm.q == 4) & sig])

        return SpatialResult(
            variable=variable_col,
            global_moran_i=float(mi.I),
            global_moran_p=float(mi.p_sim),
            global_moran_z=float(mi.z_sim),
            lisa_hh_counties=hh,
            lisa_ll_counties=ll,
            lisa_hl_counties=hl,
            lisa_lh_counties=lh,
        )

    # ------------------------------------------------------------------
    # Bivariate spatial
    # ------------------------------------------------------------------

    def bivariate_spatial(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        exposure_col: str,
        county_fips_col: str = "county_fips",
    ) -> BivariateResult:
        """
        Compute bivariate Local Moran's I between outcome and exposure.
        """
        from esda.moran import Moran_BV, Moran_Local_BV  # type: ignore

        df = _ensure_fips_column(df, county_fips_col)
        gdf = self._get_county_gdf()

        gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
        df[county_fips_col] = df[county_fips_col].astype(str).str.zfill(5)

        merged = gdf.merge(
            df[[county_fips_col, outcome_col, exposure_col]],
            left_on="GEOID",
            right_on=county_fips_col,
            how="inner",
        )
        merged = merged.dropna(subset=[outcome_col, exposure_col])

        w = self._get_spatial_weights(merged)
        w.transform = "r"

        y = merged[outcome_col].to_numpy(dtype=float)
        x = merged[exposure_col].to_numpy(dtype=float)
        fips = merged["GEOID"].to_numpy()

        bv = Moran_BV(y, x, w)
        lm_bv = Moran_Local_BV(y, x, w)

        # HH clusters: q==1 and significant
        sig = lm_bv.p_sim < 0.05
        co_hh = list(fips[(lm_bv.q == 1) & sig])

        return BivariateResult(
            outcome_col=outcome_col,
            exposure_col=exposure_col,
            bivariate_moran_i=float(bv.I),
            bivariate_moran_p=float(bv.p_sim),
            co_cluster_hh=co_hh,
        )

    # ------------------------------------------------------------------
    # Choropleth map
    # ------------------------------------------------------------------

    def make_choropleth_map(
        self,
        df: pd.DataFrame,
        value_col: str,
        county_fips_col: str = "county_fips",
        title: str = "",
        output_path: str | None = None,
    ) -> "folium.Map":
        """
        Create folium choropleth map of the given variable by county.
        Uses Census TIGER county GeoJSON.
        Optionally save to output_path as HTML.
        """
        import folium  # type: ignore

        df = _ensure_fips_column(df, county_fips_col)
        gdf = self._get_county_gdf()

        gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
        df[county_fips_col] = df[county_fips_col].astype(str).str.zfill(5)

        merged = gdf.merge(
            df[[county_fips_col, value_col]],
            left_on="GEOID",
            right_on=county_fips_col,
            how="left",
        )

        geojson = merged.__geo_interface__

        m = folium.Map(location=[37.8, -96], zoom_start=4)

        folium.Choropleth(
            geo_data=geojson,
            name=title or value_col,
            data=df,
            columns=[county_fips_col, value_col],
            key_on="feature.properties.GEOID",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=title or value_col,
            nan_fill_color="white",
        ).add_to(m)

        folium.LayerControl().add_to(m)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(output_path)

        return m

    # ------------------------------------------------------------------
    # LISA map
    # ------------------------------------------------------------------

    def make_lisa_map(
        self,
        df: pd.DataFrame,
        spatial_result: SpatialResult,
        county_fips_col: str = "county_fips",
        title: str = "",
        output_path: str | None = None,
    ) -> "folium.Map":
        """
        Create LISA cluster map: HH=red, LL=blue, HL=orange, LH=lightblue, NS=gray.
        """
        import folium  # type: ignore

        df = _ensure_fips_column(df, county_fips_col)
        gdf = self._get_county_gdf()
        gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)

        color_map = {
            "HH": "#d7191c",
            "LL": "#2c7bb6",
            "HL": "#fdae61",
            "LH": "#abd9e9",
            "NS": "#cccccc",
        }

        def _classify(fips: str) -> str:
            if fips in spatial_result.lisa_hh_counties:
                return "HH"
            if fips in spatial_result.lisa_ll_counties:
                return "LL"
            if fips in spatial_result.lisa_hl_counties:
                return "HL"
            if fips in spatial_result.lisa_lh_counties:
                return "LH"
            return "NS"

        gdf["lisa_class"] = gdf["GEOID"].apply(_classify)
        gdf["color"] = gdf["lisa_class"].map(color_map)

        m = folium.Map(location=[37.8, -96], zoom_start=4)

        for _, row in gdf.iterrows():
            try:
                import shapely  # type: ignore

                style = {
                    "fillColor": row["color"],
                    "color": "black",
                    "weight": 0.3,
                    "fillOpacity": 0.7,
                }
                geojson_feature = {
                    "type": "Feature",
                    "geometry": row["geometry"].__geo_interface__,
                    "properties": {
                        "GEOID": row["GEOID"],
                        "lisa_class": row["lisa_class"],
                    },
                }
                folium.GeoJson(
                    geojson_feature,
                    style_function=lambda feat, s=style: s,
                    tooltip=f"{row['GEOID']}: {row['lisa_class']}",
                ).add_to(m)
            except Exception:
                continue

        # Add a simple legend
        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                    padding:10px;border:1px solid gray;font-size:13px;">
        <b>{title}</b><br>
        <span style="background:{hh};padding:2px 10px;"></span> HH (high-high)<br>
        <span style="background:{ll};padding:2px 10px;"></span> LL (low-low)<br>
        <span style="background:{hl};padding:2px 10px;"></span> HL (high-low)<br>
        <span style="background:{lh};padding:2px 10px;"></span> LH (low-high)<br>
        <span style="background:{ns};padding:2px 10px;"></span> NS (not significant)<br>
        </div>
        """.format(
            title=title or spatial_result.variable,
            hh=color_map["HH"],
            ll=color_map["LL"],
            hl=color_map["HL"],
            lh=color_map["LH"],
            ns=color_map["NS"],
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            m.save(output_path)

        return m


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    fips = [f"{i:05d}" for i in range(1001, 1101)]
    outcome = pd.DataFrame({"county_fips": fips, "parkinsons_rate": np.random.normal(5, 1, 100)})
    exposure = pd.DataFrame({"county_fips": fips, "paraquat_kg": np.random.normal(100, 30, 100)})

    agent = CorrelationAgent()
    req = CorrelationRequest(outcome_col="parkinsons_rate", exposure_col="paraquat_kg")
    result = agent.correlate(outcome, exposure, req)
    print(result.model_dump_json(indent=2))
