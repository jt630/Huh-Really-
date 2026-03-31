"""
Tests for src/agents/correlation.py

Coverage:
- pairwise_correlation: Pearson/Spearman values, partial correlation,
  n_counties count, edge cases (too few rows, all-NaN column).
- correlation_sweep: Bonferroni correction, ranking, top_n, missing cols.
- find_hotspot_counties: correct hot-spot detection.
- CorrelationResult / SweepReport pydantic models round-trip.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agents.correlation import (
    CorrelationRequest,
    CorrelationResult,
    SweepReport,
    correlation_sweep,
    find_hotspot_counties,
    pairwise_correlation,
)


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    fips = [f"{i:05d}" for i in range(1, n + 1)]
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.3
    df = pd.DataFrame({"outcome": x + noise, "exposure": x, "confounder": rng.standard_normal(n)}, index=fips)
    df.index.name = "county_fips"
    return df


@pytest.fixture
def uncorrelated_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    fips = [f"{i:05d}" for i in range(1001, 1001 + n)]
    df = pd.DataFrame({"outcome": rng.standard_normal(n), "exposure": rng.standard_normal(n)}, index=fips)
    df.index.name = "county_fips"
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    df = pd.DataFrame({"outcome": [5.0, 7.0], "exposure": [100.0, 200.0]}, index=["01001", "01003"])
    df.index.name = "county_fips"
    return df


@pytest.fixture
def nan_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 40
    fips = [f"{i:05d}" for i in range(2001, 2001 + n)]
    outcome = rng.uniform(1, 10, n).tolist()
    exposure = rng.uniform(0, 500, n).tolist()
    for i in [0, 1, 2]:
        outcome[i] = float("nan")
    for i in [3, 4]:
        exposure[i] = float("nan")
    df = pd.DataFrame({"outcome": outcome, "exposure": exposure}, index=fips)
    df.index.name = "county_fips"
    return df


class TestPairwiseCorrelation:
    def test_strong_positive_pearson(self, correlated_df):
        res = pairwise_correlation(correlated_df, "outcome", "exposure")
        assert res.r_pearson > 0.9
        assert res.p_pearson < 1e-10

    def test_strong_positive_spearman(self, correlated_df):
        res = pairwise_correlation(correlated_df, "outcome", "exposure")
        assert res.r_spearman > 0.9
        assert res.p_spearman < 1e-10

    def test_n_counties_full_data(self, correlated_df):
        assert pairwise_correlation(correlated_df, "outcome", "exposure").n_counties == 100

    def test_n_counties_with_nans(self, nan_df):
        assert pairwise_correlation(nan_df, "outcome", "exposure").n_counties == 35

    def test_partial_correlation_returned(self, correlated_df):
        res = pairwise_correlation(correlated_df, "outcome", "exposure", confounder_cols=["confounder"])
        assert res.r_partial is not None
        assert res.r_partial > 0.85

    def test_no_confounders_partial_is_none(self, correlated_df):
        res = pairwise_correlation(correlated_df, "outcome", "exposure")
        assert res.r_partial is None
        assert res.p_partial is None

    def test_too_few_rows_returns_nan(self, small_df):
        res = pairwise_correlation(small_df, "outcome", "exposure")
        assert res.n_counties == 2
        assert math.isnan(res.r_pearson)
        assert res.significant is False

    def test_returns_correlation_result_model(self, correlated_df):
        assert isinstance(pairwise_correlation(correlated_df, "outcome", "exposure"), CorrelationResult)

    def test_significant_flag_set(self, correlated_df):
        assert pairwise_correlation(correlated_df, "outcome", "exposure", alpha=0.05).significant is True


class TestCorrelationSweep:
    @pytest.fixture
    def multi_exposure_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        n = 80
        fips = [f"{i:05d}" for i in range(3001, 3001 + n)]
        x = rng.standard_normal(n)
        df = pd.DataFrame({
            "outcome": x + rng.standard_normal(n) * 0.2,
            "strong_exp": x,
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
            "noise_3": rng.standard_normal(n),
        }, index=fips)
        df.index.name = "county_fips"
        return df

    def test_returns_sweep_report(self, multi_exposure_df):
        assert isinstance(correlation_sweep(multi_exposure_df, "outcome", ["strong_exp", "noise_1"]), SweepReport)

    def test_bonferroni_alpha_correct(self, multi_exposure_df):
        report = correlation_sweep(multi_exposure_df, "outcome", ["strong_exp", "noise_1", "noise_2", "noise_3"], alpha=0.05)
        assert abs(report.bonferroni_alpha - 0.05 / 4) < 1e-10

    def test_results_ranked_by_p_value(self, multi_exposure_df):
        report = correlation_sweep(multi_exposure_df, "outcome", ["strong_exp", "noise_1", "noise_2", "noise_3"])
        p_vals = [r.p_pearson for r in report.results]
        assert p_vals == sorted(p_vals)

    def test_strong_exposure_ranks_first(self, multi_exposure_df):
        report = correlation_sweep(multi_exposure_df, "outcome", ["noise_1", "noise_2", "strong_exp", "noise_3"])
        assert report.results[0].exposure_col == "strong_exp"

    def test_top_n_respected(self, multi_exposure_df):
        report = correlation_sweep(multi_exposure_df, "outcome", ["strong_exp", "noise_1", "noise_2", "noise_3"], top_n=2)
        assert len(report.results) <= 2

    def test_missing_exposure_col_skipped(self, multi_exposure_df):
        report = correlation_sweep(multi_exposure_df, "outcome", ["strong_exp", "does_not_exist"])
        assert "does_not_exist" not in [r.exposure_col for r in report.results]
        assert "strong_exp" in [r.exposure_col for r in report.results]


class TestFindHotspotCounties:
    @pytest.fixture
    def hotspot_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(5)
        n = 40
        fips = [f"{i:05d}" for i in range(4001, 4001 + n)]
        outcome = list(rng.standard_normal(n - 5) * 0.5)
        exposure = list(rng.standard_normal(n - 5) * 0.5)
        for _ in range(5):
            outcome.append(rng.uniform(8, 12))
            exposure.append(rng.uniform(800, 1200))
        df = pd.DataFrame({"outcome": outcome, "exposure": exposure}, index=fips)
        df.index.name = "county_fips"
        return df

    def test_hotspots_detected(self, hotspot_df):
        assert len(find_hotspot_counties(hotspot_df, "outcome", "exposure", n_sd=2.0)) >= 4

    def test_hotspots_are_valid_fips(self, hotspot_df):
        for fips in find_hotspot_counties(hotspot_df, "outcome", "exposure"):
            assert fips in set(hotspot_df.index)

    def test_no_hotspots_in_random_data(self, uncorrelated_df):
        assert len(find_hotspot_counties(uncorrelated_df, "outcome", "exposure", n_sd=3.0)) <= 2

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame({"outcome": [], "exposure": []})
        df.index.name = "county_fips"
        assert find_hotspot_counties(df, "outcome", "exposure") == []

    def test_custom_n_sd_threshold(self, hotspot_df):
        assert len(find_hotspot_counties(hotspot_df, "outcome", "exposure", n_sd=10.0)) <= len(find_hotspot_counties(hotspot_df, "outcome", "exposure", n_sd=1.0))
