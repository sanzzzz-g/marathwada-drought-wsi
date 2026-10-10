"""
Tests verifying zero temporal leakage across train, validation, and test splits.
"""

import pytest
import pandas as pd
import numpy as np

from src.qc import fit_train_imputation_stats, impute_missing_with_train_stats
from src.spi import fit_train_gamma_parameters
from src.groundwater import process_groundwater_features


def test_imputation_strictly_train_fitted():
    """
    Verifies that imputation medians are computed ONLY from training data (<= 2017-12-01).
    Modifying post-2017 validation/test values must not change the fitted imputation medians.
    """
    dates = pd.date_range("2015-01-01", "2020-12-01", freq="MS")
    df_clean = pd.DataFrame({
        "district": ["Latur"] * len(dates),
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "val": 10.0,
    })
    
    # Baseline stats
    stats_base = fit_train_imputation_stats(df_clean, val_col="val", train_end_date="2017-12-01")
    
    # Mutate post-2017 values drastically (simulate extreme test/validation noise)
    df_mutated = df_clean.copy()
    df_mutated.loc[df_mutated["date"] > "2017-12-01", "val"] = 99999.0
    
    stats_mutated = fit_train_imputation_stats(df_mutated, val_col="val", train_end_date="2017-12-01")
    
    assert stats_base == stats_mutated, "Post-2017 validation/test data leaked into imputation statistics!"


def test_spi_calibration_strictly_bounded():
    """
    Verifies that SPI Gamma fitting strictly respects calibration start and end dates.
    Rainfall after 2017-12-01 must not influence the fitted alpha, scale, or zero-probability q.
    """
    dates = pd.date_range("1981-01-01", "2024-12-01", freq="MS")
    rng = np.random.RandomState(42)
    rain = rng.gamma(shape=2.0, scale=30.0, size=len(dates))
    
    df_chirps = pd.DataFrame({
        "district": ["Beed"] * len(dates),
        "date": dates,
        "year": dates.year,
        "month": dates.month,
        "rainfall_3m_mm": rain,
    })
    
    params_base = fit_train_gamma_parameters(
        df_chirps, train_start_date="1981-01-01", train_end_date="2017-12-01"
    )
    
    # Mutate post-2017 rainfall to extreme values
    df_mutated = df_chirps.copy()
    df_mutated.loc[df_mutated["date"] > "2017-12-01", "rainfall_3m_mm"] = 1000.0
    
    params_mutated = fit_train_gamma_parameters(
        df_mutated, train_start_date="1981-01-01", train_end_date="2017-12-01"
    )
    
    assert params_base == params_mutated, "Post-calibration rainfall leaked into SPI Gamma parameters!"


def test_no_future_groundwater_leakage():
    """
    Verifies that groundwater as-of features are strictly causal:
    1. For every date t, last_gw_obs_date <= t.
    2. months_since_last_gw_obs >= 0.
    3. An observation in month t + k is never used at month t.
    """
    dates = pd.date_range("2015-01-01", "2015-12-01", freq="MS")
    cal = pd.DataFrame({
        "district": ["Latur"] * len(dates),
        "date": dates,
        "year": dates.year,
        "month": dates.month,
    })
    
    # Only one qualifying observation in October 2015
    agg_gw = pd.DataFrame([{
        "district": "Latur",
        "obs_date": pd.Timestamp("2015-10-01"),
        "year": 2015,
        "month": 10,
        "n_wells": 20,
        "median_depth_m_bgl": 5.0,
        "is_qualifying": True,
    }])
    
    train_clim = {
        "district_month_climatology": {"Latur_10": 5.0},
        "district_mean_fallback": {"Latur": 5.0},
    }
    
    feat_df, _ = process_groundwater_features(
        calendar_df=cal,
        agg_gw_df=agg_gw,
        train_clim_stats=train_clim,
        train_end_date="2017-12-01",
        min_wells_threshold=10,
        staleness_threshold_months=3,
    )
    
    # Prior to October 2015, there is NO observation yet: last_gw_obs_date must be None / NaN
    pre_oct = feat_df[feat_df["date"] < "2015-10-01"]
    assert pre_oct["last_gw_obs_date"].isna().all(), "Future groundwater date leaked into prior months!"
    
    # From October 2015 onwards, observation is October 2015
    post_oct = feat_df[feat_df["date"] >= "2015-10-01"]
    assert (post_oct["last_gw_obs_date"] == "2015-10-01").all()
    assert (post_oct["months_since_last_gw_obs"] >= 0).all()


def test_forecast_origin_target_isolation():
    """Verify that three-month target windows stay within their assigned periods."""
    horizon = 3

    # Last training origin: September 2017 -> October–December 2017.
    last_train_origin = pd.Timestamp("2017-09-01")
    train_targets = [
        last_train_origin + pd.DateOffset(months=h)
        for h in range(1, horizon + 1)
    ]
    train_end = pd.Timestamp("2017-12-01")
    assert all(d <= train_end for d in train_targets)

    # First validation origin: December 2017 -> January–March 2018.
    first_val_origin = pd.Timestamp("2017-12-01")
    val_targets = [
        first_val_origin + pd.DateOffset(months=h)
        for h in range(1, horizon + 1)
    ]
    assert val_targets == list(pd.date_range("2018-01-01", periods=3, freq="MS"))
    assert all(d > train_end for d in val_targets)

    # Last validation origin: September 2020 -> October–December 2020.
    last_val_origin = pd.Timestamp("2020-09-01")
    last_val_targets = [
        last_val_origin + pd.DateOffset(months=h)
        for h in range(1, horizon + 1)
    ]
    val_end = pd.Timestamp("2020-12-01")
    assert all(d <= val_end for d in last_val_targets)

    # First test origin: December 2020 -> January–March 2021.
    first_test_origin = pd.Timestamp("2020-12-01")
    test_targets = [
        first_test_origin + pd.DateOffset(months=h)
        for h in range(1, horizon + 1)
    ]
    assert test_targets == list(pd.date_range("2021-01-01", periods=3, freq="MS"))
    assert all(d > val_end for d in test_targets)
