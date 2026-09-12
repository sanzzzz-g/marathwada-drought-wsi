"""
Tests for GSDA groundwater deduplication, coverage audit, causal propagation, and standardization.
"""

import pytest
import pandas as pd
import numpy as np

from src.groundwater import (
    clean_groundwater_records,
    aggregate_district_monthly_groundwater,
    measure_groundwater_coverage,
    measure_groundwater_sensitivity,
    fit_train_groundwater_climatology,
    process_groundwater_features,
)


def test_groundwater_cleaning_conflicts_and_negatives():
    """Verifies that negative depths are dropped and conflicting measurements on the same date are excluded."""
    raw = pd.DataFrame([
        # Valid
        {"well_id": "W1", "district": "Latur", "obs_date": pd.Timestamp("2015-05-01"), "water_level_m_bgl": 5.0},
        # Impossible negative depth
        {"well_id": "W2", "district": "Latur", "obs_date": pd.Timestamp("2015-05-01"), "water_level_m_bgl": -2.0},
        # Conflicting readings for same well on same date
        {"well_id": "W3", "district": "Latur", "obs_date": pd.Timestamp("2015-05-01"), "water_level_m_bgl": 4.0},
        {"well_id": "W3", "district": "Latur", "obs_date": pd.Timestamp("2015-05-01"), "water_level_m_bgl": 8.0},
    ])
    
    cleaned = clean_groundwater_records(raw)
    
    # Only W1 should survive
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["well_id"] == "W1"
    assert cleaned.iloc[0]["water_level_m_bgl"] == 5.0


def test_groundwater_sensitivity_analysis():
    """Verifies that sensitivity analysis computes diagnostics for multiple thresholds (5, 10, 15)."""
    cal = pd.DataFrame([
        {"district": "Latur", "date": pd.Timestamp("2015-05-01"), "year": 2015, "month": 5},
        {"district": "Latur", "date": pd.Timestamp("2015-10-01"), "year": 2015, "month": 10},
    ])
    raw = pd.DataFrame([
        {"well_id": f"W{i}", "district": "Latur", "obs_date": pd.Timestamp("2015-05-01"), "water_level_m_bgl": 5.0}
        for i in range(12)  # 12 wells in May (qualifies for N=5 and N=10, but not N=15)
    ] + [
        {"well_id": f"W{i}", "district": "Latur", "obs_date": pd.Timestamp("2015-10-01"), "water_level_m_bgl": 3.0}
        for i in range(7)   # 7 wells in Oct (qualifies for N=5, not N=10, not N=15)
    ])
    cleaned = clean_groundwater_records(raw)
    sens = measure_groundwater_sensitivity(cal, cleaned, thresholds=[5, 10, 15])
    
    assert sens["N_5"]["qualifying_observations_count"] == 2
    assert sens["N_10"]["qualifying_observations_count"] == 1
    assert sens["N_15"]["qualifying_observations_count"] == 0


def test_observation_month_standardization_formula():
    """
    Verifies that:
    1. Physical anomaly is computed against observation month climatology.
    2. Anomaly is causally forward-carried without seasonal mismatch.
    3. Final standardization on the training as-of series produces exact mean=0 and std=1 (ddof=0).
    4. Staleness flag is set if months_since_last_gw_obs > 3.
    """
    dates = pd.date_range("2015-01-01", "2015-12-01", freq="MS")
    cal = pd.DataFrame({
        "district": ["Latur"] * len(dates),
        "date": dates,
        "year": dates.year,
        "month": dates.month,
    })
    
    # Qualifying observations in May (depth=12.0m) and October (depth=4.0m)
    agg_gw = pd.DataFrame([
        {
            "district": "Latur",
            "obs_date": pd.Timestamp("2015-05-01"),
            "year": 2015,
            "month": 5,
            "n_wells": 25,
            "median_depth_m_bgl": 12.0,
            "is_qualifying": True,
        },
        {
            "district": "Latur",
            "obs_date": pd.Timestamp("2015-10-01"),
            "year": 2015,
            "month": 10,
            "n_wells": 25,
            "median_depth_m_bgl": 4.0,
            "is_qualifying": True,
        },
    ])
    
    # Climatology: May=8.0m (May anom = +4.0m), October=6.0m (Oct anom = -2.0m)
    train_clim = {
        "district_month_climatology": {"Latur_5": 8.0, "Latur_10": 6.0},
        "district_mean_fallback": {"Latur": 7.0},
    }
    
    feat_df, asof_stats = process_groundwater_features(
        calendar_df=cal,
        agg_gw_df=agg_gw,
        train_clim_stats=train_clim,
        train_end_date="2017-12-01",
        min_wells_threshold=10,
        staleness_threshold_months=3,
    )
    
    # Training as-of series must have mean=0 and std=1 (ddof=0)
    train_feat = feat_df[feat_df["date"] <= "2017-12-01"]
    assert abs(train_feat["groundwater_stress_z"].mean()) < 1e-10
    assert abs(train_feat["groundwater_stress_z"].std(ddof=0) - 1.0) < 1e-10
    
    # Check causal propagation in June, July, August (carried from May)
    june_row = feat_df[feat_df["date"] == pd.Timestamp("2015-06-01")].iloc[0]
    may_row = feat_df[feat_df["date"] == pd.Timestamp("2015-05-01")].iloc[0]
    assert june_row["groundwater_is_qualifying"] == 0
    assert june_row["groundwater_anomaly_m_bgl"] == may_row["groundwater_anomaly_m_bgl"]
    assert june_row["groundwater_stress_z"] == may_row["groundwater_stress_z"]
    assert june_row["months_since_last_gw_obs"] == 1
    assert june_row["groundwater_stale_flag"] == 0
    
    # September is 4 months after May -> stale flag = 1
    sept_row = feat_df[feat_df["date"] == pd.Timestamp("2015-09-01")].iloc[0]
    assert sept_row["months_since_last_gw_obs"] == 4
    assert sept_row["groundwater_stale_flag"] == 1
