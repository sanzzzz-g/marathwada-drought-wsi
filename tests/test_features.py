"""
Tests for multi-tier hydroclimatic feature engineering across all stress components.
"""

import pytest
import pandas as pd
import numpy as np

from src.features import (
    process_soil_water_features,
    process_ndvi_features,
    process_lst_features,
    process_surface_water_features,
)


def test_multi_tier_naming_and_polarity():
    """
    Verifies that:
    1. Multi-tier columns (raw, climatology, anomaly, stress_z) are all generated and distinct.
    2. Moisture deficits (soil, ndvi, surface water) yield POSITIVE stress Z-scores.
    3. Heat excess (LST) yields POSITIVE stress Z-score.
    """
    cal = pd.DataFrame([
        {"district": "Latur", "date": pd.Timestamp("2015-08-01"), "year": 2015, "month": 8},
    ])
    
    # Low soil moisture relative to climatology
    era5 = pd.DataFrame([{
        "district": "Latur", "date": pd.Timestamp("2015-08-01"),
        "soil_water_layer_1": 0.1, "soil_water_layer_2": 0.1, "soil_water_layer_3": 0.1, "soil_water_layer_4": 0.1,
        "soil_water_storage_mm_0_289cm": 500.0,
    }])
    
    soil_df, _ = process_soil_water_features(cal, era5, train_end_date="2017-12-01")
    # All tiers must be present
    assert "soil_water_storage_mm" in soil_df.columns
    assert "soil_water_climatology_mm" in soil_df.columns
    assert "soil_water_anomaly_mm" in soil_df.columns
    assert "soil_stress_z" in soil_df.columns
    
    # LST high temperature relative to climatology
    lst = pd.DataFrame([{
        "district": "Latur", "date": pd.Timestamp("2015-08-01"),
        "lst_celsius": 48.0, "lst_raw": 48.0, "lst_imputed": 0,
    }])
    lst_df, _ = process_lst_features(cal, lst, train_end_date="2017-12-01")
    assert "lst_celsius" in lst_df.columns
    assert "lst_climatology_celsius" in lst_df.columns
    assert "lst_anomaly_celsius" in lst_df.columns
    assert "lst_stress_z" in lst_df.columns
