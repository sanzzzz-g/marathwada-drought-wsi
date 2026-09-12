"""
Tests for WSI calculation, equal component weighting, and discrete classification.
"""

import pytest
import pandas as pd
import numpy as np

from src.wsi import (
    calculate_wsi,
    classify_wsi,
    WSI_COMPONENT_COLS,
)


def test_wsi_equal_weighting_and_exact_math():
    """Verifies that WSI is exactly 1/6 sum of the 6 stress components."""
    df = pd.DataFrame([{
        "spi_stress_z": 1.2,
        "soil_stress_z": 1.5,
        "ndvi_stress_z": 0.9,
        "lst_stress_z": 1.8,
        "groundwater_stress_z": 2.1,
        "surface_water_stress_z": 0.9,
    }])
    
    expected_wsi = (1.2 + 1.5 + 0.9 + 1.8 + 2.1 + 0.9) / 6.0
    res = calculate_wsi(df)
    
    assert abs(res.iloc[0]["wsi"] - expected_wsi) < 1e-12
    assert res.iloc[0]["wsi_category"] == "Moderate Stress"  # 1.4 is [0.5, 1.5)


def test_wsi_categorical_thresholds():
    """Verifies all project classification boundary thresholds."""
    assert classify_wsi(-2.0) == "Very Wet"
    assert classify_wsi(-1.5) == "Wet"
    assert classify_wsi(-1.0) == "Wet"
    assert classify_wsi(-0.5) == "Normal"
    assert classify_wsi(0.0) == "Normal"
    assert classify_wsi(0.49) == "Normal"
    assert classify_wsi(0.5) == "Moderate Stress"
    assert classify_wsi(1.49) == "Moderate Stress"
    assert classify_wsi(1.5) == "Severe Stress"
    assert classify_wsi(3.0) == "Severe Stress"
