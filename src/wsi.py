"""
src/wsi.py
==========
Module for calculating the authoritative Water Stress Index (WSI)
and categorizing drought severity according to project thresholds.

Formula:
  WSI = (spi_stress_z + soil_stress_z + ndvi_stress_z +
         lst_stress_z + groundwater_stress_z + surface_water_stress_z) / 6.0

Direction:
  Higher value indicates greater drought stress.
  All 6 components are strictly dimensionless standardized Z-scores.

Project Classification Thresholds:
  WSI < -1.5              : Very Wet
  -1.5 <= WSI < -0.5      : Wet
  -0.5 <= WSI <  0.5      : Normal
   0.5 <= WSI <  1.5      : Moderate Stress
  WSI >=  1.5             : Severe Stress
"""

from typing import List, Dict
import pandas as pd
import numpy as np

WSI_COMPONENT_COLS: List[str] = [
    "spi_stress_z",
    "soil_stress_z",
    "ndvi_stress_z",
    "lst_stress_z",
    "groundwater_stress_z",
    "surface_water_stress_z",
]


def classify_wsi(wsi_val: float) -> str:
    """Classifies a continuous WSI value into its discrete project drought category."""
    if pd.isna(wsi_val) or np.isnan(wsi_val):
        return "Unknown"
    if wsi_val < -1.5:
        return "Very Wet"
    elif wsi_val < -0.5:
        return "Wet"
    elif wsi_val < 0.5:
        return "Normal"
    elif wsi_val < 1.5:
        return "Moderate Stress"
    else:
        return "Severe Stress"


def calculate_wsi(
    df: pd.DataFrame,
    components: List[str] = WSI_COMPONENT_COLS,
) -> pd.DataFrame:
    """
    Computes equal-weighted WSI from the 6 standardized stress components:
    1. Verifies that all 6 components exist and contain no NaNs.
    2. Computes exact row mean across the 6 components.
    3. Categorizes each row into wsi_category.
    4. Mathematically validates that max absolute error is < 1e-12.
    """
    out_df = df.copy()
    
    # Check column presence
    for col in components:
        if col not in out_df.columns:
            raise KeyError(f"Required WSI component column '{col}' is missing from DataFrame.")
        if out_df[col].isna().any():
            nan_count = out_df[col].isna().sum()
            raise ValueError(f"WSI component '{col}' contains {nan_count} unexpected NaN values.")
            
    # Calculate exact unweighted mean
    out_df["wsi"] = out_df[components].mean(axis=1)
    
    # Classify
    out_df["wsi_category"] = out_df["wsi"].apply(classify_wsi)
    
    # Mathematical assertion: exact arithmetic verification
    manual_sum = (
        out_df[components[0]]
        + out_df[components[1]]
        + out_df[components[2]]
        + out_df[components[3]]
        + out_df[components[4]]
        + out_df[components[5]]
    ) / 6.0
    max_error = (out_df["wsi"] - manual_sum).abs().max()
    if max_error > 1e-12:
        raise ValueError(f"WSI calculation violates numerical precision tolerance: max error = {max_error}")
        
    return out_df
