"""
Authoritative Water Stress Index (WSI) calculation and drought classification.

Formula:
  WSI = (spi_stress_z + soil_stress_z + ndvi_stress_z +
         lst_stress_z + groundwater_stress_z + surface_water_stress_z) / 6.0

Categories:
  WSI < -1.5         : Very Wet
  -1.5 <= WSI < -0.5 : Wet
  -0.5 <= WSI < 0.5  : Normal
  0.5 <= WSI < 1.5   : Moderate Stress
  WSI >= 1.5         : Severe Stress
"""

from typing import List
import numpy as np
import pandas as pd

WSI_COMPONENT_COLS: List[str] = [
    "spi_stress_z",
    "soil_stress_z",
    "ndvi_stress_z",
    "lst_stress_z",
    "groundwater_stress_z",
    "surface_water_stress_z",
]


def classify_wsi(wsi_val: float) -> str:
    """Classifies continuous WSI into drought category."""
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
    """Computes equal-weighted WSI from the 6 standardized stress components."""
    out_df = df.copy()

    for col in components:
        if col not in out_df.columns:
            raise KeyError(f"Required WSI component column '{col}' is missing.")
        if out_df[col].isna().any():
            nan_count = out_df[col].isna().sum()
            raise ValueError(f"WSI component '{col}' contains {nan_count} unexpected NaN values.")

    out_df["wsi"] = out_df[components].mean(axis=1)
    out_df["wsi_category"] = out_df["wsi"].apply(classify_wsi)

    # Numerical verification
    manual_sum = sum(out_df[col] for col in components) / float(len(components))
    max_error = (out_df["wsi"] - manual_sum).abs().max()
    if max_error > 1e-12:
        raise ValueError(f"WSI calculation precision violation: max error = {max_error}")

    return out_df
