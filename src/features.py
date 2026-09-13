"""
Multi-tier hydroclimatic anomaly features for soil moisture, vegetation condition,
land surface temperature, and surface water extent.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np


def process_soil_water_features(
    calendar_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    train_end_date: str = "2017-12-01",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds ERA5-Land soil moisture features:
    - soil_water_storage_mm (RAW 0-289cm)
    - soil_water_climatology_mm (TRAIN district-month mean)
    - soil_water_anomaly_mm = raw - climatology (physical deficit/surplus in mm)
    - soil_stress_z = -anomaly / sigma_train(district)  (less moisture = higher stress)
    """
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        era5_df[[
            "district", "date",
            "soil_water_layer_1", "soil_water_layer_2", "soil_water_layer_3", "soil_water_layer_4",
            "soil_water_storage_mm_0_289cm",
        ]],
        on=["district", "date"],
        how="left",
    )
    
    # Rename column to canonical clean name
    merged["soil_water_storage_mm"] = merged["soil_water_storage_mm_0_289cm"]
    
    train_mask = merged["date"] <= train_end_date
    train_df = merged[train_mask]
    
    # 1. Fit TRAIN district-month climatology
    clim_lookup = train_df.groupby(["district", "month"])["soil_water_storage_mm"].mean().to_dict()
    
    merged["soil_water_climatology_mm"] = [
        clim_lookup.get((r["district"], int(r["month"])), np.nan)
        for _, r in merged.iterrows()
    ]
    
    # 2. Physical anomaly in mm
    merged["soil_water_anomaly_mm"] = (
        merged["soil_water_storage_mm"] - merged["soil_water_climatology_mm"]
    )
    
    # 3. Fit TRAIN standard deviation of physical anomaly per district (ddof=0)
    train_anoms = merged[train_mask]
    std_lookup = train_anoms.groupby("district")["soil_water_anomaly_mm"].std(ddof=0).to_dict()
    
    # Guard against zero std
    for d, s in std_lookup.items():
        if pd.isna(s) or s <= 1e-6:
            std_lookup[d] = 1.0
            
    # 4. Standardize and invert for stress orientation (negative anomaly = positive stress)
    merged["soil_stress_z"] = [
        -(r["soil_water_anomaly_mm"]) / std_lookup.get(r["district"], 1.0)
        for _, r in merged.iterrows()
    ]
    
    train_stats = {
        "soil_water_climatology_mm": {f"{k[0]}_{k[1]}": float(v) for k, v in clim_lookup.items()},
        "soil_water_anomaly_std_mm": {str(k): float(v) for k, v in std_lookup.items()},
    }
    
    cols = [
        "district", "date",
        "soil_water_layer_1", "soil_water_layer_2", "soil_water_layer_3", "soil_water_layer_4",
        "soil_water_storage_mm", "soil_water_climatology_mm", "soil_water_anomaly_mm", "soil_stress_z",
    ]
    return merged[cols], train_stats


def process_ndvi_features(
    calendar_df: pd.DataFrame,
    ndvi_df: pd.DataFrame,
    train_end_date: str = "2017-12-01",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds MOD13Q1 NDVI features:
    - ndvi (RAW / imputed)
    - ndvi_climatology (TRAIN district-month mean)
    - ndvi_anomaly = ndvi - climatology (physical greenness deficit/surplus)
    - ndvi_stress_z = -anomaly / sigma_train(district)  (less vegetation = higher stress)
    """
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        ndvi_df[["district", "date", "ndvi", "ndvi_raw", "ndvi_artifact_masked", "ndvi_imputed"]],
        on=["district", "date"],
        how="left",
    )
    
    train_mask = merged["date"] <= train_end_date
    train_df = merged[train_mask]
    
    # 1. Fit TRAIN district-month climatology
    clim_lookup = train_df.groupby(["district", "month"])["ndvi"].mean().to_dict()
    
    merged["ndvi_climatology"] = [
        clim_lookup.get((r["district"], int(r["month"])), np.nan)
        for _, r in merged.iterrows()
    ]
    
    # 2. Physical anomaly
    merged["ndvi_anomaly"] = merged["ndvi"] - merged["ndvi_climatology"]
    
    # 3. Fit TRAIN standard deviation of anomaly per district (ddof=0)
    train_anoms = merged[train_mask]
    std_lookup = train_anoms.groupby("district")["ndvi_anomaly"].std(ddof=0).to_dict()
    
    for d, s in std_lookup.items():
        if pd.isna(s) or s <= 1e-6:
            std_lookup[d] = 1.0
            
    # 4. Standardize and invert for stress orientation (negative NDVI anomaly = positive stress)
    merged["ndvi_stress_z"] = [
        -(r["ndvi_anomaly"]) / std_lookup.get(r["district"], 1.0)
        for _, r in merged.iterrows()
    ]
    
    train_stats = {
        "ndvi_climatology": {f"{k[0]}_{k[1]}": float(v) for k, v in clim_lookup.items()},
        "ndvi_anomaly_std": {str(k): float(v) for k, v in std_lookup.items()},
    }
    
    cols = [
        "district", "date", "ndvi_raw", "ndvi", "ndvi_imputed", "ndvi_artifact_masked",
        "ndvi_climatology", "ndvi_anomaly", "ndvi_stress_z",
    ]
    return merged[cols], train_stats


def process_lst_features(
    calendar_df: pd.DataFrame,
    lst_df: pd.DataFrame,
    train_end_date: str = "2017-12-01",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds MOD11A2 LST features:
    - lst_celsius (RAW / imputed)
    - lst_climatology_celsius (TRAIN district-month mean)
    - lst_anomaly_celsius = lst - climatology (physical temperature anomaly in C)
    - lst_stress_z = +anomaly / sigma_train(district)  (higher temperature = higher stress)
    """
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        lst_df[["district", "date", "lst_celsius", "lst_raw", "lst_imputed"]],
        on=["district", "date"],
        how="left",
    )
    
    train_mask = merged["date"] <= train_end_date
    train_df = merged[train_mask]
    
    # 1. Fit TRAIN district-month climatology
    clim_lookup = train_df.groupby(["district", "month"])["lst_celsius"].mean().to_dict()
    
    merged["lst_climatology_celsius"] = [
        clim_lookup.get((r["district"], int(r["month"])), np.nan)
        for _, r in merged.iterrows()
    ]
    
    # 2. Physical anomaly in Celsius
    merged["lst_anomaly_celsius"] = (
        merged["lst_celsius"] - merged["lst_climatology_celsius"]
    )
    
    # 3. Fit TRAIN standard deviation of anomaly per district (ddof=0)
    train_anoms = merged[train_mask]
    std_lookup = train_anoms.groupby("district")["lst_anomaly_celsius"].std(ddof=0).to_dict()
    
    for d, s in std_lookup.items():
        if pd.isna(s) or s <= 1e-6:
            std_lookup[d] = 1.0
            
    # 4. Standardize (higher LST = positive stress)
    merged["lst_stress_z"] = [
        +(r["lst_anomaly_celsius"]) / std_lookup.get(r["district"], 1.0)
        for _, r in merged.iterrows()
    ]
    
    train_stats = {
        "lst_climatology_celsius": {f"{k[0]}_{k[1]}": float(v) for k, v in clim_lookup.items()},
        "lst_anomaly_std_celsius": {str(k): float(v) for k, v in std_lookup.items()},
    }
    
    cols = [
        "district", "date", "lst_raw", "lst_celsius", "lst_imputed",
        "lst_climatology_celsius", "lst_anomaly_celsius", "lst_stress_z",
    ]
    return merged[cols], train_stats


def process_surface_water_features(
    calendar_df: pd.DataFrame,
    gwp_df: pd.DataFrame,
    train_end_date: str = "2017-12-01",
    use_lag_1: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds Global WaterPack (GWP) surface water features:
    - surface_water_frequency (RAW fraction [0, 1])
    - surface_water_climatology (TRAIN district-month mean)
    - surface_water_anomaly = frequency - climatology (physical water fraction anomaly)
    - surface_water_stress_z = -anomaly / sigma_train(district)  (less water = higher stress)
    
    Data availability rule:
    If use_lag_1 is True, uses GWP(t-1) as the input feature to guarantee zero lookahead.
    """
    df = gwp_df.sort_values(["district", "date"]).copy()
    
    if use_lag_1:
        # Causal lag-1 feature
        df["surface_water_frequency_raw"] = df["surface_water_frequency"]
        df["surface_water_frequency"] = (
            df.groupby("district")["surface_water_frequency_raw"].shift(1)
        )
    else:
        df["surface_water_frequency_raw"] = df["surface_water_frequency"]
        
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        df[["district", "date", "surface_water_frequency", "surface_water_frequency_raw", "valid_pixels"]],
        on=["district", "date"],
        how="left",
    )
    
    # If lag-1 caused first row (2003-01) to be NaN, backfill from Jan 2003 raw
    if merged["surface_water_frequency"].isna().any():
        merged["surface_water_frequency"] = merged["surface_water_frequency"].fillna(
            merged["surface_water_frequency_raw"]
        )
        
    train_mask = merged["date"] <= train_end_date
    train_df = merged[train_mask]
    
    # 1. Fit TRAIN district-month climatology
    clim_lookup = train_df.groupby(["district", "month"])["surface_water_frequency"].mean().to_dict()
    
    merged["surface_water_climatology"] = [
        clim_lookup.get((r["district"], int(r["month"])), np.nan)
        for _, r in merged.iterrows()
    ]
    
    # 2. Physical anomaly
    merged["surface_water_anomaly"] = (
        merged["surface_water_frequency"] - merged["surface_water_climatology"]
    )
    
    # 3. Fit TRAIN standard deviation of anomaly per district (ddof=0)
    train_anoms = merged[train_mask]
    std_lookup = train_anoms.groupby("district")["surface_water_anomaly"].std(ddof=0).to_dict()
    
    for d, s in std_lookup.items():
        if pd.isna(s) or s <= 1e-6:
            std_lookup[d] = 1.0
            
    # 4. Standardize and invert for stress orientation (less water = positive stress)
    merged["surface_water_stress_z"] = [
        -(r["surface_water_anomaly"]) / std_lookup.get(r["district"], 1.0)
        for _, r in merged.iterrows()
    ]
    
    train_stats = {
        "surface_water_climatology": {f"{k[0]}_{k[1]}": float(v) for k, v in clim_lookup.items()},
        "surface_water_anomaly_std": {str(k): float(v) for k, v in std_lookup.items()},
        "gwp_use_lag_1": use_lag_1,
    }
    
    cols = [
        "district", "date", "surface_water_frequency_raw", "surface_water_frequency", "valid_pixels",
        "surface_water_climatology", "surface_water_anomaly", "surface_water_stress_z",
    ]
    return merged[cols], train_stats
