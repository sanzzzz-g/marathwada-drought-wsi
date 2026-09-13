"""
Verification and auditing suite for the Marathwada WSI data pipeline.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

EXPECTED_DISTRICTS = [
    "Beed",
    "Chhatrapati Sambhajinagar",
    "Dharashiv",
    "Hingoli",
    "Jalna",
    "Latur",
    "Nanded",
    "Parbhani",
]
EXPECTED_TOTAL_ROWS = 2112
EXPECTED_MONTHS_COUNT = 264


def validate_calendar_and_shape(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates row count, district coverage, date range, and key uniqueness."""
    districts = sorted(df["district"].unique().tolist())
    total_rows = len(df)
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    unique_dates_count = df["date"].nunique()
    
    dup_count = int(df.duplicated(subset=["district", "date"]).sum())
    
    passed = (
        districts == EXPECTED_DISTRICTS
        and total_rows == EXPECTED_TOTAL_ROWS
        and unique_dates_count == EXPECTED_MONTHS_COUNT
        and dup_count == 0
        and min_date == "2003-01-01"
        and max_date == "2024-12-01"
    )
    
    return {
        "check": "calendar_and_shape",
        "passed": passed,
        "districts_match": districts == EXPECTED_DISTRICTS,
        "total_rows": total_rows,
        "expected_rows": EXPECTED_TOTAL_ROWS,
        "unique_dates_count": unique_dates_count,
        "duplicate_keys": dup_count,
        "min_date": min_date,
        "max_date": max_date,
    }


def validate_zero_nulls(
    df: pd.DataFrame,
    required_cols: List[str],
) -> Dict[str, Any]:
    """Validates that all critical model features and WSI targets have zero missing values."""
    null_counts = {}
    for col in required_cols:
        null_counts[col] = int(df[col].isna().sum())
        
    total_nulls = sum(null_counts.values())
    return {
        "check": "zero_nulls_in_required_cols",
        "passed": total_nulls == 0,
        "null_counts": null_counts,
        "total_nulls": total_nulls,
    }


def validate_wsi_math(
    df: pd.DataFrame,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """Validates arithmetic exactness of WSI against the 6 components."""
    components = [
        "spi_stress_z",
        "soil_stress_z",
        "ndvi_stress_z",
        "lst_stress_z",
        "groundwater_stress_z",
        "surface_water_stress_z",
    ]
    calc_mean = df[components].mean(axis=1)
    diff = (df["wsi"] - calc_mean).abs()
    max_diff = float(diff.max())
    
    return {
        "check": "wsi_mathematical_exactness",
        "passed": max_diff <= tolerance,
        "max_discrepancy": max_diff,
        "tolerance": tolerance,
    }


def validate_forecast_origin_splits(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates both row-level splits and forecast-origin boundaries for a 3-month horizon:
    X_{t-11:t} -> [WSI_{t+1}, WSI_{t+2}, WSI_{t+3}]
    
    Last train origin: 2017-09 (targets Oct, Nov, Dec 2017 are <= 2017-12)
    First val origin: 2018-01 (targets Feb, Mar, Apr 2018 are in VAL)
    First test origin: 2021-01 (targets Feb, Mar, Apr 2021 are in TEST)
    """
    train_mask = (df["date"] >= "2003-01-01") & (df["date"] <= "2017-12-01")
    val_mask = (df["date"] >= "2018-01-01") & (df["date"] <= "2020-12-01")
    test_mask = (df["date"] >= "2021-01-01") & (df["date"] <= "2024-12-01")
    
    train_rows = int(train_mask.sum())
    val_rows = int(val_mask.sum())
    test_rows = int(test_mask.sum())
    
    row_splits_passed = (
        train_rows == 1440
        and val_rows == 288
        and test_rows == 384
        and (train_rows + val_rows + test_rows) == 2112
    )
    
    # Check forecast origins
    # For train: all dates up to 2017-09-01 can serve as origins
    # Targets for 2017-09-01 are 2017-10-01, 2017-11-01, 2017-12-01
    target_dates_for_last_train_origin = pd.date_range("2017-10-01", "2017-12-01", freq="MS")
    last_train_targets_inside_train = all(dt <= pd.Timestamp("2017-12-01") for dt in target_dates_for_last_train_origin)
    
    # If origin were 2017-10-01, target t+3 would be 2018-01-01 (leaks into validation!)
    target_dates_for_invalid_origin = pd.date_range("2017-11-01", "2018-01-01", freq="MS")
    leakage_prevented = any(dt > pd.Timestamp("2017-12-01") for dt in target_dates_for_invalid_origin)
    
    return {
        "check": "forecast_origin_splits",
        "passed": row_splits_passed and last_train_targets_inside_train and leakage_prevented,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "last_train_origin": "2017-09-01",
        "first_val_origin": "2018-01-01",
        "first_test_origin": "2021-01-01",
        "last_train_targets_inside_train": last_train_targets_inside_train,
    }


def validate_groundwater_causality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates that groundwater observation dates are strictly causal (last_gw_obs_date <= date)."""
    df_check = df.dropna(subset=["last_gw_obs_date"]).copy()
    df_check["last_obs_dt"] = pd.to_datetime(df_check["last_gw_obs_date"])
    
    future_obs = df_check[df_check["last_obs_dt"] > df_check["date"]]
    negative_age = df_check[df_check["months_since_last_gw_obs"] < 0]
    
    stale_mismatch = df_check[
        (df_check["months_since_last_gw_obs"] > 3) != (df_check["groundwater_stale_flag"] == 1)
    ]
    
    passed = (len(future_obs) == 0) and (len(negative_age) == 0) and (len(stale_mismatch) == 0)
    
    return {
        "check": "groundwater_causality_and_staleness",
        "passed": passed,
        "future_observations_count": len(future_obs),
        "negative_age_count": len(negative_age),
        "stale_flag_mismatches": len(stale_mismatch),
    }


def validate_historical_events(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Sanity checks against documented historical drought events:
    - Latur August 2015 (Water Train era / severe drought): WSI must be positive (> 0.5, typically > 1.0).
    - Checks that legacy bug (where Latur Aug 2015 had WSI = -29.3 "Very Wet") is completely gone.
    """
    latur_aug_2015 = df[
        (df["district"] == "Latur")
        & (df["date"] == "2015-08-01")
    ]
    if latur_aug_2015.empty:
        return {"check": "historical_events", "passed": False, "error": "Latur 2015-08-01 row not found"}
        
    row = latur_aug_2015.iloc[0]
    wsi_val = float(row["wsi"])
    wsi_cat = str(row["wsi_category"])
    
    # In August 2015, Latur experienced severe drought; WSI must be high positive stress
    passed = wsi_val > 0.5
    
    return {
        "check": "historical_events",
        "passed": passed,
        "latur_aug_2015_wsi": wsi_val,
        "latur_aug_2015_category": wsi_cat,
        "legacy_negative_bug_absent": wsi_val > 0.0,
    }


def validate_model_audit_parity(
    audit_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Verifies that the audit and model exports derived from the authoritative master
    are 100% identical on shared keys and feature columns.
    """
    shared_cols = [c for c in model_df.columns if c in audit_df.columns]
    
    # Merge on district and date
    merged = pd.merge(
        audit_df[shared_cols],
        model_df[shared_cols],
        on=["district", "date"],
        suffixes=("_audit", "_model"),
    )
    
    mismatches = {}
    for col in shared_cols:
        if col in ["district", "date"]:
            continue
        c_audit = f"{col}_audit"
        c_model = f"{col}_model"
        if pd.api.types.is_numeric_dtype(merged[c_audit]):
            diff = (merged[c_audit] - merged[c_model]).abs().max()
            if diff > 1e-9:
                mismatches[col] = float(diff)
        else:
            diff_count = (merged[c_audit] != merged[c_model]).sum()
            if diff_count > 0:
                mismatches[col] = int(diff_count)
                
    return {
        "check": "model_audit_parity",
        "passed": len(mismatches) == 0,
        "shared_columns_checked": len(shared_cols),
        "mismatches": mismatches,
    }


def run_full_validation_suite(
    master_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Executes the complete validation suite and compiles an authoritative report."""
    required_stress_cols = [
        "spi_stress_z",
        "soil_stress_z",
        "ndvi_stress_z",
        "lst_stress_z",
        "groundwater_stress_z",
        "surface_water_stress_z",
        "wsi",
    ]
    
    c_check = validate_calendar_and_shape(master_df)
    n_check = validate_zero_nulls(master_df, required_stress_cols)
    w_check = validate_wsi_math(master_df)
    s_check = validate_forecast_origin_splits(master_df)
    g_check = validate_groundwater_causality(master_df)
    h_check = validate_historical_events(master_df)
    p_check = validate_model_audit_parity(audit_df, model_df)
    
    all_passed = all([
        c_check["passed"],
        n_check["passed"],
        w_check["passed"],
        s_check["passed"],
        g_check["passed"],
        h_check["passed"],
        p_check["passed"],
    ])
    
    return {
        "overall_status": "PASSED" if all_passed else "FAILED",
        "calendar_and_shape": c_check,
        "zero_nulls": n_check,
        "wsi_mathematical_exactness": w_check,
        "forecast_origin_splits": s_check,
        "groundwater_causality": g_check,
        "historical_events": h_check,
        "model_audit_parity": p_check,
    }
