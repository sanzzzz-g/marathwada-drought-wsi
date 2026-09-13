"""
Source-specific Quality Control (QC), artifact masking, and train-only imputation.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np

PARBHANI_JULY_2022_ARTIFACT_VALUE = 0.122442


def apply_ndvi_source_qc(
    ndvi_df: pd.DataFrame,
    artifact_threshold: float = PARBHANI_JULY_2022_ARTIFACT_VALUE,
    tolerance: float = 1e-4,
) -> pd.DataFrame:
    """
    Applies source-specific QC to MOD13Q1 NDVI:
    - Preserves ndvi_raw.
    - Detects the unflagged cloud contamination artifact in Parbhani July 2022 (value ~0.122442).
    - Sets ndvi to NaN for the artifact observation.
    - Sets ndvi_artifact_masked flag (1 if masked, 0 otherwise).
    """
    df = ndvi_df.copy()
    if "ndvi_raw" not in df.columns:
        df["ndvi_raw"] = df["ndvi"]
        
    df["ndvi_artifact_masked"] = 0
    
    # Mask Parbhani July 2022 artifact
    mask = (
        (df["district"] == "Parbhani")
        & (df["year"] == 2022)
        & (df["month"] == 7)
        & ((df["ndvi_raw"] - artifact_threshold).abs() < tolerance)
    )
    if mask.any():
        df.loc[mask, "ndvi"] = np.nan
        df.loc[mask, "ndvi_artifact_masked"] = 1
        
    return df


def fit_train_imputation_stats(
    df: pd.DataFrame,
    val_col: str,
    train_end_date: str = "2017-12-01",
) -> Dict[str, Dict]:
    """
    Computes TRAIN-only hierarchical imputation statistics for a feature:
    Level 1: (district, month) median
    Level 2: district median (fallback)
    Level 3: regional month median (fallback)
    Level 4: global train median (ultimate fallback)
    """
    train_df = df[df["date"] <= train_end_date].copy()
    if "month" not in train_df.columns and "date" in train_df.columns:
        train_df["month"] = pd.to_datetime(train_df["date"]).dt.month
    valid_train = train_df.dropna(subset=[val_col])
    
    # 1. District-Month median
    dist_month_med = (
        valid_train.groupby(["district", "month"])[val_col].median().to_dict()
    )
    # 2. District median
    dist_med = valid_train.groupby("district")[val_col].median().to_dict()
    # 3. Month median across all districts
    month_med = valid_train.groupby("month")[val_col].median().to_dict()
    # 4. Global train median
    global_med = float(valid_train[val_col].median())
    
    return {
        "district_month_median": {f"{k[0]}_{k[1]}": float(v) for k, v in dist_month_med.items()},
        "district_median": {str(k): float(v) for k, v in dist_med.items()},
        "month_median": {int(k): float(v) for k, v in month_med.items()},
        "global_median": global_med,
    }


def impute_missing_with_train_stats(
    df: pd.DataFrame,
    val_col: str,
    impute_stats: Dict[str, Dict],
    flag_col_name: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Applies TRAIN-fitted imputation statistics to the entire series (TRAIN, VAL, TEST).
    Sets flag_col_name to 1 if imputed, 0 otherwise.
    Returns the imputed dataframe and count of imputed cells.
    """
    df = df.copy()
    df[flag_col_name] = 0
    
    missing_mask = df[val_col].isna()
    imputed_count = int(missing_mask.sum())
    
    if imputed_count > 0:
        df.loc[missing_mask, flag_col_name] = 1
        
        dm_dict = impute_stats["district_month_median"]
        d_dict = impute_stats["district_median"]
        m_dict = impute_stats["month_median"]
        g_med = impute_stats["global_median"]
        
        for idx in df[missing_mask].index:
            dist = df.loc[idx, "district"]
            m = int(df.loc[idx, "month"])
            dm_key = f"{dist}_{m}"
            
            # Hierarchical lookup
            if dm_key in dm_dict and not np.isnan(dm_dict[dm_key]):
                replacement = dm_dict[dm_key]
            elif dist in d_dict and not np.isnan(d_dict[dist]):
                replacement = d_dict[dist]
            elif m in m_dict and not np.isnan(m_dict[m]):
                replacement = m_dict[m]
            else:
                replacement = g_med
                
            df.loc[idx, val_col] = replacement
            
    return df, imputed_count
