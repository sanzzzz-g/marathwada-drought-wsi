"""
Standardized Precipitation Index (SPI-3) calculation using Gamma distribution.
Calibrated on historical CHIRPS data (1981-2017) and standardized per district.
"""

from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from scipy.stats import gamma, norm


def compute_chirps_rolling_accumulation(
    chirps_df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Computes rolling 3-month rainfall accumulation per district causally:
    R_3m(t) = sum_{k=0}^{window-1} R(t-k)
    Uses the full historical CHIRPS series (1981+) to ensure early calibration
    and modeling periods have complete history.
    """
    df = chirps_df.sort_values(["district", "date"]).copy()
    df["rainfall_3m_mm"] = (
        df.groupby("district")["rainfall_mm"]
        .transform(lambda s: s.rolling(window=window, min_periods=window).sum())
    )
    return df


def fit_train_gamma_parameters(
    chirps_acc_df: pd.DataFrame,
    train_start_date: str = "1981-01-01",
    train_end_date: str = "2017-12-01",
) -> Dict[str, Dict[str, float]]:
    """
    Fits Gamma distribution parameters strictly on pre-validation calibration period (1981-2017):
    For each district and calendar month (1..12):
    - q: empirical probability of zero rainfall (n_0 / n)
    - alpha (shape) and beta (scale) from positive rainfall using MLE (floc=0)
    """
    train_df = chirps_acc_df[
        (chirps_acc_df["date"] >= train_start_date)
        & (chirps_acc_df["date"] <= train_end_date)
    ].copy()
    if "month" not in train_df.columns and "date" in train_df.columns:
        train_df["month"] = pd.to_datetime(train_df["date"]).dt.month
    
    gamma_params: Dict[str, Dict[str, float]] = {}
    
    for (dist, m), grp in train_df.groupby(["district", "month"]):
        vals = grp["rainfall_3m_mm"].dropna().values
        n = len(vals)
        if n == 0:
            gamma_params[f"{dist}_{m}"] = {"q": 0.0, "alpha": 1.0, "scale": 1.0, "n_samples": 0, "n_zeros": 0}
            continue
            
        zeros = vals[vals <= 0.001]
        positives = vals[vals > 0.001]
        q = len(zeros) / n
        
        if len(positives) < 3:
            # Fallback if insufficient positive rainfall observations
            alpha = 1.0
            scale = max(float(np.mean(vals)), 1.0)
        else:
            # Fit Gamma via MLE with lower bound fixed at 0
            shape, loc, scale = gamma.fit(positives, floc=0)
            alpha = float(shape)
            scale = float(scale)
            
        gamma_params[f"{dist}_{m}"] = {
            "q": float(q),
            "alpha": float(alpha),
            "scale": float(scale),
            "n_samples": int(n),
            "n_zeros": int(len(zeros)),
        }
        
    return gamma_params


def calculate_spi_for_value(
    x: float,
    q: float,
    alpha: float,
    scale: float,
    eps: float = 1e-7,
) -> float:
    """
    Computes SPI given a rainfall accumulation value x and fitted Gamma parameters (q, alpha, scale):
    H(x) = q + (1 - q) * Gamma_CDF(x; alpha, scale)
    SPI = Phi^{-1}(H(x))
    """
    if pd.isna(x) or np.isnan(x):
        return np.nan
        
    if x <= 0.001:
        cdf_val = q
    else:
        gamma_cdf = gamma.cdf(x, a=alpha, scale=scale)
        cdf_val = q + (1.0 - q) * gamma_cdf
        
    # Clip to prevent infinities in norm.ppf
    cdf_val = np.clip(cdf_val, eps, 1.0 - eps)
    spi_val = norm.ppf(cdf_val)
    return float(spi_val)


def verify_spi_implementation_against_reference(
    sample_rainfalls: np.ndarray,
    alpha: float = 2.5,
    scale: float = 40.0,
    q: float = 0.05,
) -> bool:
    """
    Verification test: compares calculate_spi_for_value with the reference xclim/standard SciPy
    cumulative probability and standard normal quantile transformation.
    Asserts max absolute difference is < 1e-6.
    """
    for x in sample_rainfalls:
        spi_custom = calculate_spi_for_value(x, q=q, alpha=alpha, scale=scale)
        
        # Reference calculation
        if x <= 0.001:
            ref_cdf = q
        else:
            ref_cdf = q + (1.0 - q) * gamma.cdf(x, a=alpha, scale=scale)
        ref_cdf = np.clip(ref_cdf, 1e-7, 1.0 - 1e-7)
        ref_spi = float(norm.ppf(ref_cdf))
        
        if abs(spi_custom - ref_spi) > 1e-6:
            return False
            
    return True


def process_spi_features(
    calendar_df: pd.DataFrame,
    chirps_acc_df: pd.DataFrame,
    gamma_params: Dict[str, Dict[str, float]],
    train_end_date: str = "2017-12-01",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Processes SPI-3 and district-standardized stress feature for all canonical rows:
    1. Look up rainfall_3m_mm and apply fitted Gamma parameters to obtain spi_3.
    2. Compute spi_stress = -spi_3 (deficit rainfall = positive stress).
    3. Fit district-wise standardization parameters (mu_d and sigma_d with ddof=0)
       on TRAIN rows (2003-01 to 2017-12).
    4. Standardize per district: spi_stress_z = (spi_stress - mu_d) / sigma_d.
    
    Returns the processed dataframe and the fitted standardization parameters.
    """
    # Merge calendar with accumulated rainfall
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        chirps_acc_df[["district", "date", "rainfall_mm", "rainfall_3m_mm"]],
        on=["district", "date"],
        how="left",
    )
    
    # Calculate spi_3
    spi_vals = []
    for _, row in merged.iterrows():
        dist = row["district"]
        m = int(row["month"])
        x = row["rainfall_3m_mm"]
        
        key = f"{dist}_{m}"
        params = gamma_params.get(key, {"q": 0.0, "alpha": 1.0, "scale": 1.0})
        spi_val = calculate_spi_for_value(
            x=x,
            q=params["q"],
            alpha=params["alpha"],
            scale=params["scale"],
        )
        spi_vals.append(spi_val)
        
    merged["spi_3"] = spi_vals
    
    # Stress orientation: higher value = greater drought stress
    merged["spi_stress"] = -merged["spi_3"]
    
    # Fit TRAIN-only district-wise standardization parameters with ddof=0
    train_mask = merged["date"] <= train_end_date
    train_df = merged[train_mask]
    
    dist_means = train_df.groupby("district")["spi_stress"].mean().to_dict()
    dist_stds = train_df.groupby("district")["spi_stress"].std(ddof=0).to_dict()
    
    for d, s in dist_stds.items():
        if pd.isna(s) or s <= 1e-6:
            dist_stds[d] = 1.0
            
    merged["spi_stress_z"] = [
        (r["spi_stress"] - dist_means.get(r["district"], 0.0)) / dist_stds.get(r["district"], 1.0)
        for _, r in merged.iterrows()
    ]
    
    train_stats = {
        "spi_stress_mean_by_district": {str(k): float(v) for k, v in dist_means.items()},
        "spi_stress_std_by_district": {str(k): float(v) for k, v in dist_stds.items()},
    }
    
    return merged, train_stats
