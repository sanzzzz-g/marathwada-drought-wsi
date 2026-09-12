"""
Tests for SPI-3 calculation, Gamma fitting, stress inversion, and standardization.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from src.spi import (
    calculate_spi_for_value,
    verify_spi_implementation_against_reference,
    fit_train_gamma_parameters,
    compute_chirps_rolling_accumulation,
    process_spi_features,
)


def test_spi_reference_implementation_equivalence():
    """Verifies that calculate_spi_for_value matches the standard SciPy/xclim reference formulation."""
    sample_rainfalls = np.array([0.0, 2.5, 15.0, 45.0, 100.0, 250.0, 400.0])
    is_valid = verify_spi_implementation_against_reference(
        sample_rainfalls, alpha=2.0, scale=35.0, q=0.08
    )
    assert is_valid, "Custom SPI implementation diverged from reference implementation!"


def test_spi_stress_direction():
    """
    Verifies that lower rainfall accumulation corresponds to lower SPI (meteorological drought)
    and higher spi_stress_z (drought stress).
    """
    # Low rainfall vs High rainfall
    spi_drought = calculate_spi_for_value(x=5.0, q=0.01, alpha=2.0, scale=50.0)
    spi_wet = calculate_spi_for_value(x=200.0, q=0.01, alpha=2.0, scale=50.0)
    
    # Drought rainfall should have negative SPI
    assert spi_drought < 0
    # Wet rainfall should have positive SPI
    assert spi_wet > 0
    
    # Drought stress is inverted: -SPI
    stress_drought = -spi_drought
    stress_wet = -spi_wet
    assert stress_drought > stress_wet, "Lower rainfall must produce higher drought stress!"


def test_rolling_accumulation_causality():
    """Verifies that rolling 3-month rainfall accumulation is strictly causal (sum of t, t-1, t-2)."""
    dates = pd.date_range("2010-01-01", "2010-05-01", freq="MS")
    df = pd.DataFrame({
        "district": ["Beed"] * 5,
        "date": dates,
        "rainfall_mm": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    
    res = compute_chirps_rolling_accumulation(df, window=3)
    
    # March (idx 2): 10 + 20 + 30 = 60
    assert abs(res.loc[2, "rainfall_3m_mm"] - 60.0) < 1e-6
    # April (idx 3): 20 + 30 + 40 = 90
    assert abs(res.loc[3, "rainfall_3m_mm"] - 90.0) < 1e-6
    # May (idx 4): 30 + 40 + 50 = 120
    assert abs(res.loc[4, "rainfall_3m_mm"] - 120.0) < 1e-6


def test_spi_district_wise_standardization_ddof0():
    """Verifies that spi_stress_z is standardized per district with ddof=0 to exact mean=0 and std=1 on TRAIN."""
    dates = pd.date_range("2010-01-01", "2015-12-01", freq="MS")
    cal = pd.DataFrame({
        "district": ["Beed"] * len(dates) + ["Latur"] * len(dates),
        "date": list(dates) * 2,
        "year": list(dates.year) * 2,
        "month": list(dates.month) * 2,
    })
    
    chirps_acc = cal.copy()
    # Distinct rainfall series per district
    chirps_acc["rainfall_mm"] = np.random.RandomState(42).uniform(10, 100, len(chirps_acc))
    chirps_acc["rainfall_3m_mm"] = chirps_acc["rainfall_mm"] * 3
    
    gamma_params = {
        f"Beed_{m}": {"q": 0.0, "alpha": 2.0, "scale": 30.0} for m in range(1, 13)
    }
    gamma_params.update({
        f"Latur_{m}": {"q": 0.0, "alpha": 2.5, "scale": 25.0} for m in range(1, 13)
    })
    
    res_df, stats = process_spi_features(
        calendar_df=cal,
        chirps_acc_df=chirps_acc,
        gamma_params=gamma_params,
        train_end_date="2015-12-01",
    )
    
    # Check that both Beed and Latur have mean=0 and std=1 on train
    for dist in ["Beed", "Latur"]:
        d_df = res_df[res_df["district"] == dist]
        assert abs(d_df["spi_stress_z"].mean()) < 1e-10
        assert abs(d_df["spi_stress_z"].std(ddof=0) - 1.0) < 1e-10
