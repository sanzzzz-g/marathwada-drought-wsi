"""
Tests for evaluation metrics, statistical significance, and benchmark diagnostics.
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation import (
    calc_r2,
    calc_nrmse,
    calc_event_f1,
    diebold_mariano_test,
    moving_block_bootstrap_significance,
)


def test_metric_calculations_math():
    """Verifies that MAE, RMSE, R^2, and NRMSE match known values."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    assert calc_r2(y_true, y_pred) == 1.0
    assert calc_nrmse(y_true, y_pred) == 0.0

    # Introduce systematic error
    y_pred_err = y_true + 1.0
    assert np.isclose(calc_r2(y_true, y_pred_err), 0.2, atol=1e-5)
    # RMSE is 1.0, std of [1, 2, 3, 4] is sqrt(1.25) = 1.11803
    expected_nrmse = 1.0 / np.std(y_true)
    assert np.isclose(calc_nrmse(y_true, y_pred_err), expected_nrmse, atol=1e-4)


def test_event_f1_threshold_logic():
    """Verifies event precision, recall, and F1 calculations without class rounding."""
    y_true = np.array([0.2, 0.6, 1.4, 1.8], dtype=np.float32)
    y_pred = np.array([0.4, 0.7, 0.3, 1.6], dtype=np.float32)

    # Threshold 0.5:
    # Actual >= 0.5: [False, True, True, True] (3 events)
    # Pred   >= 0.5: [False, True, False, True] (2 events)
    # TP = 2 (index 1, 3), FP = 0, FN = 1 (index 2)
    # Precision = 2/2 = 1.0, Recall = 2/3 = 0.6667
    res = calc_event_f1(y_true, y_pred, threshold=0.5)
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 1
    assert res["precision"] == 1.0
    assert np.isclose(res["recall"], 2 / 3, atol=1e-4)
    expected_f1 = 2 * (1.0 * (2 / 3)) / (1.0 + (2 / 3))
    assert np.isclose(res["f1"], expected_f1, atol=1e-4)


def test_diebold_mariano_hac_lags():
    """Verifies Diebold-Mariano test execution across lags 0, 1, 2."""
    np.random.seed(42)
    n = 100
    e1 = np.random.normal(0, 1, n)
    e2 = np.random.normal(0, 1.5, n)  # e1 has smaller loss

    for h in [1, 2, 3]:
        stat, p_val = diebold_mariano_test(e1, e2, h=h, loss_type="absolute")
        assert isinstance(stat, float)
        assert isinstance(p_val, float)
        assert 0.0 <= p_val <= 1.0
        # e1 has smaller error, so mean(abs(e1) - abs(e2)) < 0 => stat < 0
        assert stat < 0


def test_moving_block_bootstrap_clusters_districts():
    """Verifies moving block bootstrap groups all 8 districts within consecutive monthly blocks."""
    # Synthetic dataset with 12 consecutive months, 8 districts, 3 horizons
    dates = pd.date_range("2021-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
    districts = [f"District_{i}" for i in range(8)]
    horizons = ["h1", "h2", "h3"]

    rows = []
    for d in dates:
        for dist in districts:
            for h in horizons:
                # Model A (Champion)
                rows.append({
                    "model": "ModelA",
                    "forecast_origin": d,
                    "district": dist,
                    "horizon": h,
                    "actual_wsi": 0.5,
                    "predicted_wsi": 0.4,
                    "absolute_error": 0.1,
                })
                # Model B (Candidate, higher error)
                rows.append({
                    "model": "ModelB",
                    "forecast_origin": d,
                    "district": dist,
                    "horizon": h,
                    "actual_wsi": 0.5,
                    "predicted_wsi": 0.2,
                    "absolute_error": 0.3,
                })

    df = pd.DataFrame(rows)
    res = moving_block_bootstrap_significance(
        df=df,
        candidate_model="ModelB",
        champion_model="ModelA",
        block_length=3,
        n_boot=100,
    )

    assert "observed_mae_diff" in res
    assert np.isclose(res["observed_mae_diff"], 0.2, atol=1e-4)
    assert res["ci_95_lower"] > 0  # Model B is significantly worse than Model A
    assert not res["candidate_is_better"]
