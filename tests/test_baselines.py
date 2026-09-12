"""
Tests for baseline forecasting models and deterministic champion selection.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.baselines import (
    BaseForecaster,
    PersistenceForecaster,
    SeasonalPersistenceForecaster,
    WSIAutoregressiveForecaster,
    RidgeForecaster,
    XGBoostForecaster,
    load_sequence_arrays,
    assemble_prediction_records,
)


@pytest.fixture(scope="module")
def sequence_arrays():
    """Loads Step-5 sequence arrays and metadata."""
    data_dir = "data_model"
    assert os.path.exists(os.path.join(data_dir, "X_train.npy")), "data_model arrays missing! Run sequence_builder.py first."
    return load_sequence_arrays(data_dir=data_dir)


def test_persistence_forecaster_exact_values(sequence_arrays):
    """Verifies that PersistenceForecaster outputs [WSI_t, WSI_t, WSI_t] exactly."""
    X_val = sequence_arrays["X_val"]
    model = PersistenceForecaster()
    model.fit(sequence_arrays["X_train"], sequence_arrays["y_train"])
    preds = model.predict(X_val)

    assert preds.shape == (len(X_val), 3)
    # Check that all 3 columns are identical to origin WSI (mean of 6 features at step 11)
    origin_wsi = X_val[:, -1, :].mean(axis=-1)
    for h in range(3):
        assert np.allclose(preds[:, h], origin_wsi, atol=1e-6)


def test_seasonal_persistence_shape_and_values(sequence_arrays):
    """Verifies SeasonalPersistenceForecaster shape (N, 3) and lag-12 retrieval."""
    X_val = sequence_arrays["X_val"]
    model = SeasonalPersistenceForecaster()
    model.fit(sequence_arrays["X_train"], sequence_arrays["y_train"])
    preds = model.predict(X_val)

    assert preds.shape == (len(X_val), 3)
    # Default fallback maps lookback steps 0, 1, 2 to h1, h2, h3
    for h in range(3):
        assert np.allclose(preds[:, h], X_val[:, h, :].mean(axis=-1), atol=1e-6)


def test_wsi_autoregression_univariate_contract(sequence_arrays):
    """Verifies that WSIAutoregressiveForecaster consumes univariate (N, 12) inputs."""
    X_tr = sequence_arrays["X_train"]
    y_tr = sequence_arrays["y_train"]
    X_vl = sequence_arrays["X_val"]
    y_vl = sequence_arrays["y_val"]

    wsi_tr = X_tr.mean(axis=-1)
    wsi_vl = X_vl.mean(axis=-1)
    assert wsi_tr.shape == (len(X_tr), 12)

    model = WSIAutoregressiveForecaster(alphas=[0.1, 1.0, 10.0])
    model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl, wsi_train=wsi_tr, wsi_val=wsi_vl)

    assert model.is_fitted
    assert "alpha" in model.best_params_
    preds = model.predict(X_vl, wsi_input=wsi_vl)
    assert preds.shape == (len(X_vl), 3)
    assert not np.isnan(preds).any()


def test_ridge_multivariate_shape_and_refit(sequence_arrays):
    """Verifies RidgeForecaster flattened (72 -> 3) fitting, tuning, and refitting."""
    X_tr = sequence_arrays["X_train"]
    y_tr = sequence_arrays["y_train"]
    X_vl = sequence_arrays["X_val"]
    y_vl = sequence_arrays["y_val"]

    model = RidgeForecaster(alphas=[0.01, 1.0, 10.0])
    model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)

    assert model.is_fitted
    assert "alpha" in model.best_params_
    val_preds = model.predict(X_vl)
    assert val_preds.shape == (len(X_vl), 3)

    # Refit on train + val
    X_tr_vl = np.concatenate([X_tr, X_vl], axis=0)
    y_tr_vl = np.concatenate([y_tr, y_vl], axis=0)
    model.refit(X_tr_vl, y_tr_vl)
    test_preds = model.predict(sequence_arrays["X_test"])
    assert test_preds.shape == (len(sequence_arrays["X_test"]), 3)
    assert not np.isnan(test_preds).any()


def test_xgboost_decoupled_horizon_models(sequence_arrays):
    """Verifies that XGBoostForecaster builds 3 independent horizon regressors."""
    X_tr = sequence_arrays["X_train"][:100]  # Small slice for fast test execution
    y_tr = sequence_arrays["y_train"][:100]
    X_vl = sequence_arrays["X_val"][:50]
    y_vl = sequence_arrays["y_val"][:50]

    model = XGBoostForecaster()
    model.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)

    assert model.is_fitted
    assert len(model.models) == 3
    preds = model.predict(X_vl)
    assert preds.shape == (len(X_vl), 3)
    assert not np.isnan(preds).any()


def test_assemble_prediction_records_schema(sequence_arrays):
    """Verifies schema and row count of assemble_prediction_records."""
    meta = sequence_arrays["meta"]
    meta_val = meta[meta["split"] == "val"].reset_index(drop=True)
    n_samples = len(meta_val)

    dummy_preds = np.zeros((n_samples, 3), dtype=np.float32)
    y_actual = sequence_arrays["y_val"]

    records = assemble_prediction_records(
        model_name="TestModel",
        evaluation_type="official_validation",
        preds=dummy_preds,
        y_actual=y_actual,
        meta_subset=meta_val,
    )

    # Each sample generates 3 horizon rows
    assert len(records) == n_samples * 3
    first = records[0]
    expected_keys = [
        "evaluation_type", "model", "sequence_id", "district", "forecast_origin",
        "horizon", "actual_wsi", "predicted_wsi", "error", "absolute_error", "squared_error"
    ]
    for k in expected_keys:
        assert k in first, f"Missing key in prediction record: {k}"
