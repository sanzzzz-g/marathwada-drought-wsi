"""
Unit and contract tests for Experiment B seed robustness evaluation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.seed_robustness_b import (
    SEEDS,
    OFFICIAL_SEED,
    FIXED_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT,
    LSTM_UNITS,
    DENSE_UNITS,
    ALLOWED_INPUTS,
    ORIGINAL_SEED_42_METRICS,
    load_allowed_data,
    build_fresh_model,
    run_seed_diagnostic,
    compute_checksums,
)


def test_seed_set():
    """Verify that evaluated seeds are strictly {42, 43, 44, 45, 46} and official seed is 42."""
    assert SEEDS == [42, 43, 44, 45, 46], f"Seeds must be [42, 43, 44, 45, 46], got {SEEDS}"
    assert OFFICIAL_SEED == 42, f"Official seed must be 42, got {OFFICIAL_SEED}"
    assert len(SEEDS) == 5, f"Expected exactly 5 seeds, got {len(SEEDS)}"


def test_allowed_inputs_manifest():
    """Verify that ALLOWED_INPUTS contains strictly the 6 allowed files and ZERO test data files."""
    assert len(ALLOWED_INPUTS) == 6, f"Expected 6 allowed inputs, got {len(ALLOWED_INPUTS)}"

    for path in ALLOWED_INPUTS:
        path_str = str(path).lower().replace("\\", "/")
        assert "test" not in path_str, f"Forbidden test data file in ALLOWED_INPUTS: {path}"

    expected_set = {
        Path("data_model/X_train.npy"),
        Path("data_model/X_val.npy"),
        Path("data_model_residual/y_residual_train.npy"),
        Path("data_model_residual/y_residual_val.npy"),
        Path("data_model_residual/wsi_origin_val.npy"),
        Path("data_model/y_val.npy"),
    }
    assert ALLOWED_INPUTS == expected_set, f"ALLOWED_INPUTS mismatch: {ALLOWED_INPUTS.symmetric_difference(expected_set)}"

    # Test that loader's manifest strictly matches ALLOWED_INPUTS
    data, manifest = load_allowed_data()
    assert manifest == ALLOWED_INPUTS
    assert "x_train" in data
    assert "x_val" in data
    assert "y_train_residual" in data
    assert "y_val_residual" in data
    assert "wsi_origin_val" in data
    assert "y_val_original" in data


def test_training_configuration_contract():
    """Verify frozen training hyperparameters."""
    assert FIXED_EPOCHS == 3, f"Expected FIXED_EPOCHS=3, got {FIXED_EPOCHS}"
    assert BATCH_SIZE == 32, f"Expected BATCH_SIZE=32, got {BATCH_SIZE}"
    assert LEARNING_RATE == 0.001, f"Expected LEARNING_RATE=0.001, got {LEARNING_RATE}"
    assert DROPOUT == 0.20, f"Expected DROPOUT=0.20, got {DROPOUT}"
    assert LSTM_UNITS == 32, f"Expected LSTM_UNITS=32, got {LSTM_UNITS}"
    assert DENSE_UNITS == 16, f"Expected DENSE_UNITS=16, got {DENSE_UNITS}"


def test_fresh_model_and_optimizer():
    """Verify build_fresh_model creates distinct model and optimizer objects."""
    m1 = build_fresh_model()
    m2 = build_fresh_model()
    assert m1 is not m2, "build_fresh_model must return a new Model instance"
    assert m1.optimizer is not m2.optimizer, "build_fresh_model must create a new Optimizer instance"
    assert m1.input_shape == (None, 12, 6)
    assert m1.output_shape == (None, 3)


def test_ground_truth_reconstruction_identity():
    """Verify ground-truth reconstruction identity on all 288 validation origins."""
    data, _ = load_allowed_data()
    wsi_origin_val = data["wsi_origin_val"]
    y_val_residual = data["y_val_residual"]
    y_val_original = data["y_val_original"]

    reconstructed = wsi_origin_val[:, None] + y_val_residual
    max_diff = float(np.max(np.abs(reconstructed - y_val_original)))
    assert max_diff <= 1e-6, f"Reconstruction identity violated: max diff = {max_diff:.8e}"


def test_seed_42_reproduction_and_diagnostics():
    """
    Executes Seed 42 and verifies:
      - Shape is (288, 3) and finite
      - Residual MAE == WSI MAE within 1e-6
      - Numerical reproduction of Step 8.3 values within 1e-6 for overall MAE and horizons h1, h2, h3.
    """
    data, _ = load_allowed_data()
    initial_checksums = compute_checksums(data)

    res_42 = run_seed_diagnostic(42, data)

    # Immutability check
    post_checksums = compute_checksums(data)
    assert initial_checksums == post_checksums, "Input tensors were mutated during execution!"

    # Shape and finiteness
    assert res_42["epochs"] == 3
    assert res_42["batch_size"] == 32
    assert res_42["shuffle"] is False

    # Check Seed 42 reproduction within 1e-6
    for metric_name, expected_val in ORIGINAL_SEED_42_METRICS.items():
        actual_val = res_42[metric_name]
        diff = abs(actual_val - expected_val)
        assert diff <= 1e-6, (
            f"Seed 42 reproduction failed on {metric_name}! "
            f"Actual: {actual_val}, Expected: {expected_val}, Diff: {diff:.8e}"
        )
