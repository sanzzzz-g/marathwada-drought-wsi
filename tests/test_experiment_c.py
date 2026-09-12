"""
Unit and contract tests for Experiment C (Residual LSTM + explicit current WSI state).
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from src.experiment_c import (
    SEEDS,
    OFFICIAL_SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT,
    LSTM_UNITS,
    DENSE_UNITS,
    EXPECTED_TRAINABLE_PARAMS,
    ALLOWED_INPUTS,
    load_allowed_data,
    build_functional_c_model,
    compute_checksums,
    evaluate_predictions,
)


def test_seed_and_contract_constants():
    """Verify seed set and training hyperparameters."""
    assert SEEDS == [42, 43, 44, 45, 46]
    assert OFFICIAL_SEED == 42
    assert BATCH_SIZE == 32
    assert LEARNING_RATE == 0.001
    assert DROPOUT == 0.20
    assert LSTM_UNITS == 32
    assert DENSE_UNITS == 16
    assert EXPECTED_TRAINABLE_PARAMS == 5587


def test_allowed_inputs_manifest():
    """Verify that ALLOWED_INPUTS contains strictly the 7 allowed files and ZERO test data files."""
    assert len(ALLOWED_INPUTS) == 7, f"Expected 7 allowed inputs, got {len(ALLOWED_INPUTS)}"

    for path in ALLOWED_INPUTS:
        path_str = str(path).lower().replace("\\", "/")
        assert "test" not in path_str, f"Forbidden test data file in ALLOWED_INPUTS: {path}"

    expected_set = {
        Path("data_model/X_train.npy"),
        Path("data_model/X_val.npy"),
        Path("data_model/y_val.npy"),
        Path("data_model_residual/y_residual_train.npy"),
        Path("data_model_residual/y_residual_val.npy"),
        Path("data_model_residual/wsi_origin_train.npy"),
        Path("data_model_residual/wsi_origin_val.npy"),
    }
    assert ALLOWED_INPUTS == expected_set, f"ALLOWED_INPUTS mismatch: {ALLOWED_INPUTS.symmetric_difference(expected_set)}"

    data, manifest = load_allowed_data()
    assert manifest == ALLOWED_INPUTS
    assert "x_train" in data
    assert "x_val" in data
    assert "y_val_original" in data
    assert "y_train_residual" in data
    assert "y_val_residual" in data
    assert "wsi_origin_train" in data
    assert "wsi_origin_val" in data


def test_parameter_count_and_architecture():
    """
    Verifies the Functional API architecture and exact parameter count:
      LSTM(32) on input_dim=6: 4 * (6*32 + 32*32 + 32) = 4,992
      Dense(16) on concat=33:   33 * 16 + 16           =   544
      Dense(3) on 16:          16 * 3 + 3             =    51
      Total:                   4,992 + 544 + 51        = 5,587
    """
    model = build_functional_c_model()

    # Verify input layers
    assert len(model.inputs) == 2, f"Expected 2 inputs, got {len(model.inputs)}"
    input_shapes = {inp.name.split(":")[0]: tuple(inp.shape) for inp in model.inputs}
    assert (None, 12, 6) in input_shapes.values(), f"Missing (None, 12, 6) sequence input: {input_shapes}"
    assert (None, 1) in input_shapes.values(), f"Missing (None, 1) scalar WSI input: {input_shapes}"

    # Verify output layer
    assert model.output_shape == (None, 3)

    # Verify concat dimension
    concat_layer = next(l for l in model.layers if "concat" in l.name.lower())
    assert tuple(concat_layer.output.shape) == (None, 33), f"Expected concat shape (None, 33), got {concat_layer.output.shape}"

    # Verify exact trainable parameters
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    assert trainable_params == EXPECTED_TRAINABLE_PARAMS, (
        f"Trainable parameters mismatch: got {trainable_params}, expected {EXPECTED_TRAINABLE_PARAMS}"
    )


def test_ground_truth_reconstruction_identity():
    """Verify target reconstruction identity on validation ground truth."""
    data, _ = load_allowed_data()
    reconstructed = data["wsi_origin_val"][:, None] + data["y_val_residual"]
    max_diff = float(np.max(np.abs(reconstructed - data["y_val_original"])))
    assert max_diff <= 1e-6, f"Ground truth reconstruction identity violated: max diff = {max_diff:.8e}"


def test_fresh_model_and_optimizer():
    """Verify build_functional_c_model creates independent instances."""
    m1 = build_functional_c_model()
    m2 = build_functional_c_model()
    assert m1 is not m2
    assert m1.optimizer is not m2.optimizer


def test_input_arrays_immutability():
    """Verify input tensors retain identical SHA-256 hashes before and after inference."""
    data, _ = load_allowed_data()
    initial_hashes = compute_checksums(data)

    model = build_functional_c_model()
    val_inputs = {
        "sequence_input": data["x_val"],
        "wsi_origin_input": data["wsi_origin_val"][:, None],
    }
    preds = model.predict(val_inputs, verbose=0)
    assert preds.shape == (288, 3)
    assert np.isfinite(preds).all()

    eval_res = evaluate_predictions(preds, data)
    assert eval_res["mae_equivalence_diff"] <= 1e-6

    final_hashes = compute_checksums(data)
    assert initial_hashes == final_hashes, "Input arrays mutated during model inference!"
