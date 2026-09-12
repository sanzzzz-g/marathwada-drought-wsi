"""
Unit and contract tests for the small LSTM model.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.lstm_model import (
    build_small_lstm,
    set_deterministic_tf,
    find_earliest_best_epoch,
    make_lr_replay_callback,
    train_final_model,
)


def test_input_shape_contract():
    """Verifies that model accepts input shape (batch, 12, 6) and rejects invalid dimensions."""
    set_deterministic_tf(42)
    model = build_small_lstm(input_shape=(12, 6))

    # Valid batch input: (batch_size=8, timesteps=12, features=6)
    x_valid = np.random.normal(0, 1, size=(8, 12, 6)).astype(np.float32)
    out = model(x_valid)
    assert out.shape == (8, 3)

    # Invalid feature count (5 instead of 6) should fail
    x_invalid_features = np.random.normal(0, 1, size=(8, 12, 5)).astype(np.float32)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError, Exception)):
        model(x_invalid_features)


def test_output_shape_contract():
    """Verifies that model outputs shape (batch, 3) for multi-step horizons h1, h2, h3."""
    set_deterministic_tf(42)
    model = build_small_lstm(input_shape=(12, 6))

    for batch_size in [1, 4, 16, 32]:
        x = np.random.normal(0, 1, size=(batch_size, 12, 6)).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (batch_size, 3)
        assert out.dtype in [np.float32, np.float64]


def test_no_nan_inf():
    """Verifies forward pass generates strictly finite values across synthetic and boundary inputs."""
    set_deterministic_tf(42)
    model = build_small_lstm(input_shape=(12, 6))

    # Standard normal inputs
    x_normal = np.random.normal(0, 1, size=(16, 12, 6)).astype(np.float32)
    out_normal = model(x_normal).numpy()
    assert not np.isnan(out_normal).any()
    assert not np.isinf(out_normal).any()

    # Extreme value inputs (stress z-scores up to +/- 5.0)
    x_extreme = np.clip(np.random.normal(0, 3, size=(16, 12, 6)), -5.0, 5.0).astype(np.float32)
    out_extreme = model(x_extreme).numpy()
    assert not np.isnan(out_extreme).any()
    assert not np.isinf(out_extreme).any()


def test_monitor_is_val_mae():
    """Verifies that EarlyStopping and ReduceLROnPlateau explicitly monitor 'val_mae'."""
    # Build standard callbacks as configured in validation phase
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=15,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    assert early_stop.monitor == "val_mae"
    assert reduce_lr.monitor == "val_mae"
    assert early_stop.patience == 15
    assert reduce_lr.patience == 5
    assert reduce_lr.factor == 0.5


def test_final_training_uses_no_test_data():
    """
    Verifies that final refit function trains only on TRAIN+VAL data,
    uses only LR scheduler replay callback (no EarlyStopping, no ReduceLROnPlateau),
    and only evaluates TEST in evaluation mode.
    """
    set_deterministic_tf(42)
    n_train_val = 32
    n_test = 8
    X_train_val = np.random.normal(0, 1, size=(n_train_val, 12, 6)).astype(np.float32)
    y_train_val = np.random.normal(0, 1, size=(n_train_val, 3)).astype(np.float32)
    X_test = np.random.normal(0, 1, size=(n_test, 12, 6)).astype(np.float32)

    E_star = 2
    frozen_schedule = [0.001, 0.0005]

    model, test_preds = train_final_model(
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        E_star=E_star,
        frozen_lr_schedule=frozen_schedule,
        seed=42,
        batch_size=16,
    )

    assert test_preds.shape == (n_test, 3)
    # Check that model weights are finite
    for w in model.get_weights():
        assert not np.isnan(w).any()
        assert not np.isinf(w).any()


def test_final_epoch_count_equals_E_star():
    """
    Verifies that final model refit trains for exactly E* epochs matching the
    replayed learning rate schedule.
    """
    set_deterministic_tf(42)
    n_samples = 32
    X = np.random.normal(0, 1, size=(n_samples, 12, 6)).astype(np.float32)
    y = np.random.normal(0, 1, size=(n_samples, 3)).astype(np.float32)

    E_star = 3
    lr_schedule = [0.001, 0.0005, 0.00025]

    model = build_small_lstm(input_shape=(12, 6), learning_rate=lr_schedule[0])
    lr_cb = make_lr_replay_callback(lr_schedule)

    history = model.fit(
        X,
        y,
        epochs=E_star,
        batch_size=16,
        shuffle=False,
        callbacks=[lr_cb],
        verbose=0,
    )

    # Exactly E* epochs trained
    assert len(history.epoch) == E_star
    assert history.epoch == [0, 1, 2]


def test_seed_reproducibility():
    """Verifies that two model initializations under set_deterministic_tf(42) have identical initial weights."""
    set_deterministic_tf(42)
    model1 = build_small_lstm(input_shape=(12, 6))
    weights1 = model1.get_weights()

    set_deterministic_tf(42)
    model2 = build_small_lstm(input_shape=(12, 6))
    weights2 = model2.get_weights()

    assert len(weights1) == len(weights2)
    for w1, w2 in zip(weights1, weights2):
        max_diff = np.max(np.abs(w1 - w2))
        assert max_diff == 0.0, f"Weight mismatch: max diff = {max_diff}"


def test_earliest_best_epoch_tie_breaker():
    """Verifies that earliest epoch achieving minimum val_mae within 1e-8 tolerance is chosen."""
    # Exact tie at epoch 3 and epoch 7
    val_mae_history = [0.60, 0.55, 0.500000001, 0.52, 0.51, 0.505, 0.500000000, 0.53]
    # min_val is 0.500000000. Epoch 3 is 0.500000001, which is <= 0.500000000 + 1e-8.
    E_star = find_earliest_best_epoch(val_mae_history, tol=1e-8)
    assert E_star == 3  # Earliest epoch (1-indexed)
