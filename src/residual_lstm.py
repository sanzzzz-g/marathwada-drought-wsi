"""
Experiment B: 6-feature small LSTM trained on residual targets (WSI_{t+h} - WSI_t).
Architecture: Input (12, 6) -> LSTM(32) -> Dropout(0.20) -> Dense(16, relu) -> Dense(3, linear).
"""

import os
import sys
import json
import random
import platform
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


DATA_DIR = Path("data_model")
RESIDUAL_DIR = Path("data_model_residual")
OUTPUT_DIR = Path("results/experiments/B")

FEATURE_COUNT = 6
LOOKBACK = 12
HORIZON = 3
SEED = 42

BATCH_SIZE = 32
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT = 0.20
LSTM_UNITS = 32
DENSE_UNITS = 16


def set_deterministic(seed: int = 42) -> None:
    """Enforces deterministic random state and op determinism."""
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow deterministic execution could not be enabled."
        ) from exc


def load_train_val_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads strictly TRAIN and VAL data.
    Zero test data is loaded in this phase.
    Also loads original Step 5 y_val.npy as authoritative ground-truth.
    """
    x_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    x_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

    y_train_residual = np.load(RESIDUAL_DIR / "y_residual_train.npy").astype(np.float32)
    y_val_residual = np.load(RESIDUAL_DIR / "y_residual_val.npy").astype(np.float32)

    wsi_origin_val = np.load(RESIDUAL_DIR / "wsi_origin_val.npy").astype(np.float32)
    y_val_original = np.load(DATA_DIR / "y_val.npy").astype(np.float32)

    # Shape contracts
    assert x_train.shape == (1328, 12, 6), f"Unexpected x_train shape: {x_train.shape}"
    assert x_val.shape == (288, 12, 6), f"Unexpected x_val shape: {x_val.shape}"
    assert y_train_residual.shape == (1328, 3), f"Unexpected y_train_residual shape: {y_train_residual.shape}"
    assert y_val_residual.shape == (288, 3), f"Unexpected y_val_residual shape: {y_val_residual.shape}"
    assert wsi_origin_val.shape == (288,), f"Unexpected wsi_origin_val shape: {wsi_origin_val.shape}"
    assert y_val_original.shape == (288, 3), f"Unexpected y_val_original shape: {y_val_original.shape}"

    # Finiteness contracts
    for name, arr in {
        "X_train": x_train,
        "X_val": x_val,
        "y_train_residual": y_train_residual,
        "y_val_residual": y_val_residual,
        "wsi_origin_val": wsi_origin_val,
        "y_val_original": y_val_original,
    }.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

    # Hard assertion 1: verify target reconstruction matches Step 5 ground truth
    reconstructed_truth = wsi_origin_val[:, None] + y_val_residual
    max_target_diff = float(np.max(np.abs(reconstructed_truth - y_val_original)))
    assert max_target_diff <= 1e-6, f"Residual + origin does not equal Step 5 y_val! Max diff: {max_target_diff}"

    return (
        x_train,
        x_val,
        y_train_residual,
        y_val_residual,
        wsi_origin_val,
        y_val_original,
    )


def build_model() -> tf.keras.Model:
    """Builds and compiles the frozen small LSTM architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, FEATURE_COUNT)),
        tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=False, name="lstm_32"),
        tf.keras.layers.Dropout(DROPOUT, name="dropout_20"),
        tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_16"),
        tf.keras.layers.Dense(HORIZON, activation="linear", name="output_3"),
    ], name="residual_small_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def find_earliest_best_epoch(val_mae_history: List[float], tol: float = 1e-8) -> int:
    """Identifies the 1-indexed epoch E* corresponding to the earliest minimum val_mae within tol."""
    min_val = min(val_mae_history)
    for idx, val in enumerate(val_mae_history):
        if val <= min_val + tol:
            return idx + 1
    return 1


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Callback to record learning rate at every epoch."""

    def __init__(self):
        super().__init__()
        self.lr_history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is not None and "learning_rate" in logs:
            lr_val = float(logs["learning_rate"])
        elif logs is not None and "lr" in logs:
            lr_val = float(logs["lr"])
        elif hasattr(self.model.optimizer, "learning_rate"):
            cur_lr = self.model.optimizer.learning_rate
            if hasattr(cur_lr, "numpy"):
                lr_val = float(cur_lr.numpy())
            elif callable(cur_lr):
                lr_val = float(cur_lr(self.model.optimizer.iterations))
            else:
                lr_val = float(cur_lr)
        else:
            lr_val = 0.001
        self.lr_history.append(lr_val)


def long_predictions(
    metadata: pd.DataFrame,
    predicted_wsi: np.ndarray,
    actual_wsi: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Constructs long-format validation prediction records."""
    records = []
    meta_reset = metadata.reset_index(drop=True)
    assert len(meta_reset) == len(predicted_wsi), "Metadata and prediction length mismatch"

    for i, row in meta_reset.iterrows():
        seq_id = row["sequence_id"]
        dist = row["district"]
        origin = str(row["forecast_origin"]).split("T")[0]

        for h in range(HORIZON):
            pred = float(predicted_wsi[i, h])
            act = float(actual_wsi[i, h])
            err = pred - act

            records.append({
                "model": model_name,
                "sequence_id": seq_id,
                "district": dist,
                "forecast_origin": origin,
                "horizon": f"h{h + 1}",
                "actual_wsi": round(act, 6),
                "predicted_wsi": round(pred, 6),
                "error": round(err, 6),
                "absolute_error": round(abs(err), 6),
                "squared_error": round(err ** 2, 6),
            })

    return pd.DataFrame(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running Experiment B Residual LSTM validation run (Output: {OUTPUT_DIR})")

    set_deterministic(SEED)

    # 1. Load strictly TRAIN and VAL data
    (
        x_train,
        x_val,
        y_train_residual,
        y_val_residual,
        wsi_origin_val,
        y_val_original,
    ) = load_train_val_data()

    print(f"Loaded training data:   {x_train.shape} -> {y_train_residual.shape}")
    print(f"Loaded validation data: {x_val.shape} -> {y_val_residual.shape}")
    print(f"Loaded validation origin: {wsi_origin_val.shape}")
    print(f"Authoritative Step 5 y_val verified: {y_val_original.shape}")

    # 2. Load validation metadata (handling both 'val' and 'validation')
    meta_raw = pd.read_csv(DATA_DIR / "sequence_metadata.csv")
    val_metadata = (
        meta_raw[meta_raw["split"].isin(["val", "validation"])]
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )
    if len(val_metadata) != 288:
        raise ValueError(f"Validation metadata count mismatch: expected 288, got {len(val_metadata)}")

    # 3. Build model and callbacks
    model = build_model()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode="min",
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=15,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    lr_logger = LearningRateLogger()

    print("\nStarting validation training (monitoring standard val_mae)...")
    history = model.fit(
        x_train,
        y_train_residual,
        validation_data=(x_val, y_val_residual),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[reduce_lr, early_stop, lr_logger],
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", list(range(1, len(history_df) + 1)))
    history_df["learning_rate"] = lr_logger.lr_history

    # 4. Determine E* via deterministic tie-breaking (earliest minimum within 1e-8)
    val_maes = history_df["val_mae"].tolist()
    best_epoch = find_earliest_best_epoch(val_maes, tol=1e-8)
    best_val_residual_mae = float(val_maes[best_epoch - 1])
    train_residual_mae_at_best_epoch = float(history_df.loc[best_epoch - 1, "mae"])

    print(f"\nTraining complete. Optimal epoch E* = {best_epoch} (out of {len(history_df)} trained).")
    print(f"Restoring best weights from epoch {best_epoch}...")

    # 5. Predict on validation set using restored best weights
    val_residual_pred = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0)

    # Reconstruct WSI: WSI_hat = WSI_t + r_hat
    val_pred_wsi = wsi_origin_val[:, None] + val_residual_pred

    # Authoritative ground truth: Step 5 y_val
    val_actual_wsi = y_val_original

    # Compute explicit metrics
    abs_err_residual = np.abs(val_residual_pred - y_val_residual)
    abs_err_wsi = np.abs(val_pred_wsi - val_actual_wsi)

    val_residual_mae = float(abs_err_residual.mean())
    val_wsi_mae = float(abs_err_wsi.mean())

    # Hard assertion 2: Exact mathematical equivalence over all 288 x 3 values
    max_metric_diff = float(np.max(np.abs(abs_err_residual - abs_err_wsi)))
    print(f"Mathematical equivalence check: Max error difference = {max_metric_diff:.8e}")
    assert max_metric_diff <= 1e-6, f"Residual and WSI errors diverge! Max diff: {max_metric_diff}"
    assert abs(val_residual_mae - val_wsi_mae) <= 1e-6, "Aggregated MAE mismatch"

    # Per-horizon reconstructed WSI MAEs
    val_mae_h1 = float(abs_err_wsi[:, 0].mean())
    val_mae_h2 = float(abs_err_wsi[:, 1].mean())
    val_mae_h3 = float(abs_err_wsi[:, 2].mean())

    # 6. Save artifacts
    # a. Training history
    history_path = OUTPUT_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    # b. Validation predictions (864 rows)
    val_long = long_predictions(
        val_metadata,
        val_pred_wsi,
        val_actual_wsi,
        "B_residual_LSTM",
    )
    val_pred_path = OUTPUT_DIR / "validation_predictions.csv"
    val_long.to_csv(val_pred_path, index=False)
    assert len(val_long) == 864, f"Expected 864 rows, got {len(val_long)}"
    print(f"Saved validation predictions: {val_pred_path} ({len(val_long)} rows)")

    # c. Saved model
    model_path = OUTPUT_DIR / "model.keras"
    model.save(model_path)
    print(f"Saved restored model artifact: {model_path}")

    # d. Experiment config and environment audit
    devices = [d.device_type for d in tf.config.list_physical_devices()]
    config = {
        "experiment": "B",
        "description": "6-feature small LSTM trained on residual targets r_{t,h} = WSI_{t+h} - WSI_t",
        "input_shape": [LOOKBACK, FEATURE_COUNT],
        "target": "WSI(t+h) - WSI(t)",
        "architecture": {
            "lstm_units": LSTM_UNITS,
            "dropout": DROPOUT,
            "dense_units": DENSE_UNITS,
            "output_units": HORIZON,
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "loss": "Huber(delta=1.0)",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "shuffle": False,
            "seed": SEED,
            "monitor": "val_mae",
        },
        "validation_results": {
            "best_epoch": int(best_epoch),
            "train_residual_mae": round(train_residual_mae_at_best_epoch, 6),
            "val_residual_mae": round(val_residual_mae, 6),
            "val_wsi_mae": round(val_wsi_mae, 6),
            "val_wsi_mae_h1": round(val_mae_h1, 6),
            "val_wsi_mae_h2": round(val_mae_h2, 6),
            "val_wsi_mae_h3": round(val_mae_h3, 6),
        },
        "environment": {
            "tensorflow_version": tf.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "device": devices,
            "seed": SEED,
            "deterministic_mode": True,
        },
    }

    config_path = OUTPUT_DIR / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved experiment configuration: {config_path}")

    # 7. Print summary metrics
    print("\nExperiment B Validation Diagnostics:")
    print(f"Best epoch:                  {best_epoch}")
    print(f"Best validation WSI MAE:     {val_wsi_mae:.6f}")
    print(f"Validation MAE h1:           {val_mae_h1:.6f}")
    print(f"Validation MAE h2:           {val_mae_h2:.6f}")
    print(f"Validation MAE h3:           {val_mae_h3:.6f}")
    print(f"Training MAE at best epoch:  {train_residual_mae_at_best_epoch:.6f}")
    print(f"Validation residual MAE:     {val_residual_mae:.6f}")


if __name__ == "__main__":
    main()
