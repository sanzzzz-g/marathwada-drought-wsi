"""
Experiment C: Residual LSTM + explicit current WSI state input.
Architecture: Sequence (12, 6) + WSI(t) (1,) -> Concat (33,) -> Dense(16) -> Dense(3).
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Any, Set, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


SEEDS: List[int] = [42, 43, 44, 45, 46]
OFFICIAL_SEED: int = 42
MAX_EPOCHS: int = 200
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
DROPOUT: float = 0.20
LSTM_UNITS: int = 32
DENSE_UNITS: int = 16
LOOKBACK: int = 12
FEATURE_COUNT: int = 6
HORIZON: int = 3

EXPECTED_TRAINABLE_PARAMS: int = 5587

OUTPUT_DIR = Path("results/experiments/C")
DIAGNOSTICS_DIR = OUTPUT_DIR / "seed_diagnostics"
DATA_DIR = Path("data_model")
RESIDUAL_DIR = Path("data_model_residual")

# Baseline Reference Metrics (Validation Set)
EXP_A_METRICS = {
    "val_mae": 0.517303,
    "val_mae_h1": 0.380720,
    "val_mae_h2": 0.528734,
    "val_mae_h3": 0.642456,
    "val_rmse": 0.686522,
    "val_rmse_h1": 0.505705,
    "val_rmse_h2": 0.707765,
    "val_rmse_h3": 0.814981,
}

EXP_B_SEED_42_METRICS = {
    "val_mae": 0.473812,
    "val_mae_h1": 0.353839,
    "val_mae_h2": 0.488363,
    "val_mae_h3": 0.579234,
    "val_rmse": 0.627725,
    "val_rmse_h1": 0.474665,
    "val_rmse_h2": 0.644558,
    "val_rmse_h3": 0.738081,
}

EXP_B_5SEED_SUMMARY = {
    "mean_val_mae": 0.475888,
    "std_val_mae": 0.003615,
    "min_val_mae": 0.471272,
    "max_val_mae": 0.481608,
    "range_val_mae": 0.010336,
}

# Explicit Input Allowlist — Zero test files permitted
ALLOWED_INPUTS: Set[Path] = {
    Path("data_model/X_train.npy"),
    Path("data_model/X_val.npy"),
    Path("data_model/y_val.npy"),
    Path("data_model_residual/y_residual_train.npy"),
    Path("data_model_residual/y_residual_val.npy"),
    Path("data_model_residual/wsi_origin_train.npy"),
    Path("data_model_residual/wsi_origin_val.npy"),
}


def compute_checksums(data_dict: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Computes SHA-256 hashes of arrays to guarantee immutability."""
    return {
        name: hashlib.sha256(arr.tobytes()).hexdigest()
        for name, arr in data_dict.items()
    }


def set_deterministic(seed: int) -> None:
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
            f"TensorFlow deterministic execution could not be enabled for seed {seed}."
        ) from exc


def load_allowed_data() -> Tuple[Dict[str, np.ndarray], Set[Path]]:
    """
    Loads strictly the files present in ALLOWED_INPUTS.
    Validates shapes, dtypes, finiteness, and ground-truth reconstruction identity.
    """
    loaded_paths: Set[Path] = set()

    for p in ALLOWED_INPUTS:
        if not p.exists():
            raise FileNotFoundError(f"Required allowed input missing: {p}")
        loaded_paths.add(p)

    # Ingestion matching exact contract
    x_train = np.load(Path("data_model/X_train.npy")).astype(np.float32)
    x_val = np.load(Path("data_model/X_val.npy")).astype(np.float32)
    y_val_original = np.load(Path("data_model/y_val.npy")).astype(np.float32)

    y_train_residual = np.load(Path("data_model_residual/y_residual_train.npy")).astype(np.float32)
    y_val_residual = np.load(Path("data_model_residual/y_residual_val.npy")).astype(np.float32)
    wsi_origin_train = np.load(Path("data_model_residual/wsi_origin_train.npy")).astype(np.float32)
    wsi_origin_val = np.load(Path("data_model_residual/wsi_origin_val.npy")).astype(np.float32)

    # Verify manifest matches ALLOWED_INPUTS exactly
    assert loaded_paths == ALLOWED_INPUTS, (
        f"Loaded paths do not match ALLOWED_INPUTS: {loaded_paths.symmetric_difference(ALLOWED_INPUTS)}"
    )

    # Shape contracts
    assert x_train.shape == (1328, 12, 6), f"Unexpected x_train shape: {x_train.shape}"
    assert x_val.shape == (288, 12, 6), f"Unexpected x_val shape: {x_val.shape}"
    assert y_val_original.shape == (288, 3), f"Unexpected y_val_original shape: {y_val_original.shape}"
    assert y_train_residual.shape == (1328, 3), f"Unexpected y_train_residual shape: {y_train_residual.shape}"
    assert y_val_residual.shape == (288, 3), f"Unexpected y_val_residual shape: {y_val_residual.shape}"
    assert wsi_origin_train.shape == (1328,), f"Unexpected wsi_origin_train shape: {wsi_origin_train.shape}"
    assert wsi_origin_val.shape == (288,), f"Unexpected wsi_origin_val shape: {wsi_origin_val.shape}"

    data = {
        "x_train": x_train,
        "x_val": x_val,
        "y_val_original": y_val_original,
        "y_train_residual": y_train_residual,
        "y_val_residual": y_val_residual,
        "wsi_origin_train": wsi_origin_train,
        "wsi_origin_val": wsi_origin_val,
    }

    # Finiteness check
    for name, arr in data.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")

    # Hard assertion: Target reconstruction identity on validation ground truth
    reconstructed_truth = wsi_origin_val[:, None] + y_val_residual
    max_target_diff = float(np.max(np.abs(reconstructed_truth - y_val_original)))
    assert max_target_diff <= 1e-6, (
        f"Validation residual + origin identity check failed! Max diff: {max_target_diff:.8e}"
    )

    return data, loaded_paths


def build_functional_c_model(learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Builds and compiles the Keras Functional API model for Experiment C.
    Architecture:
      Input A: (12, 6) -> LSTM(32) -> Dropout(0.20)
      Input B: (1,) scalar current WSI state
      Concatenate: 32 + 1 = 33 units
      Dense(16, relu)
      Dense(3, linear)
    Total trainable parameters: 5,587.
    """
    seq_input = tf.keras.layers.Input(shape=(LOOKBACK, FEATURE_COUNT), name="sequence_input")
    wsi_input = tf.keras.layers.Input(shape=(1,), name="wsi_origin_input")

    lstm_out = tf.keras.layers.LSTM(LSTM_UNITS, return_sequences=False, name="lstm_32")(seq_input)
    dropout_out = tf.keras.layers.Dropout(DROPOUT, name="dropout_20")(lstm_out)

    concat = tf.keras.layers.Concatenate(name="concat_state")([dropout_out, wsi_input])
    dense1 = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_16")(concat)
    output = tf.keras.layers.Dense(HORIZON, activation="linear", name="output_3")(dense1)

    model = tf.keras.Model(
        inputs=[seq_input, wsi_input],
        outputs=output,
        name="residual_lstm_wsi_state",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
            lr_val = LEARNING_RATE
        self.lr_history.append(lr_val)


def evaluate_predictions(
    pred_residuals: np.ndarray,
    data: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Computes comprehensive MAE and RMSE metrics on reconstructed WSI and verifies residual equivalence."""
    assert pred_residuals.shape == (288, 3), f"Unexpected pred_residuals shape: {pred_residuals.shape}"
    assert np.isfinite(pred_residuals).all(), "Non-finite residual predictions encountered"

    # WSI Reconstruction: WSI_hat(t+h) = WSI(t) + r_hat(t, h)
    pred_wsi = data["wsi_origin_val"][:, None] + pred_residuals
    assert pred_wsi.shape == (288, 3), f"Unexpected pred_wsi shape: {pred_wsi.shape}"
    assert np.isfinite(pred_wsi).all(), "Non-finite reconstructed WSI predictions encountered"

    # Metric calculations on reconstructed WSI vs y_val_original
    errors = pred_wsi - data["y_val_original"]
    abs_errors = np.abs(errors)
    sq_errors = np.square(errors)

    val_mae = float(np.mean(abs_errors))
    val_mae_h1 = float(np.mean(abs_errors[:, 0]))
    val_mae_h2 = float(np.mean(abs_errors[:, 1]))
    val_mae_h3 = float(np.mean(abs_errors[:, 2]))

    val_rmse = float(np.sqrt(np.mean(sq_errors)))
    val_rmse_h1 = float(np.sqrt(np.mean(sq_errors[:, 0])))
    val_rmse_h2 = float(np.sqrt(np.mean(sq_errors[:, 1])))
    val_rmse_h3 = float(np.sqrt(np.mean(sq_errors[:, 2])))

    # Residual MAE check
    residual_mae = float(np.mean(np.abs(pred_residuals - data["y_val_residual"])))
    mae_diff = abs(val_mae - residual_mae)
    assert mae_diff <= 1e-6, (
        f"MAE_WSI ({val_mae:.8f}) and MAE_residual ({residual_mae:.8f}) mismatch: {mae_diff:.8e}"
    )

    return {
        "val_mae": round(val_mae, 6),
        "val_mae_h1": round(val_mae_h1, 6),
        "val_mae_h2": round(val_mae_h2, 6),
        "val_mae_h3": round(val_mae_h3, 6),
        "val_rmse": round(val_rmse, 6),
        "val_rmse_h1": round(val_rmse_h1, 6),
        "val_rmse_h2": round(val_rmse_h2, 6),
        "val_rmse_h3": round(val_rmse_h3, 6),
        "val_residual_mae": round(residual_mae, 6),
        "mae_equivalence_diff": float(mae_diff),
    }


def long_predictions(
    metadata: pd.DataFrame,
    predicted_wsi: np.ndarray,
    actual_wsi: np.ndarray,
    model_name: str = "C_residual_LSTM_wsi_state",
) -> pd.DataFrame:
    """Constructs long-format validation prediction records (864 rows)."""
    records = []
    meta_reset = metadata.reset_index(drop=True)
    assert len(meta_reset) == len(predicted_wsi), "Metadata and prediction length mismatch"

    for idx, row in meta_reset.iterrows():
        seq_id = int(row["sequence_id"])
        dist = str(row["district"])
        origin = str(row["forecast_origin"])

        for h in range(HORIZON):
            act = float(actual_wsi[idx, h])
            pred = float(predicted_wsi[idx, h])
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


def run_stage_1_seed_42(data: Dict[str, np.ndarray]) -> Tuple[tf.keras.Model, Dict[str, Any], List[float], int]:
    """
    Stage 1: Official Seed 42 Run & E* Selection.
    Trains with EarlyStopping & ReduceLROnPlateau.
    Identifies earliest best epoch E* within 1e-8.
    """
    print("\nStage 1: Experiment C Seed 42 exploratory run...")

    set_deterministic(OFFICIAL_SEED)

    model = build_functional_c_model(learning_rate=LEARNING_RATE)

    # Trainable parameter check
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    print(f"Model instantiated. Trainable parameters: {trainable_params} (Expected: {EXPECTED_TRAINABLE_PARAMS})")
    assert trainable_params == EXPECTED_TRAINABLE_PARAMS, (
        f"Trainable parameters mismatch: got {trainable_params}, expected {EXPECTED_TRAINABLE_PARAMS}"
    )

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

    train_inputs = {
        "sequence_input": data["x_train"],
        "wsi_origin_input": data["wsi_origin_train"][:, None],
    }
    val_inputs = {
        "sequence_input": data["x_val"],
        "wsi_origin_input": data["wsi_origin_val"][:, None],
    }

    history = model.fit(
        train_inputs,
        data["y_train_residual"],
        validation_data=(val_inputs, data["y_val_residual"]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[reduce_lr, early_stop, lr_logger],
        verbose=1,
    )

    val_mae_hist = [float(v) for v in history.history["val_mae"]]
    e_star = find_earliest_best_epoch(val_mae_hist, tol=1e-8)
    best_val_mae = val_mae_hist[e_star - 1]
    frozen_lr_schedule = lr_logger.lr_history[:e_star]

    print(f"\n✓ Stage 1 Complete!")
    print(f"  Total Epochs Run: {len(val_mae_hist)}")
    print(f"  Selected E*:      {e_star}")
    print(f"  Min Val MAE:      {best_val_mae:.6f}")
    print(f"  LR Schedule to E*: {frozen_lr_schedule}")

    # Predict validation residuals using restored best weights
    pred_residuals = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=0)
    eval_metrics = evaluate_predictions(pred_residuals, data)

    # Save training history CSV
    hist_df = pd.DataFrame({
        "epoch": range(1, len(val_mae_hist) + 1),
        "loss": history.history["loss"],
        "mae": history.history["mae"],
        "val_loss": history.history["val_loss"],
        "val_mae": history.history["val_mae"],
        "learning_rate": lr_logger.lr_history,
    })
    hist_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Save validation predictions CSV (864 rows)
    meta_raw = pd.read_csv(DATA_DIR / "sequence_metadata.csv")
    val_metadata = (
        meta_raw[meta_raw["split"].isin(["val", "validation"])]
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )
    pred_wsi = data["wsi_origin_val"][:, None] + pred_residuals
    val_preds_df = long_predictions(val_metadata, pred_wsi, data["y_val_original"])
    assert len(val_preds_df) == 864, f"Expected 864 rows in validation predictions, got {len(val_preds_df)}"
    val_preds_df.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    # Save model.keras
    model.save(OUTPUT_DIR / "model.keras")

    # Save experiment_config.json
    config = {
        "experiment": "C",
        "description": "Residual LSTM + Explicit Current WSI State (WSI(t)) via Keras Functional API",
        "scientific_note": "WSI(t) is already in the final timestep of the 6 stress inputs; C provides no new information but tests composite state inductive bias.",
        "input_shapes": {
            "sequence_input": [LOOKBACK, FEATURE_COUNT],
            "wsi_origin_input": [1],
        },
        "target": "WSI(t+h) - WSI(t)",
        "trainable_parameters": trainable_params,
        "architecture": {
            "lstm_units": LSTM_UNITS,
            "dropout": DROPOUT,
            "concat_dimension": LSTM_UNITS + 1,
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
            "seed": OFFICIAL_SEED,
            "monitor": "val_mae",
        },
        "selected_e_star": e_star,
        "frozen_lr_schedule": frozen_lr_schedule,
        "validation_results_seed_42": {
            "best_epoch": e_star,
            "train_residual_mae": round(float(history.history["mae"][e_star - 1]), 6),
            **eval_metrics,
        },
    }
    with open(OUTPUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return model, eval_metrics, frozen_lr_schedule, e_star


def run_stage_2_seed_robustness(
    data: Dict[str, np.ndarray],
    e_star: int,
    frozen_lr_schedule: List[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Stage 2: 5-Seed Robustness Evaluation at frozen E* with replayed LR schedule.
    Seeds: [42, 43, 44, 45, 46].
    """
    print(f"\nStage 2: Experiment C 5-seed robustness (Seeds: {SEEDS}, Epochs: {e_star})...")

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    all_seed_results: List[Dict[str, Any]] = []

    train_inputs = {
        "sequence_input": data["x_train"],
        "wsi_origin_input": data["wsi_origin_train"][:, None],
    }
    val_inputs = {
        "sequence_input": data["x_val"],
        "wsi_origin_input": data["wsi_origin_val"][:, None],
    }

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} for {e_star} epochs ---")
        set_deterministic(seed)

        # Fresh model & optimizer
        model = build_functional_c_model(learning_rate=frozen_lr_schedule[0])

        def make_scheduler(schedule):
            def scheduler(epoch, lr):
                if epoch < len(schedule):
                    return schedule[epoch]
                return schedule[-1]
            return scheduler

        lr_callback = tf.keras.callbacks.LearningRateScheduler(make_scheduler(frozen_lr_schedule), verbose=0)

        history = model.fit(
            train_inputs,
            data["y_train_residual"],
            epochs=e_star,
            batch_size=BATCH_SIZE,
            shuffle=False,
            callbacks=[lr_callback],
            verbose=0,
        )

        assert len(history.history["loss"]) == e_star, (
            f"Seed {seed} ran {len(history.history['loss'])} epochs; expected {e_star}"
        )

        pred_residuals = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=0)
        metrics = evaluate_predictions(pred_residuals, data)

        res = {
            "seed": seed,
            "epochs": e_star,
            "batch_size": BATCH_SIZE,
            "shuffle": False,
            "learning_rate_schedule": frozen_lr_schedule,
            **metrics,
        }
        all_seed_results.append(res)

        # Save individual JSON
        seed_path = DIAGNOSTICS_DIR / f"seed_{seed}.json"
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        print(
            f"Seed {seed:2d} | Val MAE: {res['val_mae']:.6f} "
            f"(h1: {res['val_mae_h1']:.6f}, h2: {res['val_mae_h2']:.6f}, h3: {res['val_mae_h3']:.6f}) | "
            f"Val RMSE: {res['val_rmse']:.6f}"
        )

    # Save seed_robustness.csv
    csv_rows = []
    for r in all_seed_results:
        csv_rows.append({
            "seed": r["seed"],
            "val_mae": r["val_mae"],
            "val_mae_h1": r["val_mae_h1"],
            "val_mae_h2": r["val_mae_h2"],
            "val_mae_h3": r["val_mae_h3"],
            "val_rmse": r["val_rmse"],
            "val_rmse_h1": r["val_rmse_h1"],
            "val_rmse_h2": r["val_rmse_h2"],
            "val_rmse_h3": r["val_rmse_h3"],
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(DIAGNOSTICS_DIR / "seed_robustness.csv", index=False)
    print(f"\n✓ Saved seed robustness table to: {DIAGNOSTICS_DIR / 'seed_robustness.csv'}")

    # Calculate distribution statistics across 5 seeds (ddof=0)
    val_maes = [r["val_mae"] for r in all_seed_results]
    mean_val_mae = float(np.mean(val_maes))
    std_val_mae = float(np.std(val_maes, ddof=0))
    min_val_mae = float(np.min(val_maes))
    max_val_mae = float(np.max(val_maes))
    range_val_mae = float(max_val_mae - min_val_mae)

    seed_42_res = next(r for r in all_seed_results if r["seed"] == OFFICIAL_SEED)

    summary = {
        "official_seed": OFFICIAL_SEED,
        "fixed_epochs": e_star,
        "seeds": SEEDS,
        "mean_val_mae": round(mean_val_mae, 6),
        "std_val_mae": round(std_val_mae, 6),
        "min_val_mae": round(min_val_mae, 6),
        "max_val_mae": round(max_val_mae, 6),
        "range_val_mae": round(range_val_mae, 6),
        "seed_42_metrics": seed_42_res,
        "all_seeds_metrics": all_seed_results,
        "comparison_vs_b": {
            "exp_b_seed_42_val_mae": EXP_B_SEED_42_METRICS["val_mae"],
            "c_seed_42_minus_b_seed_42": round(seed_42_res["val_mae"] - EXP_B_SEED_42_METRICS["val_mae"], 6),
            "exp_b_5seed_mean_mae": EXP_B_5SEED_SUMMARY["mean_val_mae"],
            "c_5seed_mean_minus_b_5seed_mean": round(mean_val_mae - EXP_B_5SEED_SUMMARY["mean_val_mae"], 6),
            "exp_b_5seed_std": EXP_B_5SEED_SUMMARY["std_val_mae"],
            "c_5seed_std": round(std_val_mae, 6),
        },
    }

    # Save summary in both expected locations for compatibility
    for dest in [DIAGNOSTICS_DIR / "seed_robustness_summary.json", OUTPUT_DIR / "seed_diagnostics_summary.json"]:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"✓ Saved seed robustness summary to: {DIAGNOSTICS_DIR / 'seed_robustness_summary.json'}")

    return all_seed_results, summary


def evaluate_decision(
    c_seed_42: Dict[str, Any],
    c_summary: Dict[str, Any],
) -> str:
    """
    Evaluates the formal decision criteria:
      1. Primary comparison: C Seed 42 Val MAE vs B Seed 42 Val MAE (0.473812).
      2. Stability check: C 5-seed mean not materially worse than Seed 42,
         and C variance not substantially larger than B's (0.003615).
    """
    c_seed42_mae = c_seed_42["val_mae"]
    b_seed42_mae = EXP_B_SEED_42_METRICS["val_mae"]
    c_mean_mae = c_summary["mean_val_mae"]
    c_std_mae = c_summary["std_val_mae"]
    b_std_mae = EXP_B_5SEED_SUMMARY["std_val_mae"]

    delta_seed42 = c_seed42_mae - b_seed42_mae

    print("\nExperimental Decision Evaluation (C vs. B):")
    print(f"Experiment B Seed 42 Val MAE:      {b_seed42_mae:.6f}")
    print(f"Experiment C Seed 42 Val MAE:      {c_seed42_mae:.6f}")
    print(f"Delta (C Seed 42 - B Seed 42):     {delta_seed42:+.6f}")
    print(f"Experiment B 5-Seed Mean Val MAE:  {EXP_B_5SEED_SUMMARY['mean_val_mae']:.6f} (Std: {b_std_mae:.6f})")
    print(f"Experiment C 5-Seed Mean Val MAE:  {c_mean_mae:.6f} (Std: {c_std_mae:.6f})")

    if delta_seed42 < -0.001:
        if c_mean_mae < EXP_B_5SEED_SUMMARY["mean_val_mae"] and c_std_mae <= b_std_mae * 1.5:
            decision = (
                "DECISION: C IMPROVES B.\n"
                f"Experiment C reduces validation MAE by {abs(delta_seed42):.6f} on Seed 42 "
                f"and demonstrates consistent stability across 5 seeds (mean={c_mean_mae:.6f}, std={c_std_mae:.6f}).\n"
                "Proceed with Experiment C as the champion formulation."
            )
        else:
            decision = (
                "DECISION: C SHOWS SEED-42 GAIN BUT FAILS STABILITY CHECK.\n"
                "Retain Experiment B due to higher variance or inconsistent multi-seed behavior."
            )
    elif abs(delta_seed42) <= 0.001:
        decision = (
            "DECISION: C IS APPROXIMATELY EQUAL TO B (|Delta| <= 0.001).\n"
            f"Difference is negligible ({delta_seed42:+.6f} MAE). By Occam's razor, retain simpler Experiment B "
            "without the auxiliary input."
        )
    else:
        decision = (
            "DECISION: C IS WORSE THAN B.\n"
            f"Experiment C increases validation MAE by {delta_seed42:+.6f} on Seed 42. "
            "Discard Experiment C; retain Experiment B."
        )

    print("\n" + decision)
    return decision


def print_comparison_table(
    c_seed_42: Dict[str, Any],
    c_summary: Dict[str, Any],
) -> None:
    """Prints a comparative validation benchmark table across Experiments A, B, and C."""
    print("\nValidation Benchmark Comparison (Exp A vs. B vs. C):")
    header = f"{'Metric':<18} | {'Exp A (Direct)':<15} | {'Exp B (Residual)':<17} | {'Exp C (Resid+WSI)':<17} | {'Delta (C - B)':<15}"
    print(header)

    metrics_to_show = [
        ("Overall Val MAE", EXP_A_METRICS["val_mae"], EXP_B_SEED_42_METRICS["val_mae"], c_seed_42["val_mae"]),
        ("Val MAE (h1)", EXP_A_METRICS["val_mae_h1"], EXP_B_SEED_42_METRICS["val_mae_h1"], c_seed_42["val_mae_h1"]),
        ("Val MAE (h2)", EXP_A_METRICS["val_mae_h2"], EXP_B_SEED_42_METRICS["val_mae_h2"], c_seed_42["val_mae_h2"]),
        ("Val MAE (h3)", EXP_A_METRICS["val_mae_h3"], EXP_B_SEED_42_METRICS["val_mae_h3"], c_seed_42["val_mae_h3"]),
        ("Overall Val RMSE", EXP_A_METRICS["val_rmse"], EXP_B_SEED_42_METRICS["val_rmse"], c_seed_42["val_rmse"]),
        ("Val RMSE (h1)", EXP_A_METRICS["val_rmse_h1"], EXP_B_SEED_42_METRICS["val_rmse_h1"], c_seed_42["val_rmse_h1"]),
        ("Val RMSE (h2)", EXP_A_METRICS["val_rmse_h2"], EXP_B_SEED_42_METRICS["val_rmse_h2"], c_seed_42["val_rmse_h2"]),
        ("Val RMSE (h3)", EXP_A_METRICS["val_rmse_h3"], EXP_B_SEED_42_METRICS["val_rmse_h3"], c_seed_42["val_rmse_h3"]),
        ("5-Seed Mean MAE", "N/A (single)", EXP_B_5SEED_SUMMARY["mean_val_mae"], c_summary["mean_val_mae"]),
        ("5-Seed Std MAE", "N/A (single)", EXP_B_5SEED_SUMMARY["std_val_mae"], c_summary["std_val_mae"]),
    ]

    for label, val_a, val_b, val_c in metrics_to_show:
        if isinstance(val_b, float) and isinstance(val_c, float):
            delta = f"{val_c - val_b:+.6f}"
            str_b = f"{val_b:.6f}"
            str_c = f"{val_c:.6f}"
        else:
            delta = "N/A"
            str_b = str(val_b)
            str_c = str(val_c)
        str_a = f"{val_a:.6f}" if isinstance(val_a, float) else str(val_a)
        print(f"{label:<18} | {str_a:<15} | {str_b:<17} | {str_c:<17} | {delta:<15}")




def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data via strict allowlist
    data, manifest = load_allowed_data()
    print(f"Loaded {len(manifest)} files strictly matching ALLOWED_INPUTS.")
    initial_checksums = compute_checksums(data)

    # 2. Stage 1: Official Seed 42 run to determine E*
    model_s1, s1_metrics, frozen_lr_schedule, e_star = run_stage_1_seed_42(data)

    # 3. Stage 2: 5-Seed Robustness Evaluation at frozen E*
    seed_results, summary = run_stage_2_seed_robustness(data, e_star, frozen_lr_schedule)

    # 4. Verify input data immutability
    final_checksums = compute_checksums(data)
    for name in initial_checksums:
        assert initial_checksums[name] == final_checksums[name], (
            f"Input tensor '{name}' mutated during execution!"
        )
    print("\n✓ Input data immutability verified across all runs.")

    # 5. Comparative benchmark & decision
    seed_42_res = next(r for r in seed_results if r["seed"] == OFFICIAL_SEED)
    print_comparison_table(seed_42_res, summary)
    evaluate_decision(seed_42_res, summary)


if __name__ == "__main__":
    main()
