"""
Seed robustness evaluation for Experiment B across random weight initializations.
Runs seeds 42, 43, 44, 45, 46 for 3 epochs with frozen learning rate and architecture.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, Any, Set

import numpy as np
import pandas as pd
import tensorflow as tf


SEEDS: List[int] = [42, 43, 44, 45, 46]
OFFICIAL_SEED: int = 42
FIXED_EPOCHS: int = 3
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
DROPOUT: float = 0.20
LSTM_UNITS: int = 32
DENSE_UNITS: int = 16
LOOKBACK: int = 12
FEATURE_COUNT: int = 6
HORIZON: int = 3

EXPERIMENT_A_VAL_MAE: float = 0.517303
ORIGINAL_SEED_42_METRICS = {
    "val_mae": 0.473812,
    "val_mae_h1": 0.353839,
    "val_mae_h2": 0.488363,
    "val_mae_h3": 0.579234,
}

OUTPUT_DIR = Path("results/experiments/B")
DIAGNOSTICS_DIR = OUTPUT_DIR / "seed_diagnostics"

# Explicit Input Allowlist — Zero test files permitted
ALLOWED_INPUTS: Set[Path] = {
    Path("data_model/X_train.npy"),
    Path("data_model/X_val.npy"),
    Path("data_model_residual/y_residual_train.npy"),
    Path("data_model_residual/y_residual_val.npy"),
    Path("data_model_residual/wsi_origin_val.npy"),
    Path("data_model/y_val.npy"),
}


def compute_checksums(data_dict: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Computes SHA-256 hashes of arrays to guarantee immutability across seed runs."""
    return {
        name: hashlib.sha256(arr.tobytes()).hexdigest()
        for name, arr in data_dict.items()
    }


def set_deterministic(seed: int) -> None:
    """Enforces deterministic random state and op determinism for a specific seed."""
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
    Returns data dictionary and the manifest set of loaded paths.
    """
    loaded_paths: Set[Path] = set()

    for p in ALLOWED_INPUTS:
        if not p.exists():
            raise FileNotFoundError(f"Required allowed input missing: {p}")
        loaded_paths.add(p)

    # Ingestion matching exact contract
    x_train = np.load(Path("data_model/X_train.npy")).astype(np.float32)
    x_val = np.load(Path("data_model/X_val.npy")).astype(np.float32)
    y_train_residual = np.load(Path("data_model_residual/y_residual_train.npy")).astype(np.float32)
    y_val_residual = np.load(Path("data_model_residual/y_residual_val.npy")).astype(np.float32)
    wsi_origin_val = np.load(Path("data_model_residual/wsi_origin_val.npy")).astype(np.float32)
    y_val_original = np.load(Path("data_model/y_val.npy")).astype(np.float32)

    # Verify manifest matches ALLOWED_INPUTS exactly
    assert loaded_paths == ALLOWED_INPUTS, (
        f"Loaded paths do not match ALLOWED_INPUTS: {loaded_paths.symmetric_difference(ALLOWED_INPUTS)}"
    )

    # Shape contracts
    assert x_train.shape == (1328, 12, 6), f"Unexpected x_train shape: {x_train.shape}"
    assert x_val.shape == (288, 12, 6), f"Unexpected x_val shape: {x_val.shape}"
    assert y_train_residual.shape == (1328, 3), f"Unexpected y_train_residual shape: {y_train_residual.shape}"
    assert y_val_residual.shape == (288, 3), f"Unexpected y_val_residual shape: {y_val_residual.shape}"
    assert wsi_origin_val.shape == (288,), f"Unexpected wsi_origin_val shape: {wsi_origin_val.shape}"
    assert y_val_original.shape == (288, 3), f"Unexpected y_val_original shape: {y_val_original.shape}"

    data = {
        "x_train": x_train,
        "x_val": x_val,
        "y_train_residual": y_train_residual,
        "y_val_residual": y_val_residual,
        "wsi_origin_val": wsi_origin_val,
        "y_val_original": y_val_original,
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


def build_fresh_model() -> tf.keras.Model:
    """Builds and compiles a fresh model and fresh Adam optimizer."""
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


def run_seed_diagnostic(seed: int, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Executes a single seed diagnostic run:
      - Enforces determinism
      - Builds fresh model & optimizer
      - Trains for exactly FIXED_EPOCHS with shuffle=False
      - Predicts validation residuals and reconstructs WSI
      - Evaluates MAE and RMSE across all horizons
      - Validates residual MAE equivalence and reconstruction identity
    """
    print(f"\n--- Running Seed {seed} ---")
    set_deterministic(seed)

    model = build_fresh_model()

    history = model.fit(
        data["x_train"],
        data["y_train_residual"],
        epochs=FIXED_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )

    actual_epochs = len(history.history["loss"])
    assert actual_epochs == FIXED_EPOCHS, (
        f"Seed {seed} ran {actual_epochs} epochs; expected exactly {FIXED_EPOCHS}."
    )

    # Validation predictions (residuals)
    pred_residuals = model.predict(data["x_val"], batch_size=BATCH_SIZE, verbose=0)
    assert pred_residuals.shape == (288, 3), f"Unexpected pred_residuals shape: {pred_residuals.shape}"
    assert np.isfinite(pred_residuals).all(), f"Seed {seed} produced non-finite residual predictions"

    # WSI Reconstruction: WSI_hat(t+h) = WSI(t) + r_hat(t, h)
    pred_wsi = data["wsi_origin_val"][:, None] + pred_residuals
    assert pred_wsi.shape == (288, 3), f"Unexpected pred_wsi shape: {pred_wsi.shape}"
    assert np.isfinite(pred_wsi).all(), f"Seed {seed} produced non-finite reconstructed WSI predictions"

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
        f"Seed {seed}: MAE_WSI ({val_mae:.8f}) and MAE_residual ({residual_mae:.8f}) mismatch: {mae_diff:.8e}"
    )

    result = {
        "seed": seed,
        "epochs": FIXED_EPOCHS,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "learning_rate_schedule": [LEARNING_RATE] * FIXED_EPOCHS,
        "val_mae": round(val_mae, 6),
        "val_mae_h1": round(val_mae_h1, 6),
        "val_mae_h2": round(val_mae_h2, 6),
        "val_mae_h3": round(val_mae_h3, 6),
        "val_rmse": round(val_rmse, 6),
        "val_rmse_h1": round(val_rmse_h1, 6),
        "val_rmse_h2": round(val_rmse_h2, 6),
        "val_rmse_h3": round(val_rmse_h3, 6),
        "val_residual_mae": round(residual_mae, 6),
    }

    print(
        f"Seed {seed:2d} | Val MAE: {result['val_mae']:.6f} "
        f"(h1: {result['val_mae_h1']:.6f}, h2: {result['val_mae_h2']:.6f}, h3: {result['val_mae_h3']:.6f}) | "
        f"Val RMSE: {result['val_rmse']:.6f}"
    )

    return result


def main() -> None:
    print(f"Running Experiment B Seed Robustness (Seeds: {SEEDS}, Epochs: {FIXED_EPOCHS})")

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data via strict allowlist
    data, manifest = load_allowed_data()
    print(f"Loaded {len(manifest)} files strictly matching ALLOWED_INPUTS.")

    # 2. Record initial checksums to verify immutability
    initial_checksums = compute_checksums(data)

    # 3. Execute all seed runs
    seed_results: List[Dict[str, Any]] = []

    for seed in SEEDS:
        res = run_seed_diagnostic(seed, data)
        seed_results.append(res)

        # Save individual seed diagnostic JSON
        seed_path = DIAGNOSTICS_DIR / f"seed_{seed}.json"
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    # 4. Verify input data immutability
    final_checksums = compute_checksums(data)
    for name in initial_checksums:
        assert initial_checksums[name] == final_checksums[name], (
            f"Input tensor '{name}' mutated during seed runs!"
        )
    print("\n✓ Input data immutability verified across all runs.")

    # 5. Verify Seed 42 reproduction against Step 8.3 within 1e-6
    seed_42_res = next(r for r in seed_results if r["seed"] == OFFICIAL_SEED)
    print(f"\n--- Checking Seed 42 Reproduction (Tolerance <= 1e-6) ---")
    for metric_name, expected_val in ORIGINAL_SEED_42_METRICS.items():
        actual_val = seed_42_res[metric_name]
        diff = abs(actual_val - expected_val)
        print(f"  {metric_name:15s}: actual={actual_val:.6f}, expected={expected_val:.6f}, diff={diff:.8e}")
        assert diff <= 1e-6, (
            f"Seed 42 reproduction failed for {metric_name}! Actual={actual_val}, Expected={expected_val}, Diff={diff}"
        )
    print("✓ Seed 42 reproduced the Step 8.3 results across all horizons within 1e-6.")

    # 6. Save seed_robustness.csv
    csv_rows = []
    for r in seed_results:
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
    robustness_df = pd.DataFrame(csv_rows)
    csv_path = DIAGNOSTICS_DIR / "seed_robustness.csv"
    robustness_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved seed robustness table to: {csv_path}")

    # 7. Compute distribution metrics across the 5 seeds (ddof=0)
    val_maes = [r["val_mae"] for r in seed_results]
    mean_val_mae = float(np.mean(val_maes))
    std_val_mae = float(np.std(val_maes, ddof=0))
    min_val_mae = float(np.min(val_maes))
    max_val_mae = float(np.max(val_maes))
    range_val_mae = float(max_val_mae - min_val_mae)

    summary = {
        "official_seed": OFFICIAL_SEED,
        "fixed_epochs": FIXED_EPOCHS,
        "seeds": SEEDS,
        "mean_val_mae": round(mean_val_mae, 6),
        "std_val_mae": round(std_val_mae, 6),
        "min_val_mae": round(min_val_mae, 6),
        "max_val_mae": round(max_val_mae, 6),
        "range_val_mae": round(range_val_mae, 6),
        "seed_42_reproduced": True,
        "seed_42_metrics": seed_42_res,
        "all_seeds_metrics": seed_results,
        "reference_comparison": {
            "experiment_a_val_mae": round(EXPERIMENT_A_VAL_MAE, 6),
            "delta_mean_vs_a": round(mean_val_mae - EXPERIMENT_A_VAL_MAE, 6),
            "delta_seed42_vs_a": round(seed_42_res["val_mae"] - EXPERIMENT_A_VAL_MAE, 6),
        },
    }

    summary_path = OUTPUT_DIR / "seed_diagnostics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved seed diagnostics summary to: {summary_path}")

    # 8. Report results
    print("\nExperiment B Seed Robustness Summary:")
    print(f"Official Seed 42 Val MAE:  {seed_42_res['val_mae']:.6f}")
    print(f"5-Seed Mean Val MAE:      {mean_val_mae:.6f}")
    print(f"5-Seed Std Val MAE (N=5): {std_val_mae:.6f} (ddof=0)")
    print(f"5-Seed Min Val MAE:       {min_val_mae:.6f}")
    print(f"5-Seed Max Val MAE:       {max_val_mae:.6f}")
    print(f"5-Seed Spread / Range:    {range_val_mae:.6f}")
    print(f"Reference Exp A Val MAE:  {EXPERIMENT_A_VAL_MAE:.6f}")
    print(f"Difference (Mean - Exp A): {mean_val_mae - EXPERIMENT_A_VAL_MAE:+.6f}")


if __name__ == "__main__":
    main()
