"""
Small LSTM model implementation and final benchmark comparison for 3-month WSI forecasting.
Architecture: Input (12, 6) -> LSTM(32) -> Dropout(0.20) -> Dense(16, relu) -> Dense(3, linear).
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Tuple, Any, Optional

# Ensure workspace root is in sys.path when running as a standalone script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.baselines import load_sequence_arrays, assemble_prediction_records
from src.evaluation import calc_r2, calc_nrmse, run_evaluation_suite


# Deterministic Settings

def set_deterministic_tf(seed: int = 42) -> None:
    """
    Enforces deterministic random seed across Python, NumPy, and TensorFlow,
    and requires TensorFlow op determinism.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.keras.backend.clear_session()
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        raise RuntimeError(
            f"TensorFlow op determinism could not be enabled: {e}. "
            "Deterministic backend execution is required for this experiment."
        )


# Model Architecture

def build_small_lstm(input_shape: Tuple[int, int] = (12, 6), learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Constructs and compiles the frozen single-layer small LSTM architecture.
    Input (12, 6) -> LSTM(32) -> Dropout(0.20) -> Dense(16, relu) -> Dense(3, linear).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(32, return_sequences=False, name="lstm_32"),
        tf.keras.layers.Dropout(0.20, name="dropout_20"),
        tf.keras.layers.Dense(16, activation="relu", name="dense_16"),
        tf.keras.layers.Dense(3, activation="linear", name="output_3"),
    ], name="small_lstm")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.Huber(delta=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Callback to record optimizer learning rate at the end of each epoch."""

    def __init__(self):
        super().__init__()
        self.lr_history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        # Extract learning rate from logs if present, otherwise directly from optimizer
        if logs is not None and "lr" in logs:
            lr_val = float(logs["lr"])
        elif logs is not None and "learning_rate" in logs:
            lr_val = float(logs["learning_rate"])
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


def make_lr_replay_callback(lr_schedule: List[float]) -> tf.keras.callbacks.LearningRateScheduler:
    """
    Creates a LearningRateScheduler callback that strictly replays the recorded
    epoch-wise learning rate schedule for epochs 0 ... len(lr_schedule)-1.
    """
    def schedule_fn(epoch: int, current_lr: float) -> float:
        if epoch < len(lr_schedule):
            return float(lr_schedule[epoch])
        return float(lr_schedule[-1])

    return tf.keras.callbacks.LearningRateScheduler(schedule_fn)


def find_earliest_best_epoch(val_mae_history: List[float], tol: float = 1e-8) -> int:
    """
    Identifies the 1-indexed epoch E* corresponding to the earliest minimum val_mae
    within floating-point tolerance (tol).
    """
    min_val = min(val_mae_history)
    for idx, val in enumerate(val_mae_history):
        if val <= min_val + tol:
            return idx + 1  # 1-indexed epoch count
    return 1


# Phase 1: Validation Run & Early Stopping E* Search

def train_validation_phase(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    batch_size: int = 32,
    max_epochs: int = 200,
    output_dir: str = "results/lstm",
) -> Dict[str, Any]:
    """
    Phase 1: Trains Seed 42 on TRAIN, monitors val_mae on VAL, records LR schedule,
    selects E*, and exports training history and validation predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_deterministic_tf(seed)

    model = build_small_lstm(input_shape=(12, 6), learning_rate=0.001)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
    lr_logger = LearningRateLogger()

    print(f"\n--- Phase 1: Training Validation Model (Seed {seed}) ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop, reduce_lr, lr_logger],
        verbose=1,
    )

    val_mae_history = [float(v) for v in history.history["val_mae"]]
    loss_history = [float(v) for v in history.history["loss"]]
    val_loss_history = [float(v) for v in history.history["val_loss"]]
    mae_history = [float(v) for v in history.history["mae"]]
    lr_history = lr_logger.lr_history

    # Determine E* (1-indexed) as earliest epoch achieving min(val_mae) within 1e-8
    E_star = find_earliest_best_epoch(val_mae_history, tol=1e-8)
    best_val_mae = float(val_mae_history[E_star - 1])

    # Extract frozen LR schedule for epochs 1...E*
    selected_lr_schedule = lr_history[:E_star]

    print(f"Optimal Stopping Epoch E*: {E_star} (out of {len(val_mae_history)} trained)")
    print(f"Best Validation MAE (Seed {seed}): {best_val_mae:.4f}")
    print(f"Frozen LR Schedule (length {len(selected_lr_schedule)}): {selected_lr_schedule}")

    # Save training history CSV
    hist_df = pd.DataFrame({
        "epoch": list(range(1, len(val_mae_history) + 1)),
        "loss": loss_history,
        "val_loss": val_loss_history,
        "mae": mae_history,
        "val_mae": val_mae_history,
        "learning_rate": lr_history,
    })
    hist_path = os.path.join(output_dir, "training_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved training history: {hist_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_df["epoch"], hist_df["loss"], label="Train Huber Loss", color="#1f77b4", linewidth=1.8)
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Huber Loss", color="#ff7f0e", linewidth=1.8)
    plt.axvline(E_star, color="black", linestyle="--", alpha=0.7, label=f"E* = {E_star}")
    plt.title("Huber Loss vs Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_df["epoch"], hist_df["mae"], label="Train MAE", color="#2ca02c", linewidth=1.8)
    plt.plot(hist_df["epoch"], hist_df["val_mae"], label="Val MAE", color="#d62728", linewidth=1.8)
    plt.axvline(E_star, color="black", linestyle="--", alpha=0.7, label=f"E* = {E_star}")
    plt.axhline(best_val_mae, color="gray", linestyle=":", alpha=0.6, label=f"Best Val MAE = {best_val_mae:.4f}")
    plt.title("MAE vs Epoch (Primary Selection Metric)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("MAE", fontsize=10)
    plt.legend()

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved training curve: {curve_path}")

    # Generate validation predictions using restored best weights
    val_preds = model.predict(X_val, batch_size=batch_size, verbose=0)

    # Per-horizon validation MAEs for audit
    val_mae_h1 = float(np.mean(np.abs(val_preds[:, 0] - y_val[:, 0])))
    val_mae_h2 = float(np.mean(np.abs(val_preds[:, 1] - y_val[:, 1])))
    val_mae_h3 = float(np.mean(np.abs(val_preds[:, 2] - y_val[:, 2])))

    return {
        "model": model,
        "history_df": hist_df,
        "E_star": E_star,
        "best_val_mae": best_val_mae,
        "best_val_mae_h1": val_mae_h1,
        "best_val_mae_h2": val_mae_h2,
        "best_val_mae_h3": val_mae_h3,
        "selected_lr_schedule": selected_lr_schedule,
        "val_preds": val_preds,
    }


# Phase 2: Seed Diagnostics (Seeds 43–46, Pure Initialization Sensitivity)

def run_seed_diagnostics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    E_star: int,
    frozen_lr_schedule: List[float],
    diagnostic_seeds: List[int] = [43, 44, 45, 46],
    batch_size: int = 32,
    seed_42_val_mae: float = 0.0,
    seed_42_val_maes_h: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Phase 2: Runs validation training for diagnostic seeds at exactly E* epochs
    with identical frozen LR schedule. Isolates weight-initialization sensitivity.
    Seeds 43-46 do NOT alter E*, do NOT alter final model weights, and are NOT ensembled.
    """
    print(f"\n--- Phase 2: Seed Diagnostics (Seeds {diagnostic_seeds}) at E*={E_star} ---")

    all_seed_results = [{
        "seed": 42,
        "val_mae": seed_42_val_mae,
        "val_mae_h1": seed_42_val_maes_h["h1"] if seed_42_val_maes_h else 0.0,
        "val_mae_h2": seed_42_val_maes_h["h2"] if seed_42_val_maes_h else 0.0,
        "val_mae_h3": seed_42_val_maes_h["h3"] if seed_42_val_maes_h else 0.0,
    }]

    for s in diagnostic_seeds:
        print(f"  -> Training diagnostic seed {s} for {E_star} epochs...")
        set_deterministic_tf(s)
        diag_model = build_small_lstm(input_shape=(12, 6), learning_rate=frozen_lr_schedule[0])
        lr_cb = make_lr_replay_callback(frozen_lr_schedule)

        diag_model.fit(
            X_train,
            y_train,
            epochs=E_star,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[lr_cb],
            verbose=0,
        )

        preds = diag_model.predict(X_val, batch_size=batch_size, verbose=0)
        h1 = float(np.mean(np.abs(preds[:, 0] - y_val[:, 0])))
        h2 = float(np.mean(np.abs(preds[:, 1] - y_val[:, 1])))
        h3 = float(np.mean(np.abs(preds[:, 2] - y_val[:, 2])))
        overall = float((h1 + h2 + h3) / 3.0)

        all_seed_results.append({
            "seed": s,
            "val_mae": overall,
            "val_mae_h1": h1,
            "val_mae_h2": h2,
            "val_mae_h3": h3,
        })
        print(f"     Seed {s} Val MAE: {overall:.4f} (h1: {h1:.4f}, h2: {h2:.4f}, h3: {h3:.4f})")

    maes = [r["val_mae"] for r in all_seed_results]
    h1s = [r["val_mae_h1"] for r in all_seed_results]
    h2s = [r["val_mae_h2"] for r in all_seed_results]
    h3s = [r["val_mae_h3"] for r in all_seed_results]

    summary = {
        "seeds_evaluated": [r["seed"] for r in all_seed_results],
        "all_seeds_val_mae": [round(m, 4) for m in maes],
        "mean_val_mae": round(float(np.mean(maes)), 4),
        "std_val_mae": round(float(np.std(maes)), 4),
        "mean_val_mae_by_horizon": {
            "h1": round(float(np.mean(h1s)), 4),
            "h2": round(float(np.mean(h2s)), 4),
            "h3": round(float(np.mean(h3s)), 4),
        },
        "std_val_mae_by_horizon": {
            "h1": round(float(np.std(h1s)), 4),
            "h2": round(float(np.std(h2s)), 4),
            "h3": round(float(np.std(h3s)), 4),
        },
        "seed_details": all_seed_results,
    }
    print(f"Seed Robustness Summary (5 seeds): Mean Val MAE = {summary['mean_val_mae']} ± {summary['std_val_mae']}")
    return summary


# Phase 3: Final Model Refit on Train + Validation

def train_final_model(
    X_train_val: np.ndarray,
    y_train_val: np.ndarray,
    X_test: np.ndarray,
    E_star: int,
    frozen_lr_schedule: List[float],
    seed: int = 42,
    batch_size: int = 32,
) -> Tuple[tf.keras.Model, np.ndarray]:
    """
    Phase 3: Refits fresh seed 42 model on combined TRAIN + VAL (1,616 samples)
    for exactly E* epochs, replaying the frozen LR schedule via LearningRateScheduler.
    No early stopping, no ReduceLROnPlateau, and no test data used.
    Predicts once on TEST (360 samples -> 1,080 predictions).
    """
    print(f"\n--- Phase 3: Final Model Refit on TRAIN+VAL ({len(X_train_val)} samples) for {E_star} epochs ---")
    set_deterministic_tf(seed)

    model = build_small_lstm(input_shape=(12, 6), learning_rate=frozen_lr_schedule[0])
    lr_callback = make_lr_replay_callback(frozen_lr_schedule)

    model.fit(
        X_train_val,
        y_train_val,
        epochs=E_star,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[lr_callback],
        verbose=1,
    )

    print("Generating official predictions on TEST set (360 samples)...")
    test_preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    assert test_preds.shape == (len(X_test), 3), f"Unexpected test preds shape: {test_preds.shape}"
    return model, test_preds


# Phase 4: Final Comparison and Benchmark Assembly

def compile_final_model_reports(
    lstm_test_records: List[Dict[str, Any]],
    val_summary_data: Dict[str, Any],
    seed_diagnostics: Dict[str, Any],
    baseline_predictions_path: str = "results/baselines/baseline_predictions.csv",
    baseline_metrics_path: str = "results/baselines/baseline_metrics.csv",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Phase 4: Integrates LSTM predictions into the baseline suite, runs evaluation,
    computes skill vs Persistence and WSI-AR, and outputs final comparison tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    lstm_dir = os.path.join(output_dir, "lstm")
    os.makedirs(lstm_dir, exist_ok=True)

    lstm_df = pd.DataFrame(lstm_test_records)
    test_pred_path = os.path.join(lstm_dir, "test_predictions.csv")
    lstm_df.to_csv(test_pred_path, index=False)
    print(f"\nSaved LSTM test predictions: {test_pred_path} ({len(lstm_df)} rows)")

    # Run full Step 6 evaluation suite on LSTM test predictions
    print("\nRunning standard evaluation suite on LSTM test predictions...")
    lstm_eval = run_evaluation_suite(predictions_csv=test_pred_path, output_dir=lstm_dir)
    lstm_metrics = lstm_eval["metrics"].iloc[0].to_dict()

    # Load baseline metrics to build unified final_model_comparison.csv
    if os.path.exists(baseline_metrics_path):
        base_metrics_df = pd.read_csv(baseline_metrics_path)
    else:
        raise FileNotFoundError(f"Baseline metrics not found at: {baseline_metrics_path}")

    # Read Persistence and WSI-AR baseline metrics for reference
    persist_row = base_metrics_df[base_metrics_df["model"] == "Persistence"].iloc[0]
    wsi_ar_row = base_metrics_df[base_metrics_df["model"] == "WSI_Autoregression"].iloc[0]

    persist_mae_all = float(persist_row["overall_mae"])
    persist_mae_h = {f"h{h}": float(persist_row[f"mae_h{h}"]) for h in range(1, 4)}

    wsi_ar_mae_all = float(wsi_ar_row["overall_mae"])
    wsi_ar_mae_h = {f"h{h}": float(wsi_ar_row[f"mae_h{h}"]) for h in range(1, 4)}

    # Build 6-model comparison table
    comparison_rows = []
    # 1. Existing 5 baselines
    for _, row in base_metrics_df.iterrows():
        m_name = row["model"]
        m_mae_all = float(row["overall_mae"])
        skill_wsi_ar = 1.0 - (m_mae_all / wsi_ar_mae_all) if wsi_ar_mae_all > 0 else 0.0

        r_dict = {
            "model": m_name,
            "overall_mae": round(m_mae_all, 4),
            "mae_h1": round(float(row["mae_h1"]), 4),
            "mae_h2": round(float(row["mae_h2"]), 4),
            "mae_h3": round(float(row["mae_h3"]), 4),
            "overall_rmse": round(float(row["overall_rmse"]), 4),
            "rmse_h1": round(float(row["rmse_h1"]), 4),
            "rmse_h2": round(float(row["rmse_h2"]), 4),
            "rmse_h3": round(float(row["rmse_h3"]), 4),
            "overall_r2": round(float(row["overall_r2"]), 4),
            "r2_h1": round(float(row["r2_h1"]), 4),
            "r2_h2": round(float(row["r2_h2"]), 4),
            "r2_h3": round(float(row["r2_h3"]), 4),
            "skill_vs_persistence": round(float(row["skill_mae_vs_persistence"]), 4),
            "skill_vs_persistence_h1": round(float(row["skill_mae_h1"]), 4),
            "skill_vs_persistence_h2": round(float(row["skill_mae_h2"]), 4),
            "skill_vs_persistence_h3": round(float(row["skill_mae_h3"]), 4),
            "skill_vs_wsi_ar": round(float(skill_wsi_ar), 4),
            "skill_vs_wsi_ar_h1": round(float(1.0 - (float(row["mae_h1"]) / wsi_ar_mae_h["h1"])), 4),
            "skill_vs_wsi_ar_h2": round(float(1.0 - (float(row["mae_h2"]) / wsi_ar_mae_h["h2"])), 4),
            "skill_vs_wsi_ar_h3": round(float(1.0 - (float(row["mae_h3"]) / wsi_ar_mae_h["h3"])), 4),
        }
        comparison_rows.append(r_dict)

    # 2. LSTM row
    lstm_mae_all = float(lstm_metrics["overall_mae"])
    lstm_skill_persist = 1.0 - (lstm_mae_all / persist_mae_all) if persist_mae_all > 0 else 0.0
    lstm_skill_wsi_ar = 1.0 - (lstm_mae_all / wsi_ar_mae_all) if wsi_ar_mae_all > 0 else 0.0

    lstm_comp_row = {
        "model": "LSTM",
        "overall_mae": round(lstm_mae_all, 4),
        "mae_h1": round(float(lstm_metrics["mae_h1"]), 4),
        "mae_h2": round(float(lstm_metrics["mae_h2"]), 4),
        "mae_h3": round(float(lstm_metrics["mae_h3"]), 4),
        "overall_rmse": round(float(lstm_metrics["overall_rmse"]), 4),
        "rmse_h1": round(float(lstm_metrics["rmse_h1"]), 4),
        "rmse_h2": round(float(lstm_metrics["rmse_h2"]), 4),
        "rmse_h3": round(float(lstm_metrics["rmse_h3"]), 4),
        "overall_r2": round(float(lstm_metrics["overall_r2"]), 4),
        "r2_h1": round(float(lstm_metrics["r2_h1"]), 4),
        "r2_h2": round(float(lstm_metrics["r2_h2"]), 4),
        "r2_h3": round(float(lstm_metrics["r2_h3"]), 4),
        "skill_vs_persistence": round(float(lstm_skill_persist), 4),
        "skill_vs_persistence_h1": round(float(1.0 - (float(lstm_metrics["mae_h1"]) / persist_mae_h["h1"])), 4),
        "skill_vs_persistence_h2": round(float(1.0 - (float(lstm_metrics["mae_h2"]) / persist_mae_h["h2"])), 4),
        "skill_vs_persistence_h3": round(float(1.0 - (float(lstm_metrics["mae_h3"]) / persist_mae_h["h3"])), 4),
        "skill_vs_wsi_ar": round(float(lstm_skill_wsi_ar), 4),
        "skill_vs_wsi_ar_h1": round(float(1.0 - (float(lstm_metrics["mae_h1"]) / wsi_ar_mae_h["h1"])), 4),
        "skill_vs_wsi_ar_h2": round(float(1.0 - (float(lstm_metrics["mae_h2"]) / wsi_ar_mae_h["h2"])), 4),
        "skill_vs_wsi_ar_h3": round(float(1.0 - (float(lstm_metrics["mae_h3"]) / wsi_ar_mae_h["h3"])), 4),
    }
    comparison_rows.append(lstm_comp_row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(output_dir, "final_model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\nSaved 6-model final comparison: {comp_path}")

    # Determine test winner strictly by lowest test Mean MAE
    sorted_by_test = comp_df.sort_values("overall_mae").reset_index(drop=True)
    test_winner = sorted_by_test.loc[0, "model"]
    test_winner_mae = sorted_by_test.loc[0, "overall_mae"]

    # Calculate diagnostic improvements over Persistence and WSI-AR
    delta_persist = persist_mae_all - lstm_mae_all
    pct_persist = (delta_persist / persist_mae_all) * 100.0

    delta_wsi_ar = wsi_ar_mae_all - lstm_mae_all
    pct_wsi_ar = (delta_wsi_ar / wsi_ar_mae_all) * 100.0

    # Build decoupled final summary JSON
    summary_dict = {
        "baseline_champion_by_validation": "WSI_Autoregression",
        "baseline_champion_validation_mae": 0.5404,
        "lstm_validation_result": {
            "best_epoch": int(val_summary_data["E_star"]),
            "best_val_mae": round(float(val_summary_data["best_val_mae"]), 4),
            "best_val_mae_h1": round(float(val_summary_data["best_val_mae_h1"]), 4),
            "best_val_mae_h2": round(float(val_summary_data["best_val_mae_h2"]), 4),
            "best_val_mae_h3": round(float(val_summary_data["best_val_mae_h3"]), 4),
            "selected_lr_schedule": [float(x) for x in val_summary_data["selected_lr_schedule"]],
            "seed_robustness": {
                "seeds_evaluated": seed_diagnostics["seeds_evaluated"],
                "all_seeds_val_mae": seed_diagnostics["all_seeds_val_mae"],
                "mean_val_mae": seed_diagnostics["mean_val_mae"],
                "std_val_mae": seed_diagnostics["std_val_mae"],
                "mean_val_mae_by_horizon": seed_diagnostics["mean_val_mae_by_horizon"],
                "std_val_mae_by_horizon": seed_diagnostics["std_val_mae_by_horizon"],
            },
        },
        "lstm_test_result": {
            "overall_mae": round(lstm_mae_all, 4),
            "mae_h1": round(float(lstm_metrics["mae_h1"]), 4),
            "mae_h2": round(float(lstm_metrics["mae_h2"]), 4),
            "mae_h3": round(float(lstm_metrics["mae_h3"]), 4),
            "overall_rmse": round(float(lstm_metrics["overall_rmse"]), 4),
            "overall_r2": round(float(lstm_metrics["overall_r2"]), 4),
        },
        "test_winner": str(test_winner),
        "test_winner_mae": round(float(test_winner_mae), 4),
        "diagnostic_comparisons": {
            "persistence_benchmark": {
                "overall_mae": round(persist_mae_all, 4),
                "mae_h1": round(persist_mae_h["h1"], 4),
                "mae_h2": round(persist_mae_h["h2"], 4),
                "mae_h3": round(persist_mae_h["h3"], 4),
            },
            "wsi_ar_champion": {
                "overall_mae": round(wsi_ar_mae_all, 4),
                "mae_h1": round(wsi_ar_mae_h["h1"], 4),
                "mae_h2": round(wsi_ar_mae_h["h2"], 4),
                "mae_h3": round(wsi_ar_mae_h["h3"], 4),
            },
            "improvement_over_persistence": {
                "overall_delta_mae": round(float(delta_persist), 4),
                "overall_skill_pct": round(float(pct_persist), 2),
                "h1_delta_mae": round(float(persist_mae_h["h1"] - float(lstm_metrics["mae_h1"])), 4),
                "h1_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h1"]) / persist_mae_h["h1"])) * 100.0, 2),
                "h2_delta_mae": round(float(persist_mae_h["h2"] - float(lstm_metrics["mae_h2"])), 4),
                "h2_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h2"]) / persist_mae_h["h2"])) * 100.0, 2),
                "h3_delta_mae": round(float(persist_mae_h["h3"] - float(lstm_metrics["mae_h3"])), 4),
                "h3_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h3"]) / persist_mae_h["h3"])) * 100.0, 2),
                "lstm_beats_persistence_test": bool(lstm_mae_all < persist_mae_all),
            },
            "improvement_over_wsi_ar": {
                "overall_delta_mae": round(float(delta_wsi_ar), 4),
                "overall_skill_pct": round(float(pct_wsi_ar), 2),
                "h1_delta_mae": round(float(wsi_ar_mae_h["h1"] - float(lstm_metrics["mae_h1"])), 4),
                "h1_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h1"]) / wsi_ar_mae_h["h1"])) * 100.0, 2),
                "h2_delta_mae": round(float(wsi_ar_mae_h["h2"] - float(lstm_metrics["mae_h2"])), 4),
                "h2_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h2"]) / wsi_ar_mae_h["h2"])) * 100.0, 2),
                "h3_delta_mae": round(float(wsi_ar_mae_h["h3"] - float(lstm_metrics["mae_h3"])), 4),
                "h3_skill_pct": round(float(1.0 - (float(lstm_metrics["mae_h3"]) / wsi_ar_mae_h["h3"])) * 100.0, 2),
                "lstm_beats_wsi_ar_test": bool(lstm_mae_all < wsi_ar_mae_all),
            },
        },
        "reproducibility": {
            "framework": f"tensorflow {tf.__version__}",
            "primary_seed": 42,
            "diagnostic_seeds": seed_diagnostics["seeds_evaluated"],
            "op_determinism_enabled": True,
            "shuffle": False,
            "lr_schedule_replayed": True,
            "stopping_metric": "val_mae",
        },
    }

    summary_path = os.path.join(output_dir, "final_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved final model summary: {summary_path}")

    return summary_dict


# Main Step 7 Pipeline Entry Point

def run_lstm_pipeline(
    data_dir: str = "data_model",
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Orchestrates the end-to-end Step 7 LSTM training and evaluation pipeline.
    """
    print("Running Small LSTM training and benchmark comparison...")

    data = load_sequence_arrays(data_dir=data_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    meta_df = data["meta"]

    meta_val = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    meta_test = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    print(f"Loaded Sequence Tensors:")
    print(f"  TRAIN: {X_train.shape} -> {y_train.shape}")
    print(f"  VAL:   {X_val.shape} -> {y_val.shape}")
    print(f"  TEST:  {X_test.shape} -> {y_test.shape}")

    # Phase 1: Validation Run (Seed 42)
    val_out = train_validation_phase(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=42,
        batch_size=32,
        max_epochs=200,
        output_dir=os.path.join(results_dir, "lstm"),
    )

    # Save validation predictions (864 rows)
    val_records = assemble_prediction_records(
        model_name="LSTM",
        evaluation_type="validation",
        preds=val_out["val_preds"],
        y_actual=y_val,
        meta_subset=meta_val,
    )
    val_df = pd.DataFrame(val_records)
    val_pred_path = os.path.join(results_dir, "lstm", "validation_predictions.csv")
    val_df.to_csv(val_pred_path, index=False)
    print(f"Saved LSTM validation predictions: {val_pred_path} ({len(val_df)} rows)")

    # Phase 2: Diagnostic Runs (Seeds 43-46)
    seed_summary = run_seed_diagnostics(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        E_star=val_out["E_star"],
        frozen_lr_schedule=val_out["selected_lr_schedule"],
        diagnostic_seeds=[43, 44, 45, 46],
        batch_size=32,
        seed_42_val_mae=val_out["best_val_mae"],
        seed_42_val_maes_h={
            "h1": val_out["best_val_mae_h1"],
            "h2": val_out["best_val_mae_h2"],
            "h3": val_out["best_val_mae_h3"],
        },
    )

    # Phase 3: Final Model Refit on TRAIN + VAL
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    final_model, test_preds = train_final_model(
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        X_test=X_test,
        E_star=val_out["E_star"],
        frozen_lr_schedule=val_out["selected_lr_schedule"],
        seed=42,
        batch_size=32,
    )

    # Save final trained model artifact directly
    model_save_path = os.path.join(results_dir, "lstm", "model.keras")
    final_model.save(model_save_path)
    print(f"Saved trained model artifact: {model_save_path}")

    # Convert test predictions to standard long format (1,080 rows)
    test_records = assemble_prediction_records(
        model_name="LSTM",
        evaluation_type="test",
        preds=test_preds,
        y_actual=y_test,
        meta_subset=meta_test,
    )

    # Phase 4: Compile Final Reports & Comparison
    final_summary = compile_final_model_reports(
        lstm_test_records=test_records,
        val_summary_data=val_out,
        seed_diagnostics=seed_summary,
        baseline_predictions_path=os.path.join(results_dir, "baselines", "baseline_predictions.csv"),
        baseline_metrics_path=os.path.join(results_dir, "baselines", "baseline_metrics.csv"),
        output_dir=results_dir,
    )

    print("\n" + "=" * 75)
    print("STEP 7 COMPLETED SUCCESSFULLY")
    print(f"Validation Champion Baseline: {final_summary['baseline_champion_by_validation']}")
    print(f"LSTM Validation MAE:          {final_summary['lstm_validation_result']['best_val_mae']:.4f}")
    print(f"LSTM Test MAE:                {final_summary['lstm_test_result']['overall_mae']:.4f}")
    print(f"Persistence Test MAE:         {final_summary['diagnostic_comparisons']['persistence_benchmark']['overall_mae']:.4f}")
    print(f"WSI-AR Test MAE:              {final_summary['diagnostic_comparisons']['wsi_ar_champion']['overall_mae']:.4f}")
    print(f"Test Winner:                  {final_summary['test_winner']}")
    print(f"LSTM beats Persistence?       {final_summary['diagnostic_comparisons']['improvement_over_persistence']['lstm_beats_persistence_test']}")
    print(f"LSTM beats WSI-AR?            {final_summary['diagnostic_comparisons']['improvement_over_wsi_ar']['lstm_beats_wsi_ar_test']}")
    print(f"LSTM Test MAE:          {final_summary['lstm_test_result']['overall_mae']:.4f}")
    return final_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate small LSTM for Marathwada WSI forecasting")
    parser.add_argument("--data-dir", default="data_model", help="Directory containing Step 5 sequence npy arrays")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    args = parser.parse_args()

    run_lstm_pipeline(data_dir=args.data_dir, results_dir=args.results_dir)
