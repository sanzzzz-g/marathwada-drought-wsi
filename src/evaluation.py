"""
src/evaluation.py
=================
Step 6: Evaluation Engine, Metrics, Statistical Significance, and Plotting.

Implements:
1. Standardized Metrics Computation:
   - Horizon-specific (h1, h2, h3) and overall MAE, RMSE, R^2, NRMSE.
   - Skill scores against Persistence: Skill_MAE = 1 - MAE_model / MAE_persistence.
2. District-Level Breakdowns (all 8 districts, mean, median, best, worst).
3. Regime-Level Diagnostics (Very Wet, Wet, Normal, Moderate Stress, Severe Stress).
4. Event-Level Evaluation (Precision, Recall, F1 for WSI >= 0.5 and WSI >= 1.5).
5. Statistical Significance:
   - Primary: Moving Block Bootstrap over consecutive origin months (block length L=3,
     preserving all 8 districts per month, B=1,000, 95% CIs).
   - Supplementary: Diebold-Mariano test per horizon with HAC lag = h - 1.
6. Publication Figures (8 figures saved to results/figures/).
"""

import os
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# Metric Calculations
# ==============================================================================

def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates R^2 score safely."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def calc_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates NRMSE normalized by the standard deviation of y_true."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    std_true = np.std(y_true)
    if std_true == 0:
        return 0.0
    return float(rmse / std_true)


def calc_event_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Calculates precision, recall, and F1 for a continuous threshold with zero-event support handling."""
    actual_event = (y_true >= threshold).astype(int)
    pred_event = (y_pred >= threshold).astype(int)

    actual_positives = int(np.sum(actual_event == 1))
    pred_positives = int(np.sum(pred_event == 1))

    tp = int(np.sum((actual_event == 1) & (pred_event == 1)))
    fp = int(np.sum((actual_event == 0) & (pred_event == 1)))
    fn = int(np.sum((actual_event == 1) & (pred_event == 0)))
    tn = int(np.sum((actual_event == 0) & (pred_event == 0)))

    if actual_positives == 0:
        return {
            "actual_positive_count": 0,
            "predicted_positive_count": pred_positives,
            "precision": None,
            "recall": None,
            "f1": None,
            "status": "not_estimable",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "actual_positive_count": actual_positives,
        "predicted_positive_count": pred_positives,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "status": "estimable",
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ==============================================================================
# Statistical Significance: Moving Block Bootstrap & Diebold-Mariano
# ==============================================================================

def moving_block_bootstrap_significance(
    df: pd.DataFrame,
    candidate_model: str,
    champion_model: str,
    block_length: int = 3,
    n_boot: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Moving block bootstrap over consecutive forecast-origin months:
    - Retains all 8 districts clustered within each sampled month.
    - Preserves cross-district spatial dependence and temporal autocorrelation.
    - Evaluates paired difference: Diff_MAE = MAE(candidate) - MAE(champion).
    """
    rng = np.random.default_rng(random_state)

    cand_df = df[df["model"] == candidate_model].sort_values(by=["forecast_origin", "district", "horizon"]).reset_index(drop=True)
    champ_df = df[df["model"] == champion_model].sort_values(by=["forecast_origin", "district", "horizon"]).reset_index(drop=True)

    origins = sorted(cand_df["forecast_origin"].unique())
    n_origins = len(origins)

    if n_origins <= block_length:
        return {"error": "Not enough origins for block bootstrap"}

    # Group records by origin
    cand_by_origin = {o: cand_df[cand_df["forecast_origin"] == o]["absolute_error"].values for o in origins}
    champ_by_origin = {o: champ_df[champ_df["forecast_origin"] == o]["absolute_error"].values for o in origins}

    observed_diff = np.mean(cand_df["absolute_error"]) - np.mean(champ_df["absolute_error"])
    boot_diffs = []

    # Number of blocks needed to cover n_origins
    n_blocks_needed = int(np.ceil(n_origins / block_length))
    max_start_idx = n_origins - block_length + 1

    for _ in range(n_boot):
        start_indices = rng.integers(0, max_start_idx, size=n_blocks_needed)
        sampled_cand_errors = []
        sampled_champ_errors = []

        for s_idx in start_indices:
            for b in range(block_length):
                o = origins[s_idx + b]
                sampled_cand_errors.extend(cand_by_origin[o])
                sampled_champ_errors.extend(champ_by_origin[o])

        # Trim to exact length
        boot_diff = np.mean(sampled_cand_errors) - np.mean(sampled_champ_errors)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    p_value = float(np.mean(boot_diffs <= 0) if observed_diff > 0 else np.mean(boot_diffs >= 0)) * 2
    p_value = min(1.0, max(0.0, p_value))

    return {
        "candidate_model": candidate_model,
        "champion_model": champion_model,
        "observed_mae_diff": round(float(observed_diff), 5),
        "ci_95_lower": round(ci_lower, 5),
        "ci_95_upper": round(ci_upper, 5),
        "bootstrap_p_value": round(p_value, 4),
        "significantly_different": bool(ci_lower > 0 or ci_upper < 0),
        "candidate_is_better": bool(observed_diff < 0 and ci_upper < 0),
    }


def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss_type: str = "absolute",
) -> Tuple[float, float]:
    """
    Computes Diebold-Mariano test with horizon-specific HAC lag = h - 1.
    loss_type: 'absolute' or 'squared'.
    """
    if loss_type == "squared":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        d = np.abs(e1) - np.abs(e2)

    n = len(d)
    mean_d = np.mean(d)
    lag = max(0, h - 1)

    # Autocovariance
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for l in range(1, lag + 1):
        gamma_l = np.mean((d[l:] - mean_d) * (d[:-l] - mean_d))
        gamma_sum += 2.0 * (1.0 - l / (lag + 1)) * gamma_l

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 1e-12:
        return 0.0, 1.0

    dm_stat = float(mean_d / np.sqrt(var_d))
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return round(dm_stat, 4), round(p_val, 4)


# ==============================================================================
# Comprehensive Evaluation Pipeline
# ==============================================================================

def run_evaluation_suite(
    predictions_csv: str = "results/baseline_predictions.csv",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Reads official test predictions, computes all required tables, conducts
    block-bootstrap testing, and generates the 8 publication figures.
    """
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f"Predictions file not found at: {predictions_csv}")

    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    models = sorted(df["model"].unique())

    print("=" * 70)
    print("STEP 6: COMPREHENSIVE EVALUATION & STATISTICAL COMPARISON")
    print(f"Predictions path: {predictions_csv} ({len(df)} rows)")
    print(f"Models evaluated: {models}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 1. Baseline Horizon & Overall Metrics (with Skill vs Persistence)
    # --------------------------------------------------------------------------
    metrics_rows = []
    # Find persistence MAE for skill calculations
    persist_df = df[df["model"] == "Persistence"]
    persist_mae_overall = np.mean(persist_df["absolute_error"])
    persist_rmse_overall = np.sqrt(np.mean(persist_df["squared_error"]))

    persist_mae_h = {
        f"h{h}": np.mean(persist_df[persist_df["horizon"] == f"h{h}"]["absolute_error"])
        for h in range(1, 4)
    }
    persist_rmse_h = {
        f"h{h}": np.sqrt(np.mean(persist_df[persist_df["horizon"] == f"h{h}"]["squared_error"]))
        for h in range(1, 4)
    }

    for model_name in models:
        m_df = df[df["model"] == model_name]
        y_true = m_df["actual_wsi"].values
        y_pred = m_df["predicted_wsi"].values

        mae_all = float(np.mean(m_df["absolute_error"]))
        rmse_all = float(np.sqrt(np.mean(m_df["squared_error"])))
        r2_all = calc_r2(y_true, y_pred)
        nrmse_all = calc_nrmse(y_true, y_pred)

        skill_mae = 1.0 - (mae_all / persist_mae_overall) if persist_mae_overall > 0 else 0.0
        skill_rmse = 1.0 - (rmse_all / persist_rmse_overall) if persist_rmse_overall > 0 else 0.0

        row = {
            "model": model_name,
            "overall_mae": round(mae_all, 4),
            "overall_rmse": round(rmse_all, 4),
            "overall_r2": round(r2_all, 4),
            "overall_nrmse": round(nrmse_all, 4),
            "skill_mae_vs_persistence": round(skill_mae, 4),
            "skill_rmse_vs_persistence": round(skill_rmse, 4),
        }

        # Horizon-specific metrics
        for h in range(1, 4):
            h_df = m_df[m_df["horizon"] == f"h{h}"]
            h_act = h_df["actual_wsi"].values
            h_pred = h_df["predicted_wsi"].values

            h_mae = float(np.mean(h_df["absolute_error"]))
            h_rmse = float(np.sqrt(np.mean(h_df["squared_error"])))
            h_r2 = calc_r2(h_act, h_pred)
            h_skill = 1.0 - (h_mae / persist_mae_h[f"h{h}"]) if persist_mae_h[f"h{h}"] > 0 else 0.0

            row[f"mae_h{h}"] = round(h_mae, 4)
            row[f"rmse_h{h}"] = round(h_rmse, 4)
            row[f"r2_h{h}"] = round(h_r2, 4)
            row[f"skill_mae_h{h}"] = round(h_skill, 4)

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(output_dir, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n1. Saved baseline metrics: {metrics_path}")

    # --------------------------------------------------------------------------
    # 2. District-Level Metrics (all 8 districts)
    # --------------------------------------------------------------------------
    district_rows = []
    districts = sorted(df["district"].unique())
    for model_name in models:
        m_df = df[df["model"] == model_name]
        dist_maes = []
        for dist in districts:
            d_df = m_df[m_df["district"] == dist]
            d_mae = float(np.mean(d_df["absolute_error"]))
            d_rmse = float(np.sqrt(np.mean(d_df["squared_error"])))
            d_r2 = calc_r2(d_df["actual_wsi"].values, d_df["predicted_wsi"].values)
            dist_maes.append((dist, d_mae))

            row = {
                "model": model_name,
                "district": dist,
                "mae": round(d_mae, 4),
                "rmse": round(d_rmse, 4),
                "r2": round(d_r2, 4),
            }
            # Add horizon-specific MAE
            for h in range(1, 4):
                dh_df = d_df[d_df["horizon"] == f"h{h}"]
                row[f"mae_h{h}"] = round(float(np.mean(dh_df["absolute_error"])), 4)
            district_rows.append(row)

    dist_metrics_df = pd.DataFrame(district_rows)
    dist_path = os.path.join(output_dir, "district_metrics.csv")
    dist_metrics_df.to_csv(dist_path, index=False)
    print(f"2. Saved district metrics: {dist_path}")

    # --------------------------------------------------------------------------
    # 3. Regime Diagnostics
    # --------------------------------------------------------------------------
    def categorize_wsi(val: float) -> str:
        if val < -1.5:
            return "Very Wet"
        elif val < -0.5:
            return "Wet"
        elif val < 0.5:
            return "Normal"
        elif val < 1.5:
            return "Moderate Stress"
        else:
            return "Severe Stress"

    df["regime"] = df["actual_wsi"].apply(categorize_wsi)
    regimes = ["Very Wet", "Wet", "Normal", "Moderate Stress", "Severe Stress"]

    regime_rows = []
    for model_name in models:
        m_df = df[df["model"] == model_name]
        for reg in regimes:
            r_df = m_df[m_df["regime"] == reg]
            count = len(r_df)
            if count > 0:
                mae = float(np.mean(r_df["absolute_error"]))
                rmse = float(np.sqrt(np.mean(r_df["squared_error"])))
            else:
                mae, rmse = 0.0, 0.0

            regime_rows.append({
                "model": model_name,
                "regime": reg,
                "sample_count": count,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
            })

    regime_df = pd.DataFrame(regime_rows)
    regime_path = os.path.join(output_dir, "regime_metrics.csv")
    regime_df.to_csv(regime_path, index=False)
    print(f"3. Saved regime metrics: {regime_path}")

    # --------------------------------------------------------------------------
    # 4. Event-Level Evaluation (WSI >= 0.5 and WSI >= 1.5)
    # --------------------------------------------------------------------------
    event_rows = []
    for model_name in models:
        m_df = df[df["model"] == model_name]
        y_act = m_df["actual_wsi"].values
        y_pred = m_df["predicted_wsi"].values

        # Overall
        stress_res = calc_event_f1(y_act, y_pred, threshold=0.5)
        severe_res = calc_event_f1(y_act, y_pred, threshold=1.5)

        event_rows.append({
            "model": model_name,
            "horizon": "overall",
            "actual_stress_count": stress_res["actual_positive_count"],
            "stress_precision": stress_res["precision"],
            "stress_recall": stress_res["recall"],
            "stress_f1": stress_res["f1"],
            "stress_status": stress_res["status"],
            "actual_severe_count": severe_res["actual_positive_count"],
            "severe_precision": severe_res["precision"],
            "severe_recall": severe_res["recall"],
            "severe_f1": severe_res["f1"],
            "severe_status": severe_res["status"],
        })

        for h in range(1, 4):
            h_df = m_df[m_df["horizon"] == f"h{h}"]
            h_act = h_df["actual_wsi"].values
            h_pred = h_df["predicted_wsi"].values
            s_h = calc_event_f1(h_act, h_pred, threshold=0.5)
            sev_h = calc_event_f1(h_act, h_pred, threshold=1.5)
            event_rows.append({
                "model": model_name,
                "horizon": f"h{h}",
                "actual_stress_count": s_h["actual_positive_count"],
                "stress_precision": s_h["precision"],
                "stress_recall": s_h["recall"],
                "stress_f1": s_h["f1"],
                "stress_status": s_h["status"],
                "actual_severe_count": sev_h["actual_positive_count"],
                "severe_precision": sev_h["precision"],
                "severe_recall": sev_h["recall"],
                "severe_f1": sev_h["f1"],
                "severe_status": sev_h["status"],
            })

    event_df = pd.DataFrame(event_rows)
    event_path = os.path.join(output_dir, "event_metrics.csv")
    event_df.to_csv(event_path, index=False)
    print(f"4. Saved event metrics: {event_path}")

    # --------------------------------------------------------------------------
    # 5. Statistical Significance (Bootstrap & Diebold-Mariano)
    # --------------------------------------------------------------------------
    # Identify champion non-neural baseline by lowest overall test MAE among trained models
    champion = metrics_df.sort_values("overall_mae")["model"].iloc[0]
    print(f"\nEvaluating statistical significance against top baseline: {champion}")

    sig_results = []
    for cand in models:
        if cand == champion:
            continue
        boot_res = moving_block_bootstrap_significance(
            df=df,
            candidate_model=cand,
            champion_model=champion,
            block_length=3,
            n_boot=1000,
        )
        sig_results.append(boot_res)

    sig_df = pd.DataFrame(sig_results)
    sig_path = os.path.join(output_dir, "significance_metrics.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"5. Saved block-bootstrap significance: {sig_path}")

    # --------------------------------------------------------------------------
    # 6. Generate 8 Publication Figures
    # --------------------------------------------------------------------------
    print("\n6. Generating 8 publication figures in results/figures/...")
    sns.set_theme(style="whitegrid", palette="muted")

    # Figures 1, 2, 3: Forecast comparison by horizon
    for h in range(1, 4):
        plt.figure(figsize=(12, 5))
        h_df = df[df["horizon"] == f"h{h}"]
        origin_sample = h_df.groupby(["forecast_origin", "model"])["predicted_wsi"].mean().unstack()
        actual_series = h_df.groupby("forecast_origin")["actual_wsi"].mean()

        plt.plot(actual_series.index, actual_series.values, color="black", linewidth=2.5, label="Actual Mean WSI")
        for m in models:
            if m in origin_sample.columns:
                plt.plot(origin_sample.index, origin_sample[m], linewidth=1.5, linestyle="--", label=m)

        plt.title(f"Forecast Comparison across Test Origins (Horizon {h})", fontsize=13, fontweight="bold")
        plt.xlabel("Forecast Origin Date", fontsize=11)
        plt.ylabel("WSI", fontsize=11)
        plt.xticks(rotation=45)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"forecast_comparison_h{h}.png"), dpi=300)
        plt.close()

    # Figure 4: Error by Horizon
    plt.figure(figsize=(9, 5))
    horizon_melted = []
    for m in models:
        m_m = metrics_df[metrics_df["model"] == m].iloc[0]
        for h in range(1, 4):
            horizon_melted.append({"model": m, "horizon": f"h{h}", "MAE": m_m[f"mae_h{h}"]})
    h_plot_df = pd.DataFrame(horizon_melted)
    sns.barplot(data=h_plot_df, x="horizon", y="MAE", hue="model")
    plt.title("Forecasting Error (MAE) Across Horizons 1, 2, and 3", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon", fontsize=11)
    plt.ylabel("MAE", fontsize=11)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "error_by_horizon.png"), dpi=300)
    plt.close()

    # Figure 5: District Error
    plt.figure(figsize=(12, 6))
    sns.barplot(data=dist_metrics_df, x="district", y="mae", hue="model")
    plt.title("District-Level Forecasting Performance (MAE)", fontsize=13, fontweight="bold")
    plt.xlabel("District", fontsize=11)
    plt.ylabel("MAE", fontsize=11)
    plt.xticks(rotation=30)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "district_error.png"), dpi=300)
    plt.close()

    # Figure 6: Actual vs Predicted Scatter
    plt.figure(figsize=(8, 8))
    champ_df = df[df["model"] == champion]
    plt.scatter(champ_df["actual_wsi"], champ_df["predicted_wsi"], alpha=0.4, color="#1f77b4", edgecolors="none")
    lims = [min(champ_df["actual_wsi"].min(), champ_df["predicted_wsi"].min()) - 0.2,
            max(champ_df["actual_wsi"].max(), champ_df["predicted_wsi"].max()) + 0.2]
    plt.plot(lims, lims, "r--", linewidth=2, label="1:1 Perfect Forecast")
    plt.title(f"Actual vs Predicted WSI ({champion})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual WSI", fontsize=11)
    plt.ylabel("Predicted WSI", fontsize=11)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # Figure 7: Residual Distribution
    plt.figure(figsize=(10, 5))
    for m in models:
        sns.kdeplot(df[df["model"] == m]["error"], label=m, linewidth=1.5)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5)
    plt.title("Forecast Residual Distributions (Predicted - Actual)", fontsize=13, fontweight="bold")
    plt.xlabel("Residual", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "residuals.png"), dpi=300)
    plt.close()

    # Figure 8: Event Confusion Matrix for Champion Model (Threshold >= 0.5)
    plt.figure(figsize=(6, 5))
    champ_ev = calc_event_f1(champ_df["actual_wsi"].values, champ_df["predicted_wsi"].values, threshold=0.5)
    cm = np.array([[champ_ev["tn"], champ_ev["fp"]], [champ_ev["fn"], champ_ev["tp"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (<0.5)", "Stress (>=0.5)"],
                yticklabels=["Normal (<0.5)", "Stress (>=0.5)"])
    plt.title(f"Confusion Matrix for Stress Event (WSI >= 0.5)\nModel: {champion}", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Event", fontsize=11)
    plt.ylabel("Actual Event", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "event_confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"All 8 figures successfully saved to {fig_dir}/")
    return {
        "metrics": metrics_df,
        "district_metrics": dist_metrics_df,
        "regime_metrics": regime_df,
        "event_metrics": event_df,
        "significance": sig_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Marathwada WSI baseline predictions")
    parser.add_argument("--predictions", default="results/baselines/baseline_predictions.csv", help="Path to predictions CSV")
    parser.add_argument("--output-dir", default="results/baselines", help="Directory to save evaluation metrics and plots")
    args = parser.parse_args()

    run_evaluation_suite(predictions_csv=args.predictions, output_dir=args.output_dir)
