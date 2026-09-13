"""
Historical monthly rolling-origin diagnostic backtesting (2011–2020).
Strict causal training boundaries with causal historical preprocessing.
"""

import os
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


FEATURES = [
    "spi_stress_z",
    "soil_stress_z",
    "ndvi_stress_z",
    "lst_stress_z",
    "groundwater_stress_z",
    "surface_water_stress_z",
]


def build_causal_sequences_at_origin(
    panel_df: pd.DataFrame,
    origin_t: pd.Timestamp,
    lookback: int = 12,
    horizon: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Constructs training sequences and target forecast input at backtest origin t:
    - Filters data strictly to date <= origin_t.
    - Standardizes the 6 features causally using statistics up to origin_t.
    - Training sequences have origin o <= origin_t - 3 months (targets <= origin_t).
    - Current sample for origin_t has input [origin_t - 11 ... origin_t].
    """
    hist_df = panel_df[panel_df["date"] <= origin_t].copy()
    districts = sorted(hist_df["district"].unique())

    # Causal Standardization per district
    # Features in panel_df already have sign-oriented stress. We standardize each feature by (dist) mean & std up to t
    std_df = hist_df.copy()
    for dist in districts:
        d_mask = std_df["district"] == dist
        for feat in FEATURES:
            vals = std_df.loc[d_mask, feat].values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if std_val < 1e-6:
                std_val = 1.0
            std_df.loc[d_mask, feat] = (vals - mean_val) / std_val

    # Recompute WSI as exact mean of 6 standardized features
    std_df["wsi"] = std_df[FEATURES].mean(axis=1)

    X_train_list = []
    y_train_list = []
    wsi_train_list = []

    X_curr_list = []
    wsi_curr_list = []
    meta_curr_list = []

    latest_train_origin = origin_t - pd.DateOffset(months=horizon)

    for dist in districts:
        dist_df = std_df[std_df["district"] == dist].sort_values("date").reset_index(drop=True)
        dates = dist_df["date"].tolist()
        feat_mat = dist_df[FEATURES].values.astype(np.float32)
        wsi_vec = dist_df["wsi"].values.astype(np.float32)
        n_rows = len(dist_df)

        # 1. Historical training samples (origins <= latest_train_origin)
        for i in range(lookback - 1, n_rows - horizon):
            o_date = dates[i]
            if o_date <= latest_train_origin:
                X_sample = feat_mat[i - 11 : i + 1]  # (12, 6)
                y_sample = wsi_vec[i + 1 : i + 4]    # (3,)
                wsi_sample = wsi_vec[i - 11 : i + 1] # (12,)

                X_train_list.append(X_sample)
                y_train_list.append(y_sample)
                wsi_train_list.append(wsi_sample)

        # 2. Current origin forecast sample
        origin_matches = dist_df[dist_df["date"] == origin_t].index
        if len(origin_matches) > 0:
            idx_t = origin_matches[0]
            if idx_t >= lookback - 1:
                X_curr = feat_mat[idx_t - 11 : idx_t + 1]
                wsi_curr = wsi_vec[idx_t - 11 : idx_t + 1]
                X_curr_list.append(X_curr)
                wsi_curr_list.append(wsi_curr)

                meta_curr_list.append({
                    "district": dist,
                    "forecast_origin": origin_t.strftime("%Y-%m-%d"),
                    "target_t1": (origin_t + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
                    "target_t2": (origin_t + pd.DateOffset(months=2)).strftime("%Y-%m-%d"),
                    "target_t3": (origin_t + pd.DateOffset(months=3)).strftime("%Y-%m-%d"),
                })

    return (
        np.array(X_train_list, dtype=np.float32),
        np.array(y_train_list, dtype=np.float32),
        np.array(wsi_train_list, dtype=np.float32),
        np.array(X_curr_list, dtype=np.float32),
        meta_curr_list,
        {"wsi_curr": np.array(wsi_curr_list, dtype=np.float32)},
    )


def run_historical_rolling_backtest(
    panel_csv: str = "outputs/Marathwada_MODEL_DATASET_2003_2024.csv",
    output_dir: str = "results/backtesting",
    start_year: int = 2011,
    end_year: int = 2020,
    stride_months: int = 1,  # Monthly rolling origins
) -> Dict[str, Any]:
    """
    Runs systematic historical rolling-origin diagnostic backtest across start_year to end_year.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(panel_csv):
        raise FileNotFoundError(f"Model panel dataset not found at: {panel_csv}")

    panel_df = pd.read_csv(panel_csv)
    panel_df["date"] = pd.to_datetime(panel_df["date"])

    # Create target truth lookup (unmodified true WSI)
    truth_lookup = {}
    for _, row in panel_df.iterrows():
        truth_lookup[(row["district"], row["date"].strftime("%Y-%m-%d"))] = float(row["wsi"])

    # Define monthly origin range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-01")
    origin_dates = pd.date_range(start=start_date, end=end_date, freq="MS")[::stride_months]

    print(f"Running historical rolling-origin backtest ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}, {len(origin_dates)} origins)")

    backtest_records = []

    for step_num, origin_t in enumerate(origin_dates, start=1):
        if step_num % 12 == 0 or step_num == 1:
            print(f"  -> Processing origin {step_num}/{len(origin_dates)}: {origin_t.strftime('%Y-%m-%d')}...")

        X_tr, y_tr, wsi_tr, X_curr, meta_curr, extras = build_causal_sequences_at_origin(
            panel_df=panel_df,
            origin_t=origin_t,
        )
        if len(X_curr) == 0 or len(X_tr) == 0:
            continue

        wsi_curr = extras["wsi_curr"]
        n_dist = len(meta_curr)

        # Flattened inputs
        X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
        X_curr_flat = X_curr.reshape(X_curr.shape[0], -1)

        # 1. Persistence
        persist_preds = np.repeat(wsi_curr[:, -1:], repeats=3, axis=1)

        # 2. Seasonal Persistence (lookback step 0, 1, 2 = t-11, t-10, t-9)
        seasonal_preds = np.zeros((n_dist, 3), dtype=np.float32)
        for i, meta in enumerate(meta_curr):
            dist = meta["district"]
            for h in range(1, 4):
                t_date = pd.Timestamp(meta[f"target_t{h}"])
                lag_key = (dist, (t_date - pd.DateOffset(months=12)).strftime("%Y-%m-%d"))
                seasonal_preds[i, h - 1] = truth_lookup.get(lag_key, wsi_curr[i, h - 1])

        # 3. WSI Autoregression (AR-12 with Ridge)
        scaler_wsi = StandardScaler()
        wsi_tr_scaled = scaler_wsi.fit_transform(wsi_tr)
        wsi_curr_scaled = scaler_wsi.transform(wsi_curr)
        ar_model = Ridge(alpha=1.0)
        ar_model.fit(wsi_tr_scaled, y_tr)
        ar_preds = ar_model.predict(wsi_curr_scaled).astype(np.float32)

        # 4. Multivariate Ridge
        scaler_x = StandardScaler()
        X_tr_scaled = scaler_x.fit_transform(X_tr_flat)
        X_curr_scaled = scaler_x.transform(X_curr_flat)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_tr_scaled, y_tr)
        ridge_preds = ridge_model.predict(X_curr_scaled).astype(np.float32)

        # 5. Fast XGBoost (50 estimators for backtest efficiency)
        xgb_preds = np.zeros((n_dist, 3), dtype=np.float32)
        for h in range(3):
            xgb_m = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.08,
                random_state=42,
                n_jobs=-1,
            )
            xgb_m.fit(X_tr_flat, y_tr[:, h], verbose=False)
            xgb_preds[:, h] = xgb_m.predict(X_curr_flat)

        model_predictions = {
            "Persistence": persist_preds,
            "Seasonal_Persistence": seasonal_preds,
            "WSI_Autoregression": ar_preds,
            "Ridge_Regression": ridge_preds,
            "XGBoost": xgb_preds,
        }

        # Assemble records
        for i, meta in enumerate(meta_curr):
            dist = meta["district"]
            origin_str = meta["forecast_origin"]

            for h in range(1, 4):
                target_str = meta[f"target_t{h}"]
                actual_val = truth_lookup.get((dist, target_str), np.nan)
                if np.isnan(actual_val):
                    continue

                for m_name, p_array in model_predictions.items():
                    pred_val = float(p_array[i, h - 1])
                    err = pred_val - actual_val
                    backtest_records.append({
                        "evaluation_type": "historical_diagnostic",
                        "model": m_name,
                        "district": dist,
                        "forecast_origin": origin_str,
                        "target_date": target_str,
                        "horizon": f"h{h}",
                        "actual_wsi": round(actual_val, 6),
                        "predicted_wsi": round(pred_val, 6),
                        "error": round(err, 6),
                        "absolute_error": round(abs(err), 6),
                        "squared_error": round(err ** 2, 6),
                    })

    backtest_df = pd.DataFrame(backtest_records)
    preds_out_path = os.path.join(output_dir, "rolling_backtest_predictions.csv")
    backtest_df.to_csv(preds_out_path, index=False)

    # Compute rolling backtest summary metrics
    metrics_summary = []
    models = sorted(backtest_df["model"].unique())

    # Acute drought period subset (2014–2016)
    drought_subset = backtest_df[backtest_df["forecast_origin"].between("2014-01-01", "2016-12-31")]

    for m in models:
        m_df = backtest_df[backtest_df["model"] == m]
        d_df = drought_subset[drought_subset["model"] == m]

        row = {
            "model": m,
            "total_evaluations": len(m_df),
            "overall_mae": round(float(np.mean(m_df["absolute_error"])), 4),
            "overall_rmse": round(float(np.sqrt(np.mean(m_df["squared_error"]))), 4),
            "drought_2014_2016_mae": round(float(np.mean(d_df["absolute_error"])), 4) if len(d_df) > 0 else 0.0,
            "drought_2014_2016_rmse": round(float(np.sqrt(np.mean(d_df["squared_error"]))), 4) if len(d_df) > 0 else 0.0,
        }
        for h in range(1, 4):
            h_df = m_df[m_df["horizon"] == f"h{h}"]
            row[f"mae_h{h}"] = round(float(np.mean(h_df["absolute_error"])), 4)
            row[f"rmse_h{h}"] = round(float(np.sqrt(np.mean(h_df["squared_error"]))), 4)

        metrics_summary.append(row)

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_out_path = os.path.join(output_dir, "rolling_backtest_metrics.csv")
    metrics_df.to_csv(metrics_out_path, index=False)

    print(f"\nHistorical rolling backtest complete:")
    print(f"  - Saved predictions: {preds_out_path} ({len(backtest_df)} rows)")
    print(f"  - Saved summary metrics: {metrics_out_path}")

    return {
        "predictions": backtest_df,
        "metrics": metrics_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marathwada WSI Historical Monthly Rolling Backtest")
    parser.add_argument("--panel-path", default="outputs/Marathwada_MODEL_DATASET_2003_2024.csv", help="Path to frozen model panel")
    parser.add_argument("--output-dir", default="results/backtesting", help="Directory to save backtest predictions and metrics")
    parser.add_argument("--start-year", type=int, default=2011, help="Start year for backtesting")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for backtesting")
    parser.add_argument("--stride", type=int, default=1, help="Month stride (default 1 for monthly)")
    args = parser.parse_args()

    run_historical_rolling_backtest(
        panel_csv=args.panel_path,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        stride_months=args.stride,
    )
