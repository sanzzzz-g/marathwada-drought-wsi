"""
src/baselines.py
================
Step 6: Baseline Forecasting Suite for 3-Month Water Stress Index (WSI) Prediction.

Implements five frozen baseline forecasters:
1. PersistenceForecaster: Hat{Y}_t = [WSI_t, WSI_t, WSI_t]
2. SeasonalPersistenceForecaster: Hat{WSI}_{t+h} = WSI_{t+h-12}
3. WSIAutoregressiveForecaster: Univariate AR(12) on historical WSI trajectory
4. RidgeForecaster: Multivariate flattened Ridge (72 features -> 3 outputs)
5. XGBoostForecaster: Three independent horizon regressors (XGB_h1, XGB_h2, XGB_h3)

Enforces:
- Strict split isolation: Fit on TRAIN -> Tune on VAL -> Select champion on VAL -> Refit on TRAIN+VAL -> Evaluate once on TEST.
- Deterministic Champion Selection:
    Primary: min(mean validation MAE across h1, h2, h3)
    Tie-breaker: min(mean validation RMSE across h1, h2, h3)
- Output namespace separation:
    results/validation_predictions.csv (4,320 rows)
    results/baseline_predictions.csv   (5,400 rows)
    results/baseline_summary.json
"""

import os
import json
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb


# ==============================================================================
# Abstract Base Forecaster Interface
# ==============================================================================

class BaseForecaster(ABC):
    """Abstract base class for all 3-month multi-step WSI forecasters."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.best_params_: Dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "BaseForecaster":
        """Fit model on training data, optionally using validation data for hyperparameter tuning."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Generate 3-step predictions of shape (N, 3)."""
        pass

    def refit(self, X_train_val: np.ndarray, y_train_val: np.ndarray, **kwargs) -> "BaseForecaster":
        """Refit model using selected hyperparameters on combined Train + Validation data."""
        return self.fit(X_train_val, y_train_val, **kwargs)


# ==============================================================================
# Baseline 1: Persistence Forecaster
# ==============================================================================

class PersistenceForecaster(BaseForecaster):
    """
    Persistence baseline: Hat{Y}_t = [WSI_t, WSI_t, WSI_t].
    WSI_t is the value at forecast origin (step 12 of the lookback window).
    Zero parameters, no fitting, no tuning. Mathematical skill == 0.00.
    """

    def __init__(self):
        super().__init__(name="Persistence")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "PersistenceForecaster":
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, origin_wsi: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        If origin_wsi is provided, uses shape (N,).
        Otherwise, computes WSI as the mean of the 6 components at the last timestep: X[:, -1, :].mean(axis=-1).
        """
        if origin_wsi is not None:
            wsi_t = np.asarray(origin_wsi).reshape(-1, 1)
        else:
            wsi_t = X[:, -1, :].mean(axis=-1, keepdims=True)
        return np.repeat(wsi_t, repeats=3, axis=1).astype(np.float32)


# ==============================================================================
# Baseline 2: Seasonal Persistence Forecaster
# ==============================================================================

class SeasonalPersistenceForecaster(BaseForecaster):
    """
    Seasonal persistence baseline: Hat{WSI}_{t+h} = WSI_{t+h-12} for h in {1, 2, 3}.
    Uses historical WSI from exactly 12 months prior to each forecast horizon.
    Tests whether models genuinely add skill beyond annual hydrological seasonality.
    """

    def __init__(self, panel_df: Optional[pd.DataFrame] = None):
        super().__init__(name="Seasonal_Persistence")
        self.panel_lookup: Dict[Tuple[str, str], float] = {}
        if panel_df is not None:
            self._build_lookup(panel_df)

    def _build_lookup(self, panel_df: pd.DataFrame) -> None:
        df = panel_df.copy()
        wsi_col = "wsi" if "wsi" in df.columns else "WSI"
        df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for _, row in df.iterrows():
            self.panel_lookup[(row["district"], row["date_str"])] = float(row[wsi_col])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "SeasonalPersistenceForecaster":
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        metadata_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predicts 3 horizons using WSI at (target_date - 12 months).
        If metadata_df is not provided, falls back to lookback steps:
        h=1 -> t-11 (step 0), h=2 -> t-10 (step 1), h=3 -> t-9 (step 2).
        """
        n_samples = len(X)
        preds = np.zeros((n_samples, 3), dtype=np.float32)

        if metadata_df is not None and len(self.panel_lookup) > 0:
            for i, (_, row) in enumerate(metadata_df.iterrows()):
                dist = row["district"]
                for h in range(1, 4):
                    target_date = pd.Timestamp(row[f"target_t{h}"])
                    lag12_date = (target_date - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
                    key = (dist, lag12_date)
                    if key in self.panel_lookup:
                        preds[i, h - 1] = self.panel_lookup[key]
                    else:
                        # Fallback to lookback window step (h-1)
                        preds[i, h - 1] = X[i, h - 1, :].mean()
        else:
            # Lookback step 0 is t-11 (target_t1 - 12 months)
            # Lookback step 1 is t-10 (target_t2 - 12 months)
            # Lookback step 2 is t-9  (target_t3 - 12 months)
            for h in range(1, 4):
                preds[:, h - 1] = X[:, h - 1, :].mean(axis=-1)

        return preds


# ==============================================================================
# Baseline 3: WSI Autoregression Forecaster (Univariate AR)
# ==============================================================================

class WSIAutoregressiveForecaster(BaseForecaster):
    """
    Direct univariate baseline using historical WSI trajectory:
    WSI(t-11:t) in R^12 -> [WSI(t+1), WSI(t+2), WSI(t+3)] in R^3.
    Evaluates whether the 6 environmental features beat historical WSI alone.
    """

    def __init__(self, alphas: Optional[List[float]] = None):
        super().__init__(name="WSI_Autoregression")
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        wsi_train: Optional[np.ndarray] = None,
        wsi_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "WSIAutoregressiveForecaster":
        # Extract univariate WSI input: shape (N, 12)
        W_tr = wsi_train if wsi_train is not None else X_train.mean(axis=-1)
        best_alpha = 1.0
        best_val_mae = float("inf")

        if X_val is not None and y_val is not None:
            W_vl = wsi_val if wsi_val is not None else X_val.mean(axis=-1)
            scaler_temp = StandardScaler()
            W_tr_scaled = scaler_temp.fit_transform(W_tr)
            W_vl_scaled = scaler_temp.transform(W_vl)

            for alpha in self.alphas:
                m = Ridge(alpha=alpha)
                m.fit(W_tr_scaled, y_train)
                preds = m.predict(W_vl_scaled)
                mae = np.mean(np.abs(preds - y_val))
                if mae < best_val_mae:
                    best_val_mae = mae
                    best_alpha = alpha

        self.best_params_ = {"alpha": best_alpha}
        W_scaled = self.scaler.fit_transform(W_tr)
        self.model = Ridge(alpha=best_alpha)
        self.model.fit(W_scaled, y_train)
        self.is_fitted = True
        return self

    def refit(
        self,
        X_train_val: np.ndarray,
        y_train_val: np.ndarray,
        wsi_train_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "WSIAutoregressiveForecaster":
        W_tr_vl = wsi_train_val if wsi_train_val is not None else X_train_val.mean(axis=-1)
        alpha = self.best_params_.get("alpha", 1.0)
        W_scaled = self.scaler.fit_transform(W_tr_vl)
        self.model = Ridge(alpha=alpha)
        self.model.fit(W_scaled, y_train_val)
        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        wsi_input: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        W = wsi_input if wsi_input is not None else X.mean(axis=-1)
        W_scaled = self.scaler.transform(W)
        return self.model.predict(W_scaled).astype(np.float32)


# ==============================================================================
# Baseline 4: Linear Multivariate Model (Ridge)
# ==============================================================================

class RidgeForecaster(BaseForecaster):
    """
    Linear autoregressive multivariate baseline:
    Flattened 12 x 6 = 72 features -> 3 outputs.
    StandardScaler + Ridge(alpha) fit strictly on train, tuned on validation,
    refit on train+val.
    """

    def __init__(self, alphas: Optional[List[float]] = None):
        super().__init__(name="Ridge_Regression")
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "RidgeForecaster":
        X_tr_flat = self._flatten(X_train)
        best_alpha = 1.0
        best_val_mae = float("inf")

        if X_val is not None and y_val is not None:
            X_vl_flat = self._flatten(X_val)
            scaler_temp = StandardScaler()
            X_tr_scaled = scaler_temp.fit_transform(X_tr_flat)
            X_vl_scaled = scaler_temp.transform(X_vl_flat)

            for alpha in self.alphas:
                m = Ridge(alpha=alpha)
                m.fit(X_tr_scaled, y_train)
                preds = m.predict(X_vl_scaled)
                mae = np.mean(np.abs(preds - y_val))
                if mae < best_val_mae:
                    best_val_mae = mae
                    best_alpha = alpha

        self.best_params_ = {"alpha": best_alpha}
        X_scaled = self.scaler.fit_transform(X_tr_flat)
        self.model = Ridge(alpha=best_alpha)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def refit(
        self,
        X_train_val: np.ndarray,
        y_train_val: np.ndarray,
        **kwargs,
    ) -> "RidgeForecaster":
        X_flat = self._flatten(X_train_val)
        alpha = self.best_params_.get("alpha", 1.0)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y_train_val)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled).astype(np.float32)


# ==============================================================================
# Baseline 5: Nonlinear Decoupled Tree Models (XGBoost)
# ==============================================================================

class XGBoostForecaster(BaseForecaster):
    """
    Nonlinear baseline: Three separate XGBoost regressors (XGB_h1, XGB_h2, XGB_h3).
    Input: Flattened 72 features.
    Outputs: WSI(t+1), WSI(t+2), WSI(t+3) predicted by independent models.
    Each gets its own early stopping and hyperparameter selection.
    """

    def __init__(self, random_state: int = 42):
        super().__init__(name="XGBoost")
        self.random_state = random_state
        self.models: List[xgb.XGBRegressor] = []
        self.best_iterations: List[int] = [100, 100, 100]

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "XGBoostForecaster":
        X_tr_flat = self._flatten(X_train)
        X_vl_flat = self._flatten(X_val) if X_val is not None else None

        self.models = []
        self.best_params_ = {}

        candidate_params = [
            {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
            {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.5, "reg_lambda": 1.0},
            {"max_depth": 3, "learning_rate": 0.10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_alpha": 0.1, "reg_lambda": 1.0},
        ]

        for h in range(3):
            y_tr_h = y_train[:, h]
            y_vl_h = y_val[:, h] if y_val is not None else None

            best_h_params = candidate_params[0]
            best_h_mae = float("inf")
            best_n_trees = 100

            if X_vl_flat is not None and y_vl_h is not None:
                for params in candidate_params:
                    model = xgb.XGBRegressor(
                        n_estimators=300,
                        early_stopping_rounds=15,
                        random_state=self.random_state,
                        n_jobs=-1,
                        **params,
                    )
                    model.fit(
                        X_tr_flat,
                        y_tr_h,
                        eval_set=[(X_vl_flat, y_vl_h)],
                        verbose=False,
                    )
                    preds = model.predict(X_vl_flat)
                    mae = np.mean(np.abs(preds - y_vl_h))
                    if mae < best_h_mae:
                        best_h_mae = mae
                        best_h_params = params
                        best_n_trees = model.best_iteration + 1 if hasattr(model, "best_iteration") and model.best_iteration is not None else 100

            self.best_params_[f"h{h+1}"] = {**best_h_params, "n_estimators": best_n_trees}
            self.best_iterations[h] = best_n_trees

            final_model = xgb.XGBRegressor(
                n_estimators=best_n_trees,
                random_state=self.random_state,
                n_jobs=-1,
                **best_h_params,
            )
            final_model.fit(X_tr_flat, y_tr_h, verbose=False)
            self.models.append(final_model)

        self.is_fitted = True
        return self

    def refit(
        self,
        X_train_val: np.ndarray,
        y_train_val: np.ndarray,
        **kwargs,
    ) -> "XGBoostForecaster":
        X_flat = self._flatten(X_train_val)
        self.models = []

        for h in range(3):
            params = self.best_params_.get(f"h{h+1}", {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100})
            n_trees = params.get("n_estimators", 100)
            clean_params = {k: v for k, v in params.items() if k != "n_estimators"}

            model = xgb.XGBRegressor(
                n_estimators=n_trees,
                random_state=self.random_state,
                n_jobs=-1,
                **clean_params,
            )
            model.fit(X_flat, y_train_val[:, h], verbose=False)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        X_flat = self._flatten(X)
        n_samples = len(X)
        preds = np.zeros((n_samples, 3), dtype=np.float32)
        for h in range(3):
            preds[:, h] = self.models[h].predict(X_flat)
        return preds


# ==============================================================================
# Helper Functions: Data Loading & Long-Format Table Assembly
# ==============================================================================

def load_sequence_arrays(data_dir: str = "data_model") -> Dict[str, Any]:
    """Loads all pre-built Step-5 sequence arrays and metadata."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    meta_df = pd.read_csv(os.path.join(data_dir, "sequence_metadata.csv"))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta": meta_df,
    }


def assemble_prediction_records(
    model_name: str,
    evaluation_type: str,
    preds: np.ndarray,
    y_actual: np.ndarray,
    meta_subset: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Transforms wide (N, 3) predictions into standard long-format evaluation records:
    [evaluation_type, model, sequence_id, district, forecast_origin, horizon,
     actual_wsi, predicted_wsi, error, absolute_error, squared_error]
    """
    records = []
    meta_reset = meta_subset.reset_index(drop=True)
    assert len(preds) == len(meta_reset), f"Shape mismatch: preds {len(preds)} vs meta {len(meta_reset)}"

    for i in range(len(preds)):
        seq_id = meta_reset.loc[i, "sequence_id"]
        dist = meta_reset.loc[i, "district"]
        origin = meta_reset.loc[i, "forecast_origin"]

        for h in range(1, 4):
            act = float(y_actual[i, h - 1])
            pred = float(preds[i, h - 1])
            err = pred - act
            records.append({
                "evaluation_type": evaluation_type,
                "model": model_name,
                "sequence_id": seq_id,
                "district": dist,
                "forecast_origin": origin,
                "horizon": f"h{h}",
                "actual_wsi": round(act, 6),
                "predicted_wsi": round(pred, 6),
                "error": round(err, 6),
                "absolute_error": round(abs(err), 6),
                "squared_error": round(err ** 2, 6),
            })
    return records


# ==============================================================================
# Pipeline Execution & Champion Selection
# ==============================================================================

def run_baseline_suite(
    data_dir: str = "data_model",
    output_dir: str = "results/baselines",
    model_dataset_path: str = "outputs/Marathwada_MODEL_DATASET_2003_2024.csv",
) -> Dict[str, Any]:
    """
    Orchestrates baseline training, validation champion selection, refitting on Train+Val,
    and official test evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_sequence_arrays(data_dir=data_dir)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    meta_df = data["meta"]

    meta_train = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    meta_val = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    meta_test = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    panel_df = None
    if os.path.exists(model_dataset_path):
        panel_df = pd.read_csv(model_dataset_path)

    forecasters: List[BaseForecaster] = [
        PersistenceForecaster(),
        SeasonalPersistenceForecaster(panel_df=panel_df),
        WSIAutoregressiveForecaster(),
        RidgeForecaster(),
        XGBoostForecaster(),
    ]

    print("=" * 70)
    print("STEP 6: BASELINE FORECASTING SUITE & EVALUATION")
    print(f"Data directory: {data_dir}")
    print(f"Train samples:  {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print("=" * 70)

    val_records: List[Dict[str, Any]] = []
    val_model_metrics: Dict[str, Dict[str, float]] = {}

    print("\nPhase 1: Fitting baselines on TRAIN and generating VALIDATION predictions...")
    for model in forecasters:
        print(f"  -> Fitting {model.name}...")
        if isinstance(model, SeasonalPersistenceForecaster):
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val, metadata_df=meta_val)
        else:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            val_preds = model.predict(X_val)

        val_recs = assemble_prediction_records(
            model_name=model.name,
            evaluation_type="official_validation",
            preds=val_preds,
            y_actual=y_val,
            meta_subset=meta_val,
        )
        val_records.extend(val_recs)

        mae_h = [np.mean(np.abs(val_preds[:, h] - y_val[:, h])) for h in range(3)]
        rmse_h = [np.sqrt(np.mean((val_preds[:, h] - y_val[:, h]) ** 2)) for h in range(3)]
        mean_mae = float(np.mean(mae_h))
        mean_rmse = float(np.mean(rmse_h))

        val_model_metrics[model.name] = {
            "val_mae_h1": round(float(mae_h[0]), 5),
            "val_mae_h2": round(float(mae_h[1]), 5),
            "val_mae_h3": round(float(mae_h[2]), 5),
            "mean_val_mae": round(float(mean_mae), 5),
            "mean_val_rmse": round(float(mean_rmse), 5),
        }
        print(f"     Validation Mean MAE: {mean_mae:.4f} | Mean RMSE: {mean_rmse:.4f}")

    sorted_candidates = sorted(
        val_model_metrics.items(),
        key=lambda x: (x[1]["mean_val_mae"], x[1]["mean_val_rmse"]),
    )
    champion_name = sorted_candidates[0][0]
    champion_scores = sorted_candidates[0][1]
    print(f"\nPhase 2: Champion Baseline Selection:")
    print(f"  *** Champion: {champion_name} ***")
    print(f"      Validation MAE: {champion_scores['mean_val_mae']:.4f} (Tie-breaker RMSE: {champion_scores['mean_val_rmse']:.4f})")

    print("\nPhase 3: Refitting baselines on TRAIN + VALIDATION and evaluating on TEST...")
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    test_records: List[Dict[str, Any]] = []
    test_model_metrics: Dict[str, Dict[str, float]] = {}

    for model in forecasters:
        print(f"  -> Refitting {model.name} and predicting TEST...")
        if isinstance(model, SeasonalPersistenceForecaster):
            test_preds = model.predict(X_test, metadata_df=meta_test)
        else:
            model.refit(X_train_val, y_train_val)
            test_preds = model.predict(X_test)

        test_recs = assemble_prediction_records(
            model_name=model.name,
            evaluation_type="official_test",
            preds=test_preds,
            y_actual=y_test,
            meta_subset=meta_test,
        )
        test_records.extend(test_recs)

        mae_h = [np.mean(np.abs(test_preds[:, h] - y_test[:, h])) for h in range(3)]
        rmse_h = [np.sqrt(np.mean((test_preds[:, h] - y_test[:, h]) ** 2)) for h in range(3)]
        test_model_metrics[model.name] = {
            "test_mae_h1": round(float(mae_h[0]), 5),
            "test_mae_h2": round(float(mae_h[1]), 5),
            "test_mae_h3": round(float(mae_h[2]), 5),
            "mean_test_mae": round(float(np.mean(mae_h)), 5),
            "mean_test_rmse": round(float(np.mean(rmse_h)), 5),
        }

    val_pred_df = pd.DataFrame(val_records)
    test_pred_df = pd.DataFrame(test_records)

    val_csv_path = os.path.join(output_dir, "validation_predictions.csv")
    test_csv_path = os.path.join(output_dir, "baseline_predictions.csv")

    val_pred_df.to_csv(val_csv_path, index=False)
    test_pred_df.to_csv(test_csv_path, index=False)

    print(f"\nPhase 4: Artifact Persistence:")
    print(f"  - Saved validation predictions: {val_csv_path} ({len(val_pred_df)} rows; expected 4,320)")
    print(f"  - Saved official test predictions: {test_csv_path} ({len(test_pred_df)} rows; expected 5,400)")

    assert len(val_pred_df) == 5 * 288 * 3, f"Unexpected validation rows: {len(val_pred_df)}"
    assert len(test_pred_df) == 5 * 360 * 3, f"Unexpected test rows: {len(test_pred_df)}"

    summary_data = {
        "models": [m.name for m in forecasters],
        "champion_baseline": {
            "name": champion_name,
            "selection_criteria": "min(mean_val_mae) on VALIDATION with mean_val_rmse tie-breaker",
            "validation_scores": champion_scores,
            "test_scores": test_model_metrics.get(champion_name, {}),
        },
        "all_validation_metrics": val_model_metrics,
        "all_test_metrics": test_model_metrics,
    }
    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"  - Saved baseline summary: {summary_path}")

    return {
        "val_predictions": val_pred_df,
        "test_predictions": test_pred_df,
        "summary": summary_data,
        "champion_name": champion_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marathwada WSI Baseline Forecasting Suite")
    parser.add_argument("--data-dir", default="data_model", help="Directory containing pre-built Step-5 sequence arrays")
    parser.add_argument("--output-dir", default="results/baselines", help="Directory to save baseline predictions and metrics")
    parser.add_argument("--panel-path", default="outputs/Marathwada_MODEL_DATASET_2003_2024.csv", help="Path to frozen model panel dataset")
    args = parser.parse_args()

    run_baseline_suite(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dataset_path=args.panel_path,
    )
