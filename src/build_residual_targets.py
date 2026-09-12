"""
Residual target tensor builder for 3-month WSI forecasting.
Computes r_{t,h} = WSI_{t+h} - WSI_t and origin vectors WSI_t.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


MODEL_DATASET = Path("outputs/Marathwada_MODEL_DATASET_2003_2024.csv")
SEQUENCE_DIR = Path("data_model")
OUTPUT_DIR = Path("data_model_residual")

REQUIRED_META = [
    "sequence_id",
    "district",
    "forecast_origin",
    "target_t1",
    "target_t2",
    "target_t3",
    "split",
]

EXPECTED_COUNTS = {
    "train": 1328,
    "validation": 288,
    "test": 360,
}


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads model dataset and sequence metadata with strict schema validation."""
    if not MODEL_DATASET.exists():
        raise FileNotFoundError(f"Model dataset not found at {MODEL_DATASET}")
    if not (SEQUENCE_DIR / "sequence_metadata.csv").exists():
        raise FileNotFoundError(f"Sequence metadata not found in {SEQUENCE_DIR}")

    model_df = pd.read_csv(MODEL_DATASET, parse_dates=["date"])
    meta = pd.read_csv(
        SEQUENCE_DIR / "sequence_metadata.csv",
        parse_dates=[
            "forecast_origin",
            "input_start",
            "input_end",
            "target_t1",
            "target_t2",
            "target_t3",
        ],
    )

    # 1. Metadata column schema check
    for col in REQUIRED_META:
        if col not in meta.columns:
            raise ValueError(f"Missing required metadata column: {col}")

    # 2. Strict WSI column schema check on model dataset
    has_upper = "WSI" in model_df.columns
    has_lower = "wsi" in model_df.columns

    if has_upper and has_lower:
        raise ValueError("Ambiguous schema: both 'WSI' and 'wsi' columns present in model dataset.")
    if not has_upper and not has_lower:
        raise ValueError("Schema error: neither 'WSI' nor 'wsi' column found in model dataset.")

    if has_lower:
        model_df = model_df.rename(columns={"wsi": "WSI"})

    required_model_cols = ["district", "date", "WSI"]
    for col in required_model_cols:
        if col not in model_df.columns:
            raise ValueError(f"Missing required model column: {col}")

    return model_df, meta


def build_origin_wsi(model_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Matches each sequence's (district, forecast_origin) to extract WSI_t."""
    lookup = model_df[["district", "date", "WSI"]].copy()
    origin = meta[["sequence_id", "district", "forecast_origin"]].copy()

    merged = origin.merge(
        lookup,
        left_on=["district", "forecast_origin"],
        right_on=["district", "date"],
        how="left",
        validate="one_to_one",
    )

    if merged["WSI"].isna().any():
        missing = merged[merged["WSI"].isna()]
        raise ValueError(
            f"Missing WSI at forecast origin for {len(missing)} sequences: {missing[['district', 'forecast_origin']].head()}"
        )

    merged = merged.rename(columns={"WSI": "wsi_at_origin"})
    merged = merged.drop(columns=["date"])
    return merged


def build_residual_targets(
    meta: pd.DataFrame,
    origin_wsi: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], pd.DataFrame]:
    """
    Computes residual target matrices and standalone origin vectors per split.
    Strictly asserts that metadata row ordering matches Step 5 array rows.
    """
    y_train = np.load(SEQUENCE_DIR / "y_train.npy")
    y_val = np.load(SEQUENCE_DIR / "y_val.npy")
    y_test = np.load(SEQUENCE_DIR / "y_test.npy")

    split_arrays = {
        "train": y_train,
        "validation": y_val,
        "test": y_test,
    }

    residual_arrays = {}
    origin_arrays = {}
    meta_parts = []

    for split, y in split_arrays.items():
        meta_split_val = "val" if split == "validation" else split
        split_meta = meta[meta["split"] == meta_split_val].copy()
        expected = EXPECTED_COUNTS[split]

        if len(split_meta) != expected:
            raise ValueError(f"{split}: expected {expected} metadata rows, got {len(split_meta)}")

        # Step 5 arrays were created strictly in sequence_metadata order for that split
        split_meta = split_meta.sort_values("sequence_id").reset_index(drop=True)

        split_origin = (
            origin_wsi[origin_wsi["sequence_id"].isin(split_meta["sequence_id"])]
            .sort_values("sequence_id")
            .reset_index(drop=True)
        )

        # Enforce exact sequence ID match and ordering
        if not np.array_equal(split_meta["sequence_id"].to_numpy(), split_origin["sequence_id"].to_numpy()):
            raise ValueError(f"{split}: sequence_id ordering mismatch between metadata and origin lookup")

        wsi_origin = split_origin["wsi_at_origin"].to_numpy(dtype=np.float64)

        if y.shape != (expected, 3):
            raise ValueError(f"{split}: expected y shape {(expected, 3)}, got {y.shape}")

        # Compute residual: r_{t,h} = WSI_{t+h} - WSI_t
        residual = y.astype(np.float64) - wsi_origin[:, None]

        if not np.isfinite(residual).all():
            raise ValueError(f"{split}: residual target contains NaN or Inf")
        if not np.isfinite(wsi_origin).all():
            raise ValueError(f"{split}: origin WSI contains NaN or Inf")

        split_meta["wsi_at_origin"] = wsi_origin
        split_meta["residual_t1"] = residual[:, 0]
        split_meta["residual_t2"] = residual[:, 1]
        split_meta["residual_t3"] = residual[:, 2]

        residual_arrays[split] = residual.astype(np.float32)
        origin_arrays[split] = wsi_origin.astype(np.float32)
        meta_parts.append(split_meta)

    residual_meta = pd.concat(meta_parts, ignore_index=True)
    residual_meta = residual_meta.sort_values("sequence_id").reset_index(drop=True)

    return split_arrays, origin_arrays, residual_arrays, residual_meta


def validate_residual_pipeline(
    split_arrays: Dict[str, np.ndarray],
    origin_arrays: Dict[str, np.ndarray],
    residual_arrays: Dict[str, np.ndarray],
    residual_meta: pd.DataFrame,
) -> None:
    """
    Exhaustive vectorized validation across all 1,976 sequences.
    Asserts max |R_stored - (Y_original - WSI_origin)| <= 1e-6.
    """
    print("\n--- Running Exhaustive Vectorized Validation ---")

    # 1. Shape and dimension validation
    for split, exp_count in EXPECTED_COUNTS.items():
        res_arr = residual_arrays[split]
        orig_arr = origin_arrays[split]
        raw_arr = split_arrays[split]

        if res_arr.shape != (exp_count, 3):
            raise ValueError(f"{split}: invalid residual shape {res_arr.shape}, expected ({exp_count}, 3)")
        if orig_arr.shape != (exp_count,):
            raise ValueError(f"{split}: invalid origin shape {orig_arr.shape}, expected ({exp_count},)")

        # 2. Universal Vectorized Identity Check across all sequences
        expected_residual = raw_arr.astype(np.float64) - orig_arr.astype(np.float64)[:, None]
        max_identity_diff = np.max(np.abs(res_arr.astype(np.float64) - expected_residual))

        print(f"  [{split.upper()}] Vectorized Identity Check ({exp_count} sequences x 3 horizons):")
        print(f"       Max absolute difference: {max_identity_diff:.8e}")

        if max_identity_diff > 1e-6:
            raise ValueError(f"{split}: residual identity check failed! Max diff = {max_identity_diff}")

    # 3. Metadata validation
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(residual_meta) != expected_total:
        raise ValueError(f"Expected {expected_total} metadata rows, got {len(residual_meta)}")

    if residual_meta["sequence_id"].duplicated().any():
        raise ValueError("Duplicate sequence_id values detected in residual metadata")

    # 4. Check for null or non-finite values in metadata
    cols_to_check = ["wsi_at_origin", "residual_t1", "residual_t2", "residual_t3"]
    for col in cols_to_check:
        if residual_meta[col].isna().any():
            raise ValueError(f"NaN detected in metadata column: {col}")
        if not np.isfinite(residual_meta[col].to_numpy()).all():
            raise ValueError(f"Non-finite value detected in metadata column: {col}")

    print("\nALL 1,976 SEQUENCES PASSED VECTORIZED RESIDUAL IDENTITY VALIDATION (error <= 1e-6).")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building residual target tensors in {OUTPUT_DIR}")

    # 1. Load inputs with strict schema
    model_df, meta = load_inputs()
    print(f"Loaded model dataset: {len(model_df)} monthly rows")
    print(f"Loaded sequence metadata: {len(meta)} sequences")

    # 2. Extract WSI at forecast origin
    origin_wsi = build_origin_wsi(model_df, meta)
    print(f"Matched forecast-origin WSI_t for all {len(origin_wsi)} sequences.")

    # 3. Compute residual arrays and origin vectors
    split_arrays, origin_arrays, residual_arrays, residual_meta = build_residual_targets(meta, origin_wsi)

    # 4. Exhaustive vectorized validation
    validate_residual_pipeline(split_arrays, origin_arrays, residual_arrays, residual_meta)

    # 5. Save all tensors and metadata
    np.save(OUTPUT_DIR / "y_residual_train.npy", residual_arrays["train"])
    np.save(OUTPUT_DIR / "y_residual_val.npy", residual_arrays["validation"])
    np.save(OUTPUT_DIR / "y_residual_test.npy", residual_arrays["test"])

    np.save(OUTPUT_DIR / "wsi_origin_train.npy", origin_arrays["train"])
    np.save(OUTPUT_DIR / "wsi_origin_val.npy", origin_arrays["validation"])
    np.save(OUTPUT_DIR / "wsi_origin_test.npy", origin_arrays["test"])

    # Reorder columns cleanly
    final_meta_cols = [
        "sequence_id",
        "district",
        "forecast_origin",
        "wsi_at_origin",
        "residual_t1",
        "residual_t2",
        "residual_t3",
        "split",
    ]
    residual_meta[final_meta_cols].to_csv(OUTPUT_DIR / "residual_metadata.csv", index=False)

    print("\nSuccessfully Generated Artifacts in data_model_residual/:")
    print(f"  y_residual_train.npy:   {residual_arrays['train'].shape}  [float32]")
    print(f"  y_residual_val.npy:     {residual_arrays['validation'].shape}  [float32]")
    print(f"  y_residual_test.npy:    {residual_arrays['test'].shape}  [float32, EVALUATION ONLY]")
    print(f"  wsi_origin_train.npy:   {origin_arrays['train'].shape}  [float32]")
    print(f"  wsi_origin_val.npy:     {origin_arrays['validation'].shape}  [float32]")
    print(f"  wsi_origin_test.npy:    {origin_arrays['test'].shape}  [float32]")
    print(f"  residual_metadata.csv:  {len(residual_meta)} rows")
    print("\nSTEP 8.2 COMPLETE. Target transformation successful.")


if __name__ == "__main__":
    main()
