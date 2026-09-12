"""
Supervised sequence tensor builder for rolling-origin 3-month WSI forecasting.
Constructs X (12, 6) input arrays and y (3,) target arrays.
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

# Fixed feature order (never rely on alphabetical ordering)
FEATURES: List[str] = [
    "spi_stress_z",
    "soil_stress_z",
    "ndvi_stress_z",
    "lst_stress_z",
    "groundwater_stress_z",
    "surface_water_stress_z",
]

LOOKBACK: int = 12
HORIZON: int = 3

# Origin bounds
FIRST_POSSIBLE_ORIGIN = pd.Timestamp("2003-12-01")
FINAL_POSSIBLE_ORIGIN = pd.Timestamp("2024-09-01")

# Split origin boundaries
TRAIN_START_ORIGIN = pd.Timestamp("2003-12-01")
TRAIN_END_ORIGIN = pd.Timestamp("2017-09-01")
VAL_START_ORIGIN = pd.Timestamp("2018-01-01")
VAL_END_ORIGIN = pd.Timestamp("2020-12-01")
TEST_START_ORIGIN = pd.Timestamp("2021-01-01")
TEST_END_ORIGIN = pd.Timestamp("2024-09-01")


def load_and_validate_input(csv_path: str) -> pd.DataFrame:
    """
    Loads and validates the frozen model dataset.
    Ensures 2,112 rows, 8 districts, 264 dates, zero nulls/inf, and proper sorting.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input model dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Resolve WSI column name (support 'wsi' or 'WSI')
    wsi_col = None
    for candidate in ["wsi", "WSI"]:
        if candidate in df.columns:
            wsi_col = candidate
            break
    if wsi_col is None:
        raise KeyError(f"Neither 'wsi' nor 'WSI' found in columns: {list(df.columns)}")

    # Ensure required columns
    required_cols = ["district", "date"] + FEATURES + [wsi_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    # Standardize column name to 'wsi'
    if wsi_col != "wsi":
        df["wsi"] = df[wsi_col]

    # Convert date to Timestamp
    df["date"] = pd.to_datetime(df["date"])

    # 1. Total shape checks
    assert len(df) == 2112, f"Expected 2,112 rows, found {len(df)}"
    assert df["district"].nunique() == 8, f"Expected 8 districts, found {df['district'].nunique()}"
    assert df["date"].nunique() == 264, f"Expected 264 unique dates, found {df['date'].nunique()}"

    # 2. Key uniqueness
    assert not df.duplicated(subset=["district", "date"]).any(), "Duplicate (district, date) keys found!"

    # 3. Missing and infinite values
    check_cols = FEATURES + ["wsi"]
    null_counts = df[check_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Null values found in features/wsi: {null_counts[null_counts > 0].to_dict()}"

    for col in check_cols:
        assert not np.isinf(df[col]).any(), f"Infinite values found in column: {col}"

    # 4. District continuity and exact dates
    districts = sorted(df["district"].unique())
    for dist in districts:
        dist_df = df[df["district"] == dist].sort_values("date")
        assert len(dist_df) == 264, f"District {dist} has {len(dist_df)} rows, expected 264"
        assert dist_df["date"].min() == pd.Timestamp("2003-01-01"), f"District {dist} min date != 2003-01-01"
        assert dist_df["date"].max() == pd.Timestamp("2024-12-01"), f"District {dist} max date != 2024-12-01"

        # Check monthly continuity (every step is exactly 1 month)
        diffs = dist_df["date"].diff().dropna()
        # Monthly steps: day count is either 28, 29, 30, or 31 days
        for diff in diffs:
            assert 28 <= diff.days <= 31, f"District {dist} has non-consecutive monthly dates: diff of {diff.days} days"

    # Sort strictly by district ascending, date ascending
    df = df.sort_values(by=["district", "date"], ascending=[True, True]).reset_index(drop=True)
    return df


def determine_split_by_origin(origin_date: pd.Timestamp) -> Optional[str]:
    """
    Assigns split strictly by the forecast origin date.
    Returns 'train', 'val', 'test', or None (if origin is omitted across boundaries).
    """
    if TRAIN_START_ORIGIN <= origin_date <= TRAIN_END_ORIGIN:
        return "train"
    elif VAL_START_ORIGIN <= origin_date <= VAL_END_ORIGIN:
        return "val"
    elif TEST_START_ORIGIN <= origin_date <= TEST_END_ORIGIN:
        return "test"
    else:
        # Origins 2017-10, 2017-11, 2017-12 are omitted because targets cross into validation
        return None


def build_sequences_for_district(
    dist_df: pd.DataFrame,
    start_seq_id: int = 1,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[Dict[str, Any]], int]:
    """
    Slides lookback=12 and horizon=3 across a single district's chronological panel.
    Returns X_dict, y_dict, metadata_records, next_seq_id.
    """
    dist = dist_df["district"].iloc[0]
    n_rows = len(dist_df)
    assert n_rows == 264, f"District {dist} has {n_rows} rows; expected 264"

    X_by_split: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    y_by_split: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    metadata_records: List[Dict[str, Any]] = []

    current_seq_id = start_seq_id

    # Feature matrix (264, 6) and WSI vector (264,)
    feature_matrix = dist_df[FEATURES].values.astype(np.float32)
    wsi_vector = dist_df["wsi"].values.astype(np.float32)
    dates = dist_df["date"].tolist()

    # Valid origin index i must have:
    # 1. Lookback: i - 11 >= 0 -> i >= 11 (index 11 is 2003-12-01)
    # 2. Horizon:  i + 3 <= 263 -> i <= 260 (index 260 is 2024-09-01)
    for i in range(LOOKBACK - 1, n_rows - HORIZON):
        origin_date = dates[i]
        split = determine_split_by_origin(origin_date)
        if split is None:
            continue

        # Input window: [i - 11 ... i] (12 months)
        input_start_idx = i - (LOOKBACK - 1)
        input_end_idx = i
        X_sample = feature_matrix[input_start_idx : input_end_idx + 1]  # shape (12, 6)

        # Target window: [i + 1, i + 2, i + 3] (3 months)
        target_start_idx = i + 1
        target_end_idx = i + HORIZON
        y_sample = wsi_vector[target_start_idx : target_end_idx + 1]  # shape (3,)

        # Assert shapes
        assert X_sample.shape == (LOOKBACK, len(FEATURES)), f"Unexpected X shape: {X_sample.shape}"
        assert y_sample.shape == (HORIZON,), f"Unexpected y shape: {y_sample.shape}"

        # Record metadata
        seq_id_str = f"{current_seq_id:06d}"
        rec = {
            "sequence_id": seq_id_str,
            "district": dist,
            "forecast_origin": origin_date.strftime("%Y-%m-%d"),
            "input_start": dates[input_start_idx].strftime("%Y-%m-%d"),
            "input_end": dates[input_end_idx].strftime("%Y-%m-%d"),
            "target_t1": dates[i + 1].strftime("%Y-%m-%d"),
            "target_t2": dates[i + 2].strftime("%Y-%m-%d"),
            "target_t3": dates[i + 3].strftime("%Y-%m-%d"),
            "split": split,
        }

        X_by_split[split].append(X_sample)
        y_by_split[split].append(y_sample)
        metadata_records.append(rec)
        current_seq_id += 1

    return X_by_split, y_by_split, metadata_records, current_seq_id


def build_all_sequences(
    df: pd.DataFrame,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Constructs supervised sequence arrays for all districts and partitions.
    Orders arrays and metadata consistently.
    """
    districts = sorted(df["district"].unique())
    assert len(districts) == 8, f"Expected 8 districts, got {len(districts)}"

    all_X_splits: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_y_splits: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    all_metadata: List[Dict[str, Any]] = []

    seq_id = 1
    # Iterate across districts in alphabetical order
    for dist in districts:
        dist_df = df[df["district"] == dist].sort_values("date").reset_index(drop=True)
        X_dist, y_dist, meta_dist, seq_id = build_sequences_for_district(dist_df, start_seq_id=seq_id)

        for split in ["train", "val", "test"]:
            all_X_splits[split].extend(X_dist[split])
            all_y_splits[split].extend(y_dist[split])
        all_metadata.extend(meta_dist)

    # Convert to NumPy arrays
    arrays_X = {
        "train": np.array(all_X_splits["train"], dtype=np.float32),
        "val": np.array(all_X_splits["val"], dtype=np.float32),
        "test": np.array(all_X_splits["test"], dtype=np.float32),
    }
    arrays_y = {
        "train": np.array(all_y_splits["train"], dtype=np.float32),
        "val": np.array(all_y_splits["val"], dtype=np.float32),
        "test": np.array(all_y_splits["test"], dtype=np.float32),
    }

    meta_df = pd.DataFrame(all_metadata)

    # Run complete verification
    val_report = validate_generated_sequences(arrays_X, arrays_y, meta_df)

    return arrays_X, arrays_y, meta_df, val_report


def validate_generated_sequences(
    arrays_X: Dict[str, np.ndarray],
    arrays_y: Dict[str, np.ndarray],
    meta_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Performs rigorous sequence-level validations across all generated tensors and metadata.
    """
    # 1. Sequence count checks
    n_train = len(arrays_X["train"])
    n_val = len(arrays_X["val"])
    n_test = len(arrays_X["test"])
    n_total = n_train + n_val + n_test

    assert n_train == 1328, f"Expected 1,328 train sequences, got {n_train}"
    assert n_val == 288, f"Expected 288 val sequences, got {n_val}"
    assert n_test == 360, f"Expected 360 test sequences, got {n_test}"
    assert n_total == 1976, f"Expected 1,976 total sequences, got {n_total}"
    assert len(meta_df) == 1976, f"Metadata row count {len(meta_df)} != 1976"

    # 2. Tensor shape checks
    assert arrays_X["train"].shape == (1328, 12, 6), f"Wrong X_train shape: {arrays_X['train'].shape}"
    assert arrays_y["train"].shape == (1328, 3), f"Wrong y_train shape: {arrays_y['train'].shape}"
    assert arrays_X["val"].shape == (288, 12, 6), f"Wrong X_val shape: {arrays_X['val'].shape}"
    assert arrays_y["val"].shape == (288, 3), f"Wrong y_val shape: {arrays_y['val'].shape}"
    assert arrays_X["test"].shape == (360, 12, 6), f"Wrong X_test shape: {arrays_X['test'].shape}"
    assert arrays_y["test"].shape == (360, 3), f"Wrong y_test shape: {arrays_y['test'].shape}"

    # 3. No NaN / Inf in any tensor
    for split in ["train", "val", "test"]:
        assert not np.isnan(arrays_X[split]).any(), f"NaN found in X_{split}"
        assert not np.isinf(arrays_X[split]).any(), f"Inf found in X_{split}"
        assert not np.isnan(arrays_y[split]).any(), f"NaN found in y_{split}"
        assert not np.isinf(arrays_y[split]).any(), f"Inf found in y_{split}"

    # 4. District balance check
    dist_counts = meta_df.groupby(["district", "split"]).size().unstack(fill_value=0)
    for dist in dist_counts.index:
        assert dist_counts.loc[dist, "train"] == 166, f"District {dist} train count != 166"
        assert dist_counts.loc[dist, "val"] == 36, f"District {dist} val count != 36"
        assert dist_counts.loc[dist, "test"] == 45, f"District {dist} test count != 45"

    # 5. Boundary checks
    train_meta = meta_df[meta_df["split"] == "train"]
    val_meta = meta_df[meta_df["split"] == "val"]
    test_meta = meta_df[meta_df["split"] == "test"]

    assert train_meta["forecast_origin"].max() == "2017-09-01", "Max train origin != 2017-09-01"
    assert val_meta["forecast_origin"].min() == "2018-01-01", "Min val origin != 2018-01-01"
    assert val_meta["forecast_origin"].max() == "2020-12-01", "Max val origin != 2020-12-01"
    assert test_meta["forecast_origin"].min() == "2021-01-01", "Min test origin != 2021-01-01"
    assert test_meta["forecast_origin"].max() == "2024-09-01", "Max test origin != 2024-09-01"

    # Training targets must never exceed 2017-12-01
    assert (train_meta["target_t1"] <= "2017-12-01").all()
    assert (train_meta["target_t2"] <= "2017-12-01").all()
    assert (train_meta["target_t3"] <= "2017-12-01").all()

    # Targets strictly after input
    assert (meta_df["target_t1"] > meta_df["input_end"]).all(), "Target t1 is not strictly after input_end!"

    # 6. Temporal continuity check per sequence
    for _, row in meta_df.iterrows():
        inp_start = pd.Timestamp(row["input_start"])
        inp_end = pd.Timestamp(row["input_end"])
        origin = pd.Timestamp(row["forecast_origin"])
        t1 = pd.Timestamp(row["target_t1"])
        t2 = pd.Timestamp(row["target_t2"])
        t3 = pd.Timestamp(row["target_t3"])

        assert inp_end == origin, f"input_end {inp_end} != origin {origin}"
        # input_end should be exactly 11 months after input_start
        expected_inp_end = inp_start + pd.DateOffset(months=11)
        assert inp_end == expected_inp_end, f"Input dates not 12 consecutive months: {inp_start} to {inp_end}"

        assert t1 == origin + pd.DateOffset(months=1), f"target_t1 {t1} != origin + 1 month"
        assert t2 == origin + pd.DateOffset(months=2), f"target_t2 {t2} != origin + 2 months"
        assert t3 == origin + pd.DateOffset(months=3), f"target_t3 {t3} != origin + 3 months"

    validation_report = {
        "lookback_months": LOOKBACK,
        "horizon_months": HORIZON,
        "train_sequences": n_train,
        "validation_sequences": n_val,
        "test_sequences": n_test,
        "total_sequences": n_total,
        "train_shape": list(arrays_X["train"].shape),
        "validation_shape": list(arrays_X["val"].shape),
        "test_shape": list(arrays_X["test"].shape),
        "target_shapes": {
            "train": list(arrays_y["train"].shape),
            "validation": list(arrays_y["val"].shape),
            "test": list(arrays_y["test"].shape),
        },
        "districts": 8,
        "district_balance": True,
        "temporal_continuity": True,
        "target_after_input": True,
    }
    return validation_report


def verify_sample_sequence_values(
    df: pd.DataFrame,
    arrays_X: Dict[str, np.ndarray],
    arrays_y: Dict[str, np.ndarray],
    meta_df: pd.DataFrame,
) -> None:
    """
    Manually verifies that tensor slices match the exact raw values from the source dataframe.
    Verifies train, val, test, known drought periods (e.g. Latur 2015-08-01), and different districts.
    """
    test_cases = [
        # (district, origin_date, split)
        ("Beed", "2003-12-01", "train"),
        ("Latur", "2015-08-01", "train"),  # Historical acute drought
        ("Jalna", "2018-01-01", "val"),     # First validation origin
        ("Parbhani", "2020-05-01", "val"),
        ("Nanded", "2021-01-01", "test"),   # First test origin
        ("Dharashiv", "2024-09-01", "test"),# Final test origin
    ]

    for dist, origin_str, split in test_cases:
        meta_match = meta_df[
            (meta_df["district"] == dist)
            & (meta_df["forecast_origin"] == origin_str)
            & (meta_df["split"] == split)
        ]
        assert len(meta_match) == 1, f"No unique metadata record found for ({dist}, {origin_str}, {split})"

        # Find sample index within the split array
        split_meta = meta_df[meta_df["split"] == split].reset_index(drop=True)
        sample_idx = split_meta[
            (split_meta["district"] == dist) & (split_meta["forecast_origin"] == origin_str)
        ].index[0]

        X_sample = arrays_X[split][sample_idx]  # (12, 6)
        y_sample = arrays_y[split][sample_idx]  # (3,)

        # Extract directly from source DataFrame
        dist_df = df[df["district"] == dist].sort_values("date").reset_index(drop=True)
        origin_idx = dist_df[dist_df["date"] == pd.Timestamp(origin_str)].index[0]

        expected_X = dist_df.iloc[origin_idx - 11 : origin_idx + 1][FEATURES].values.astype(np.float32)
        expected_y = dist_df.iloc[origin_idx + 1 : origin_idx + 4]["wsi"].values.astype(np.float32)

        # Bitwise or float close comparison
        diff_X = np.max(np.abs(X_sample - expected_X))
        diff_y = np.max(np.abs(y_sample - expected_y))

        assert diff_X < 1e-6, f"Value mismatch in X for ({dist}, {origin_str}): max diff = {diff_X}"
        assert diff_y < 1e-6, f"Value mismatch in y for ({dist}, {origin_str}): max diff = {diff_y}"

    print(f"Sample value verification PASSED across all {len(test_cases)} spot-checked cases!")


def save_sequence_artifacts(
    arrays_X: Dict[str, np.ndarray],
    arrays_y: Dict[str, np.ndarray],
    meta_df: pd.DataFrame,
    val_report: Dict[str, Any],
    output_dir: str = "data_model",
) -> None:
    """
    Saves all tensors, metadata, and validation JSON into the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save NumPy tensors
    np.save(os.path.join(output_dir, "X_train.npy"), arrays_X["train"])
    np.save(os.path.join(output_dir, "y_train.npy"), arrays_y["train"])
    np.save(os.path.join(output_dir, "X_val.npy"), arrays_X["val"])
    np.save(os.path.join(output_dir, "y_val.npy"), arrays_y["val"])
    np.save(os.path.join(output_dir, "X_test.npy"), arrays_X["test"])
    np.save(os.path.join(output_dir, "y_test.npy"), arrays_y["test"])

    # Save sequence metadata CSV
    meta_path = os.path.join(output_dir, "sequence_metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    # Save sequence validation report JSON
    val_path = os.path.join(output_dir, "sequence_validation.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    print(f"Sequence artifacts successfully saved to: {output_dir}/")
    print(f"  - X_train.npy: {arrays_X['train'].shape}")
    print(f"  - y_train.npy: {arrays_y['train'].shape}")
    print(f"  - X_val.npy:   {arrays_X['val'].shape}")
    print(f"  - y_val.npy:   {arrays_y['val'].shape}")
    print(f"  - X_test.npy:  {arrays_X['test'].shape}")
    print(f"  - y_test.npy:  {arrays_y['test'].shape}")
    print(f"  - sequence_metadata.csv: {len(meta_df)} rows")
    print(f"  - sequence_validation.json")


def build_sequences(
    input_path: Optional[str] = None,
    output_dir: str = "data_model",
) -> Dict[str, Any]:
    """
    Main orchestration entry point.
    """
    if input_path is None:
        # Preferred default is outputs/Marathwada_MODEL_DATASET_2003_2024.csv
        candidates = [
            "outputs/Marathwada_MODEL_DATASET_2003_2024.csv",
            "Marathwada_MODEL_DATASET_2003_2024.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break
        if input_path is None:
            raise FileNotFoundError(f"Model dataset not found in default locations: {candidates}")

    print(f"Building supervised sequences from {input_path} to {output_dir}")
    print(f"Lookback: {LOOKBACK} months | Horizon: {HORIZON} months")

    # 1. Load and validate input
    print("Step 5.1-5.3: Loading and validating model dataset...")
    df = load_and_validate_input(input_path)
    print(f"Dataset validated: {len(df)} rows, {df['district'].nunique()} districts, {df['date'].nunique()} dates.")

    # 2. Build sequences
    print("Step 5.4-5.11: Generating rolling-origin supervised sequences...")
    arrays_X, arrays_y, meta_df, val_report = build_all_sequences(df)

    # 3. Spot-check actual values against source DataFrame
    print("Step 5.15: Verifying actual values across sample sequences...")
    verify_sample_sequence_values(df, arrays_X, arrays_y, meta_df)

    # 4. Save artifacts
    print("Step 5.11 & 5.18: Saving NumPy tensors, metadata CSV, and validation JSON...")
    save_sequence_artifacts(arrays_X, arrays_y, meta_df, val_report, output_dir=output_dir)

    print("\nStep 5 sequence generation complete! Ready for model training.")
    return {
        "arrays_X": arrays_X,
        "arrays_y": arrays_y,
        "metadata": meta_df,
        "validation_report": val_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marathwada WSI Supervised Sequence Builder")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to Marathwada_MODEL_DATASET_2003_2024.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data_model",
        help="Directory to save generated tensors and metadata",
    )
    args = parser.parse_args()

    build_sequences(input_path=args.input, output_dir=args.output_dir)
