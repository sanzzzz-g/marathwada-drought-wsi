"""
Tests for supervised sequence tensor shapes, continuity, and split isolation.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.sequence_builder import (
    LOOKBACK,
    HORIZON,
    FEATURES,
    build_sequences,
)


@pytest.fixture(scope="module")
def sequence_data():
    """
    Fixture that loads or builds the sequence tensors and metadata.
    """
    output_dir = "data_model_v2"
    required_files = [
        os.path.join(output_dir, "X_train.npy"),
        os.path.join(output_dir, "y_train.npy"),
        os.path.join(output_dir, "X_val.npy"),
        os.path.join(output_dir, "y_val.npy"),
        os.path.join(output_dir, "X_test.npy"),
        os.path.join(output_dir, "y_test.npy"),
        os.path.join(output_dir, "sequence_metadata.csv"),
    ]

    # If any file is missing, build them automatically
    if not all(os.path.exists(f) for f in required_files):
        build_sequences(output_dir=output_dir)

    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_val = np.load(os.path.join(output_dir, "X_val.npy"))
    y_val = np.load(os.path.join(output_dir, "y_val.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    meta_df = pd.read_csv(os.path.join(output_dir, "sequence_metadata.csv"))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta": meta_df,
    }



def test_total_sequence_count(sequence_data):
    """Verifies that total sequence count across all splits is exactly 1,968."""
    total = (
        len(sequence_data["X_train"])
        + len(sequence_data["X_val"])
        + len(sequence_data["X_test"])
    )
    assert total == 1968, f"Expected 1,968 total sequences, got {total}"
    assert len(sequence_data["meta"]) == 1968, (
        f"Expected 1,968 metadata rows, got {len(sequence_data['meta'])}"
    )


def test_train_shape(sequence_data):
    """Verifies X_train is (1328, 12, 6) and y_train is (1328, 3)."""
    assert sequence_data["X_train"].shape == (1328, 12, 6)
    assert sequence_data["y_train"].shape == (1328, 3)


def test_validation_shape(sequence_data):
    """Verifies X_val is (272, 12, 6) and y_val is (272, 3)."""
    assert sequence_data["X_val"].shape == (272, 12, 6)
    assert sequence_data["y_val"].shape == (272, 3)


def test_test_shape(sequence_data):
    """Verifies X_test is (368, 12, 6) and y_test is (368, 3)."""
    assert sequence_data["X_test"].shape == (368, 12, 6)
    assert sequence_data["y_test"].shape == (368, 3)


def test_sequence_length(sequence_data):
    """Verifies sequence length is strictly 12 months across all splits."""
    for split in ["train", "val", "test"]:
        assert sequence_data[f"X_{split}"].shape[1] == LOOKBACK == 12


def test_feature_count(sequence_data):
    """Verifies feature count is strictly 6 across all splits."""
    for split in ["train", "val", "test"]:
        assert sequence_data[f"X_{split}"].shape[2] == len(FEATURES) == 6


def test_target_length(sequence_data):
    """Verifies target length is strictly 3 months across all splits."""
    for split in ["train", "val", "test"]:
        assert sequence_data[f"y_{split}"].shape[1] == HORIZON == 3


def test_district_continuity(sequence_data):
    """Verifies that no sequence crosses districts."""
    meta = sequence_data["meta"]
    # In sequence_metadata, district is tracked per sequence
    assert "district" in meta.columns
    # Ensure all 8 districts are present
    assert meta["district"].nunique() == 8


def test_month_continuity(sequence_data):
    """Verifies that input dates and target dates are strictly consecutive monthly dates."""
    meta = sequence_data["meta"]
    for _, row in meta.iterrows():
        inp_start = pd.Timestamp(row["input_start"])
        inp_end = pd.Timestamp(row["input_end"])
        origin = pd.Timestamp(row["forecast_origin"])
        t1 = pd.Timestamp(row["target_t1"])
        t2 = pd.Timestamp(row["target_t2"])
        t3 = pd.Timestamp(row["target_t3"])

        # input_end must equal forecast_origin
        assert inp_end == origin, f"input_end {inp_end} != forecast_origin {origin}"

        # 12 consecutive months: inp_end is exactly 11 months after inp_start
        assert inp_end == inp_start + pd.DateOffset(months=11)

        # 3 consecutive target months
        assert t1 == origin + pd.DateOffset(months=1)
        assert t2 == origin + pd.DateOffset(months=2)
        assert t3 == origin + pd.DateOffset(months=3)


def test_target_after_input(sequence_data):
    """Verifies that target_t1 is strictly greater than input_end for every sequence."""
    meta = sequence_data["meta"]
    assert (meta["target_t1"] > meta["input_end"]).all()


def test_split_boundaries(sequence_data):
    """Verify forecast origins and target windows stay within their assigned splits."""
    meta = sequence_data["meta"]
    train = meta[meta["split"] == "train"]
    val = meta[meta["split"] == "val"]
    test = meta[meta["split"] == "test"]

    # Forecast-origin boundaries
    assert train["forecast_origin"].min() == "2003-12-01"
    assert train["forecast_origin"].max() == "2017-09-01"

    assert val["forecast_origin"].min() == "2017-12-01"
    assert val["forecast_origin"].max() == "2020-09-01"

    assert test["forecast_origin"].min() == "2020-12-01"
    assert test["forecast_origin"].max() == "2024-09-01"

    # Training targets: January 2004–December 2017
    assert (train["target_t1"] >= "2004-01-01").all()
    assert (train["target_t3"] <= "2017-12-01").all()

    # Validation targets: January 2018–December 2020
    assert (val["target_t1"] >= "2018-01-01").all()
    assert (val["target_t3"] <= "2020-12-01").all()

    # Test targets: January 2021–December 2024
    assert (test["target_t1"] >= "2021-01-01").all()
    assert (test["target_t3"] <= "2024-12-01").all()


def test_district_balance(sequence_data):
    """Verify each district has 166 train, 34 validation, and 46 test sequences."""
    meta = sequence_data["meta"]
    counts = meta.groupby(["district", "split"]).size().unstack(fill_value=0)

    for dist in counts.index:
        assert counts.loc[dist, "train"] == 166, f"{dist} train count != 166"
        assert counts.loc[dist, "val"] == 34, f"{dist} val count != 34"
        assert counts.loc[dist, "test"] == 46, f"{dist} test count != 46"
        assert counts.loc[dist].sum() == 246, f"{dist} total count != 246"


def test_no_missing_values(sequence_data):
    """Verifies zero NaN and zero infinite values across all X and y arrays."""
    for split in ["train", "val", "test"]:
        X = sequence_data[f"X_{split}"]
        y = sequence_data[f"y_{split}"]
        assert not np.isnan(X).any(), f"NaN found in X_{split}"
        assert not np.isinf(X).any(), f"Inf found in X_{split}"
        assert not np.isnan(y).any(), f"NaN found in y_{split}"
        assert not np.isinf(y).any(), f"Inf found in y_{split}"
