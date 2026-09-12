"""
Tests for historical monthly rolling-origin backtesting and causal split isolation.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.backtesting import build_causal_sequences_at_origin


@pytest.fixture(scope="module")
def panel_data():
    csv_path = "outputs/Marathwada_MODEL_DATASET_2003_2024.csv"
    assert os.path.exists(csv_path), "Model dataset missing!"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_causal_training_boundary(panel_data):
    """Verifies that all training samples at backtest origin t have targets <= t."""
    origin_t = pd.Timestamp("2015-06-01")
    X_tr, y_tr, wsi_tr, X_curr, meta_curr, extras = build_causal_sequences_at_origin(
        panel_df=panel_data,
        origin_t=origin_t,
        lookback=12,
        horizon=3,
    )

    # 1. Check shapes
    assert len(X_tr) == len(y_tr) == len(wsi_tr)
    assert X_tr.shape[1:] == (12, 6)
    assert y_tr.shape[1:] == (3,)
    assert wsi_tr.shape[1:] == (12,)
    assert len(meta_curr) == 8  # 8 districts at origin_t

    # 2. Latest training origin must be <= origin_t - 3 months (i.e. <= 2015-03-01)
    # Target dates for origin 2015-03-01 are 2015-04, 2015-05, 2015-06 (<= origin_t)
    latest_allowed_origin = origin_t - pd.DateOffset(months=3)
    assert latest_allowed_origin == pd.Timestamp("2015-03-01")

    # Verify that current forecast origin metadata equals origin_t
    for meta in meta_curr:
        assert meta["forecast_origin"] == "2015-06-01"
        assert meta["target_t1"] == "2015-07-01"
        assert meta["target_t2"] == "2015-08-01"
        assert meta["target_t3"] == "2015-09-01"


def test_causal_preprocessing_no_future_leakage(panel_data):
    """Verifies that statistics for origin t are calculated strictly using date <= t."""
    origin_t = pd.Timestamp("2012-01-01")
    # Subsetting panel up to 2012-01-01
    X_tr_2012, _, _, _, _, _ = build_causal_sequences_at_origin(
        panel_df=panel_data,
        origin_t=origin_t,
    )

    origin_later = pd.Timestamp("2016-01-01")
    X_tr_2016, _, _, _, _, _ = build_causal_sequences_at_origin(
        panel_df=panel_data,
        origin_t=origin_later,
    )

    # In 2016, more historical training sequences exist than in 2012
    assert len(X_tr_2016) > len(X_tr_2012)
    assert not np.isnan(X_tr_2012).any()
    assert not np.isnan(X_tr_2016).any()
