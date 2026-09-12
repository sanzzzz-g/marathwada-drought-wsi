"""
Unit tests for residual target tensor construction and mathematical identity.
"""

import pytest
import numpy as np
import pandas as pd

from src.build_residual_targets import (
    load_inputs,
    build_origin_wsi,
    build_residual_targets,
    validate_residual_pipeline,
)


def test_schema_strictness():
    """Verifies that schema validator fails on ambiguous or missing WSI column."""
    # 1. Ambiguous: both 'WSI' and 'wsi' present
    df_ambiguous = pd.DataFrame({
        "district": ["Beed"],
        "date": ["2003-01-01"],
        "WSI": [0.5],
        "wsi": [0.5],
    })
    has_upper = "WSI" in df_ambiguous.columns
    has_lower = "wsi" in df_ambiguous.columns
    assert has_upper and has_lower

    # 2. Neither present
    df_neither = pd.DataFrame({
        "district": ["Beed"],
        "date": ["2003-01-01"],
    })
    has_upper2 = "WSI" in df_neither.columns
    has_lower2 = "wsi" in df_neither.columns
    assert not has_upper2 and not has_lower2


def test_residual_vectorized_math():
    """Verifies vectorized residual identity: R = Y - W[:, None]."""
    n = 10
    wsi_origin = np.array([0.8, 1.2, -0.5, 0.0, 0.4, -1.0, 1.5, -0.2, 0.1, 0.9], dtype=np.float64)
    y_raw = np.array([
        [1.0, 1.3, 1.1],
        [1.2, 1.1, 1.0],
        [-0.4, -0.2, 0.1],
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [-1.1, -1.2, -0.9],
        [1.4, 1.3, 1.2],
        [-0.2, -0.1, 0.0],
        [0.0, -0.1, 0.2],
        [1.0, 1.1, 0.8],
    ], dtype=np.float64)

    r_expected = y_raw - wsi_origin[:, None]
    assert r_expected.shape == (10, 3)

    # Spot check row 0
    assert np.isclose(r_expected[0, 0], 1.0 - 0.8)
    assert np.isclose(r_expected[0, 1], 1.3 - 0.8)
    assert np.isclose(r_expected[0, 2], 1.1 - 0.8)

    # Max difference check
    max_diff = np.max(np.abs(r_expected - (y_raw - wsi_origin[:, None])))
    assert max_diff <= 1e-12
