"""
Tests for authoritative pipeline output artifacts and schema parity.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.ingestion import CANONICAL_DISTRICTS
from src.wsi import WSI_COMPONENT_COLS


def get_dataset_paths():
    """Locates model and audit datasets in outputs/ or workspace root."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check outputs/ first, fallback to root
    model_path = os.path.join(root, "outputs", "Marathwada_MODEL_DATASET_2003_2024.csv")
    if not os.path.exists(model_path):
        model_path = os.path.join(root, "Marathwada_MODEL_DATASET_2003_2024.csv")
        
    audit_path = os.path.join(root, "outputs", "Marathwada_MASTER_AUDIT_DATASET_2003_2024.csv")
    if not os.path.exists(audit_path):
        audit_path = os.path.join(root, "Marathwada_MASTER_AUDIT_DATASET_2003_2024.csv")
        
    return model_path, audit_path


def test_authoritative_dataset_dimensions_and_schema():
    """Verifies that the generated model dataset has exactly 2,112 rows, 8 districts, and 264 months."""
    model_path, _ = get_dataset_paths()
    if not os.path.exists(model_path):
        pytest.skip(f"Output dataset not found at {model_path}; run pipeline first.")
        
    df = pd.read_csv(model_path)
    
    # 1. Total row count: 2,112
    assert len(df) == 2112, f"Expected 2,112 rows, got {len(df)}"
    
    # 2. Canonical districts
    districts = sorted(df["district"].unique().tolist())
    assert districts == sorted(CANONICAL_DISTRICTS), f"Districts mismatch: {districts}"
    
    # 3. Canonical months: 264
    assert df["date"].nunique() == 264, f"Expected 264 months, got {df['date'].nunique()}"
    assert df["date"].min() == "2003-01-01"
    assert df["date"].max() == "2024-12-01"
    
    # 4. Zero duplicate keys
    assert not df.duplicated(subset=["district", "date"]).any(), "Duplicate district-month keys detected!"


def test_zero_nulls_in_required_features():
    """Verifies that all 6 stress components and WSI have zero nulls across all 2,112 rows."""
    model_path, _ = get_dataset_paths()
    if not os.path.exists(model_path):
        pytest.skip("Output dataset not found; run pipeline first.")
        
    df = pd.read_csv(model_path)
    
    for col in WSI_COMPONENT_COLS + ["wsi", "wsi_category"]:
        assert col in df.columns, f"Missing required column: {col}"
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column {col} has {null_count} nulls; expected 0."


def test_mathematical_wsi_calculation_exact():
    """Verifies that WSI is identically equal to (1/6) * sum(six components) within float precision."""
    model_path, _ = get_dataset_paths()
    if not os.path.exists(model_path):
        pytest.skip("Output dataset not found; run pipeline first.")
        
    df = pd.read_csv(model_path)
    calculated_wsi = df[WSI_COMPONENT_COLS].mean(axis=1)
    diff = (df["wsi"] - calculated_wsi).abs().max()
    assert diff < 1e-12, f"WSI mathematical deviation exceeds tolerance: max diff = {diff}"


def test_audit_model_dataset_parity():
    """Verifies that shared columns between Audit and Model datasets are strictly identical."""
    model_path, audit_path = get_dataset_paths()
    if not os.path.exists(model_path) or not os.path.exists(audit_path):
        pytest.skip("Authoritative datasets not found; run pipeline first.")
        
    model_df = pd.read_csv(model_path)
    audit_df = pd.read_csv(audit_path)
    
    shared_cols = [col for col in model_df.columns if col in audit_df.columns]
    assert len(shared_cols) >= 10, f"Expected at least 10 shared columns, found {len(shared_cols)}"
    
    for col in shared_cols:
        if pd.api.types.is_numeric_dtype(model_df[col]):
            diff = (model_df[col] - audit_df[col]).abs().max()
            assert diff < 1e-6, f"Numerical divergence between audit and model on column '{col}': max diff = {diff}"
        else:
            mismatch = (model_df[col] != audit_df[col]).sum()
            assert mismatch == 0, f"Categorical/string mismatch between audit and model on column '{col}': {mismatch} rows"
