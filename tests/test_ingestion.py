"""
Tests for raw data ingestion, district normalization, and calendar construction.
"""

import pytest
import pandas as pd
import numpy as np

from src.ingestion import (
    CANONICAL_DISTRICTS,
    DISTRICT_ALIASES,
    normalize_district_name,
    create_canonical_calendar,
)


def test_district_normalization():
    """Verifies that all historical and administrative aliases map to canonical names."""
    assert normalize_district_name("Bid") == "Beed"
    assert normalize_district_name("Aurangabad") == "Chhatrapati Sambhajinagar"
    assert normalize_district_name("Osmanabad") == "Dharashiv"
    assert normalize_district_name("Latur") == "Latur"
    assert normalize_district_name("Parbhani") == "Parbhani"


def test_canonical_calendar_shape_and_keys():
    """Verifies the canonical rectangular calendar dimensions and properties."""
    cal = create_canonical_calendar(
        start_date="2003-01-01",
        end_date="2024-12-01",
        districts=CANONICAL_DISTRICTS,
    )
    
    # 8 districts x 264 months = 2,112 rows
    assert len(cal) == 2112
    assert cal["district"].nunique() == 8
    assert cal["date"].nunique() == 264
    
    # Check bounds
    assert cal["date"].min() == pd.Timestamp("2003-01-01")
    assert cal["date"].max() == pd.Timestamp("2024-12-01")
    
    # Check key uniqueness (zero duplicates)
    assert not cal.duplicated(subset=["district", "date"]).any()
    
    # Check that districts match exactly
    assert sorted(cal["district"].unique().tolist()) == sorted(CANONICAL_DISTRICTS)
