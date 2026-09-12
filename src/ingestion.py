"""
src/ingestion.py
================
Module for loading, structurally validating, normalizing aliases,
and building the canonical calendar for the 6 raw Marathwada hydro-climatic datasets.

Strictly adheres to:
- RAW DATA -> structural QC only (no statistical calculations before split)
- Canonical study area: Exactly 8 districts
- Canonical timeline: 2003-01 to 2024-12 (264 months, 2,112 district-month cells)
"""

from typing import Dict, List, Tuple, Optional
import os
import pandas as pd
import numpy as np

# Canonical 8 districts of Marathwada
CANONICAL_DISTRICTS: List[str] = [
    "Beed",
    "Chhatrapati Sambhajinagar",
    "Dharashiv",
    "Hingoli",
    "Jalna",
    "Latur",
    "Nanded",
    "Parbhani",
]

# Historical / administrative aliases mapping
DISTRICT_ALIASES: Dict[str, str] = {
    "Bid": "Beed",
    "Aurangabad": "Chhatrapati Sambhajinagar",
    "Osmanabad": "Dharashiv",
}

CANONICAL_START_DATE = "2003-01-01"
CANONICAL_END_DATE = "2024-12-01"


def normalize_district_name(district: str) -> str:
    """Standardizes district name against known historical/administrative aliases."""
    if not isinstance(district, str):
        return str(district)
    clean_name = district.strip()
    return DISTRICT_ALIASES.get(clean_name, clean_name)


def create_canonical_calendar(
    start_date: str = CANONICAL_START_DATE,
    end_date: str = CANONICAL_END_DATE,
    districts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds the complete canonical rectangular calendar:
    8 districts x 264 months = 2,112 rows.
    """
    if districts is None:
        districts = CANONICAL_DISTRICTS

    dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    records = []
    for d in sorted(districts):
        for dt in dates:
            records.append({
                "district": d,
                "date": dt,
                "year": dt.year,
                "month": dt.month,
            })

    calendar_df = pd.DataFrame(records)
    calendar_df = calendar_df.sort_values(["district", "date"]).reset_index(drop=True)
    return calendar_df


def load_raw_chirps(filepath: str) -> pd.DataFrame:
    """
    Loads raw CHIRPS rainfall CSV.
    Columns expected: district, [district_gaul], year, month, rainfall_mm
    Retains full history (1981+) so rolling 3-month windows for early 2003 can be computed causally.
    """
    df = pd.read_csv(filepath)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    # Structural checks
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    
    # Check for invalid values
    if (df["rainfall_mm"] < 0).any():
        raise ValueError("CHIRPS contains physically impossible negative rainfall values.")
        
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    
    # Check for duplicates on (district, date)
    if df.duplicated(subset=["district", "date"]).any():
        dups = df[df.duplicated(subset=["district", "date"], keep=False)]
        raise ValueError(f"CHIRPS contains duplicate district-date entries:\n{dups.head()}")
        
    return df[["district", "date", "year", "month", "rainfall_mm"]].sort_values(
        ["district", "date"]
    ).reset_index(drop=True)


def load_raw_era5land(filepath: str) -> pd.DataFrame:
    """
    Loads raw ERA5-Land soil moisture CSV.
    Columns expected: district, [district_gaul], year, month,
                      soil_water_layer_1, soil_water_layer_2, soil_water_layer_3, soil_water_layer_4,
                      soil_water_storage_mm_0_289cm
    """
    df = pd.read_csv(filepath)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    
    for col in [
        "soil_water_layer_1",
        "soil_water_layer_2",
        "soil_water_layer_3",
        "soil_water_layer_4",
        "soil_water_storage_mm_0_289cm",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Boundary checks: volumetric soil water must be in [0, 1] m3/m3
    for col in ["soil_water_layer_1", "soil_water_layer_2", "soil_water_layer_3", "soil_water_layer_4"]:
        if ((df[col] < 0) | (df[col] > 1.0)).any():
            raise ValueError(f"ERA5-Land {col} has values outside physical bounds [0, 1].")
            
    if (df["soil_water_storage_mm_0_289cm"] < 0).any():
        raise ValueError("ERA5-Land soil water storage cannot be negative.")
        
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    
    if df.duplicated(subset=["district", "date"]).any():
        dups = df[df.duplicated(subset=["district", "date"], keep=False)]
        raise ValueError(f"ERA5-Land contains duplicate district-date entries:\n{dups.head()}")
        
    cols = [
        "district", "date", "year", "month",
        "soil_water_layer_1", "soil_water_layer_2", "soil_water_layer_3", "soil_water_layer_4",
        "soil_water_storage_mm_0_289cm",
    ]
    return df[cols].sort_values(["district", "date"]).reset_index(drop=True)


def load_raw_ndvi(filepath: str) -> pd.DataFrame:
    """
    Loads raw MOD13Q1 NDVI CSV.
    Columns expected: district, [district_gaul], year, month, ndvi
    """
    df = pd.read_csv(filepath)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
    
    # Bound check: NDVI must be in [-1, 1]
    valid_ndvi = df["ndvi"].dropna()
    if ((valid_ndvi < -1.0) | (valid_ndvi > 1.0)).any():
        raise ValueError("MODIS NDVI contains values outside physical bounds [-1, 1].")
        
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    
    if df.duplicated(subset=["district", "date"]).any():
        dups = df[df.duplicated(subset=["district", "date"], keep=False)]
        raise ValueError(f"NDVI contains duplicate district-date entries:\n{dups.head()}")
        
    return df[["district", "date", "year", "month", "ndvi"]].sort_values(
        ["district", "date"]
    ).reset_index(drop=True)


def load_raw_lst(filepath: str) -> pd.DataFrame:
    """
    Loads raw MOD11A2 LST CSV.
    Columns expected: district, [district_gaul], year, month, lst_celsius
    """
    df = pd.read_csv(filepath)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["lst_celsius"] = pd.to_numeric(df["lst_celsius"], errors="coerce")
    
    # Bound check: LST in Marathwada should be between 0 C and 70 C
    valid_lst = df["lst_celsius"].dropna()
    if ((valid_lst < 0.0) | (valid_lst > 70.0)).any():
        raise ValueError("MODIS LST contains values outside terrestrial bounds [0, 70] C.")
        
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    
    if df.duplicated(subset=["district", "date"]).any():
        dups = df[df.duplicated(subset=["district", "date"], keep=False)]
        raise ValueError(f"LST contains duplicate district-date entries:\n{dups.head()}")
        
    return df[["district", "date", "year", "month", "lst_celsius"]].sort_values(
        ["district", "date"]
    ).reset_index(drop=True)


def load_raw_gwp(filepath: str) -> pd.DataFrame:
    """
    Loads raw Global WaterPack surface water CSV.
    Columns expected: district, date, year, month, surface_water_frequency, valid_pixels
    """
    df = pd.read_csv(filepath)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["surface_water_frequency"] = pd.to_numeric(df["surface_water_frequency"], errors="coerce")
    df["valid_pixels"] = pd.to_numeric(df["valid_pixels"], errors="coerce")
    
    # Bound check: frequency in [0, 1]
    if ((df["surface_water_frequency"] < 0.0) | (df["surface_water_frequency"] > 1.0)).any():
        raise ValueError("GWP surface water frequency outside valid range [0, 1].")
    if (df["valid_pixels"] <= 0).any():
        raise ValueError("GWP valid pixels must be strictly positive.")
        
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    
    if df.duplicated(subset=["district", "date"]).any():
        dups = df[df.duplicated(subset=["district", "date"], keep=False)]
        raise ValueError(f"GWP contains duplicate district-date entries:\n{dups.head()}")
        
    cols = ["district", "date", "year", "month", "surface_water_frequency", "valid_pixels"]
    return df[cols].sort_values(["district", "date"]).reset_index(drop=True)


def load_raw_groundwater(filepath: str) -> pd.DataFrame:
    """
    Loads raw GSDA groundwater station-level well observations.
    Performs structural cleaning:
    - Normalizes district names
    - Parses dates to datetime
    - Drops observations with missing water_level_m_bgl
    - Drops impossible negative depths (depth below ground level must be >= 0)
    - Deduplicates exact pairs (well_id, date, water_level_m_bgl)
    - Excludes conflicting pairs (same well_id and date with different water_level_m_bgl)
    """
    df = pd.read_csv(filepath, low_memory=False)
    df["district"] = df["district"].apply(normalize_district_name)
    df = df[df["district"].isin(CANONICAL_DISTRICTS)].copy()
    
    # Parse dates
    df["raw_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["raw_date"]).copy()
    
    # Cast depth
    df["water_level_m_bgl"] = pd.to_numeric(df["water_level_m_bgl"], errors="coerce")
    df = df.dropna(subset=["water_level_m_bgl"]).copy()
    
    # Depth must be >= 0
    df = df[df["water_level_m_bgl"] >= 0.0].copy()
    
    # Create month anchor
    df["obs_date"] = pd.to_datetime(
        df["raw_date"].dt.year.astype(str)
        + "-"
        + df["raw_date"].dt.month.astype(str).str.zfill(2)
        + "-01"
    )
    df["year"] = df["obs_date"].dt.year
    df["month"] = df["obs_date"].dt.month
    
    # Deduplicate exact pairs (well_id, raw_date, water_level_m_bgl)
    df = df.drop_duplicates(subset=["well_id", "raw_date", "water_level_m_bgl"]).copy()
    
    # Identify conflicting pairs: same well_id and raw_date with differing water_level_m_bgl
    conflict_mask = df.duplicated(subset=["well_id", "raw_date"], keep=False)
    if conflict_mask.any():
        df = df[~conflict_mask].copy()
        
    cols = [
        "well_id", "district", "block", "village", "latitude", "longitude",
        "well_depth_m", "raw_date", "obs_date", "year", "month", "water_level_m_bgl",
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols].sort_values(["district", "obs_date"]).reset_index(drop=True)
