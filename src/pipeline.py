"""
src/pipeline.py
===============
Master orchestration pipeline for the Marathwada Drought Water Stress Index (WSI).

Executes the exact 28-step implementation sequence:
 1. Load raw six datasets
 2. Structural validation
 3. Normalize district names
 4. Normalize dates
 5. Freeze 2003-01 -> 2024-12
 6. Build canonical 2,112-row district-month calendar
 7. Split TRAIN / VALIDATION / TEST
 8. Perform source-specific QC
 9. Fit TRAIN-only imputation rules
10. Transform missing values
11. Build CHIRPS 3-month accumulation
12. Fit TRAIN-only SPI parameters
13. Generate SPI for all periods
14. Build ERA5 soil-water anomaly
15. Build NDVI anomaly
16. Build LST anomaly
17. Build groundwater qualifying observations
18. Build groundwater as-of stress feature
19. Build GWP surface-water anomaly
20. Standardize all six stress components using TRAIN-only parameters
21. Validate six-component polarity
22. Calculate WSI
23. Calculate WSI category
24. Validate WSI mathematically
25. Validate temporal leakage
26. Validate groundwater causality
27. Validate model/audit parity
28. Export master + audit + model + parameters + validation report
"""

from typing import Dict, Any, Optional
import os
import json
import yaml
import pandas as pd
import numpy as np

from src.ingestion import (
    CANONICAL_DISTRICTS,
    CANONICAL_START_DATE,
    CANONICAL_END_DATE,
    create_canonical_calendar,
    load_raw_chirps,
    load_raw_era5land,
    load_raw_ndvi,
    load_raw_lst,
    load_raw_gwp,
    load_raw_groundwater,
)
from src.qc import (
    apply_ndvi_source_qc,
    fit_train_imputation_stats,
    impute_missing_with_train_stats,
)
from src.groundwater import (
    clean_groundwater_records,
    aggregate_district_monthly_groundwater,
    measure_groundwater_coverage,
    measure_groundwater_sensitivity,
    fit_train_groundwater_climatology,
    process_groundwater_features,
)
from src.spi import (
    compute_chirps_rolling_accumulation,
    fit_train_gamma_parameters,
    verify_spi_implementation_against_reference,
    process_spi_features,
)
from src.features import (
    process_soil_water_features,
    process_ndvi_features,
    process_lst_features,
    process_surface_water_features,
)
from src.wsi import (
    calculate_wsi,
    WSI_COMPONENT_COLS,
)
from src.validation import (
    run_full_validation_suite,
)


def load_config(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """Loads pipeline configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config_path: str = "config/pipeline.yaml",
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    Executes the end-to-end 28-step Marathwada WSI rebuild pipeline.
    """
    cfg = load_config(config_path)
    raw_cfg = cfg["raw_data"]
    qc_cfg = cfg.get("qc", {})
    gw_qc_cfg = qc_cfg.get("groundwater", {})
    min_wells = gw_qc_cfg.get("min_qualifying_wells", 10)
    staleness_thresh = gw_qc_cfg.get("staleness_threshold_months", 3)
    use_gwp_lag_1 = qc_cfg.get("gwp", {}).get("use_lag_1", False)

    train_end = cfg["splits"]["row_level"]["train"]["end"]
    val_end = cfg["splits"]["row_level"]["validation"]["end"]

    # -------------------------------------------------------------
    # Step 1 - 4: Load raw six datasets, structural validation & normalization
    # -------------------------------------------------------------
    print("Step 1-4: Loading and structurally validating raw six datasets...")
    chirps_raw = load_raw_chirps(raw_cfg["chirps"])
    era5_raw = load_raw_era5land(raw_cfg["era5land"])
    ndvi_raw = load_raw_ndvi(raw_cfg["ndvi"])
    lst_raw = load_raw_lst(raw_cfg["lst"])
    gwp_raw = load_raw_gwp(raw_cfg["gwp"])
    gw_raw = load_raw_groundwater(raw_cfg["groundwater"])

    # -------------------------------------------------------------
    # Step 5 - 6: Build canonical 2,112-row district-month calendar
    # -------------------------------------------------------------
    print("Step 5-6: Building canonical 2,112-row rectangular calendar (2003-01 to 2024-12)...")
    calendar_df = create_canonical_calendar(
        start_date=cfg["dates"]["canonical_start"],
        end_date=cfg["dates"]["canonical_end"],
        districts=cfg["districts"],
    )

    # -------------------------------------------------------------
    # Step 7: Split TRAIN / VALIDATION / TEST row tags & forecast origins
    # -------------------------------------------------------------
    print("Step 7: Defining chronological split tags and forecast origins...")
    def assign_split(dt: pd.Timestamp) -> str:
        if dt <= pd.Timestamp(train_end):
            return "train"
        elif dt <= pd.Timestamp(val_end):
            return "validation"
        else:
            return "test"

    calendar_df["split"] = calendar_df["date"].apply(assign_split)
    calendar_df["is_train"] = (calendar_df["split"] == "train").astype(int)
    calendar_df["is_val"] = (calendar_df["split"] == "validation").astype(int)
    calendar_df["is_test"] = (calendar_df["split"] == "test").astype(int)

    # Forecast-origin flags for horizon 3
    # Last train origin is 2017-09-01 (targets Oct, Nov, Dec 2017)
    calendar_df["is_forecast_origin_train"] = (
        (calendar_df["date"] >= "2003-01-01") & (calendar_df["date"] <= "2017-09-01")
    ).astype(int)
    calendar_df["is_forecast_origin_val"] = (
        (calendar_df["date"] >= "2018-01-01") & (calendar_df["date"] <= "2020-09-01")
    ).astype(int)
    calendar_df["is_forecast_origin_test"] = (
        (calendar_df["date"] >= "2021-01-01") & (calendar_df["date"] <= "2024-09-01")
    ).astype(int)

    # -------------------------------------------------------------
    # Step 8 - 10: Source-specific QC, TRAIN-only imputation rules, transform
    # -------------------------------------------------------------
    print("Step 8-10: Performing source QC and TRAIN-only imputation...")
    # NDVI: Mask Parbhani July 2022 QA artifact
    ndvi_qc = apply_ndvi_source_qc(ndvi_raw)

    # Fit TRAIN-only imputation stats for NDVI
    ndvi_impute_stats = fit_train_imputation_stats(ndvi_qc, val_col="ndvi", train_end_date=train_end)
    ndvi_clean, ndvi_imputed_count = impute_missing_with_train_stats(
        ndvi_qc, val_col="ndvi", impute_stats=ndvi_impute_stats, flag_col_name="ndvi_imputed"
    )

    # LST: Preserve raw, fit TRAIN-only imputation stats
    lst_qc = lst_raw.copy()
    lst_qc["lst_raw"] = lst_qc["lst_celsius"]
    lst_impute_stats = fit_train_imputation_stats(lst_qc, val_col="lst_celsius", train_end_date=train_end)
    lst_clean, lst_imputed_count = impute_missing_with_train_stats(
        lst_qc, val_col="lst_celsius", impute_stats=lst_impute_stats, flag_col_name="lst_imputed"
    )

    # -------------------------------------------------------------
    # Step 11 - 13: CHIRPS 3-month accumulation, TRAIN-only Gamma fit, SPI
    # -------------------------------------------------------------
    print("Step 11-13: Computing SPI-3 with 1981-2017 Gamma fitting & reference check...")
    chirps_acc = compute_chirps_rolling_accumulation(chirps_raw, window=3)
    spi_cal_start = cfg.get("spi", {}).get("calibration_start", "1981-01-01")
    spi_cal_end = cfg.get("spi", {}).get("calibration_end", train_end)
    gamma_params = fit_train_gamma_parameters(
        chirps_acc,
        train_start_date=spi_cal_start,
        train_end_date=spi_cal_end,
    )

    # Verification test vs standard SciPy/xclim reference implementation
    sample_rainfalls = np.array([0.0, 5.0, 25.0, 75.0, 150.0, 300.0])
    spi_ref_verified = verify_spi_implementation_against_reference(sample_rainfalls)
    if not spi_ref_verified:
        raise RuntimeError("SPI verification test against reference implementation failed!")

    spi_df, spi_train_stats = process_spi_features(
        calendar_df=calendar_df,
        chirps_acc_df=chirps_acc,
        gamma_params=gamma_params,
        train_end_date=train_end,
    )

    # -------------------------------------------------------------
    # Step 14 - 16: ERA5 soil-water, NDVI, and LST multi-tier features
    # -------------------------------------------------------------
    print("Step 14-16: Engineering ERA5, NDVI, and LST multi-tier features...")
    soil_df, soil_train_stats = process_soil_water_features(
        calendar_df=calendar_df,
        era5_df=era5_raw,
        train_end_date=train_end,
    )
    ndvi_feat_df, ndvi_train_stats = process_ndvi_features(
        calendar_df=calendar_df,
        ndvi_df=ndvi_clean,
        train_end_date=train_end,
    )
    lst_feat_df, lst_train_stats = process_lst_features(
        calendar_df=calendar_df,
        lst_df=lst_clean,
        train_end_date=train_end,
    )

    # -------------------------------------------------------------
    # Step 17 - 18: Groundwater qualifying observations & causal as-of stress
    # -------------------------------------------------------------
    print("Step 17-18: Performing empirical groundwater audit and causal stress propagation...")
    gw_clean = clean_groundwater_records(gw_raw)
    gw_agg = aggregate_district_monthly_groundwater(gw_clean, min_wells_threshold=min_wells)

    # Measure empirical coverage consequences of N >= 10
    gw_coverage_audit = measure_groundwater_coverage(
        calendar_df=calendar_df,
        agg_gw_df=gw_agg,
        min_wells_threshold=min_wells,
    )

    # Measure sensitivity across N in [5, 10, 15]
    gw_sensitivity_thresholds = gw_qc_cfg.get("sensitivity_thresholds", [5, 10, 15])
    gw_sensitivity_audit = measure_groundwater_sensitivity(
        calendar_df=calendar_df,
        cleaned_gw_df=gw_clean,
        thresholds=gw_sensitivity_thresholds,
    )

    # Fit TRAIN-only climatology mu_GW(d, m)
    gw_train_stats = fit_train_groundwater_climatology(
        agg_gw_df=gw_agg,
        train_end_date=train_end,
    )

    # Build causal observation-standardized as-of feature with exact training series standardizer
    gw_feat_df, gw_asof_train_stats = process_groundwater_features(
        calendar_df=calendar_df,
        agg_gw_df=gw_agg,
        train_clim_stats=gw_train_stats,
        train_end_date=train_end,
        min_wells_threshold=min_wells,
        staleness_threshold_months=staleness_thresh,
    )

    # -------------------------------------------------------------
    # Step 19: Global WaterPack surface water anomaly
    # -------------------------------------------------------------
    print("Step 19: Engineering GWP surface water features (cutoff verified)...")
    gwp_feat_df, gwp_train_stats = process_surface_water_features(
        calendar_df=calendar_df,
        gwp_df=gwp_raw,
        train_end_date=train_end,
        use_lag_1=use_gwp_lag_1,
    )

    # -------------------------------------------------------------
    # Step 20 - 21: Assemble master dataframe & validate stress polarity
    # -------------------------------------------------------------
    print("Step 20-21: Assembling multi-tier master dataframe and checking polarity...")
    master = calendar_df.copy()

    # Merge SPI
    master = pd.merge(
        master,
        spi_df[["district", "date", "rainfall_mm", "rainfall_3m_mm", "spi_3", "spi_stress", "spi_stress_z"]],
        on=["district", "date"],
    )

    # Merge Soil Water
    master = pd.merge(
        master,
        soil_df[[
            "district", "date",
            "soil_water_layer_1", "soil_water_layer_2", "soil_water_layer_3", "soil_water_layer_4",
            "soil_water_storage_mm", "soil_water_climatology_mm", "soil_water_anomaly_mm", "soil_stress_z",
        ]],
        on=["district", "date"],
    )

    # Merge NDVI
    master = pd.merge(
        master,
        ndvi_feat_df[[
            "district", "date",
            "ndvi_raw", "ndvi", "ndvi_artifact_masked", "ndvi_imputed",
            "ndvi_climatology", "ndvi_anomaly", "ndvi_stress_z",
        ]],
        on=["district", "date"],
    )

    # Merge LST
    master = pd.merge(
        master,
        lst_feat_df[[
            "district", "date",
            "lst_raw", "lst_celsius", "lst_imputed",
            "lst_climatology_celsius", "lst_anomaly_celsius", "lst_stress_z",
        ]],
        on=["district", "date"],
    )

    # Merge Groundwater
    master = pd.merge(
        master,
        gw_feat_df[[
            "district", "date",
            "groundwater_depth_m_bgl_raw", "groundwater_n_wells", "groundwater_is_qualifying",
            "groundwater_low_sample_flag", "groundwater_as_of_depth_m_bgl", "groundwater_climatology_m_bgl",
            "groundwater_anomaly_m_bgl", "groundwater_stress_z", "months_since_last_gw_obs",
            "last_gw_obs_date", "groundwater_stale_flag",
        ]],
        on=["district", "date"],
    )

    # Merge Surface Water
    master = pd.merge(
        master,
        gwp_feat_df[[
            "district", "date",
            "surface_water_frequency_raw", "surface_water_frequency", "valid_pixels",
            "surface_water_climatology", "surface_water_anomaly", "surface_water_stress_z",
        ]],
        on=["district", "date"],
    )

    # -------------------------------------------------------------
    # Step 22 - 24: Calculate WSI, WSI category, and validate mathematically
    # -------------------------------------------------------------
    print("Step 22-24: Calculating equal-weighted WSI and discrete categories...")
    master = calculate_wsi(master, components=WSI_COMPONENT_COLS)

    # -------------------------------------------------------------
    # Step 25 - 27: Validate temporal leakage, causality, and parity
    # -------------------------------------------------------------
    print("Step 25-27: Running rigorous automated validation suite...")
    # Generate the model subset directly from master
    model_cols = [
        "district", "date", "year", "month", "split",
        "spi_stress_z", "soil_stress_z", "ndvi_stress_z", "lst_stress_z",
        "groundwater_stress_z", "surface_water_stress_z",
        "groundwater_stale_flag", "months_since_last_gw_obs",
        "ndvi_imputed", "lst_imputed",
        "wsi", "wsi_category",
    ]
    model_df = master[model_cols].copy()

    # Generate the audit dataset directly from master
    audit_df = master.copy()

    val_report = run_full_validation_suite(
        master_df=master,
        audit_df=audit_df,
        model_df=model_df,
    )

    if val_report["overall_status"] != "PASSED":
        print(f"Validation FAILED! Report: {json.dumps(val_report, indent=2)}")
        raise RuntimeError("Authoritative validation checks failed!")
    else:
        print("Validation PASSED across all 7 rigorous checks!")

    # -------------------------------------------------------------
    # Step 28: Export authoritative Parquet, Audit CSV, Model CSV, JSON stats
    # -------------------------------------------------------------
    print("Step 28: Exporting authoritative Parquet, Audit CSV, Model CSV, and metadata...")
    master_parquet_path = os.path.join(output_dir, cfg["outputs"]["master_parquet"])
    audit_csv_path = os.path.join(output_dir, cfg["outputs"]["master_audit_csv"])
    model_csv_path = os.path.join(output_dir, cfg["outputs"]["model_csv"])
    train_stats_path = os.path.join(output_dir, cfg["outputs"]["train_statistics_json"])
    val_report_path = os.path.join(output_dir, cfg["outputs"]["validation_report_json"])
    meta_path = os.path.join(output_dir, cfg["outputs"]["preprocessing_metadata_json"])

    # Ensure output directories exist
    for p in [master_parquet_path, audit_csv_path, model_csv_path, train_stats_path, val_report_path, meta_path]:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    # Export Parquet & CSVs
    try:
        master.to_parquet(master_parquet_path, index=False, engine="pyarrow")
    except Exception as e:
        print(f"Note: pyarrow export failed ({e}), attempting fallback engine...")
        master.to_parquet(master_parquet_path, index=False)

    audit_df.to_csv(audit_csv_path, index=False)
    model_df.to_csv(model_csv_path, index=False)

    # Compile train statistics JSON
    all_train_stats = {
        "train_period": {"start": "2003-01-01", "end": train_end},
        "spi_calibration_period": {
            "start": spi_cal_start,
            "end": spi_cal_end,
            "standard": "WMO 30+ year climatological baseline",
        },
        "spi_gamma_parameters": gamma_params,
        "spi_standardization": spi_train_stats,
        "soil_water": soil_train_stats,
        "ndvi": ndvi_train_stats,
        "ndvi_imputation_medians": ndvi_impute_stats,
        "lst": lst_train_stats,
        "lst_imputation_medians": lst_impute_stats,
        "groundwater_observation_climatology": gw_train_stats,
        "groundwater_as_of_standardization": gw_asof_train_stats,
        "surface_water": gwp_train_stats,
    }
    with open(train_stats_path, "w", encoding="utf-8") as f:
        json.dump(all_train_stats, f, indent=2, default=str)

    # Export validation report JSON
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, default=str)

    # Compile preprocessing metadata JSON
    metadata = {
        "pipeline_version": "2.0.0",
        "districts_count": len(CANONICAL_DISTRICTS),
        "total_canonical_months": 264,
        "total_rows": 2112,
        "splits": cfg["splits"],
        "wsi_formula": "WSI = (spi_stress_z + soil_stress_z + ndvi_stress_z + lst_stress_z + groundwater_stress_z + surface_water_stress_z) / 6.0",
        "component_weights": cfg["wsi"]["component_weights"],
        "wsi_category_thresholds": cfg["wsi"]["categories"],
        "imputation_counts": {
            "ndvi_imputed_cells": ndvi_imputed_count,
            "lst_imputed_cells": lst_imputed_count,
        },
        "groundwater_empirical_coverage": gw_coverage_audit,
        "groundwater_threshold_sensitivity": gw_sensitivity_audit,
        "forecast_origin_semantics": "Forecast issued after month t is complete, using data available through end of month t (X_t -> WSI_{t+1:t+3}).",
        "model_input_tensor_shape": "Lookback 12 months, 6 continuous stress features: (12, 6). Targets: WSI for horizons t+1, t+2, t+3: (3,).",
        "test_period_evaluation_note": "Test period (2021-2024) contains predominantly wet/normal conditions; continuous forecasting metrics (R^2, RMSE, MAE) are valid, but categorical drought event detection should use rolling-origin backtesting across historical droughts (e.g. 2015).",
        "artifacts_generated": {
            "master_parquet": master_parquet_path,
            "master_audit_csv": audit_csv_path,
            "model_csv": model_csv_path,
            "train_statistics": train_stats_path,
            "validation_report": val_report_path,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("Pipeline completed successfully! All authoritative artifacts created.")
    return {
        "master_df": master,
        "model_df": model_df,
        "audit_df": audit_df,
        "validation_report": val_report,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse
    import sys

    # Ensure parent directory is on sys.path for direct script execution
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    parser = argparse.ArgumentParser(description="Marathwada Drought WSI Master Pipeline")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline configuration YAML")
    parser.add_argument("--output-dir", default=".", help="Base output directory")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, output_dir=args.output_dir)

