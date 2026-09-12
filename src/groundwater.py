"""
GSDA groundwater data cleaning, district aggregation, causal monthly propagation,
and district-wise Z-standardization.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np


def clean_groundwater_records(raw_gw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw GSDA station well observations:
    - Removes null dates and depths
    - Removes physically impossible negative depths
    - Deduplicates exact triples (well_id, date, depth)
    - Excludes conflicting observations (same well_id and date with multiple different depths)
    """
    df = raw_gw_df.copy()
    
    # Drop missing
    df = df.dropna(subset=["water_level_m_bgl", "obs_date"]).copy()
    df = df[df["water_level_m_bgl"] >= 0.0].copy()
    
    # Deduplicate exact
    df = df.drop_duplicates(subset=["well_id", "obs_date", "water_level_m_bgl"]).copy()
    
    # Exclude conflicting
    conflicts = df.duplicated(subset=["well_id", "obs_date"], keep=False)
    if conflicts.any():
        df = df[~conflicts].copy()
        
    return df.reset_index(drop=True)


def aggregate_district_monthly_groundwater(
    cleaned_gw_df: pd.DataFrame,
    min_wells_threshold: int = 10,
) -> pd.DataFrame:
    """
    Aggregates cleaned station well observations to district-month level:
    - Calculates n_wells (total active monitoring wells in district-month)
    - Calculates median depth (meters below ground level)
    - Applies qualifying filter: is_qualifying = (n_wells >= min_wells_threshold)
    """
    grouped = cleaned_gw_df.groupby(["district", "obs_date"]).agg(
        n_wells=("water_level_m_bgl", "count"),
        median_depth_m_bgl=("water_level_m_bgl", "median"),
        mean_depth_m_bgl=("water_level_m_bgl", "mean"),
        std_depth_m_bgl=("water_level_m_bgl", lambda x: np.std(x, ddof=0)),
        iqr_depth_m_bgl=("water_level_m_bgl", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    ).reset_index()
    
    grouped["year"] = grouped["obs_date"].dt.year
    grouped["month"] = grouped["obs_date"].dt.month
    grouped["is_qualifying"] = grouped["n_wells"] >= min_wells_threshold
    
    return grouped.sort_values(["district", "obs_date"]).reset_index(drop=True)


def measure_groundwater_coverage(
    calendar_df: pd.DataFrame,
    agg_gw_df: pd.DataFrame,
    min_wells_threshold: int = 10,
) -> Dict[str, Any]:
    """
    Measures the empirical coverage consequences of the N >= min_wells_threshold rule:
    - Number of qualifying district-month observations
    - Percentage of canonical months with n < threshold
    - Maximum groundwater gap (months between qualifying observations)
    - Mean groundwater age under causal forward-filling
    """
    merged = pd.merge(
        calendar_df[["district", "date", "year", "month"]],
        agg_gw_df[["district", "obs_date", "n_wells", "is_qualifying"]],
        left_on=["district", "date"],
        right_on=["district", "obs_date"],
        how="left",
    )
    merged["n_wells"] = merged["n_wells"].fillna(0).astype(int)
    merged["is_qualifying"] = merged["is_qualifying"].fillna(False)
    
    total_months = len(merged)
    qualifying_count = int(merged["is_qualifying"].sum())
    unobserved_count = total_months - qualifying_count
    pct_unobserved = float((unobserved_count / total_months) * 100)
    
    district_gaps: Dict[str, int] = {}
    district_mean_ages: Dict[str, float] = {}
    all_ages = []
    
    for dist, d_df in merged.groupby("district"):
        d_sorted = d_df.sort_values("date").copy()
        current_age = 0
        has_seen_obs = False
        dist_ages = []
        max_dist_gap = 0
        cur_gap = 0
        
        for _, row in d_sorted.iterrows():
            if row["is_qualifying"]:
                has_seen_obs = True
                current_age = 0
                if cur_gap > max_dist_gap:
                    max_dist_gap = cur_gap
                cur_gap = 0
            else:
                cur_gap += 1
                if has_seen_obs:
                    current_age += 1
                else:
                    current_age = 1
            dist_ages.append(current_age)
            
        district_gaps[dist] = max_dist_gap
        district_mean_ages[dist] = float(np.mean(dist_ages))
        all_ages.extend(dist_ages)
        
    return {
        "min_wells_threshold": min_wells_threshold,
        "total_canonical_district_months": total_months,
        "qualifying_observations_count": qualifying_count,
        "low_sample_or_missing_count": unobserved_count,
        "pct_months_with_n_lt_10": pct_unobserved,
        "overall_max_gap_months": int(max(district_gaps.values())) if district_gaps else 0,
        "overall_mean_groundwater_age_months": float(np.mean(all_ages)) if all_ages else 0.0,
        "district_max_gaps": district_gaps,
        "district_mean_ages": district_mean_ages,
    }


def measure_groundwater_sensitivity(
    calendar_df: pd.DataFrame,
    cleaned_gw_df: pd.DataFrame,
    thresholds: List[int] = [5, 10, 15],
) -> Dict[str, Any]:
    """
    Sensitivity analysis for groundwater threshold N in [5, 10, 15]:
    Compares qualifying count, non-qualifying %, max gap, and mean age across thresholds.
    """
    sensitivity_results = {}
    for n in thresholds:
        agg_df = aggregate_district_monthly_groundwater(cleaned_gw_df, min_wells_threshold=n)
        cov = measure_groundwater_coverage(calendar_df, agg_df, min_wells_threshold=n)
        sensitivity_results[f"N_{n}"] = {
            "min_wells_threshold": n,
            "qualifying_observations_count": cov["qualifying_observations_count"],
            "low_sample_or_missing_count": cov["low_sample_or_missing_count"],
            "pct_months_non_qualifying": cov["pct_months_with_n_lt_10"],
            "overall_max_gap_months": cov["overall_max_gap_months"],
            "overall_mean_age_months": cov["overall_mean_groundwater_age_months"],
        }
    return sensitivity_results


def fit_train_groundwater_climatology(
    agg_gw_df: pd.DataFrame,
    train_end_date: str = "2017-12-01",
    train_start_date: str = "2003-01-01",
) -> Dict[str, Any]:
    """
    Fits groundwater observation climatology strictly on TRAIN period (2003-01 to 2017-12)
    using only qualifying observations (n_wells >= 10):
    - District-Month mean depth: mu_GW(d, m)
    - District overall mean depth (fallback): mu_GW(d)
    """
    train_df = agg_gw_df[
        (agg_gw_df["obs_date"] >= train_start_date)
        & (agg_gw_df["obs_date"] <= train_end_date)
        & (agg_gw_df["is_qualifying"])
    ].copy()
    
    # District-Month mean depth
    dm_means = train_df.groupby(["district", "month"])["median_depth_m_bgl"].mean().to_dict()
    # District mean fallback
    d_means = train_df.groupby("district")["median_depth_m_bgl"].mean().to_dict()
    
    return {
        "district_month_climatology": {f"{k[0]}_{k[1]}": float(v) for k, v in dm_means.items()},
        "district_mean_fallback": {str(k): float(v) for k, v in d_means.items()},
    }


def process_groundwater_features(
    calendar_df: pd.DataFrame,
    agg_gw_df: pd.DataFrame,
    train_clim_stats: Dict[str, Any],
    train_end_date: str = "2017-12-01",
    min_wells_threshold: int = 10,
    staleness_threshold_months: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds the authoritative groundwater features across the canonical 2,112-row calendar.
    
    1. For each qualifying observation round s (n_wells >= min_wells_threshold):
       gw_anomaly_s = GW_s - mu_GW(d, m_s)  (physical anomaly in meters)
       
    2. Causally forward-carries the as-of observation state:
       gw_anomaly_t = gw_anomaly_s (until the next qualifying observation)
       
    3. Calculates standardization parameters strictly on the actual AS-OF training monthly series (ddof=0):
       mu_GW,d = mean(GWStress_{d, t}^{train})
       sigma_GW,d = std(GWStress_{d, t}^{train}, ddof=0)
       
    4. Standardizes:
       GWStressZ_{d, t} = (gw_anomaly_{d, t} - mu_GW,d) / sigma_GW,d
       Guaranteeing exact training mean = 0.0 and training std = 1.0 for every district.
    """
    dm_clim = train_clim_stats["district_month_climatology"]
    d_fallback = train_clim_stats["district_mean_fallback"]
    
    # 1. Compute observation physical anomaly for all qualifying observations
    obs_df = agg_gw_df.copy()
    obs_df["obs_climatology"] = np.nan
    obs_df["obs_anomaly"] = np.nan
    
    for idx, row in obs_df.iterrows():
        if not row["is_qualifying"]:
            continue
        dist = row["district"]
        m = int(row["month"])
        depth = row["median_depth_m_bgl"]
        
        dm_key = f"{dist}_{m}"
        mu = dm_clim.get(dm_key, d_fallback.get(dist, depth))
        anom = depth - mu
        
        obs_df.loc[idx, "obs_climatology"] = mu
        obs_df.loc[idx, "obs_anomaly"] = anom
        
    qual_lookup = (
        obs_df[obs_df["is_qualifying"]]
        .set_index(["district", "obs_date"])
        .sort_index()
    )
    all_lookup = (
        obs_df
        .set_index(["district", "obs_date"])
        .sort_index()
    )
    
    records = []
    
    for dist in sorted(calendar_df["district"].unique()):
        dist_cal = calendar_df[calendar_df["district"] == dist].sort_values("date")
        
        # Initialize from pre-2003 qualifying observation if available
        pre_qual = obs_df[
            (obs_df["district"] == dist)
            & (obs_df["is_qualifying"])
            & (obs_df["obs_date"] < dist_cal["date"].min())
        ].sort_values("obs_date")
        
        if not pre_qual.empty:
            last_qual_row = pre_qual.iloc[-1]
            last_s_date = last_qual_row["obs_date"]
            last_depth = last_qual_row["median_depth_m_bgl"]
            last_clim = last_qual_row["obs_climatology"]
            last_anom = last_qual_row["obs_anomaly"]
        else:
            # Strictly causal: No past observation exists prior to calendar start
            last_s_date = None
            last_depth = d_fallback.get(dist, 10.0)
            last_clim = last_depth
            last_anom = 0.0
                
        for _, row in dist_cal.iterrows():
            curr_date = row["date"]
            key = (dist, curr_date)
            
            raw_depth = np.nan
            n_wells = 0
            is_qualifying = 0
            low_sample_flag = 0
            
            if key in all_lookup.index:
                row_obs = all_lookup.loc[key]
                if isinstance(row_obs, pd.DataFrame):
                    row_obs = row_obs.iloc[0]
                n_wells = int(row_obs["n_wells"])
                raw_depth = float(row_obs["median_depth_m_bgl"])
                if row_obs["is_qualifying"]:
                    is_qualifying = 1
                elif n_wells > 0:
                    low_sample_flag = 1
                    
            if key in qual_lookup.index:
                qual_row = qual_lookup.loc[key]
                if isinstance(qual_row, pd.DataFrame):
                    qual_row = qual_row.iloc[0]
                last_s_date = curr_date
                last_depth = float(qual_row["median_depth_m_bgl"])
                last_clim = float(qual_row["obs_climatology"])
                last_anom = float(qual_row["obs_anomaly"])
                
            if last_s_date is not None:
                delta_months = (curr_date.year - last_s_date.year) * 12 + (curr_date.month - last_s_date.month)
                delta_months = max(0, delta_months)
                last_s_date_str = last_s_date.strftime("%Y-%m-%d")
            else:
                delta_months = 999
                last_s_date_str = None
                
            stale_flag = 1 if delta_months > staleness_threshold_months else 0
            
            records.append({
                "district": dist,
                "date": curr_date,
                "groundwater_depth_m_bgl_raw": raw_depth,
                "groundwater_n_wells": n_wells,
                "groundwater_is_qualifying": is_qualifying,
                "groundwater_low_sample_flag": low_sample_flag,
                "groundwater_as_of_depth_m_bgl": last_depth,
                "groundwater_climatology_m_bgl": last_clim,
                "groundwater_anomaly_m_bgl": last_anom,
                "months_since_last_gw_obs": delta_months,
                "last_gw_obs_date": last_s_date_str,
                "groundwater_stale_flag": stale_flag,
            })
            
    res_df = pd.DataFrame(records)
    
    # Fit standardization parameters strictly on the actual AS-OF monthly training series (ddof=0)
    train_mask = res_df["date"] <= train_end_date
    train_asof = res_df[train_mask]
    
    gw_asof_means = train_asof.groupby("district")["groundwater_anomaly_m_bgl"].mean().to_dict()
    gw_asof_stds = train_asof.groupby("district")["groundwater_anomaly_m_bgl"].std(ddof=0).to_dict()
    
    for d, s in gw_asof_stds.items():
        if pd.isna(s) or s <= 1e-6:
            gw_asof_stds[d] = 1.0
            
    # Standardize for all periods
    res_df["groundwater_stress_z"] = [
        (r["groundwater_anomaly_m_bgl"] - gw_asof_means.get(r["district"], 0.0))
        / gw_asof_stds.get(r["district"], 1.0)
        for _, r in res_df.iterrows()
    ]
    
    train_stats = {
        "groundwater_as_of_mean_by_district": {str(k): float(v) for k, v in gw_asof_means.items()},
        "groundwater_as_of_std_by_district": {str(k): float(v) for k, v in gw_asof_stds.items()},
    }
    
    return res_df.sort_values(["district", "date"]).reset_index(drop=True), train_stats
