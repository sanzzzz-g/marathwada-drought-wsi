# Exploratory Data Analysis (EDA)

This directory contains the comprehensive empirical audit, statistical summaries, and diagnostic visualizations supporting the Marathwada Water Stress Index (WSI) data engineering pipeline.

The study area encompasses the eight administrative districts of the Marathwada region in Maharashtra, India: **Beed, Chhatrapati Sambhajinagar, Dharashiv, Hingoli, Jalna, Latur, Nanded, and Parbhani** over the 22-year period from **January 2003 to December 2024** (264 canonical months; 2,112 district-month observations).

---

## Structure

```text
EDA/
├── 01_data_quality.csv
├── 02_temporal_coverage.csv
├── 03_district_coverage.csv
├── 04_missingness.csv
├── 05_summary_statistics.csv
├── 06_monthly_climatology.csv
├── 07_outlier_report.csv
├── 08_correlation_matrix.csv
├── 09_lag_correlations.csv
├── 10_trend_statistics.csv
├── figures/
│   ├── correlation_matrix.png
│   ├── coverage.png
│   ├── district_comparisons.png
│   ├── groundwater_distributions.png
│   ├── lag_correlation.png
│   ├── lst_distributions.png
│   ├── missingness_heatmap.png
│   ├── monthly_climatologies.png
│   ├── ndvi_distributions.png
│   ├── rainfall_distributions.png
│   ├── soil_water_distributions.png
│   └── surface_water_distributions.png
└── README.md
```

---

## Statistical Summaries (`.csv`)

| File | Description | Key Findings |
| :--- | :--- | :--- |
| [`01_data_quality.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/01_data_quality.csv) | High-level dataset audit per stream | Zero nulls in CHIRPS and ERA5-Land; MODIS NDVI/LST have high completeness with localized cloud gaps; GSDA groundwater has episodic seasonal sampling. |
| [`02_temporal_coverage.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/02_temporal_coverage.csv) | Date range, continuity, and frequency per stream | Continuous monthly coverage across 2003–2024 for 5/6 streams; CHIRPS calibrated back to 1981. |
| [`03_district_coverage.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/03_district_coverage.csv) | District-level record counts and valid observations | All 8 districts have 264 continuous records in the rectangular grid. |
| [`04_missingness.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/04_missingness.csv) | District-by-month missing value breakdowns | Identifies isolated cloud-contaminated MODIS pixels and seasonal groundwater monitoring gaps (monitored in Jan, Mar, May, Oct). |
| [`05_summary_statistics.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/05_summary_statistics.csv) | Mean, std, min, percentiles, max for all raw and derived features | Establishes baselines for all 6 physical variables across training (2003–2017) and holdout periods. |
| [`06_monthly_climatology.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/06_monthly_climatology.csv) | 12-month district-wise seasonal expectations ($\mu_m, \sigma_m$) | Quantifies monsoon peak (June–September) and summer heat stress (March–May). |
| [`07_outlier_report.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/07_outlier_report.csv) | Extreme values beyond $3\sigma$ or IQR thresholds | Documents historical drought events (2015–2016) vs. sensor artifacts (e.g. Parbhani July 2022 NDVI masked). |
| [`08_correlation_matrix.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/08_correlation_matrix.csv) | Cross-stream concurrent correlation matrix | Strong alignment between rainfall, soil moisture, and NDVI; inverse relationship with LST. |
| [`09_lag_correlations.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/09_lag_correlations.csv) | Cross-correlation at lags $k \in [-6, +6]$ months | Demonstrates hydrological cascade: Rainfall ($t$) leads Soil Water ($t+0, t+1$), NDVI ($t+1, t+2$), and Groundwater ($t+2, t+3$). |
| [`10_trend_statistics.csv`](file:///c:/Users/Yash%20Bhardwaj/Desktop/wsi/marathwada-drought-wsi/EDA/10_trend_statistics.csv) | Mann-Kendall trend tests and Sen's slopes (2003–2024) | Detects rising temperatures (LST) and variable monsoon onset dynamics across the region. |

---

## Visualizations (`figures/`)

1. **`coverage.png`**: Heatmap of spatiotemporal completeness across all 8 districts and 6 data streams.
2. **`missingness_heatmap.png`**: Exact calendar matrix of observation availability highlighting groundwater seasonal monitoring rounds.
3. **`monthly_climatologies.png`**: 12-month annual cycle profiles for rainfall, soil moisture, vegetation, surface temperature, groundwater depth, and water body extent.
4. **`rainfall_distributions.png`**: CHIRPS monthly and 3-month accumulated precipitation distributions by district.
5. **`soil_water_distributions.png`**: ERA5-Land root-zone (0–289 cm) soil water storage distributions.
6. **`ndvi_distributions.png`**: MODIS MOD13Q1 vegetation health dynamics and seasonal canopy evolution.
7. **`lst_distributions.png`**: MODIS MOD11A2 daytime land surface temperature distributions and summer extremes.
8. **`groundwater_distributions.png`**: GSDA well depth-to-water distributions across pre-monsoon (May) and post-monsoon (October) monitoring rounds.
9. **`surface_water_distributions.png`**: Global WaterPack monthly water surface occurrence frequency.
10. **`correlation_matrix.png`**: Correlation heatmap confirming expected physical polarity across all six stress dimensions.
11. **`lag_correlation.png`**: Cross-lag correlograms demonstrating the multi-month delay from meteorological deficit to groundwater response.
12. **`district_comparisons.png`**: Multi-panel spatial comparison contrasting western rain-shadow districts (Beed, Dharashiv) with eastern districts (Nanded).

---

## Methodological Conclusions for Deep Learning

1. **Information Leakage Prevention**: Because climatological baselines and standardizers dictate anomaly values, all statistical fitting must be restricted strictly to the 2003–2017 training window.
2. **Causal Groundwater Representation**: Groundwater monitoring occurs episodically (~4 times/year). A causal as-of forward fill standardized against the observation month prevents future leakage and eliminates artificial seasonal step jumps.
3. **Multi-Step Horizon Alignment**: The rolling 12-month lookback ($X_{t-11:t}$) paired with a 3-month multi-step forecast target ($[WSI_{t+1}, WSI_{t+2}, WSI_{t+3}]$) maps directly to the physical hydrological lag identified in `09_lag_correlations.csv`.
