# Data Sources and Extraction Guide

This directory documents the hydroclimatic source datasets for the eight districts of the Marathwada region, Maharashtra, India.

---

## 1. Precipitation (CHIRPS)
- **Dataset**: Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS) v2.0 Monthly
- **Source**: UCSB Climate Hazards Center / Google Earth Engine (`UCSB-CHG/CHIRPS/DAILY`)
- **URL**: [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps)
- **Spatial Resolution**: 0.05° (~5.5 km)
- **Temporal Coverage**: 1981-01 to 2026-02
- **Filename**: `marathwada_chirps_monthly_rainfall_1981_2026.csv`
- **Schema**: `district, district_gaul, year, month, rainfall_mm`
- **Units**: Millimeters (mm)
- **Pipeline Role**: Evaluated as a 3-month rolling accumulation ($R_{3m}(t) = \sum_{k=0}^2 R(t-k)$) and calibrated against a 1981–2017 Gamma distribution baseline to compute SPI-3.

---

## 2. Soil Moisture (ERA5-Land)
- **Dataset**: ERA5-Land Monthly Averaged by Hour of Day
- **Source**: ECMWF Copernicus Climate Change Service (C3S) / Google Earth Engine (`ECMWF/ERA5_LAND/MONTHLY_BY_HOUR`)
- **URL**: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land-monthly-means)
- **Spatial Resolution**: 0.1° (~9 km)
- **Temporal Coverage**: 2000-01 to 2026-01
- **Filename**: `marathwada_era5land_soil_water_2000_2026.csv`
- **Schema**: `district, district_gaul, year, month, soil_water_layer_1, soil_water_layer_2, soil_water_layer_3, soil_water_layer_4, soil_water_storage_mm_0_289cm`
- **Units**: Volumetric soil moisture ($\text{m}^3/\text{m}^3$) and Storage (mm)
- **Storage Formulation**: Total storage ($0\text{--}289\text{ cm}$) $= 70\theta_1 + 210\theta_2 + 720\theta_3 + 1890\theta_4$.
- **Pipeline Role**: Transformed into standardized anomaly relative to 2003–2017 monthly climatology.

---

## 3. Vegetation Index (MODIS NDVI)
- **Dataset**: MOD13Q1.061 Terra Vegetation Indices 16-Day Global 250m
- **Source**: NASA LP DAAC / Google Earth Engine (`MODIS/061/MOD13Q1`)
- **URL**: [https://lpdaac.usgs.gov/products/mod13q1v061/](https://lpdaac.usgs.gov/products/mod13q1v061/)
- **Spatial Resolution**: 250 m
- **Temporal Coverage**: 2000-02 to 2025-01
- **Filename**: `marathwada_mod13q1_monthly_ndvi_2000_2025.csv`
- **Schema**: `district, district_gaul, year, month, ndvi`
- **Units**: Dimensionless normalized difference ratio $[-1.0, 1.0]$
- **Quality Control**: Parbhani July 2022 cloud QA artifact (`0.122442`) masked and imputed using training climatological median.

---

## 4. Land Surface Temperature (MODIS LST)
- **Dataset**: MOD11A2.061 Terra Land Surface Temperature and Emissivity 8-Day Global 1km
- **Source**: NASA LP DAAC / Google Earth Engine (`MODIS/061/MOD11A2`)
- **URL**: [https://lpdaac.usgs.gov/products/mod11a2v061/](https://lpdaac.usgs.gov/products/mod11a2v061/)
- **Spatial Resolution**: 1,000 m (1 km)
- **Temporal Coverage**: 2000-02 to 2025-01
- **Filename**: `marathwada_mod11a2_monthly_lst_2000_2025.csv`
- **Schema**: `district, district_gaul, year, month, lst_celsius`
- **Units**: Degrees Celsius (°C)
- **Pipeline Role**: Standardized daytime surface temperature anomaly relative to 2003–2017 monthly climatology.

---

## 5. Surface Water Dynamics (Global WaterPack)
- **Dataset**: Global WaterPack (GWP) Monthly Surface Water Dynamics
- **Source**: German Aerospace Center (DLR) Earth Observation Center
- **URL**: [https://doi.org/10.1038/s41597-024-03328-7](https://doi.org/10.1038/s41597-024-03328-7)
- **Spatial Resolution**: 250 m
- **Temporal Coverage**: 2003-01 to 2024-12
- **Filename**: `marathwada_gwp_monthly_surface_water_2003_2024.csv`
- **Schema**: `district, date, year, month, surface_water_frequency, valid_pixels`
- **Units**: Water occurrence frequency (fraction $[0.0, 1.0]$) and valid pixel count.

---

## 6. Groundwater Depth (GSDA)
- **Dataset**: Groundwater Surveys and Development Agency (GSDA) Maharashtra Monitoring Well Records
- **Source**: Water Resources Department, Government of Maharashtra
- **URL**: [https://gsda.maharashtra.gov.in/](https://gsda.maharashtra.gov.in/)
- **Spatial Resolution**: Observation dug wells across the 8 districts
- **Temporal Coverage**: 1989-05 to 2024-10
- **Filename**: `marathwada_groundwater_extracted.csv`
- **Schema**: `well_id, state, district, block, village, site_name, aquifer, agency, latitude, longitude, well_type, well_depth_m, date, water_level_m_bgl`
- **Units**: Meters Below Ground Level (m bgl)
- **Pipeline Role**:
  - Filter for $N \ge 10$ active wells per district-month.
  - Calculate anomaly relative to observation-month climatology: $\Delta GW_s = GW_s - \mu(d, m_s)$.
  - Causally forward-propagate as-of anomaly across the panel.
  - Standardize directly on the monthly as-of training series ($\text{ddof}=0$) to achieve exact $\sigma_{\text{train}} = 1.000$.
  - Flag observation age $> 3$ months as stale (`groundwater_stale_flag = 1`).

---

## Sample Data

A 24-row sample of the model training dataset is available in [`sample/sample_model_data.csv`](sample/sample_model_data.csv) for schema inspection without loading raw files.
