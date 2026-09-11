# Marathwada Drought Early Warning — Water Stress Index (WSI)

A reproducible data-processing pipeline for constructing a district-month Water Stress Index (WSI)
for the eight districts of Marathwada over the common period 2003–2024.

## Study period

- Period: January 2003 – December 2024
- Districts: 8
- Monthly observations: 264 per district
- Final district-month records: 2,112

## Data streams

The pipeline integrates six sources:

1. CHIRPS rainfall → SPI-3
2. ERA5-Land soil water storage → soil-moisture anomaly
3. MOD13Q1 NDVI → NDVI anomaly
4. MOD11A2 LST → LST anomaly
5. Global WaterPack surface-water frequency → surface-water anomaly
6. GSDA groundwater observations → groundwater anomaly

## Final model features

The final model dataset contains:

- `spi_3`
- `soil_moisture_anomaly`
- `ndvi_anomaly`
- `lst_anomaly`
- `surface_water_anomaly`
- `groundwater_anomaly`

The WSI is calculated from the six stress-oriented standardized components.

## Temporal validation design

The dataset uses a chronological split:

- Train: 2003–2017
- Validation: 2018–2020
- Test: 2021–2024

Feature transformation parameters are fitted on the training period and then applied unchanged
to validation and test periods.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Marathwada_Drought_WSI_Pipeline.ipynb
└── src/
    └── full_pipeline.py
```

`full_pipeline.py` is an extraction of the notebook code in notebook-cell order. The notebook remains
the primary reproducibility artifact.

## Final datasets

The final model and audit CSVs are generated separately from the notebook workflow and should be
kept outside Git history unless the project explicitly requires publishing the data.

## Reproducibility note

The notebook contains exploratory, correction, validation, and finalization cells from the development
process. The final verified model/audit outputs are the authoritative outputs from the completed
pipeline.
