# Marathwada Water Stress Index (WSI) Dataset and Pipeline

A spatiotemporal data engineering pipeline and benchmark dataset for monthly drought monitoring and multi-step water stress forecasting across the eight administrative districts of the Marathwada region in Maharashtra, India (2003–2024).

The pipeline ingests six raw hydroclimatic data streams, enforces strict chronological separation to prevent temporal data leakage, and generates a leak-free 2,112-observation rectangular panel formatted for sequence forecasting models (e.g., LSTM, Transformer).

---

## Study Area and Spatiotemporal Scope

- **Region**: Marathwada, Maharashtra, India
- **Districts (8)**: Beed, Chhatrapati Sambhajinagar, Dharashiv, Hingoli, Jalna, Latur, Nanded, Parbhani
- **Historical Alias Mapping**:
  - `Bid` &rarr; `Beed`
  - `Aurangabad` &rarr; `Chhatrapati Sambhajinagar`
  - `Osmanabad` &rarr; `Dharashiv`
- **Temporal Coverage**: January 2003 to December 2024 (264 months)
- **Panel Structure**: Balanced rectangular grid of 2,112 rows ($8 \text{ districts} \times 264 \text{ months}$) on unique primary key `(district, date)`.

---

## Data Sources

Features are derived from six remote sensing and ground monitoring datasets:

| Domain | Source | Raw Variable | Native Resolution | Aggregation |
| :--- | :--- | :--- | :--- | :--- |
| **Precipitation** | CHIRPS v2.0 | `rainfall_mm` | 0.05° (~5.5 km) | District monthly total |
| **Root-Zone Soil Moisture** | ERA5-Land | Layers 1–4 ($0\text{--}289\text{ cm}$) | 0.1° (~9 km) | Depth-weighted storage (mm) |
| **Vegetation Condition** | MODIS MOD13Q1 | `ndvi` | 250 m (16-day) | Monthly district mean |
| **Land Surface Temperature** | MODIS MOD11A2 | `lst_celsius` | 1 km (8-day) | Monthly daytime mean |
| **Surface Water Extent** | Global WaterPack | `surface_water_frequency` | 250 m (daily) | Monthly surface water fraction |
| **Groundwater Depth** | GSDA Monitoring Network | `water_level_m_bgl` | ~1,000+ well stations | District median depth-to-water |

Detailed download URLs, Google Earth Engine extraction scripts, and column schemas are documented in [`data/README.md`](data/README.md).

---

## Methodology and Feature Engineering

### Order of Operations and Split Isolation

To guarantee that no future or holdout information leaks into preprocessing:
1. **Structural Quality Control**: Format normalization, date validation, and alias resolution are performed prior to splitting.
2. **Chronological Splitting**: The dataset is partitioned into Train (2003–2017), Validation (2018–2020), and Test (2021–2024).
3. **Train-Only Statistical Calibration**: All climatological means, standard deviations, imputation medians, and distribution parameters are fitted strictly on training data ($\le 2017\text{-}12$).
4. **Causal Forward Transformation**: Validation and test sets are transformed using the fixed training baselines.

### Multi-Tier Hydroclimatic Stress Components

Each of the six dimensions is transformed into a standardized stress anomaly $z \in \mathbb{R}$ where **positive values indicate drought stress** and **negative values indicate surplus moisture**:

1. **Meteorological Deficit ($z_{\text{spi}}$)**:
   - Evaluated using 3-month accumulated precipitation ($R_{3m}(t) = \sum_{k=0}^2 R(t-k)$).
   - Fitted with a two-parameter Gamma distribution calibrated over a 37-year baseline (1981–2017; ~36–37 samples per district-month), satisfying WMO standard climatological guidelines.
   - Standardized per district on training records with $\text{ddof}=0$ and inverted so that lower precipitation yields positive stress ($z_{\text{spi}} = -\text{SPI}_3$).

2. **Soil Moisture Deficit ($z_{\text{soil}}$)**:
   - Root-zone soil water storage integrated across all four ERA5-Land layers:
     $$S = 70\,\theta_1 + 210\,\theta_2 + 720\,\theta_3 + 1890\,\theta_4 \quad (\text{mm})$$
   - Physical anomaly computed relative to the training monthly climatology $\mu_{\text{soil}}(d, m)$.
   - Standardized per district on training records ($z_{\text{soil}} = -\text{anomaly} / \sigma_{\text{train}}$).

3. **Vegetation Health Deficit ($z_{\text{ndvi}}$)**:
   - Sensor artifact in Parbhani (July 2022 cloud QA failure) is masked and imputed using the training climatological median for Parbhani in July.
   - Standardized against training monthly baseline ($z_{\text{ndvi}} = -\text{anomaly} / \sigma_{\text{train}}$).

4. **Thermal Stress ($z_{\text{lst}}$)**:
   - Daytime land surface temperature anomaly computed relative to training monthly climatology $\mu_{\text{lst}}(d, m)$.
   - High surface temperature indicates evaporative demand and thermal stress ($z_{\text{lst}} = +\text{anomaly} / \sigma_{\text{train}}$).

5. **Groundwater Stress ($z_{\text{gw}}$)**:
   - Minimum sample threshold of $N \ge 10$ active wells per district-month is required for an observation round to qualify.
   - Physical anomaly is computed at the observation month $s$: $\Delta GW_s = GW_s - \mu_{GW}(d, m_s)$.
   - For unobserved subsequent months $t > s$, the anomaly is carried forward causally ($\Delta GW_t = \Delta GW_s$), preventing seasonal step jumps.
   - Standardized directly on the monthly as-of training series ($\mu_{\text{train}} = 0.000$, $\sigma_{\text{train}} = 1.000$, $\text{ddof}=0$) to preserve equal weighting with continuous streams.
   - Observation age $\Delta t = t - s$ is tracked; $\Delta t > 3$ months flags a stale reading (`groundwater_stale_flag = 1`).

6. **Surface Water Deficit ($z_{\text{sw}}$)**:
   - Surface water frequency anomaly computed relative to training monthly baseline $\mu_{\text{sw}}(d, m)$.
   - Standardized per district on training records ($z_{\text{sw}} = -\text{anomaly} / \sigma_{\text{train}}$).

### Water Stress Index (WSI)

The composite index is computed as the unweighted mean of the six standardized stress dimensions:

$$\text{WSI}_{d, t} = \frac{1}{6} \left( z_{\text{spi}} + z_{\text{soil}} + z_{\text{ndvi}} + z_{\text{lst}} + z_{\text{gw}} + z_{\text{sw}} \right)_{d, t}$$

Because each component is calibrated to unit variance ($\sigma_{\text{train}} = 1.000$) on the training set, each environmental indicator contributes equally to the composite index.

| Stress Category | WSI Range | Physical Interpretation |
| :--- | :--- | :--- |
| **Very Wet** | $\text{WSI} < -1.5$ | Severe moisture excess / flood potential |
| **Wet** | $-1.5 \le \text{WSI} < -0.5$ | Above-normal moisture availability |
| **Normal** | $-0.5 \le \text{WSI} < 0.5$ | Near-climatological average conditions |
| **Moderate Stress** | $0.5 \le \text{WSI} < 1.5$ | Significant soil moisture and hydrological deficit |
| **Severe Stress** | $\text{WSI} \ge 1.5$ | Severe meteorological and hydrological drought |

---

## Sequence Forecasting Formulation

The dataset is structured for rolling-origin sequence models (e.g., LSTM, GRU, Temporal Fusion Transformer):

- **Input Window ($X_t$)**: 12 consecutive months of the six continuous standardized features:
  $$X_t \in \mathbb{R}^{12 \times 6}$$
- **Forecast Horizon ($Y_t$)**: Continuous WSI targets for 1-, 2-, and 3-month lead times:
  $$Y_t = \left[ \text{WSI}_{t+1},\, \text{WSI}_{t+2},\, \text{WSI}_{t+3} \right] \in \mathbb{R}^3$$
- **Information Cutoff**: Month-end semantics ($X_t$ uses observations through the final day of month $t$; $Y_t$ predicts months $t+1, t+2, t+3$).

### Partition Boundaries

| Partition | Date Range | Months | Rows | Forecast Origin Bounds |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 2003-01 to 2017-12 | 180 | 1,440 | Last origin: `2017-09` (targets Oct, Nov, Dec 2017 $\le$ 2017-12) |
| **Validation** | 2018-01 to 2020-12 | 36 | 288 | Origins: `2018-01` to `2020-09` |
| **Testing** | 2021-01 to 2024-12 | 48 | 384 | Origins: `2021-01` to `2024-09` |

Zero target values from training forecast origins cross into the validation or testing periods.

---

## Repository Structure

```text
marathwada-drought-wsi/
│
├── README.md                                  # Documentation and reproduction guide
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git exclusions (.venv, archive/, tests/, data/*.csv)
│
├── config/
│   └── pipeline.yaml                          # Model hyperparameters, thresholds, and paths
│
├── data/
│   ├── README.md                              # Data source specifications and retrieval instructions
│   └── sample/
│       └── sample_model_data.csv              # 24-row preview of the model feature matrix
│
├── EDA/
│   ├── 01_data_quality.csv                    # Quality and completeness audit
│   ├── 02_temporal_coverage.csv               # Temporal range and continuity metrics
│   ├── 03_district_coverage.csv               # District-level observation counts
│   ├── 04_missingness.csv                     # Missingness breakdown per stream
│   ├── 05_summary_statistics.csv              # Feature distributional statistics
│   ├── 06_monthly_climatology.csv             # 12-month seasonal climatologies
│   ├── 07_outlier_report.csv                  # Anomaly and outlier audit
│   ├── 08_correlation_matrix.csv              # Concurrent cross-feature correlation matrix
│   ├── 09_lag_correlations.csv                # Multi-month lead/lag cross-correlations
│   ├── 10_trend_statistics.csv                # Mann-Kendall and Sen's slope trend tests
│   ├── figures/                               # 12 publication-quality diagnostic plots
│   └── README.md                              # Detailed summary of exploratory findings
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py                           # Raw data ingestion, bounds checking, calendar construction
│   ├── qc.py                                  # Quality control, artifact masking, train-only imputation
│   ├── groundwater.py                         # Well deduplication, sensitivity audit, causal propagation
│   ├── spi.py                                 # CHIRPS rolling accumulation, Gamma fit, SPI-3 derivation
│   ├── features.py                            # Soil moisture, NDVI, LST, and surface water anomalies
│   ├── wsi.py                                 # Equal-weight WSI calculation and categorical labeling
│   ├── validation.py                          # Automated 7-point data integrity and leakage audit
│   └── pipeline.py                            # Master pipeline orchestrator (CLI entry point)
│
├── artifacts/
│   ├── train_statistics.json                  # Parameters calibrated on training set (2003–2017)
│   ├── preprocessing_metadata.json            # Tensor shapes, split definitions, and empirical stats
│   └── validation_report.json                 # Audit report across all 7 verification checks
│
└── outputs/
    ├── README.md                              # Output data dictionary and schemas
    ├── Marathwada_MASTER_2003_2024.parquet    # Columnar binary archive with full multi-tier feature audit
    ├── Marathwada_MASTER_AUDIT_DATASET_2003_2024.csv # 37-column complete audit trail (CSV)
    └── Marathwada_MODEL_DATASET_2003_2024.csv # 17-column clean standardized model training matrix (CSV)
```

---

## Reproduction

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Execute Data Pipeline

To recompute all anomalies, calibrate distributions, and generate output datasets from raw data:

```bash
python src/pipeline.py --config config/pipeline.yaml
```

The pipeline automatically runs the validation suite before exporting artifacts:
- Structural integrity (2,112 rows, 8 districts, 264 months)
- Zero nulls across all six required model features
- Exact mathematical parity between individual components and WSI
- Temporal isolation across splits and forecast origins

