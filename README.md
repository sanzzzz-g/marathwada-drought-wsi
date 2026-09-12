# Marathwada Water Stress Index (WSI) Dataset and Pipeline

A spatiotemporal data engineering pipeline and forecasting benchmark for monthly drought monitoring and multi-step water stress forecasting across the eight administrative districts of Marathwada, Maharashtra, India (2003–2024).

The pipeline ingests six hydroclimatic data streams, enforces strict chronological separation to prevent temporal data leakage, and generates a leak-free 2,112-observation rectangular panel formatted for multi-step sequence forecasting models.

---

## Study Area and Scope

- **Region**: Marathwada, Maharashtra, India
- **Districts (8)**: Beed, Chhatrapati Sambhajinagar, Dharashiv, Hingoli, Jalna, Latur, Nanded, Parbhani
- **Temporal Coverage**: January 2003 to December 2024 (264 months)
- **Panel Structure**: Balanced rectangular grid of 2,112 rows ($8 \text{ districts} \times 264 \text{ months}$) on primary key `(district, date)`.

---

## Data Sources

| Domain | Source | Raw Variable | Native Resolution | Aggregation |
| :--- | :--- | :--- | :--- | :--- |
| **Precipitation** | CHIRPS v2.0 | `rainfall_mm` | 0.05° (~5.5 km) | District monthly total |
| **Root-Zone Soil Moisture** | ERA5-Land | Layers 1–4 ($0\text{--}289\text{ cm}$) | 0.1° (~9 km) | Depth-weighted storage (mm) |
| **Vegetation Condition** | MODIS MOD13Q1 | `ndvi` | 250 m (16-day) | Monthly district mean |
| **Land Surface Temperature** | MODIS MOD11A2 | `lst_celsius` | 1 km (8-day) | Monthly daytime mean |
| **Surface Water Extent** | Global WaterPack | `surface_water_frequency` | 250 m (daily) | Monthly surface water fraction |
| **Groundwater Depth** | GSDA Monitoring Network | `water_level_m_bgl` | ~1,000+ well stations | District median depth-to-water |

Detailed download URLs and column schemas are documented in [`data/README.md`](data/README.md).

---

## Methodology

### Chronological Splits
- **Training**: 2003-01 to 2017-12 (180 months $\times$ 8 districts = 1,440 rows)
- **Validation**: 2018-01 to 2020-12 (36 months $\times$ 8 districts = 288 rows)
- **Testing**: 2021-01 to 2024-12 (48 months $\times$ 8 districts = 384 rows)

All normalization, climatologies, and distribution parameters are fitted strictly on training data ($\le 2017\text{-}12$).

### Multi-Tier Standardized Stress Components
Each dimension is transformed into a standardized stress anomaly where positive values indicate drought stress:
1. **Meteorological Deficit ($z_{\text{spi}}$)**: 3-month accumulated precipitation fitted to a Gamma distribution calibrated over 1981–2017. Inverted so lower rainfall yields positive stress ($z_{\text{spi}} = -\text{SPI}_3$).
2. **Soil Moisture Deficit ($z_{\text{soil}}$)**: Root-zone storage anomaly relative to training monthly climatology, standardized per district.
3. **Vegetation Deficit ($z_{\text{ndvi}}$)**: MODIS NDVI anomaly relative to training monthly baseline.
4. **Thermal Stress ($z_{\text{lst}}$)**: MODIS daytime land surface temperature excess anomaly.
5. **Groundwater Stress ($z_{\text{gw}}$)**: Depth-to-water anomaly carried forward causally, standardized per district on the training series.
6. **Surface Water Deficit ($z_{\text{sw}}$)**: Surface water frequency deficit anomaly.

### Composite Water Stress Index (WSI)
$$\text{WSI}_{d, t} = \frac{1}{6} \left( z_{\text{spi}} + z_{\text{soil}} + z_{\text{ndvi}} + z_{\text{lst}} + z_{\text{gw}} + z_{\text{sw}} \right)_{d, t}$$

| Category | WSI Range | Interpretation |
| :--- | :--- | :--- |
| **Very Wet** | $\text{WSI} < -1.5$ | Severe moisture excess |
| **Wet** | $-1.5 \le \text{WSI} < -0.5$ | Above-normal moisture |
| **Normal** | $-0.5 \le \text{WSI} < 0.5$ | Near-climatological average |
| **Moderate Stress** | $0.5 \le \text{WSI} < 1.5$ | Significant hydrological deficit |
| **Severe Stress** | $\text{WSI} \ge 1.5$ | Severe drought |

---

## Supervised Forecasting Setup

- **Input Window ($X_t$)**: 12 months of the 6 standardized stress components: shape $(12, 6)$.
- **Forecast Horizon ($Y_t$)**: 1-, 2-, and 3-month lead times: shape $(3,)$.
- **Sequences**:
  - Training: 1,328 sequences (origins 2003-12 to 2017-09)
  - Validation: 288 sequences (origins 2018-01 to 2020-12)
  - Testing: 360 sequences (origins 2021-01 to 2024-09)
  - Total: 1,976 sequences

### Model Development Progression
1. **Baselines (Step 6)**: Persistence, Seasonal Persistence, WSI-AR(12), Ridge Regression, and XGBoost.
2. **Experiment A (Step 7 / 8.1)**: Direct 6-feature small LSTM ($X \to \text{WSI}_{t+h}$). Validation MAE: $0.517303$.
3. **Experiment B (Step 8.3 / 8.4)**: Residual 6-feature small LSTM ($X \to \text{WSI}_{t+h} - \text{WSI}_t$). Validation MAE: $0.473812$ (5-seed mean: $0.475888 \pm 0.003615$).
4. **Experiment C (Step 8.5)**: Residual LSTM + explicit scalar $WSI(t)$ state input via Keras Functional API. Validation MAE: $0.480891$. (Experiment B retained as the cleaner, superior formulation).

---

## Directory Layout

- `config/`: Pipeline configuration (`pipeline.yaml`).
- `data/`: Raw data documentation and sample data previews.
- `data_model/`: Step 5 input tensors (`X_*.npy`, `y_*.npy`) and sequence metadata.
- `data_model_residual/`: Step 8 residual target arrays (`y_residual_*.npy`, `wsi_origin_*.npy`).
- `EDA/`: Exploratory data analysis tables and figures.
- `outputs/`: Authoritative master Parquet, audit CSV, and model dataset CSV.
- `results/`: Baseline outputs, backtesting results, and experiment logs (`A/`, `B/`, `C/`).
- `src/`: Core library code:
  - `ingestion.py`, `qc.py`, `groundwater.py`, `spi.py`, `features.py`, `wsi.py`: Feature engineering pipeline.
  - `validation.py`, `pipeline.py`: Pipeline auditing and orchestration.
  - `sequence_builder.py`: Supervised sequence generation.
  - `baselines.py`, `evaluation.py`, `backtesting.py`: Baseline modeling and evaluation.
  - `lstm_model.py`: Direct LSTM implementation (Experiment A).
  - `build_residual_targets.py`: Residual target builder.
  - `residual_lstm.py`, `seed_robustness_b.py`: Experiment B training and seed diagnostics.
  - `experiment_c.py`: Experiment C functional model and evaluation.
- `tests/`: Automated unit and contract test suite.

---

## Execution

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Data Pipeline
```bash
python run_pipeline.py
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```

### 4. Build Sequences & Run Experiments
```bash
# Build supervised sequence tensors
python src/sequence_builder.py

# Run baseline forecasting suite
python src/baselines.py

# Build residual targets
python src/build_residual_targets.py

# Run Experiment B validation and seed robustness
python src/residual_lstm.py
python src/seed_robustness_b.py

# Run Experiment C validation and seed robustness
python src/experiment_c.py
```
