

````markdown
# Marathwada Drought Early Warning & Water Stress Index (WSI)

A data-driven drought monitoring and early-warning pipeline for the Marathwada region of Maharashtra, India.

This project integrates rainfall, soil water, vegetation, land-surface temperature, surface-water, and groundwater observations to construct a monthly Water Stress Index (WSI) for eight districts from 2003–2024.

---

## 🌍 Study Region

The analysis covers eight districts of Marathwada:

- Beed
- Chhatrapati Sambhajinagar
- Dharashiv
- Hingoli
- Jalna
- Latur
- Nanded
- Parbhani

### Study Period

**January 2003 – December 2024**

- 8 districts
- 264 months
- 2,112 district-month observations

---

## 🎯 Project Objective

The objective of this project is to integrate multiple environmental and hydrological indicators into a unified Water Stress Index (WSI) for characterizing historical drought and water-stress conditions across Marathwada.

The resulting dataset provides a foundation for future drought early-warning and time-series forecasting models.

The pipeline uses chronological data splitting and training-period-only transformation parameters to reduce temporal data leakage in machine-learning applications.

---

## 🛰️ Data Sources

The pipeline integrates six environmental and hydrological data streams:

| Data Source | Variable | Processing |
|---|---|---|
| CHIRPS | Rainfall | SPI-3 |
| ERA5-Land | Soil water storage (0–289 cm) | Soil-moisture anomaly |
| MOD13Q1 | NDVI | NDVI anomaly |
| MOD11A2 | Land Surface Temperature | LST anomaly |
| Global WaterPack | Surface-water frequency | Surface-water anomaly |
| GSDA | Groundwater level | Groundwater anomaly |

---

## 🔄 Overall Processing Pipeline

```text
                    Raw Environmental Data
                             |
        +--------------------+--------------------+
        |                    |                    |
     CHIRPS               ERA5-Land          MOD13Q1
     Rainfall             Soil Water             NDVI
        |                    |                    |
      SPI-3            Soil Moisture          NDVI
                         Anomaly              Anomaly
        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------+--------+
                    |                 |
                 MOD11A2         Global WaterPack
                    |                 |
                   LST          Surface Water
                 Anomaly           Anomaly
                    |                 |
                    +--------+--------+
                             |
                         GSDA
                      Groundwater
                             |
                     As-of Processing
                             |
                             v
                  Six WSI Components
                             |
                             v
                    Feature Scaling
                             |
                             v
                    Water Stress Index
                             |
                             v
                    WSI Classification
````

---

## 🧮 Methodology

### 1. SPI-3

Three-month accumulated CHIRPS rainfall is used to calculate the Standardized Precipitation Index (SPI-3).

The SPI distribution parameters are fitted separately by district and calendar month using the training period and then applied unchanged to the validation and test periods.

Lower SPI values represent greater rainfall-related stress.

### 2. Soil Moisture Anomaly

ERA5-Land soil water storage for the 0–289 cm soil profile is transformed into a district- and calendar-month anomaly.

Lower-than-normal soil water storage represents greater water stress.

### 3. NDVI Anomaly

MOD13Q1 NDVI is transformed into a district- and calendar-month anomaly.

Lower-than-normal NDVI represents greater vegetation stress associated with reduced water availability.

### 4. LST Anomaly

MOD11A2 land-surface temperature is transformed into a district- and calendar-month anomaly.

Higher-than-normal LST represents greater water stress.

### 5. Surface Water Anomaly

Global WaterPack surface-water frequency is transformed into a district- and calendar-month anomaly.

Lower-than-normal surface-water frequency represents greater water stress.

### 6. Groundwater Anomaly

Groundwater observations are irregular rather than strictly monthly.

For every district-month, the pipeline uses the most recent groundwater observation available at or before that month:

```text
GW_as_of(t) =
most recent groundwater observation at or before month t
```

The groundwater processing also retains:

* groundwater observation date
* months since the last groundwater observation
* number of wells contributing to the observation

This prevents future groundwater observations from being used to construct earlier observations.

---

## 🧹 Data Quality Control

The pipeline includes quality-control procedures for:

* Missing satellite observations
* District-calendar-month climatological filling
* Groundwater irregular sampling
* Duplicate well-date observations
* Groundwater as-of temporal alignment
* Satellite quality-control inspection
* District-month completeness
* Duplicate district-date detection
* Chronological train/validation/test checks
* Final reproducibility checks

A quality-control correction was applied to the Parbhani July 2022 NDVI observation after identifying an anomalous QA artifact.

---

## 📊 Temporal Train / Validation / Test Split

A chronological split is used to preserve the temporal structure of the dataset and avoid temporal leakage.

| Dataset    | Period        |      Rows |
| ---------- | ------------- | --------: |
| Training   | 2003–2017     |     1,440 |
| Validation | 2018–2020     |       288 |
| Test       | 2021–2024     |       384 |
| **Total**  | **2003–2024** | **2,112** |

There is no date overlap between the training, validation, and test datasets.

### Train-Only Transformation

Feature transformation parameters are fitted using the training period only and then applied unchanged to validation and test data.

This includes:

* SPI distribution parameters
* District-calendar-month anomaly baselines
* Feature standardization parameters

---

## 📈 Final Model Features

The final model dataset contains six stress-oriented features:

```text
spi_3
soil_moisture_anomaly
ndvi_anomaly
lst_anomaly
surface_water_anomaly
groundwater_anomaly
```

### Stress Orientation

| Component     | Greater Stress             |
| ------------- | -------------------------- |
| SPI-3         | Lower rainfall / lower SPI |
| Soil moisture | Lower soil moisture        |
| NDVI          | Lower NDVI                 |
| LST           | Higher LST                 |
| Surface water | Lower surface water        |
| Groundwater   | Greater groundwater depth  |

---

## 🧮 Water Stress Index (WSI)

The Water Stress Index is constructed from the six standardized stress-oriented components.

Equal weighting is used because no externally specified component weights were provided.

```text
WSI =
mean(
    SPI stress,
    Soil Moisture stress,
    NDVI stress,
    LST stress,
    Surface Water stress,
    Groundwater stress
)
```

Higher WSI values indicate greater water stress.

---

## 🚨 WSI Categories

The project uses the following project-defined WSI categories:

|        WSI Range | Category        |
| ---------------: | --------------- |
|         `< -1.5` | Very Wet        |
| `-1.5 to < -0.5` | Wet             |
|  `-0.5 to < 0.5` | Normal          |
|   `0.5 to < 1.5` | Moderate Stress |
|         `>= 1.5` | Severe Stress   |

These thresholds are project-defined classification rules and are not presented as a universal externally validated drought standard.

---

## 🗂️ Final Dataset

The final model dataset contains:

```text
district
date
spi_3
soil_moisture_anomaly
ndvi_anomaly
lst_anomaly
surface_water_anomaly
groundwater_anomaly
WSI
WSI_category
```

### Dataset Statistics

* 2,112 district-month observations
* 8 districts
* 264 months
* January 2003 – December 2024
* 10 columns in the final model dataset

A separate audit dataset contains the raw/source variables together with the final calculated features and WSI values for traceability.

---

## 🔍 Model Dataset vs Audit Dataset

The final model dataset is the clean modeling version containing only the final features and WSI outputs.

The audit dataset contains:

* Raw rainfall
* Raw soil water storage
* Raw NDVI
* Raw LST
* Groundwater observations
* Groundwater observation dates
* Months since last groundwater observation
* Number of wells
* Surface-water frequency
* Valid pixels
* Final calculated features
* WSI
* WSI category

The calculated variables in both datasets were compared using district-date keys.

The following variables were verified to match exactly:

```text
spi_3
soil_moisture_anomaly
ndvi_anomaly
lst_anomaly
surface_water_anomaly
groundwater_anomaly
WSI
WSI_category
```

All 2,112 district-month observations passed the consistency check.

---

## ✅ Final Dataset Validation

The completed dataset passed the following validation checks:

* 2,112 total rows
* 8 unique districts
* 264 months
* 2,112 unique district-date keys
* Complete monthly sequence for every district
* No duplicate district-date keys
* No missing final model features
* No train/validation/test date overlap
* Valid WSI categories
* WSI reproducibility difference: 0.0
* Model and audit calculated variables match exactly

---

## 📁 Repository Structure

```text
marathwada-drought-wsi/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── Marathwada_Drought_WSI_Pipeline.ipynb
│
└── src/
    └── full_pipeline.py
```

### README.md

Project documentation, methodology, dataset description, validation information, and future work.

### requirements.txt

Python packages required to run the pipeline.

### notebooks/

Contains the complete Google Colab/Jupyter workflow used for:

* Data loading
* Data preprocessing
* Feature construction
* Quality control
* SPI calculation
* Anomaly calculation
* Groundwater processing
* WSI construction
* Dataset validation

### src/

Contains the Python code extracted from the notebook.

---

## 🔬 Reproducibility

The project follows a chronological data-processing strategy.

The complete notebook preserves the data-processing workflow, while the Python source file contains the extracted pipeline code.

The final dataset construction uses:

* Fixed study period
* Standardized district names
* District-month temporal alignment
* Training-period-only transformation parameters
* Chronological train/validation/test splitting
* Groundwater as-of temporal alignment
* Explicit quality-control procedures
* Structural validation
* Reproducibility checks
* Model/audit consistency verification

The final model and corrected audit datasets were verified to contain identical calculated values for every district-month observation.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* Google Colab
* Jupyter Notebook
* Remote Sensing
* Time-Series Analysis
* Geospatial Data Processing
* Statistical Anomaly Analysis
* Drought Monitoring
* Machine Learning Preparation

---

## 📚 Project Applications

This project can support research and development in:

* Drought monitoring
* Water-resource management
* Agricultural planning
* Environmental monitoring
* Climate-risk assessment
* Hydrological analysis
* Remote sensing
* Drought early-warning systems
* Time-series forecasting

---

## 📌 Project Status

**Dataset Construction: Complete**

The 2003–2024 district-month Water Stress Index dataset has successfully passed:

* Structural validation
* Temporal validation
* Missing-value validation
* Duplicate-key validation
* Train/validation/test split validation
* WSI reproducibility validation
* Model/audit consistency validation

The dataset is ready for the next stage of drought early-warning and forecasting research.

---

## 🚀 Future Work

The completed WSI dataset provides a foundation for future drought early-warning and machine-learning applications.

### Time-Series Forecasting

* LSTM-based WSI forecasting
* GRU-based forecasting
* Transformer-based time-series forecasting
* Multi-step WSI prediction
* District-level temporal forecasting

### Drought Early Warning

* Forecasting future water-stress conditions
* Early identification of severe stress events
* District-level drought alerts
* Lead-time based warning systems
* Drought severity forecasting

### Machine Learning

Potential future models include:

* Random Forest
* XGBoost
* LightGBM
* Support Vector Machines
* Neural Networks
* Explainable AI approaches
* Feature importance analysis
* Model interpretability and attribution

### Spatial Analysis

Future extensions can include:

* District-level drought-risk maps
* Spatial drought propagation analysis
* Spatial-temporal drought modelling
* GIS-based visualization
* Interactive drought-risk maps

### Decision Support

Future deployment possibilities include:

* Real-time drought monitoring dashboards
* Automated district-level alerts
* Early-warning notification systems
* Integration with weather forecasts
* Integration with agricultural decision-support systems
* Water-resource management support

---




