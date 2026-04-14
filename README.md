# CMPE188 — Flight Delay Prediction

**Course:** CMPE 188 · Machine Learning with Big Data · San José State University

---

## Team

| Name | Email | SID |
|---|---|---|
| Harinandan Kotamsetti | harinandan.kotamsetti@sjsu.edu | 016222167 |
| Angelina Ryabechenkova | angelina.ryabechenkova@sjsu.edu | 018134165 |
| Aisha Syed | aisha.syed@sjsu.edu | 016573219 |

---

## Problem Statement

Flight delays are a frequent and costly issue in air travel, disrupting passenger schedules and reducing operational efficiency. Given the complexity of contributing factors — airline performance, route congestion, geography, and weather — predicting delays in advance is a challenging but valuable problem.

This project develops a machine learning system to predict the likelihood of a flight delay using a dataset of 500,000+ domestic U.S. flights. We compare Random Forest and XGBoost classifiers, enrich the base dataset with external weather and geographic data, and evaluate models rigorously with cross-validation and standard classification metrics.

---

## Dataset

**Source:** [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)

| Feature | Type | Description |
|---|---|---|
| `Airline` | categorical | Carrier code (e.g., AA, DL, UA) |
| `Flight` | int | Flight number (dropped — no predictive signal) |
| `AirportFrom` | categorical | Origin airport IATA code |
| `AirportTo` | categorical | Destination airport IATA code |
| `DayOfWeek` | int (1–7) | Day of the week |
| `Time` | int | Scheduled departure time in minutes from midnight |
| `Length` | int | Flight duration in minutes |
| `Delay` | binary (0/1) | Target — 1 = delayed |

---

## Modeling Pipeline

```
Raw Data
  └─► Preprocessing (OneHotEncoder + MinMaxScaler)
        └─► Feature Engineering (weather + derived features)
              └─► Feature Selection (SelectKBest chi2)
                    └─► Model Training (Random Forest / XGBoost)
                          └─► Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
                                └─► Evaluation (ROC-AUC, confusion matrix, PR curves)
```

---

## Feature Engineering

### Weather Enrichment (OpenMeteo — free, no API key)

The base dataset has no actual flight dates, so we use **climate normals** (monthly averages) from the [Open-Meteo Climate API](https://open-meteo.com/) matched to each airport via its geographic coordinates.

Features added per origin and destination airport:

- `lat`, `lon`, `elevation_ft` — geographic position
- `avg_temperature` — monthly climate normal (°C)
- `avg_precipitation` — monthly average precipitation (mm)
- `avg_wind_speed` — monthly average wind speed (km/h)

### Derived Features (from existing data)

| Feature | Description |
|---|---|
| `airline_delay_rate` | Historical delay rate per airline (target encoding, train-split only) |
| `route_volume` | Flight count per origin→destination pair (proxy for congestion) |
| `time_bucket` | Departure time bucketed: morning / afternoon / evening / night |
| `is_peak_hour` | Flag for high-congestion windows (7–9 am, 5–8 pm) |

### Future Dimensions (not yet implemented)

- Aircraft type and age (FAA registry)
- Airport runway capacity and scheduled departure density (BTS data)
- Real-time METAR weather (requires actual flight dates)
- Holiday / school break calendar flags
- ATC delay codes (ASPM database)

---

## Repository Structure

```text
CMPE188-FlightDelayProject/
├── data/
│   ├── raw/
│   │   └── Airlines.csv              # Source dataset (539k rows)
│   └── processed/                    # Enriched datasets (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Weather enrichment + derived features
│   ├── 03_model_baseline.ipynb       # RF vs XGBoost on raw features
│   ├── 04_model_tuning.ipynb         # GridSearchCV / RandomizedSearchCV
│   └── 05_evaluation.ipynb           # Confusion matrices, ROC, feature importance
├── scripts/
│   └── xgboost_pipeline.py           # Original baseline script (sklearn pipeline)
├── README.md
└── .gitignore
```

---

## Setup

```bash
# Python 3.12+ recommended
pip install pandas scikit-learn xgboost matplotlib seaborn requests

# Run baseline script
python scripts/xgboost_pipeline.py

# Launch notebooks
jupyter lab notebooks/
```

---

## References

- [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)
- [Open-Meteo Climate API](https://open-meteo.com/)
- [Priyanka Khivsara — Flight Delay Prediction (GitHub)](https://github.com/PriyankaKhivsara/flight-delay-prediction)
- [Samith Sachidanandan — Airline Flight Delay Prediction (Kaggle)](https://www.kaggle.com/code/samithsachidanandan/airline-flight-delay-prediction)
