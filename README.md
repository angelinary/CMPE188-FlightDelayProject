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
├── data/                              # Data lives on Google Drive, not here
│   ├── DATA_MANIFEST.md               # Data setup instructions + file docs
│   └── .gitkeep                       # Keeps directory in git
├── notebooks/
│   ├── 00_dataset_investigation.ipynb # Date range investigation + BTS cross-reference
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb   # Weather enrichment + derived features
│   ├── 03_model_baseline.ipynb        # RF vs XGBoost on raw features
│   ├── 04_model_tuning.ipynb          # GridSearchCV / RandomizedSearchCV
│   └── 05_evaluation.ipynb            # Confusion matrices, ROC, feature importance
├── scripts/
│   └── xgboost_pipeline.py            # Baseline script with GridSearchCV
├── config.py                          # Centralized data path config (reads from .env)
├── .env.example                       # Template — copy to .env and set your Drive path
├── .env                               # Your local env (gitignored)
├── README.md
└── .gitignore
```

**Data location:** All data files (raw, processed) are stored in Google Drive.
See `data/DATA_MANIFEST.md` for setup instructions and file documentation.

---

## Setup

**All data lives on Google Drive — not in git.** Set up once, then the project works on any machine.

```bash
# 1. Clone the repo
git clone https://github.com/angelinary/CMPE188-FlightDelayProject.git
cd CMPE188-FlightDelayProject

# 2. Sync the shared Drive folder to your local machine
#    https://drive.google.com/drive/folders/1VNTXNzXciRJRgqFlq38CLtXsxcS5vvrP

# 3. Configure your local data path
cp .env.example .env
# Edit .env: set FLIGHT_DELAY_DATA to your local Google Drive sync path
#    Example (macOS): /Users/you/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/sem-8/CMPE188/flight-delay-proj-data

# 4. Install dependencies
pip install pandas scikit-learn xgboost matplotlib seaborn requests python-dotenv

# 5. Verify setup
python -c "import config; print(config.DATA_ROOT)"
python scripts/xgboost_pipeline.py

# 6. Launch notebooks
jupyter lab notebooks/
```

> **Teammates:** Ask the repo owner to share the Drive folder with you, then sync it to your local machine. Copy `.env.example` to `.env` and set your own local path.

---

## Known Limitations & Open Questions

### Dataset Date Range — Unknown, Likely Pre-2012

The dataset contains no date, month, or year column — only `DayOfWeek` (1–7). The Kaggle
dataset page does not specify the collection period, and the dataset creator has stated
"no such information has been provided."

However, the airline codes present in the data provide a strong constraint:

| Carrier | Airline | Status |
|---|---|---|
| `CO` | Continental Airlines | Ceased **2012-03-03** (merged with United Airlines) |
| `FL` | AirTran Airways | Ceased **2014-12-28** (merged with Southwest Airlines) |
| `XE` | ExpressJet | Operated until ~2018 |
| `YV` | Mesa Airlines | Still active |

The presence of `CO` is the binding constraint: **the dataset almost certainly predates March 2012**. Most likely collection window: **2008–2011**. A Kaggle commenter proposed June 2022 as the start date, but this is almost certainly incorrect.

> See `notebooks/00_dataset_investigation.ipynb` for the full analysis and BTS cross-reference attempt.

### Weather Features Are Climate Normals, Not Actual Conditions

The weather features (`avg_temperature`, `avg_precipitation`, `avg_wind_speed`) are annual climate averages from the OpenMeteo API (2019 baseline), matched to each airport by geographic coordinates. They are not day-specific — a flight during a blizzard and a flight on a clear day at the same airport receive identical weather features. This limits the signal to broad geographic and seasonal trends rather than actual meteorological conditions.

### Binary Delay Target Only

The `Delay` column is binary (0/1). There is no information about delay duration, cause (carrier, weather, NAS/ATC, security, late aircraft), or severity. Models trained on this target can predict *whether* a delay occurs but not *why* or *by how much*.

### No Tail Number — Propagation Delay Is Untrackable

The dataset has no tail number column. In practice, a significant fraction of delays propagate from earlier legs of the same aircraft's rotation ("late aircraft" delays). Without tail numbers, this causal chain cannot be reconstructed. A partial proxy — `flight_sequence_delay_rate` per `(Airline, Flight, DayOfWeek)` — is used in notebook 03 as a noisy approximation.

### Season / Holiday Flags — Blocked

Without a confirmed date range, specific holidays and week-of-year flags cannot be reliably added. Once the date range is confirmed (via BTS matching in notebook 00), actual dates can be reconstructed and features like `is_holiday`, `month`, `week_of_year`, and `is_thanksgiving_week` can be added.

### Dataset Provenance

Sourced from [Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay). The collection methodology and time period are unconfirmed by the creator. The [BTS On-Time Performance database](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD) is the most tractable path to validating the date range.

---

## References

- [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)
- [Open-Meteo Climate API](https://open-meteo.com/)
- [Priyanka Khivsara — Flight Delay Prediction (GitHub)](https://github.com/PriyankaKhivsara/flight-delay-prediction)
- [Samith Sachidanandan — Airline Flight Delay Prediction (Kaggle)](https://www.kaggle.com/code/samithsachidanandan/airline-flight-delay-prediction)
