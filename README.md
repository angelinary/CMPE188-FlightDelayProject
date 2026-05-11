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

This project develops ML models to predict the likelihood of a departure delay using two datasets: an original Kaggle benchmark (Part 1) and the full BTS 2023 domestic flight record (Part 2). We compare Random Forest, XGBoost, and a PyTorch feedforward neural network with entity embeddings across both datasets.

---

## Datasets

### Part 1 — Kaggle Airlines Dataset (benchmark)

**Source:** [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)

- 539,383 flights · 9 columns · binary delay target
- No date column — likely 2008–2011 based on carrier codes (Continental Airlines `CO` ceased 2012)
- Weather enriched with OpenMeteo climate normals (annual averages, not day-specific)
- Delay rate: ~55%

### Part 2 — BTS 2023 Domestic Flights (primary)

**Source:** Bureau of Transportation Statistics On-Time Performance 2023

- 6,743,404 flights · 52 columns after merge · binary delay target (`Dep_Delay_Tag`)
- Actual flight dates (Jan–Dec 2023), daily weather observations, aircraft metadata
- Weather merged from OpenMeteo daily records by departure airport and date
- Airport geolocation merged for lat/lon, state, elevation
- Delay rate: 38%

---

## Results

### Part 2 — BTS 2023 (primary results, 100K test set)

| Model | Accuracy | ROC-AUC | F1 (Delayed) | Notes |
|---|---|---|---|---|
| XGBoost baseline | 0.8046 | 0.8334 | 0.678 | Jan-only training sample |
| RF baseline | 0.8027 | 0.8309 | 0.661 | Jan-only training sample |
| XGBoost tuned | 0.8117 | 0.8352 | 0.681 | GridSearchCV, GPU |
| RF tuned | 0.8110 | 0.8358 | 0.678 | RandomizedSearchCV |
| MLP (sklearn) | 0.8055 | 0.8313 | 0.678 | CPU only, 50K rows |
| **PyTorch FFNN** | **0.7980** | **0.8513** | **0.719** | GPU, 400K rows, entity embeddings |

### Part 1 — Kaggle (benchmark, confirmed baselines)

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| XGBoost baseline | 0.6438 | 0.6895 | enriched features |
| RF baseline | 0.6384 | 0.6848 | enriched features |
| XGBoost tuned | 0.6201 | 0.6452 | ⚠ worse than baseline — data leakage in feature engineering |
| RF tuned | 0.6194 | 0.6478 | ⚠ worse than baseline — data leakage in feature engineering |

> **Data leakage note (Part 1):** Weather enrichment in `02_feature_engineering.ipynb` runs on the full dataset before the train/test split. CV folds during tuning have seen test-row features through the enrichment step, producing inflated CV AUC (~0.856) that does not generalize (test AUC ~0.645). Fix: move enrichment inside the CV pipeline or after splitting.

---

## Modeling Pipeline

```
Raw Data (Part 1 or Part 2)
  └─► 00 — Data Load & Merge (Part 2 only: flights + weather + airports)
        └─► 01 — EDA
              └─► 02 — Feature Engineering
                    (target encoding, temporal features, aircraft age buckets)
                          └─► 03 — Baseline Models (XGBoost, RF)
                                └─► 04 — Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
                                      └─► 05 — Evaluation (ROC, confusion matrix, feature importance)
                                            └─► 06 — PyTorch FFNN (entity embeddings, GPU)
```

---

## PyTorch FFNN Architecture (notebook 06)

Replaces `sklearn.MLPClassifier` with a proper GPU-trained feedforward network.

**Key design:**
- **Entity embeddings** for each categorical feature (airports, airlines, aircraft models, states) — learns dense delay-pattern representations instead of sparse OHE columns
- **No SelectKBest** — BatchNorm handles scale; embeddings handle cardinality
- **Full 400K training rows** (sklearn MLP used 50K)
- **BCEWithLogitsLoss with pos_weight=1.616** — corrects for 38/62 class imbalance
- **OneCycleLR** with cosine annealing

```
Categorical cols → Embedding layers (per col) ─┐
                                                concat → Linear(227→512) → BN → ReLU → Dropout(0.30)
Numeric cols (41) → StandardScaler ────────────┘       → Linear(512→256) → BN → ReLU → Dropout(0.25)
                                                        → Linear(256→128) → BN → ReLU → Dropout(0.15)
                                                        → Linear(128→1)  ← logit
```

Embedding sizes (FastAI rule: `min(50, (vocab+1)//2)`):

| Feature | Vocab | Embedding dim |
|---|---|---|
| Dep_Airport / Arr_Airport | 340 | 50 |
| dep_STATE / arr_STATE | 55 | 28 |
| Airline | 16 | 8 |
| DepTime_label / Manufacturer | 5 | 3 |
| season_label | 2 | 1 |

Training: 40 epochs · batch 2048 · AdamW · 0.7 min on RTX 5070 Ti

---

## Repository Structure

```text
CMPE188-FlightDelayProject/
├── data/                              # Placeholder structure only — data on Google Drive
│   ├── DATA_MANIFEST.md               # Data setup instructions + file inventory
│   ├── part1/{raw,processed}/.gitkeep
│   └── part2/{raw,processed}/.gitkeep
├── notebooks/
│   ├── part1/                         # Kaggle Airlines pipeline
│   │   ├── 00_dataset_investigation.ipynb
│   │   ├── 01_eda.ipynb
│   │   ├── 02_feature_engineering.ipynb   # Weather enrichment (OpenMeteo climate normals)
│   │   ├── 03_model_baseline.ipynb
│   │   ├── 04_model_tuning.ipynb          # GridSearchCV (XGBoost) + RandomizedSearchCV (RF)
│   │   └── 05_evaluation.ipynb            # TODO: stub — not yet implemented
│   └── part2/                         # BTS 2023 pipeline
│       ├── 00_data_load_and_merge.ipynb   # Merge flights + daily weather + airport geo
│       ├── 01_eda.ipynb                   # EDA on 200K sample
│       ├── 02_feature_engineering.ipynb   # Target encoding, temporal, aircraft age
│       ├── 03_model_baseline.ipynb        # XGBoost + RF baselines (300K sample)
│       ├── 04_model_tuning.ipynb          # Tuning with GPU XGBoost (50K tuning sample)
│       ├── 05_evaluation.ipynb            # Confusion matrix, ROC, feature importance
│       └── 06_neural_network.ipynb        # PyTorch FFNN with entity embeddings (GPU)
├── scripts/
│   └── xgboost_pipeline.py            # Standalone XGBoost training script
├── config.py                          # Centralized data path config (reads from .env)
├── .env.example                       # Template — copy to .env and set your Drive path
├── .env                               # Local env (gitignored)
├── AGENT_SETUP.md                     # Remote server / HPC setup guide
├── README.md
└── .gitignore
```

**Data location:** All CSVs and processed files live on Google Drive (3.7 GB).
See `data/DATA_MANIFEST.md` for the full file inventory.

---

## Setup

### Local Machine

```bash
# 1. Clone
git clone https://github.com/angelinary/CMPE188-FlightDelayProject.git
cd CMPE188-FlightDelayProject

# 2. Sync data from Google Drive using rclone
#    Configure a remote named pointing to your Drive, then:
rclone sync "gdrive-remote:sem-8/CMPE188/flight-delay-proj-data" /path/to/local/cmpe188-data -P

# 3. Configure data path
cp .env.example .env
# Edit .env: FLIGHT_DELAY_DATA=/path/to/local/cmpe188-data

# 4. Create conda environment (torch5070 or equivalent with PyTorch + CUDA)
conda create -n torch5070 python=3.12
conda activate torch5070
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install pandas scikit-learn xgboost matplotlib seaborn jupyter python-dotenv requests

# 5. Verify
python config.py

# 6. Run notebooks in order within each part
jupyter lab notebooks/
```

### Remote / HPC

See `AGENT_SETUP.md` for server-specific instructions (conda env, kernel registration, data transfer via rclone/scp).

---

## Known Issues

### Part 1 — Data Leakage in Tuning

Weather enrichment (`02_feature_engineering.ipynb`) runs on the full dataset before splitting. The CV folds during GridSearchCV see test-row enriched features, producing CV AUC ~0.856 vs test AUC ~0.645 — worse than the untuned baseline. Fix: perform enrichment inside the pipeline or after splitting.

### Part 2 — January-Only Training Sample

`02_feature_engineering.ipynb` loads `nrows=500_000` from the top of a date-sorted CSV, capturing only January 2023. Models trained on this sample are winter-biased and may underperform on summer flights. Fix: use random sampling (`skiprows`) or load the full dataset before splitting.

### Part 1 — 05_evaluation.ipynb Not Implemented

All cells are `# TODO` stubs. Evaluation plots (confusion matrix, ROC, feature importance) for Part 1 are pending.

### Dataset Date Range (Part 1)

The Kaggle dataset has no date column. Based on carrier codes (`CO` = Continental, ceased 2012-03-03), collection is likely 2008–2011. See `notebooks/part1/00_dataset_investigation.ipynb` for the full analysis.

---

## References

- [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay)
- [BTS On-Time Performance Database](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD)
- [Open-Meteo Climate API](https://open-meteo.com/)
- Guo, C. & Berkhahn, F. (2016). Entity Embeddings of Categorical Variables. *arXiv:1604.06737*
