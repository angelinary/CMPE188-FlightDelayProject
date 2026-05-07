# Data Manifest

This document describes every data file in this project — what it is, where it comes from, and how to obtain or regenerate it.

**TL;DR:** All data files live on Google Drive, not in this git repo. Set up once with `.env`, then the project works on any machine.

---

## Quick Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd CMPE188-FlightDelayProject

# 2. Sync the shared Drive folder to your local machine
#    https://drive.google.com/drive/folders/1VNTXNzXciRJRgqFlq38CLtXsxcS5vvrP

# 3. Copy .env.example to .env and set your local Drive path
cp .env.example .env
# Edit .env: set FLIGHT_DELAY_DATA to your local Google Drive sync path

# 4. Install dependencies
pip install python-dotenv pandas scikit-learn xgboost

# 5. Verify
python -c "import config; print(config.DATA_ROOT)"
python scripts/xgboost_pipeline.py
```

---

## Directory Structure

```
Google Drive (shared folder)
└── flight-delay-proj-data/          ← FLIGHT_DELAY_DATA points here
    ├── raw/
    │   └── Airlines.csv              # Source dataset
    └── processed/
        ├── airport_weather.csv       # Airport geo + climate cache
        └── Airlines_enriched.csv     # Enriched dataset

Git repo
└── data/
    ├── .gitkeep                      # Keeps directory in git
    └── DATA_MANIFEST.md             # This file
```

---

## Files

### `raw/Airlines.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `raw/Airlines.csv` |
| **Size** | ~41 MB |
| **Rows** | 539,383 |
| **Source** | [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay) |
| **Regeneration** | Download from Kaggle if missing |

The original source dataset. Contains 9 columns: `id`, `Airline`, `Flight`, `AirportFrom`, `AirportTo`, `DayOfWeek`, `Time`, `Length`, `Delay`.

> **Date provenance:** The dataset has no date column. Airline codes present in the data — notably `CO` (Continental Airlines, ceased **2012-03-03**) — strongly suggest the data was collected before March 2012, approximately **2008–2011**. See `notebooks/00_dataset_investigation.ipynb` for the full analysis.

> **Note:** If this file is missing, download it from the Kaggle link above and save to `raw/Airlines.csv` in your local Drive folder.

---

### `processed/airport_weather.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `processed/airport_weather.csv` |
| **Size** | ~17 KB |
| **Rows** | 293 (one per unique airport) |
| **Source** | [OpenMeteo Climate API](https://open-meteo.com/) — free, no key required |

A cached lookup table mapping IATA airport codes to annual climate averages. Generated once by `notebooks/02_feature_engineering.ipynb` (Section 2) and committed to Drive so teammates don't need to re-hit the API.

**Columns:** `iata_code`, `avg_temperature` (°C), `avg_precipitation` (mm/day), `avg_wind_speed` (km/h)

> **To regenerate:** Delete this file and re-run `notebooks/02_feature_engineering.ipynb`. The notebook will re-fetch from OpenMeteo (~293 API calls, ~30–60s).

---

### `processed/Airlines_enriched.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `processed/Airlines_enriched.csv` |
| **Size** | ~103 MB |
| **Rows** | 539,383 |
| **Source** | Generated locally by `notebooks/02_feature_engineering.ipynb` |

The full dataset enriched with geographic and weather features for both origin and destination airports. Adds 12 columns (`from_lat`, `from_lon`, `from_elevation_ft`, `from_avg_temperature`, `from_avg_precipitation`, `from_avg_wind_speed`, and the same for `to_*`).

> **To generate:** Run all cells in `notebooks/02_feature_engineering.ipynb`. Requires `Airlines.csv` and internet access (for OurAirports + OpenMeteo on first run). The weather cache (`airport_weather.csv`) is already in Drive so only the merge step runs.

---

### `raw/bts_YYYY_MM.csv` (optional, not in Drive)

| Field | Value |
|---|---|
| **Location** | Local only — not in Drive or git |
| **Source** | [BTS On-Time Performance](https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD) |

Monthly snapshots of U.S. domestic on-time performance data downloaded manually from the BTS Transtats portal. Used exclusively in `notebooks/00_dataset_investigation.ipynb` (Section 5) to cross-reference date range.

**Fields required:** `YEAR`, `MONTH`, `DAY_OF_WEEK`, `OP_UNIQUE_CARRIER`, `FLIGHT_NUMBER_REPORTING_AIRLINE`, `ORIGIN`, `DEST`, `DEP_DELAY`

> These files are not needed for model training — only for the date range investigation.

---

## Config System

All data paths are managed through `config.py` and `.env`:

```python
# config.py reads from .env
import config
config.DATA_ROOT    # → FLIGHT_DELAY_DATA from .env
config.DATA_RAW     # → DATA_ROOT / "raw"
config.DATA_PROCESSED  # → DATA_ROOT / "processed"
config.load_airlines(raw=True)  # Returns DataFrame with clean column names
```

Never hardcode paths in notebooks or scripts. Always use `config.py`.

---

## Maintenance Notes

- All data files are stored on Google Drive, not in git.
- Never commit `Airlines_enriched.csv` — it is gitignored.
- `airport_weather.csv` only needs to be regenerated if the list of airports changes.
- `bts_YYYY_MM.csv` files are optional — do not add them to git or Drive.
- `notebooks/00_dataset_investigation.ipynb` is not required for model training.
