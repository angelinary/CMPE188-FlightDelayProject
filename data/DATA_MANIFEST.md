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
    ├── part1/                        # Part 1: Original Airlines.csv dataset
    │   ├── raw/
    │   │   └── Airlines.csv          # Source dataset
    │   └── processed/
    │       ├── airport_weather.csv   # Airport geo + climate cache
    │       └── Airlines_enriched.csv # Enriched dataset
    └── part2/                        # Part 2: BTS 2023 dataset
        ├── raw/
        │   ├── US_flights_2023.csv
        │   ├── weather_meteo_by_airport.csv
        │   ├── airports_geolocation.csv
        │   └── Cancelled_Diverted_2023.csv
        └── processed/
            └── flights_2023_merged.csv

Git repo
└── data/
    ├── DATA_MANIFEST.md             # This file
    ├── part1/
    │   ├── raw/.gitkeep
    │   └── processed/.gitkeep
    └── part2/
        ├── raw/.gitkeep
        └── processed/.gitkeep
```

---

## Part 1 Files

### `part1/raw/Airlines.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part1/raw/Airlines.csv` |
| **Size** | ~41 MB |
| **Rows** | 539,383 |
| **Source** | [Kaggle — Airlines Dataset to Predict a Delay](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay) |
| **Regeneration** | Download from Kaggle if missing |

The original source dataset. Contains 9 columns: `id`, `Airline`, `Flight`, `AirportFrom`, `AirportTo`, `DayOfWeek`, `Time`, `Length`, `Delay`.

> **Date provenance:** The dataset has no date column. Airline codes present in the data — notably `CO` (Continental Airlines, ceased **2012-03-03**) — strongly suggest the data was collected before March 2012, approximately **2008–2011**. See `notebooks/part1/00_dataset_investigation.ipynb` for the full analysis.

> **Note:** If this file is missing, download it from the Kaggle link above and save to `part1/raw/Airlines.csv` in your local Drive folder.

---

### `part1/processed/airport_weather.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part1/processed/airport_weather.csv` |
| **Size** | ~17 KB |
| **Rows** | 293 (one per unique airport) |
| **Source** | [OpenMeteo Climate API](https://open-meteo.com/) — free, no key required |

A cached lookup table mapping IATA airport codes to annual climate averages. Generated once by `notebooks/part1/02_feature_engineering.ipynb` (Section 2) and committed to Drive so teammates don't need to re-hit the API.

**Columns:** `iata_code`, `avg_temperature` (°C), `avg_precipitation` (mm/day), `avg_wind_speed` (km/h)

> **To regenerate:** Delete this file and re-run `notebooks/part1/02_feature_engineering.ipynb`. The notebook will re-fetch from OpenMeteo (~293 API calls, ~30–60s).

---

### `part1/processed/Airlines_enriched.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part1/processed/Airlines_enriched.csv` |
| **Size** | ~103 MB |
| **Rows** | 539,383 |
| **Source** | Generated locally by `notebooks/part1/02_feature_engineering.ipynb` |

The full dataset enriched with geographic and weather features for both origin and destination airports. Adds 12 columns (`from_lat`, `from_lon`, `from_elevation_ft`, `from_avg_temperature`, `from_avg_precipitation`, `from_avg_wind_speed`, and the same for `to_*`).

> **To generate:** Run all cells in `notebooks/part1/02_feature_engineering.ipynb`. Requires `Airlines.csv` and internet access (for OurAirports + OpenMeteo on first run). The weather cache (`airport_weather.csv`) is already in Drive so only the merge step runs.

---

## Part 2 Files

### `part2/raw/US_flights_2023.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part2/raw/US_flights_2023.csv` |
| **Size** | ~1.1 GB |
| **Rows** | ~6.7M |
| **Source** | [Kaggle — 2023 US Flights W/ Weather & Aircraft](https://www.kaggle.com/datasets/arvindnaga/2023-us-flights-with-weather-and-aircraft) |
| **Regeneration** | Download from Kaggle if missing |

The main BTS 2023 flights dataset. Contains ~60 columns including flight details, delays, cancellations, aircraft info, and carrier data.

---

### `part2/raw/weather_meteo_by_airport.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part2/raw/weather_meteo_by_airport.csv` |
| **Size** | ~3.3 MB |
| **Rows** | ~70K (one per airport per day) |
| **Source** | Same Kaggle dataset |

Daily weather observations by airport for 2023.

---

### `part2/raw/airports_geolocation.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part2/raw/airports_geolocation.csv` |
| **Size** | ~14 KB |
| **Rows** | ~380 |
| **Source** | Same Kaggle dataset |

Airport geolocation metadata (lat/lon, elevation, timezone, etc.).

---

### `part2/processed/flights_2023_merged.csv`

| Field | Value |
|---|---|
| **Location** | Drive: `part2/processed/flights_2023_merged.csv` |
| **Size** | ~3 GB |
| **Rows** | ~6.7M × 52 columns |
| **Source** | Generated by `notebooks/part2/00_data_load_and_merge.ipynb` |

The merged Part 2 dataset combining flights, weather, and airport geolocation. Target variable is `Dep_Delay_Tag` (binary classification).

> **To generate:** Run `notebooks/part2/00_data_load_and_merge.ipynb`. Requires all three Part 2 raw files.

---

## Config System

All data paths are managed through `config.py` and `.env`:

```python
# config.py reads from .env
import config
config.DATA_ROOT              # → FLIGHT_DELAY_DATA from .env
config.DATA_PART1_RAW         # → DATA_ROOT / "part1" / "raw"
config.DATA_PART1_PROCESSED   # → DATA_ROOT / "part1" / "processed"
config.DATA_PART2_RAW         # → DATA_ROOT / "part2" / "raw"
config.DATA_PART2_PROCESSED   # → DATA_ROOT / "part2" / "processed"
config.load_airlines(raw=True)    # Part 1 loader
config.load_flights_2023()        # Part 2 loader
```

Never hardcode paths in notebooks or scripts. Always use `config.py`.

---

## Maintenance Notes

- All data files are stored on Google Drive, not in git.
- Never commit CSVs — they are gitignored.
- `airport_weather.csv` only needs to be regenerated if the list of airports changes.
- `flights_2023_merged.csv` only needs to be regenerated if raw Part 2 data changes.
