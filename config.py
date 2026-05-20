import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DATA_ROOT = Path(os.environ.get("FLIGHT_DELAY_DATA"))

if DATA_ROOT is None:
    raise EnvironmentError(
        "FLIGHT_DELAY_DATA is not set.\n\n"
        "Setup:\n"
        "1. Copy .env.example to .env\n"
        "2. Edit .env and set FLIGHT_DELAY_DATA to your local Google Drive sync path\n"
        "3. Sync the shared Drive folder: https://drive.google.com/drive/folders/1VNTXNzXciRJRgqFlq38CLtXsxcS5vvrP\n\n"
        "For teammates: set the environment variable or edit .env with your own path."
    )

# ── Part 1: Original Airlines.csv dataset ────────────────────────────────────
DATA_PART1_RAW = DATA_ROOT / "part1" / "raw"
DATA_PART1_PROCESSED = DATA_ROOT / "part1" / "processed"

# Backward compatibility aliases
DATA_RAW = DATA_PART1_RAW
DATA_PROCESSED = DATA_PART1_PROCESSED

# ── Part 2: BTS 2023 dataset (flights + weather + aircraft) ──────────────────
DATA_PART2_RAW = DATA_ROOT / "part2" / "raw"
DATA_PART2_PROCESSED = DATA_ROOT / "part2" / "processed"

# ── Part 3: BTS multi-year dataset (Jan 2023 – Dec 2025, ~20.6M rows) ────────
DATA_PART3_RAW = DATA_ROOT / "part3" / "raw"
DATA_PART3_PROCESSED = DATA_ROOT / "part3" / "processed"


def assert_data_exists():
    """Raise an error with setup instructions if DATA_ROOT is not accessible."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"\nData root not found: {DATA_ROOT}\n\n"
            "Setup instructions:\n"
            "1. Sync the Google Drive folder to your local machine\n"
            "2. Copy .env.example to .env and set FLIGHT_DELAY_DATA if needed\n"
            "3. Restart the kernel / re-import this module\n\n"
            "Drive share link: https://drive.google.com/drive/folders/1VNTXNzXciRJRgqFlq38CLtXsxcS5vvrP"
        )


def load_csv(path):
    """
    Load a CSV with cleaned column names and stripped string values.

    The raw CSV has leading/trailing spaces in column headers and values.
    This function strips them automatically.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


def load_airlines(raw=True):
    """
    Load the Part 1 Airlines CSV with cleaned column names.

    DEPRECATED: use load_csv(path) for new code.
    """
    assert_data_exists()
    if raw:
        path = DATA_PART1_RAW / "Airlines.csv"
    else:
        path = DATA_PART1_PROCESSED / "Airlines_enriched.csv"
    return load_csv(path)


def load_flights_2023():
    """Load the Part 2 BTS 2023 main flights table."""
    assert_data_exists()
    return load_csv(DATA_PART2_RAW / "US_flights_2023.csv")


def load_weather_meteo():
    """Load the Part 2 daily weather by airport."""
    assert_data_exists()
    return load_csv(DATA_PART2_RAW / "weather_meteo_by_airport.csv")


def load_airports_geo():
    """Load the Part 2 airport geolocation metadata."""
    assert_data_exists()
    return load_csv(DATA_PART2_RAW / "airports_geolocation.csv")


def load_bts_flights(nrows=None):
    """Load Part 3 BTS multi-year flight data (Jan 2023–Dec 2025, ~20.6M rows).

    Prefers the monthly CSVs (part3/raw/monthly/*.csv) since they cover all 3 years.
    Falls back to US_flights_processed.csv if the monthly dir is absent (2023-2024 only).

    Columns of note:
        ARR_DEL15      — target: 1 if arrival delayed ≥15 min
        OP_CARRIER     — operating carrier code
        ORIGIN, DEST   — airport codes
        CRS_DEP_TIME   — scheduled departure (HHMM int)
        DEP_DELAY_NEW  — departure delay in minutes (capped at 0 for early)
        YEAR, MONTH, DAY_OF_WEEK
    """
    assert_data_exists()
    monthly_dir = DATA_PART3_RAW / "monthly"
    files = sorted(monthly_dir.glob("*.csv"))
    if files:
        dfs = [pd.read_csv(f, low_memory=False) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        return df if nrows is None else df.head(nrows)
    combined = DATA_PART3_RAW / "US_flights_processed.csv"
    if combined.exists():
        return pd.read_csv(combined, nrows=nrows, low_memory=False)
    raise FileNotFoundError(
        f"No BTS Part 3 data found in {DATA_PART3_RAW}.\n"
        "Run: rclone sync 'gdrive-hareee234:sem-8/CMPE188/flight-delay-proj-data' "
        "/path/to/cmpe188-data"
    )


if __name__ == "__main__":
    print(f"DATA_ROOT:         {DATA_ROOT}")
    print(f"DATA_PART1_RAW:    {DATA_PART1_RAW}")
    print(f"DATA_PART1_PROC:   {DATA_PART1_PROCESSED}")
    print(f"DATA_PART2_RAW:    {DATA_PART2_RAW}")
    print(f"DATA_PART2_PROC:   {DATA_PART2_PROCESSED}")
    print(f"DATA_PART3_RAW:    {DATA_PART3_RAW}")
    print(f"DATA_PART3_PROC:   {DATA_PART3_PROCESSED}")
    try:
        assert_data_exists()
        print("Data directory accessible.")
    except FileNotFoundError as e:
        print(e)
