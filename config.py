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

DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"


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


def load_airlines(raw=True):
    """
    Load the Airlines CSV with cleaned column names.

    The raw CSV has leading/trailing spaces in column headers.
    This function strips them automatically.
    """
    assert_data_exists()
    if raw:
        path = DATA_RAW / "Airlines.csv"
    else:
        path = DATA_PROCESSED / "Airlines_enriched.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


if __name__ == "__main__":
    print(f"DATA_ROOT:      {DATA_ROOT}")
    print(f"DATA_RAW:       {DATA_RAW}")
    print(f"DATA_PROCESSED: {DATA_PROCESSED}")
    try:
        assert_data_exists()
        print("Data directory accessible.")
    except FileNotFoundError as e:
        print(e)
