# Agent Setup Guide — Remote Server (`personal-linux`)

This guide is for the agent running on the remote Linux server (alias `personal-linux`) to set up the CMPE188 Flight Delay project after receiving the project bundle.

---

## 1. Receive the Bundle

The project bundle will be transferred via SCP from the MacBook to the server.

**On the MacBook (sender):**
```bash
# Option A: Using tailscale IP (check with `tailscale status`)
zip -r flight-delay-project.zip CMPE188-FlightDelayProject/
scp flight-delay-project.zip <user>@<tailscale-ip>:~/

# Option B: Using tailscale alias
scp flight-delay-project.zip personal-linux:~/
```

**On the server (`personal-linux`):**
```bash
cd ~
unzip -q flight-delay-project.zip
cd CMPE188-FlightDelayProject
```

---

## 2. Set Up Data Directory

The zip includes all data files (both Part 1 and Part 2). The data is located inside the project at:
```
CMPE188-FlightDelayProject/flight-delay-proj-data/
```

Create a `.env` file pointing to this internal data directory:
```bash
cp .env.example .env
# Edit .env to set:
FLIGHT_DELAY_DATA=/home/<user>/CMPE188-FlightDelayProject/flight-delay-proj-data
```

Verify the structure matches `config.py` expectations:
```bash
python -c "import config; print(config.DATA_ROOT); config.assert_data_exists()"
```

Expected output:
```
/home/<user>/CMPE188-FlightDelayProject/flight-delay-proj-data
Data directory accessible.
```

---

## 3. Install Dependencies

**Option A: Use the existing conda environment (if available):**
```bash
conda run -n torch-default pip install -r requirements.txt
```

**Option B: Create a new environment:**
```bash
conda create -n flight-delay python=3.12
conda activate flight-delay
pip install -r requirements.txt
```

Core dependencies (already in `requirements.txt`):
- pandas
- numpy
- scikit-learn
- xgboost
- python-dotenv
- jupyter
- matplotlib
- seaborn

**GPU Support:** XGBoost will automatically use CUDA if available. Verify with:
```python
import xgboost as xgb
print(xgb.build_info())  # Check for CUDA support
```

---

## 4. Kernel Setup for Jupyter

Register the conda environment as a Jupyter kernel:
```bash
# If using torch-default
conda run -n torch-default python -m ipykernel install --user --name torch-default --display-name "torch-default"

# If using flight-delay environment
python -m ipykernel install --user --name flight-delay --display-name "flight-delay"
```

---

## 5. Verify Notebooks

Run a quick smoke test on Part 1:
```bash
jupyter nbconvert --to notebook --execute notebooks/part1/00_dataset_investigation.ipynb --output /tmp/test_part1.ipynb
```

Run a quick smoke test on Part 2:
```bash
jupyter nbconvert --to notebook --execute notebooks/part2/00_data_load_and_merge.ipynb --output /tmp/test_part2.ipynb
```

Or open Jupyter Lab and run interactively:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

---

## 6. Directory Structure Reference

After setup, the project should look like:
```
CMPE188-FlightDelayProject/
├── .env                           # Local env with FLIGHT_DELAY_DATA
├── config.py                      # Path configuration
├── data/                          # Git-tracked structure only
│   ├── DATA_MANIFEST.md
│   └── part1/
│   │   ├── raw/.gitkeep
│   │   └── processed/.gitkeep
│   └── part2/
│       ├── raw/.gitkeep
│       └── processed/.gitkeep
├── flight-delay-proj-data/        # Actual data (not in git)
│   ├── part1/
│   │   ├── raw/Airlines.csv
│   │   └── processed/
│   │       ├── Airlines_enriched.csv
│   │       └── airport_weather.csv
│   └── part2/
│       ├── raw/
│       │   ├── US_flights_2023.csv
│       │   ├── weather_meteo_by_airport.csv
│       │   └── airports_geolocation.csv
│       └── processed/
│           └── flights_2023_merged.csv
├── notebooks/
│   ├── part1/                     # 6 notebooks (original pipeline)
│   └── part2/                     # 6 notebooks (BTS 2023 pipeline)
└── scripts/
```

---

## 7. Troubleshooting

### `Data root not found` error
- Check `.env` exists and `FLIGHT_DELAY_DATA` points to the absolute path of `flight-delay-proj-data/`
- Verify the directory exists: `ls -la $FLIGHT_DELAY_DATA`

### `ModuleNotFoundError: No module named 'config'`
- Ensure you run notebooks from the repo root (where `config.py` lives)
- Or add the repo root to `PYTHONPATH`: `export PYTHONPATH=/home/<user>/CMPE188-FlightDelayProject:$PYTHONPATH`

### GPU not detected by XGBoost
- Check NVIDIA drivers: `nvidia-smi`
- Check CUDA in XGBoost: `python -c "import xgboost as xgb; print(xgb.build_info())"`
- If CUDA is missing, XGBoost will fall back to CPU (slower but functional)

### Part 2 merge notebook runs out of memory
- The merge loads ~1.1 GB raw + weather + geo into memory
- Resulting merged file is ~3 GB
- Ensure server has at least 16 GB RAM
- If OOM, reduce chunk size in `00_data_load_and_merge.ipynb`
