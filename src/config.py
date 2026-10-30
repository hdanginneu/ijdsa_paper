"""All experiment settings in one place.

Every value here is fixed BEFORE looking at test results.
Any change must be recorded in DECISIONS.md with the date and the reason.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# ---- run variants (set by run_all.py through environment variables) -------
# IJDSA_TAG    : name of a robustness run; its outputs go to results/robust/<tag>/
# IJDSA_SMOKE  : 1 = tiny blocks, 1 seed, 2 trials, 2 epochs (logic check only); outputs to results_smoke/
SMOKE = os.environ.get("IJDSA_SMOKE") == "1"
TAG = os.environ.get("IJDSA_TAG", "")
MAIN_RESULTS = ROOT / ("results_smoke" if SMOKE else "results")
RESULTS = MAIN_RESULTS / "robust" / TAG if TAG else MAIN_RESULTS
FORECASTS = RESULTS / "forecasts"
TUNING = MAIN_RESULTS / "tuning"          # hyperparameters are tuned once (main run) and frozen for every variant
FIGURES = MAIN_RESULTS / "figures"

# ---- data -------------------------------------------------------------
COFFEE_FILE = Path(os.environ.get("IJDSA_COFFEE_FILE",
                                  RAW / "update_2026-09" / "london_robusta_daily_2008-01-14_2026-09-30.csv"))
ARABICA_FILE = RAW / "update_2026-09" / "us_coffee_c_daily_2008-01-14_2026-09-30.csv"
# Price unit of the coffee file: Robusta USD/t; Arabica US cents/lb (converted to USD/t in data.load_coffee)
PRICE_UNIT = os.environ.get("IJDSA_PRICE_UNIT", "usd_t")
CENTS_LB_TO_USD_T = 22.0462262
COFFEE_FILE_V1 = RAW / "coffee_price.csv"       # v1 export, kept for comparison only
# 21-24/09/2026 have O=H=L=C and zero volume (possible filler rows, see DECISIONS D-17).
DROP_FLAT_ROWS = os.environ.get("IJDSA_DROP_FLAT") == "1"         # primary: keep them (closes vary, volume -> missing); sensitivity: drop
DROP_FLAT_FROM = os.environ.get("IJDSA_DROP_FLAT_FROM", "2026-01-01")   # D-17 primary; D-55 Arabica: 2008-01-01
FRED_DIR = RAW / "fred_2026-10-16"
START_DATE = "2008-01-14"
END_DATE = None                                  # None = use all coffee dates available

# Macro series are lagged so that each value is published before the
# forecast origin (ICE London close). See DECISIONS.md, D-04.
MACRO_LAG = int(os.environ.get("IJDSA_MACRO_LAG", 6))   # trading days, primary (conservative)
MACRO_LAG_SENS = 1       # sensitivity run
# D-56: separate lag for DTWEXBGS only (robustness 'usdlag7'); None = same as MACRO_LAG
USD_LAG = int(os.environ["IJDSA_USD_LAG"]) if os.environ.get("IJDSA_USD_LAG") else None

# ---- forecasting task ---------------------------------------------------
HORIZONS = list(range(1, 8))     # h = 1..7 trading days, direct strategy
GAP = 7                          # trading days between last training origin and next segment
# Training window: None = expanding (primary); an integer = rolling window of that many origins (robustness, D-31)
TRAIN_WINDOW = int(os.environ["IJDSA_TRAIN_WINDOW"]) if os.environ.get("IJDSA_TRAIN_WINDOW") else None

# Evaluation blocks are defined by FORECAST-ORIGIN dates.
# B5 origins are chosen so that the h=1 targets are exactly the v1 test window
# (2024-11-20 .. 2026-01-28, 300 days).
B5_FIRST_TARGET = "2024-11-20"
B5_N = 300
EARLY_BLOCKS = 4                 # B1..B4, placed immediately before B5
EARLY_BLOCK_N = 250
VAL_N = 250                      # validation origins immediately before each block (after a gap)
# B6 (new data after 2026-01-28) is added automatically when the data extend past B5.
B6_FIRST_TARGET = "2026-01-29"

# ---- models ----------------------------------------------------------------
SEEDS = [0, 7, 21, 42, 99]
N_TRIALS = 20                    # identical Optuna budget for every learned model
TUNE_ON_BLOCK = "B1"             # hyperparameters tuned once on B1's validation, then frozen
DL_MAX_EPOCHS = 100
DL_PATIENCE = 10

# ---- evaluation -------------------------------------------------------------
INTERVAL_LEVELS = [0.80, 0.90, 0.95]
MCS_ALPHA = 0.10

# ---- smoke test: same code paths, tiny sizes --------------------------------
if SMOKE:
    EARLY_BLOCK_N = 20
    B5_N = 20
    VAL_N = 60
    B6_MAX = 20
    SEEDS = [0]
    N_TRIALS = 2
    DL_MAX_EPOCHS = 2
    DL_PATIENCE = 1
else:
    B6_MAX = None
