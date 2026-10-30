# Forecast accuracy, uncertainty and decision value in ICE Robusta coffee futures

Code, data and results for: *Forecast Accuracy, Uncertainty and Decision Value in ICE Robusta Coffee Futures: Evidence from Rolling-Origin Evaluation* (Phung Duy Quang, Cao Ha Hai Dang), submitted to the International Journal of Data Science and Analytics.

## Contents
- `src/`: data construction, features, rolling-origin blocks, models, intervals, decision and cost layer, statistical tests.
- `scripts/`: estimation and analysis scripts (one per stage, see `run_all.py`).
- `tests/`: leakage and variance-ratio checks.
- `data/raw/`: ICE London Robusta and ICE US Coffee "C" daily series from Investing.com (to 30 Sep 2026) and the FRED series DTWEXBGS, EFFR and DCOILWTICO (retrieved 16 Oct 2026). See the README in each sub-folder.
- `results/`: results reported in the paper: `tables/`, `paper/` (Tables 1-6 and Figures 1-4 as CSV/PNG), `figures/`, `forecasts/` (every forecast, one row per origin and horizon), `tuning/` (selected hyperparameters), `decision_aware/`, `blocks.csv`, `arima_orders.csv`.

Comments in the code that refer to identifiers such as "D-17" point to the authors' internal design log, which is not part of this repository. The design is described in Section 3 of the paper.

## Evaluation design (short)
Six rolling-origin blocks (about 1,470 origins), horizons h = 1-7, direct forecasts of the cumulative log return. Models: random walk, drift, ARIMA, SVR, random forest, XGBoost, DNN, LSTM, GRU, zero-shot Chronos-Bolt (small), combination and shrinkage forecasts. Hyperparameters (Optuna, 20 trials) are tuned once on the first block; stochastic models use seeds 0, 7, 21, 42, 99.

## Reproduction
Python 3.13.
```bash
bash setup.sh                         # creates .venv and installs pinned packages
.venv/bin/python run_all.py --smoke   # tiny run to check the pipeline (numbers not used)
.venv/bin/python run_all.py --fresh --jobs 4   # full run (long; results in results/)
```
Stage by stage:
```bash
python tests/test_leakage.py
python scripts/run_benchmarks.py
python scripts/run_breaks.py
python scripts/tune.py <svr|rf|xgb|dnn|lstm|gru> <full|price>
python scripts/run_learned.py <model> <full|price> [seeds]
python scripts/run_chronos.py
python scripts/run_decision_aware.py <xgb|gru> [seeds]
python scripts/analyze.py
python scripts/analyze_decisions.py
python scripts/analyze_states.py
python scripts/analyze_v2.py
python scripts/make_figures.py
```
Tables can be rebuilt from the saved forecasts in `results/forecasts/` without retraining. A full run was done on one macOS machine; results can differ slightly across machines and library versions for the neural models. The robustness runs listed in `run_all.py` write to `results/robust/`, which is not included because of its size.

## Data terms
Investing.com data are redistributed here for replication; their use remains subject to the provider's terms. FRED series are publicly available from the Federal Reserve Bank of St. Louis.
