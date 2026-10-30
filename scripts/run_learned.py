"""Run one learned model (frozen hyperparameters from tune.py) on every block and seed.

For each block:
  cal  : fit on pre-train (sequence models early-stop on validation) -> forecasts for validation origins
  test : refit on the full fit set (sequence models: fixed epochs = best_epoch scaled by sample size)
         -> forecasts for the block origins
Usage:
    python scripts/run_learned.py gru full
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.evaluate import records                                  # noqa: E402
from src.features import targets                                  # noqa: E402
from src.folds import make_blocks                                 # noqa: E402
from src.learned import SEQUENCE, fit_predict                     # noqa: E402

NAMES = {"svr": "SVR", "rf": "RF", "xgb": "XGB", "dnn": "DNN", "lstm": "LSTM", "gru": "GRU"}


def main(kind, variant, only_seeds=None):
    table = build_table()
    R = targets(table)
    params = json.loads((C.TUNING / f"{kind}_{variant}.json").read_text())["best_params"]
    seeds = [0] if kind == "svr" else C.SEEDS          # SVR is deterministic
    if only_seeds:
        seeds = [s for s in seeds if s in only_seeds]
    name = NAMES[kind]
    epochs_log = []
    for seed in seeds:
        if (C.FORECASTS / f"{name}_{variant}_seed{seed}.csv").exists():
            print(name, variant, "seed", seed, "already done, skipped", flush=True)
            continue
        parts = []
        for b in make_blocks(table):
            t0 = time.time()
            if kind in SEQUENCE:
                cal, ep = fit_predict(table, R, kind, params, variant, seed, b.pretrain, b.val, val_o=b.val)
                n_ep = max(1, round(ep * len(b.fit) / len(b.pretrain)))
                test, _ = fit_predict(table, R, kind, params, variant, seed, b.fit, b.origins, epochs=n_ep)
                epochs_log.append({"seed": seed, "block": b.name, "best_epoch": ep, "refit_epochs": n_ep})
            else:
                cal, _ = fit_predict(table, R, kind, params, variant, seed, b.pretrain, b.val)
                test, _ = fit_predict(table, R, kind, params, variant, seed, b.fit, b.origins)
            parts.append(records(table, R, b.val, cal, name, variant, seed, b.name, "cal"))
            parts.append(records(table, R, b.origins, test, name, variant, seed, b.name, "test"))
            print(name, variant, "seed", seed, b.name, f"{time.time() - t0:.1f}s", flush=True)
        C.FORECASTS.mkdir(parents=True, exist_ok=True)
        pd.concat(parts).to_csv(C.FORECASTS / f"{name}_{variant}_seed{seed}.csv", index=False)
    if epochs_log:                               # one file per call: parallel seeds never write the same file
        (C.RESULTS / "epochs").mkdir(parents=True, exist_ok=True)
        tag = "_".join(str(s) for s in sorted({r["seed"] for r in epochs_log}))
        pd.DataFrame(epochs_log).to_csv(C.RESULTS / "epochs" / f"{kind}_{variant}_seeds{tag}.csv", index=False)

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full", seeds)
