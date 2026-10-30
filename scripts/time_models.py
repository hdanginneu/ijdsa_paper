"""Compute cost re-measured (DECISIONS D-54). Main run only; runs after the main estimation, alone.

Block B5, seed 0, frozen hyperparameters. For each model: (a) fit time on B5's fit set (DL: the fixed
refit epochs of seed 0 / B5; RW and Chronos have no fit), (b) prediction time for B5's origins.
Median of 3 repeats. Every model runs in its own process (XGBoost and torch never share a process),
2 threads, CPU only (Chronos: device_map="cpu"). Feature construction and Chronos weight loading are
not timed. Output: tables/v3_compute_cost.csv and tables/v3_compute_env.json.

    python scripts/time_models.py            # all models, one subprocess each
    python scripts/time_models.py --one XGB  # internal: time one model, print one JSON line
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402

MODELS = ["RW", "Drift", "ARIMA", "SVR", "RF", "XGB", "DNN", "LSTM", "GRU", "Chronos"]
REPEATS, THREADS = 3, 2
OUT = C.RESULTS / "tables"


def timed(fn, repeats=REPEATS):
    secs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        secs.append(time.perf_counter() - t0)
    return float(np.median(secs))


def refit_epochs(kind):
    f = C.RESULTS / "epochs" / f"{kind}_full_seeds0.csv"
    e = pd.read_csv(f)
    return int(e[(e["seed"] == 0) & (e["block"] == "B5")]["refit_epochs"].iloc[0])


def one(model):
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    from src import models as M
    from src.data import build_table
    from src.features import targets
    from src.folds import make_blocks
    table = build_table()
    R = targets(table)
    b = next(x for x in make_blocks(table) if x.name == "B5")
    lp, ret = table["logp"].to_numpy(), table["ret"].to_numpy()
    n = len(b.origins)
    fit_s, pred_s = 0.0, 0.0
    if model == "RW":
        pred_s = timed(lambda: M.random_walk(n))
    elif model == "Drift":
        fit_s = timed(lambda: np.nanmean(ret[1: b.fit[-1] + C.GAP + 1]))
        pred_s = timed(lambda: M.drift(ret[1: b.fit[-1] + C.GAP + 1], n))
    elif model == "ARIMA":
        train = lp[: b.fit[-1] + C.GAP + 1]
        fit_s = timed(lambda: M.arima_select(train))                 # includes the AIC order search
        _, res = M.arima_select(train)
        pred_s = timed(lambda: M.arima_forecasts(res, lp, b.origins))
    elif model in ("SVR", "RF", "XGB"):
        from src.learned import inputs
        kind = model.lower()
        params = json.loads((C.TUNING / f"{kind}_full.json").read_text())["best_params"]
        X = inputs(table, kind, params, "full")
        Xf, Xo = X(b.fit), X(b.origins)
        fit_s = timed(lambda: M.TabularModel(kind, params, 0).fit(Xf, R[b.fit]))
        m = M.TabularModel(kind, params, 0).fit(Xf, R[b.fit])
        pred_s = timed(lambda: m.predict(Xo))
    elif model in ("DNN", "LSTM", "GRU"):
        import torch
        torch.set_num_threads(THREADS)
        from src.learned import inputs
        from src.nn_models import SeqModel
        kind = model.lower()
        params = json.loads((C.TUNING / f"{kind}_full.json").read_text())["best_params"]
        X = inputs(table, kind, params, "full")
        Xf, Xo = X(b.fit), X(b.origins)
        ep = refit_epochs(kind)
        fit_s = timed(lambda: SeqModel(kind, params, 0).fit_fixed(Xf, R[b.fit], ep))
        m = SeqModel(kind, params, 0)
        m.fit_fixed(Xf, R[b.fit], ep)
        pred_s = timed(lambda: m.predict(Xo))
    elif model == "Chronos":
        import torch
        torch.set_num_threads(THREADS)
        from chronos import BaseChronosPipeline
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_chronos import MODEL_ID, forecast
        pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
        price = table["price"].to_numpy()
        pred_s = timed(lambda: forecast(pipe, price, b.origins))
    print(json.dumps({"model": model, "fit_seconds": fit_s, "predict_seconds": pred_s, "n_origins": n}), flush=True)


def environment():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "n/a"
    vers = {}
    for mod in ["numpy", "pandas", "sklearn", "xgboost", "statsmodels", "torch", "chronos"]:
        r = subprocess.run([sys.executable, "-c", f"import {mod}; print(getattr({mod}, '__version__', 'n/a'))"],
                           capture_output=True, text=True)
        vers[mod] = r.stdout.strip() or "n/a"
    mem = sh(["sysctl", "-n", "hw.memsize"])
    return {"cpu": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "n_cores": sh(["sysctl", "-n", "hw.ncpu"]),
            "ram_gb": round(int(mem) / 2 ** 30, 1) if mem.isdigit() else "n/a", "os": platform.platform(),
            "python": platform.python_version(), "threads_per_model": THREADS, "repeats": REPEATS, "versions": vers}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "v3_compute_cost.csv").exists():
        print("timing already done, skipped")
        return
    rows = []
    env = {**os.environ, "OMP_NUM_THREADS": str(THREADS), "PYTHONUNBUFFERED": "1"}
    for m in MODELS:
        if m == "Chronos" and not (C.FORECASTS / "Chronos_full_seed0.csv").exists():
            continue
        r = subprocess.run([sys.executable, __file__, "--one", m], capture_output=True, text=True, env=env)
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if r.returncode != 0 or not line:
            print(f"timing failed for {m}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}", flush=True)
            raise SystemExit(1)
        d = json.loads(line[-1])
        rows.append({"model": m, "fit_seconds": d["fit_seconds"], "predict_seconds_300": d["predict_seconds"],
                     "n_origins": d["n_origins"], "predict_ms_per_origin": 1000 * d["predict_seconds"] / d["n_origins"],
                     "n_threads": THREADS, "device": "cpu", "repeats": REPEATS, "note": ""})
        print(m, f"fit {d['fit_seconds']:.3f}s, predict {d['predict_seconds']:.3f}s", flush=True)
    tab = pd.DataFrame(rows).set_index("model")
    # COMB / SHRINK: sum of their members (the averaging / multiplication itself is negligible)
    members = {"COMB": ["ARIMA", "RF", "XGB", "GRU", "Chronos"]}
    for name, mem in [("COMB", members["COMB"])] + [(f"SHRINK-{m}", [m]) for m in ["RF", "XGB", "GRU", "Chronos"]] \
            + [("SHRINK-COMB", members["COMB"])]:
        mem = [m for m in mem if m in tab.index]
        if not mem:
            continue
        s = tab.loc[mem, ["fit_seconds", "predict_seconds_300", "predict_ms_per_origin"]].sum()
        tab.loc[name] = {**s.to_dict(), "n_origins": tab.loc[mem[0], "n_origins"], "n_threads": THREADS,
                         "device": "cpu", "repeats": REPEATS, "note": "sum of members: " + ", ".join(mem)}
    tab.reset_index().to_csv(OUT / "v3_compute_cost.csv", index=False)
    (OUT / "v3_compute_env.json").write_text(json.dumps(environment(), indent=1))
    print(tab.round(4).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        one(sys.argv[2])
    else:
        main()
