"""RQ2: decision-aware (cost-sensitive classification) versions of XGB and GRU (D-22).

Same frozen hyperparameters, blocks, seeds and inputs as the MSE versions; only the loss changes.
Labels use the sourced carry cost (D-38):
    gain(t,h) = P_t (exp(r) - 1) - cost(t,h)     (USD/t, payoff of waiting for a seller)
    label = 1[gain > 0], weight = |gain| / P_t
cost(t,h) uses only information at origin t; labels use realised targets of fit-set origins only.
Output: results/decision_aware/{XGB|GRU}-DA_seed{s}.csv with p_wait per origin and horizon.
Usage:
    python scripts/run_decision_aware.py xgb
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.costs import cost_matrix                                 # noqa: E402
from src.features import targets                                  # noqa: E402
from src.folds import make_blocks                                 # noqa: E402
from src.learned import inputs                                    # noqa: E402

OUT = C.RESULTS / "decision_aware"


def labels(table, R, cost):
    p0 = table["price"].to_numpy()[:, None]
    gain = p0 * (np.exp(R) - 1) - cost                    # payoff of waiting, USD/t (NaN at the end of the series)
    y = (gain > 0).astype(np.float32)
    w = np.nan_to_num(np.abs(gain) / p0).astype(np.float32)
    return y, w


def xgb_fit_predict(params, seed, Xtr, Ytr, Wtr, Xte):
    p = {k: v for k, v in params.items() if k != "n_lags"}
    out = np.empty((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        from xgboost import XGBClassifier            # lazy: never in the same process as torch
        m = XGBClassifier(tree_method="hist", n_jobs=2, random_state=seed, **p)
        m.fit(Xtr, Ytr[:, j], sample_weight=Wtr[:, j] / Wtr[:, j].mean())
        out[:, j] = m.predict_proba(Xte)[:, 1]
    return out


def main(kind):
    table = build_table()
    R = targets(table)
    cost = cost_matrix(table)
    Y, W = labels(table, R, cost)
    params = json.loads((C.TUNING / f"{kind}_full.json").read_text())["best_params"]
    X = inputs(table, kind, params, "full")
    name = {"xgb": "XGB-DA", "gru": "GRU-DA"}[kind]
    OUT.mkdir(parents=True, exist_ok=True)
    dates = table.index
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else C.SEEDS
    for seed in seeds:
        if (OUT / f"{name}_seed{seed}.csv").exists():
            print(name, "seed", seed, "already done, skipped", flush=True)
            continue
        rows = []
        for b in make_blocks(table):
            assert not np.isnan(cost[b.fit]).any() and not np.isnan(R[b.fit]).any(), b.name   # labels known
            t0 = time.time()
            if kind == "xgb":
                cal = xgb_fit_predict(params, seed, X(b.pretrain), Y[b.pretrain], W[b.pretrain], X(b.val))
                test = xgb_fit_predict(params, seed, X(b.fit), Y[b.fit], W[b.fit], X(b.origins))
            else:
                from src.nn_models import SeqClassifier
                m = SeqClassifier("gru", params, seed)
                m._run(X(b.pretrain), Y[b.pretrain], W[b.pretrain],
                       Xv=X(b.val), Yv=Y[b.val], Wv=W[b.val])
                cal = m.predict(X(b.val))
                n_ep = max(1, round(m.best_epoch * len(b.fit) / len(b.pretrain)))
                m2 = SeqClassifier("gru", params, seed)._run(X(b.fit), Y[b.fit], W[b.fit], epochs=n_ep)
                test = m2.predict(X(b.origins))
            for split, origins, p in [("cal", b.val, cal), ("test", b.origins, test)]:
                for j, h in enumerate(C.HORIZONS):
                    rows.append(pd.DataFrame({"model": name, "seed": seed, "block": b.name, "split": split,
                                              "origin_date": dates[origins], "h": h, "p_wait": p[:, j]}))
            print(name, "seed", seed, b.name, f"{time.time() - t0:.1f}s", flush=True)
        pd.concat(rows).to_csv(OUT / f"{name}_seed{seed}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
