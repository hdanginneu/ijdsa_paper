"""Pre-specified forecast combination and shrinkage towards RW (DECISIONS D-42, D-43).

COMB      : equal-weight mean of r_pred over the COMB_MEMBERS present in this run (full variant),
            on origins where every present member has a forecast; cal and test splits.
SHRINK-m  : r_pred_s = w * r_pred_m, w on a 0.05 grid in [0, 1], chosen per block and horizon to
            minimise price MAE on that block's cal rows (ties -> smaller w). Cal rows of SHRINK-m are
            in-sample for w (flag cal_in_sample in shrink_weights.csv) and are not used for model
            selection or interval calibration.
Run after every learned job and Chronos. Writes results/forecasts/{COMB|SHRINK-m}_full_seed0.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, load                                # noqa: E402

COMB_MEMBERS = ["ARIMA", "RF", "XGB", "GRU", "Chronos"]
SHRINK_BASES = ["RF", "XGB", "GRU", "Chronos", "COMB"]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
KEY = ["block", "split", "origin_date", "h"]
COLS = ["model", "variant", "seed", "block", "split", "origin_date", "h", "target_date",
        "price_origin", "r_true", "r_pred", "price_true", "price_pred"]


def choose_weight(cal):
    """w in GRID minimising mean |P0 e^r_true - P0 e^(w r_pred)| on cal rows; ties -> smaller w.

    Returns (w, mae_at_w, mae_at_1, n)."""
    cal = cal.dropna(subset=["r_true", "r_pred"])
    p0, rt, rp = (cal[c].to_numpy() for c in ["price_origin", "r_true", "r_pred"])
    mae = np.array([np.mean(np.abs(p0 * np.exp(rt) - p0 * np.exp(w * rp))) for w in GRID])
    k = int(np.argmin(np.round(mae, 9)))                 # first minimum = smallest w on the ascending grid
    return float(GRID[k]), float(mae[k]), float(mae[-1]), len(cal)


def as_forecast(df, model):
    out = df.copy()
    out["model"], out["variant"], out["seed"] = model, "full", 0
    out["price_true"] = out["price_origin"] * np.exp(out["r_true"])
    out["price_pred"] = out["price_origin"] * np.exp(out["r_pred"])
    return out[COLS]


def comb(ens, members):
    sub = ens[ens["model"].isin(members)]
    piv = sub.pivot_table(index=KEY, columns="model", values="r_pred", aggfunc="first")[members].dropna()
    info = sub.groupby(KEY)[["target_date", "price_origin", "r_true"]].first()
    df = info.loc[piv.index].assign(r_pred=piv.mean(axis=1)).reset_index()
    return as_forecast(df, "COMB")


def shrink(base, name):
    rows, weights = [], []
    for blk, gb in base.groupby("block"):
        test_first = gb.loc[gb["split"] == "test", "origin_date"].min()
        for h, g in gb.groupby("h"):
            cal = g[g["split"] == "cal"]
            used = cal.dropna(subset=["r_true", "r_pred"])
            # leakage guard: every cal target used to choose w is observed by the block's first origin.
            # '<=' because the frozen design has val[-1] + 7 = first origin (forecasts are made after the
            # close of the origin day, so that day's price is known); see tests/test_leakage.py check 4.
            assert used["target_date"].max() <= test_first, (name, blk, h)
            w, mae_w, mae_1, n = choose_weight(cal)
            weights.append({"model": f"SHRINK-{name}", "block": blk, "h": h, "w": w,
                            "cal_mae_at_w": mae_w, "cal_mae_at_1": mae_1, "n_cal": n, "cal_in_sample": True})
            rows.append(g.assign(r_pred=w * g["r_pred"]))
    return as_forecast(pd.concat(rows, ignore_index=True), f"SHRINK-{name}"), weights


def main():
    fc_dir, tab_dir = C.FORECASTS, C.RESULTS / "tables"
    full = load()
    ens = ensemble(full[(full["variant"] == "full") & ~full["model"].isin(["COMB"]) &
                        ~full["model"].str.startswith("SHRINK-")])
    present = set(ens["model"])
    members = [m for m in COMB_MEMBERS if m in present]
    bases = [m for m in SHRINK_BASES if m in present or (m == "COMB" and len(members) >= 2)]
    targets = [fc_dir / "COMB_full_seed0.csv"] if len(members) >= 2 else []
    targets += [fc_dir / f"SHRINK-{m}_full_seed0.csv" for m in bases]
    if targets and all(t.exists() for t in targets):
        print("combos already done, skipped:", [t.name for t in targets])
        return
    tab_dir.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "combos_members.json").write_text(json.dumps({"COMB": members, "SHRINK": bases}, indent=1))
    frames = {}
    if len(members) >= 2:
        frames["COMB"] = comb(ens, members)
        frames["COMB"].to_csv(fc_dir / "COMB_full_seed0.csv", index=False)
        print("COMB members:", members, "rows:", len(frames["COMB"]))
    weights = []
    for m in bases:
        base = frames["COMB"] if m == "COMB" else ens[ens["model"] == m]
        f, w = shrink(base, m)
        f.to_csv(fc_dir / f"SHRINK-{m}_full_seed0.csv", index=False)
        weights += w
        print(f"SHRINK-{m}: mean w = {np.mean([x['w'] for x in w]):.2f}")
    pd.DataFrame(weights).to_csv(tab_dir / "shrink_weights.csv", index=False)


if __name__ == "__main__":
    main()
