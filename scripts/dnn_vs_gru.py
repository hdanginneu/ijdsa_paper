"""DNN versus GRU: Diebold-Mariano tests on the seed-averaged forecasts (Online Resource 1, Table S4b).

Uses saved forecasts only. Same loss (squared and absolute price error), HAC lags h - 1 and Harvey-Leybourne-Newbold
correction as the DM tests against the RW in analyze.py; Holm adjustment across the seven horizons within each loss.
Output: <results>/tables/dnn_vs_gru_dm.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.stats_tests import dm_test, holm                         # noqa: E402

KEY = ["block", "split", "origin_date", "h"]


def seed_average(label):
    files = sorted(C.FORECASTS.glob(f"{label}_full_seed*.csv"))
    fc = pd.concat([pd.read_csv(f, parse_dates=["origin_date"]) for f in files], ignore_index=True)
    g = fc.groupby(KEY, as_index=False).agg(r_pred=("r_pred", "mean"), r_true=("r_true", "first"),
                                            price_origin=("price_origin", "first"), price_true=("price_true", "first"))
    g["price_pred"] = g["price_origin"] * np.exp(g["r_pred"])
    return g[g["split"] == "test"].dropna(subset=["r_true"])


def main():
    fc = {k: seed_average(k) for k in ["DNN", "GRU", "RW"]}
    rows = []
    for h in C.HORIZONS:
        P = {k: v[v["h"] == h].set_index(["block", "origin_date"]) for k, v in fc.items()}
        idx = P["DNN"].index.intersection(P["GRU"].index).intersection(P["RW"].index).sort_values()
        y = P["RW"].loc[idx, "price_true"]
        for loss, f in [("SE", lambda e: e ** 2), ("AE", np.abs)]:
            lo = {k: f(y - P[k].loc[idx, "price_pred"]) for k in P}
            md, st, p = dm_test(lo["DNN"], lo["GRU"], h)
            _, st0, p0 = dm_test(lo["DNN"], lo["RW"], h)
            rows.append({"h": h, "loss": loss, "n": len(idx), "mean_diff_DNN_minus_GRU": md, "DM_HLN": st, "p_raw": p,
                         "DNN_vs_RW_DM": st0, "DNN_vs_RW_p": p0,
                         "MAE_DNN": np.mean(np.abs(y - P["DNN"].loc[idx, "price_pred"])),
                         "MAE_GRU": np.mean(np.abs(y - P["GRU"].loc[idx, "price_pred"]))})
    df = pd.DataFrame(rows)
    for loss in ["SE", "AE"]:
        m = df["loss"] == loss
        df.loc[m, "p_holm"] = holm(df.loc[m, "p_raw"].tolist())
    out = C.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dnn_vs_gru_dm.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
