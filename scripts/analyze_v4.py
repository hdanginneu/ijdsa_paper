"""Round-4 analysis-only additions. Saved forecasts only; tables in <results>/tables/v4_*.csv.

E1 inventory risk limit from the 90% interval  -> v4_risk_limit*.csv          (RQ2)
E2 oracle value and break-even accuracy        -> v4_oracle_value.csv, v4_value_share.csv  (RQ3)
E3 reliability of model selection              -> v4_selection_*.csv, v4_shrink_vs_endpoints.csv (RQ4)
E4 variance-ratio tests, main and Arabica only -> v4_variance_ratio.csv       (RQ1 mechanism)
Decision configuration: seller, storage x1, basis noise 0, sourced cost. Every HAC test uses series
in (block, origin_date) order via analyze_v3.ordered / paired.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, costs as K                           # noqa: E402
from src.data import load_coffee                                  # noqa: E402
from src.stats_tests import christoffersen, dm_test, holm         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, load                                # noqa: E402
from analyze_v3 import BLOCK_ORDER, boot_idx, ordered, paired      # noqa: E402

OUT = C.RESULTS / "tables"
TAU = 0.05                                     # lower bound of the two-sided 90% interval
METHODS = ["conf", "vol", "heur"]
CANDIDATES = ["ARIMA", "Drift", "SVR", "RF", "XGB", "XGB-price", "DNN", "LSTM", "GRU", "GRU-price", "Chronos", "COMB"]
SHRINK_BASES = ["RF", "XGB", "GRU", "Chronos", "COMB"]
VR_Q = [2, 5, 10]


# ------------------------------------------------------------------ E1
def zone(x, n):
    f = stats.binom.cdf(x, n, TAU)
    return ("green" if f < 0.95 else "yellow" if f < 0.9999 else "red"), f


def e1_risk_limit():
    f = OUT / "intervals_rows.csv"
    if not f.exists():
        return
    iv = pd.read_csv(f, parse_dates=["origin_date"])
    iv = ordered(iv[(iv["level"] == 0.9) & iv["model"].isin(["RW", "COMB"])])
    rows, zones, dms, chs = [], [], [], []
    for (m, h), g in iv.groupby(["model", "h"]):
        y = g["price_true"]
        pin = {}
        for meth in METHODS:
            lo = g[f"{meth}_lo"]
            br = y < lo
            pin[meth] = (TAU - br.astype(float)) * (y - lo)                 # quantile (pinball) loss at 0.05
            frame = pd.DataFrame({"block": g["block"], "breach": br, "short": (lo - y).where(br),
                                  "capital": g["price_origin"] - lo, "pin": pin[meth]})
            for blk, gb in [("pooled", frame)] + list(frame.groupby("block", sort=False)):
                rows.append({"model": m, "method": meth, "h": h, "block": blk, "n": len(gb),
                             "breach_rate": gb["breach"].mean(), "mean_shortfall": gb["short"].mean(),
                             "max_shortfall": gb["short"].max(), "mean_capital": gb["capital"].mean(),
                             "pinball_05": gb["pin"].mean()})
                if blk != "pooled":
                    z, cdf = zone(int(gb["breach"].sum()), len(gb))
                    zones.append({"model": m, "method": meth, "h": h, "block": blk, "n": len(gb),
                                  "breaches": int(gb["breach"].sum()), "binom_cdf": cdf, "zone": z})
            if h == 1:
                hits = (~br).astype(float)
                segs = [hits[g["block"] == b].to_numpy() for b in BLOCK_ORDER if b in set(g["block"])]
                for b, s in zip([b for b in BLOCK_ORDER if b in set(g["block"])], segs):
                    chs.append({"model": m, "method": meth, "block": b, **christoffersen(s, 1 - TAU)})
                chs.append({"model": m, "method": meth, "block": "pooled", **christoffersen(segs, 1 - TAU)})
        for other in ["heur", "conf"]:
            md, st, p = dm_test(pin["vol"], pin[other], h)
            dms.append({"model": m, "pair": f"vol vs {other}", "h": h, "n": len(g),
                        "mean_pinball_diff": md, "DM_HLN": st, "p_raw": p})
    dms = pd.DataFrame(dms)
    dms["p_holm"] = np.nan
    for _, idx in dms.groupby(["model", "pair"]).groups.items():         # Holm over h within each pair
        dms.loc[idx, "p_holm"] = holm(dms.loc[idx, "p_raw"].to_numpy())
    zones = pd.DataFrame(zones)
    counts = zones.groupby(["model", "method", "zone"]).size().unstack(fill_value=0).reset_index()
    for z in ["green", "yellow", "red"]:
        if z not in counts.columns:
            counts[z] = 0
    counts = counts[["model", "method", "green", "yellow", "red"]].assign(block="count", h="all")
    pd.DataFrame(rows).to_csv(OUT / "v4_risk_limit.csv", index=False)
    pd.concat([zones, counts], ignore_index=True).to_csv(OUT / "v4_risk_limit_zones.csv", index=False)
    dms.to_csv(OUT / "v4_risk_limit_dm.csv", index=False)
    pd.DataFrame(chs).to_csv(OUT / "v4_risk_limit_christoffersen_h1.csv", index=False)


# ------------------------------------------------------------------ E2
def thresholds(pw):
    """VPI, break-even q* and the q needed for 10% / 25% of VPI (closed form, TPR = TNR = q)."""
    A, Bn = np.maximum(pw, 0).mean(), np.minimum(pw, 0).mean()
    qs = lambda s: (s * A - Bn) / (A - Bn)                          # solves q·A + (1 − q)·Bn = s·VPI, VPI = A
    return A, -Bn / (A - Bn), qs(0.10), qs(0.25)


def e2_oracle(dec, labels):
    rw = ordered(dec[dec["label"] == "RW"].reset_index())
    rows = []
    for h, g in rw.groupby("h"):
        for blk, gb in [("pooled", g)] + list(g.groupby("block", sort=False)):
            pw = gb["payoff_wait"].to_numpy()
            vpi, q0, q10, q25 = thresholds(pw)
            rows.append({"h": h, "block": blk, "n": len(gb), "VPI_usd_t": vpi,
                         "VPI_pct_price": 100 * vpi / gb["price_origin"].mean(),
                         "q_breakeven": q0 + 0.0, "q_10pct": q10, "q_25pct": q25,
                         # every pw on one side: q* is 0 (all > 0) or 1 (all < 0) by construction
                         "all_same_sign": bool((pw > 0).all() or (pw < 0).all())})
    pd.DataFrame(rows).to_csv(OUT / "v4_oracle_value.csv", index=False)
    test = dec.reset_index()
    share = []
    for m in labels:
        for h in C.HORIZONS:
            a, _ = paired(test, m, h)
            pw, pay, wait = a["payoff_wait"].to_numpy(), a["payoff"].to_numpy(), a["wait"].to_numpy(bool)
            pos, neg = pw > 0, pw < 0
            A, Bn = np.maximum(pw, 0).mean(), np.minimum(pw, 0).mean()
            idx = boot_idx(a, h)
            boot = pay[idx].mean(1) / np.maximum(pw, 0)[idx].mean(1)
            tpr = wait[pos].mean() if pos.any() else np.nan
            tnr = (~wait[neg]).mean() if neg.any() else np.nan
            share.append({"model": m, "h": h, "n": len(a), "share": pay.mean() / A,
                          "ci95_lo": np.percentile(boot, 2.5), "ci95_hi": np.percentile(boot, 97.5),
                          "accuracy": a["correct"].mean(), "balanced_accuracy": np.nanmean([tpr, tnr]),
                          "TPR": tpr, "TNR": tnr, "payoff_check": tpr * A + (1 - tnr) * Bn,
                          "payoff_realized": pay.mean()})
    pd.DataFrame(share).to_csv(OUT / "v4_value_share.csv", index=False)


# ------------------------------------------------------------------ E3
def cell_stats(frame, labels):
    """MAE and mean regret per label in one block x h cell (frame already evaluated with K.evaluate_rule)."""
    e = (frame["price_true"] - frame["price_pred"]).abs()
    return pd.DataFrame({"MAE": e.groupby(frame["label"]).mean(),
                         "regret": frame.groupby("label")["regret"].mean()}).reindex(labels)


def e3_selection(ens, labels):
    cands = [m for m in CANDIDATES if m in labels]
    cal = ens[(ens["split"] == "cal") & ens["label"].isin(cands + ["RW"])].dropna(subset=["r_true", "target_date"])
    tst = ens[(ens["split"] == "test") & ens["label"].isin(cands + ["RW"])].dropna(subset=["r_true"])
    cal, tst = K.evaluate_rule(cal, "seller"), K.evaluate_rule(tst, "seller")
    cells, ranks = [], []
    for (blk, h), gt in tst.groupby(["block", "h"]):
        gc = cal[(cal["block"] == blk) & (cal["h"] == h)]
        sc, st = cell_stats(gc, cands + ["RW"]), cell_stats(gt, cands + ["RW"])
        ok = [m for m in cands if np.isfinite(sc.loc[m, "MAE"]) and np.isfinite(st.loc[m, "MAE"])]
        if not ok or not np.isfinite(st.loc["RW", "MAE"]):
            continue
        for proc, pool, crit in [("P1", ok, "MAE"), ("P2", ok + ["RW"], "MAE"), ("P3", ok, "regret")]:
            sel = sc.loc[pool, crit].idxmin()
            if crit == "MAE":
                stat = st.loc[sel, "MAE"] / st.loc["RW", "MAE"]
                worse, better = stat > 1, stat < 1
            else:
                stat = st.loc[sel, "regret"] - st.loc["RW", "regret"]
                worse, better = stat > 0, stat < 0
            cells.append({"procedure": proc, "block": blk, "h": h, "selected": sel, "n_candidates": len(pool),
                          "test_stat": stat, "stat_kind": "MAE ratio vs RW" if crit == "MAE" else "regret diff vs RW (USD/t)",
                          "worse_than_RW": bool(worse), "better_than_RW": bool(better),
                          "selection_regret": st.loc[sel, crit] - st.loc[pool, crit].min()})
        for proc, crit in [("P1", "MAE"), ("P3", "regret")]:         # descriptive only, never mixed
            rho, p = stats.spearmanr(sc.loc[ok, crit], st.loc[ok, crit]) if len(ok) > 2 else (np.nan, np.nan)
            ranks.append({"procedure": proc, "rank_by": crit, "block": blk, "h": h, "n_candidates": len(ok),
                          "spearman_rho": rho, "p": p})
    cells = pd.DataFrame(cells)
    ranks = pd.DataFrame(ranks)
    shrink = e3_shrink(ens, labels)
    summ = []
    for proc, g in cells.groupby("procedure"):
        freq = g["selected"].value_counts()
        summ.append({"procedure": proc, "n_cells": len(g), "n_worse_than_RW": int(g["worse_than_RW"].sum()),
                     "n_better_than_RW": int(g["better_than_RW"].sum()), "share_worse_than_RW": g["worse_than_RW"].mean(),
                     "stat_kind": g["stat_kind"].iloc[0], "mean_test_stat": g["test_stat"].mean(),
                     "median_test_stat": g["test_stat"].median(), "mean_selection_regret": g["selection_regret"].mean(),
                     "n_RW_selected": int((g["selected"] == "RW").sum()),
                     "selected_frequency": "; ".join(f"{k} {v}" for k, v in freq.items()),
                     "conclusion_worse_majority": bool(g["worse_than_RW"].mean() > 0.5)})
    if len(shrink):
        cellp = shrink[shrink["block"] != "summary"]
        summ.append({"procedure": "P4", "n_cells": len(cellp), "n_worse_than_RW": int(cellp["worse_than_w0"].sum()),
                     "n_better_than_RW": int(cellp["better_than_w0"].sum()), "share_worse_than_RW": cellp["worse_than_w0"].mean(),
                     "stat_kind": "MAE ratio selected w vs w = 0", "mean_test_stat": cellp["ratio_vs_w0"].mean(),
                     "median_test_stat": cellp["ratio_vs_w0"].median(), "mean_selection_regret": np.nan,
                     "n_RW_selected": int((cellp["w"] == 0).sum()), "selected_frequency": f"w = 0 in {int((cellp['w'] == 0).sum())} cells",
                     "conclusion_worse_majority": bool(cellp["worse_than_w0"].mean() > 0.5)})
    for proc, crit in [("P1", "MAE"), ("P3", "regret")]:
        g = ranks[ranks["procedure"] == proc]
        ranks = pd.concat([ranks, pd.DataFrame([{"procedure": proc, "rank_by": crit, "block": "summary", "h": "all",
                                                 "n_candidates": g["n_candidates"].max(), "spearman_rho": g["spearman_rho"].mean(),
                                                 "p": np.nan, "median_rho": g["spearman_rho"].median(),
                                                 "n_rho_pos_p05": int(((g["spearman_rho"] > 0) & (g["p"] < 0.05)).sum()),
                                                 "n_cells": len(g)}])], ignore_index=True)
    pd.DataFrame(summ).to_csv(OUT / "v4_selection_risk.csv", index=False)
    cells.to_csv(OUT / "v4_selection_cells.csv", index=False)
    ranks.to_csv(OUT / "v4_selection_rankcorr.csv", index=False)


def e3_shrink(ens, labels):
    f = OUT / "shrink_weights.csv"
    if not f.exists():
        return pd.DataFrame()
    w = pd.read_csv(f)
    tst = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    mae = (tst.assign(ae=(tst["price_true"] - tst["price_pred"]).abs())
           .groupby(["label", "block", "h"])["ae"].mean())
    rows = []
    for _, r in w.iterrows():
        base = r["model"].replace("SHRINK-", "")
        key = (r["block"], r["h"])
        if (r["model"],) + key not in mae.index or (base,) + key not in mae.index:
            continue
        mw, m0, m1 = mae[(r["model"],) + key], mae[("RW",) + key], mae[(base,) + key]
        rows.append({"model": r["model"], "block": r["block"], "h": r["h"], "w": r["w"], "MAE_selected_w": mw,
                     "MAE_w0": m0, "MAE_w1": m1, "ratio_vs_w0": mw / m0, "ratio_vs_w1": mw / m1,
                     "better_than_w0": bool(mw < m0), "worse_than_w0": bool(mw > m0), "better_than_w1": bool(mw < m1)})
    out = pd.DataFrame(rows)
    if len(out):
        s = out.groupby("model")[["better_than_w0", "worse_than_w0", "better_than_w1"]].mean().reset_index()
        s = s.rename(columns=lambda c: "share_" + c if c != "model" else c).assign(block="summary")
        out = pd.concat([out, s], ignore_index=True)
    out.to_csv(OUT / "v4_shrink_vs_endpoints.csv", index=False)
    return out


# ------------------------------------------------------------------ E4
def variance_ratio(r, q):
    """Lo & MacKinlay (1988): overlapping q-period variance ratio with the small-sample bias corrections and the
    heteroskedasticity-robust z* = sqrt(n) (VR - 1) / sqrt(theta), n = number of daily returns.
    Returns (VR, z*, two-sided p)."""
    r = np.asarray(r, float)
    n = len(r)
    mu = r.mean()
    dev = r - mu
    var_a = dev @ dev / (n - 1)
    sums = np.convolve(r, np.ones(q), "valid")                          # overlapping q-period sums
    m = q * (n - q + 1) * (1 - q / n)
    var_c = ((sums - q * mu) ** 2).sum() / m
    vr = var_c / var_a
    d2 = dev ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * (d2[j:] @ d2[:-j]) / (d2.sum() ** 2)
        theta += (2 * (q - j) / q) ** 2 * delta
    z = np.sqrt(n) * (vr - 1) / np.sqrt(theta)                        # sqrt(n) was missing
    return vr, z, 2 * stats.norm.sf(abs(z))


def coffee_returns():
    """Daily log returns of the coffee close from C.START_DATE on (first return = the day after START_DATE).
    Taken from the coffee series itself, not from build_table(), which drops the first rows for the lagged
    macro inputs; those inputs play no role here. On the dates both cover the two series are identical."""
    c = load_coffee(None)
    c = c[c["date"] >= C.START_DATE]
    if C.END_DATE:
        c = c[c["date"] <= C.END_DATE]
    return np.log(c.set_index("date")["price"]).diff().dropna()


def e4_variance_ratio(ens):
    """Periods. Block b covers the returns in (first origin of b, last origin of b]. 'pre-B1' covers every
    return from the first one after START_DATE up to and including the first B1 origin (2020-12-01 in the main
    run: N = 3,302), so pre-B1, B1, ..., B6 are contiguous and no return falls between them. The return dated
    at the first B1 origin is known at that origin, so it is pre-test information. (The 24/10 run used
    build_table() and stopped before that origin, N = 3,294: it lost the first 6 returns and the 2020-12-01
    return.) 'pooled B1-B6' = (first B1 origin, last B6 origin]."""
    if C.TAG not in ("", "arabica"):
        return
    ret = coffee_returns()
    tst = ens[ens["split"] == "test"]
    span = tst.groupby("block")["origin_date"].agg(["min", "max"]).reindex([b for b in BLOCK_ORDER if b in set(tst["block"])])
    periods = [("pre-B1", ret[ret.index <= span["min"].iloc[0]])]
    periods += [(b, ret[(ret.index > s) & (ret.index <= e)]) for b, (s, e) in span.iterrows()]
    periods.append(("pooled B1-B6", ret[(ret.index > span["min"].iloc[0]) & (ret.index <= span["max"].iloc[-1])]))
    rows = []
    for name, r in periods:
        part = []
        for q in VR_Q:
            if len(r) <= 2 * q:
                continue
            vr, z, p = variance_ratio(r.to_numpy(), q)
            part.append({"period": name, "q": q, "n": len(r), "VR": vr, "z_star": z, "p": p})
        for x, adj in zip(part, holm([x["p"] for x in part])):
            x["p_holm"] = adj
        rows += part
    pd.DataFrame(rows).to_csv(OUT / "v4_variance_ratio.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ens = ensemble(load())
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    labels = sorted(test["label"].unique())
    dec = K.evaluate_rule(test, "seller").set_index(["block", "origin_date", "h"])
    e1_risk_limit()
    e2_oracle(dec, labels)
    e3_selection(ens, labels)
    e4_variance_ratio(ens)
    print("analyze_v4 done")


if __name__ == "__main__":
    main()
