"""Claims of the submitted manuscript (v1) vs the new evidence. Main run only.

Writes results/tables/v1_contrast.md. Numbers only; the verdict follows the rules written next to
each claim below, and is 'needs author review' when a rule cannot decide.
v1 numbers: submitted manuscript, Table 4.2 and Sections 4.3–4.5 (300-day test window, seed 42).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.stats_tests import dm_test                               # noqa: E402

T = C.RESULTS / "tables"


def read(name):
    f = T / name
    return pd.read_csv(f) if f.exists() and f.stat().st_size > 5 else None


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def claim_gru_best():
    v1 = read("v1_window_h1.csv")
    if v1 is None:
        return "n/a", "needs author review"
    v = v1.set_index("model")
    best = v["MAE"].idxmin()
    gru, rw = v["MAE"].get("GRU", np.nan), v["MAE"].get("RW", np.nan)
    ev = (f"B5 h=1 MAE: GRU {fmt(gru)}, RW {fmt(rw)}, best {best} {fmt(v['MAE'].min())}; "
          f"GRU rMAE {fmt(v['rMAE'].get('GRU', np.nan), 3)}")
    # rule: reversed if RW beats GRU on the v1 window; supported if GRU is the minimum; else not supported
    verdict = "reversed" if gru > rw else ("supported" if best == "GRU" else "not supported")
    return ev, verdict


def claim_gru_dm():
    dm, mcs = read("dm_vs_rw.csv"), read("mcs.csv")
    if dm is None:
        return "n/a", "needs author review"
    g = dm[(dm["model"] == "GRU") & (dm["loss"] == "SE") & (dm["bandwidth"] == "h-1")]
    better = int(((g["p_holm"] < 0.01) & (g["mean_diff_vs_RW"] < 0)).sum())
    worse = int(((g["p_holm"] < 0.05) & (g["mean_diff_vs_RW"] > 0)).sum())
    in_mcs = "n/a"
    if mcs is not None:
        m = mcs[(mcs["model"] == "GRU") & (mcs["loss"] == "SE")]
        in_mcs = f"{int(m['in_mcs'].sum())}/{len(m)} horizons"
    ev = (f"GRU vs RW (SE, HAC h−1, Holm): better at 1% in {better}/{len(g)} horizons, "
          f"worse at 5% in {worse}; min p_holm {fmt(g['p_holm'].min(), 3)}; GRU in 90% MCS: {in_mcs}")
    # rule: supported if better at 1% in every horizon; reversed if significantly worse anywhere and never better
    verdict = "supported" if better == len(g) else ("reversed" if worse > 0 and better == 0 else "not supported")
    return ev, verdict


def claim_stability():
    s = read("seed_spread.csv")
    if s is None:
        return "n/a", "needs author review"
    s = s.assign(cv=s["std"] / s["mean"] * 100)
    gru, lstm = s[s["label"] == "GRU"].set_index("h")["cv"], s[s["label"] == "LSTM"].set_index("h")["cv"]
    if gru.empty or lstm.empty:
        return "n/a", "needs author review"
    ev = (f"CV of MAE across 5 seeds, mean over h: GRU {fmt(gru.mean(), 1)}%, LSTM {fmt(lstm.mean(), 1)}%; "
          f"GRU < LSTM in {int((gru < lstm.reindex(gru.index)).sum())}/{len(gru)} horizons")
    if gru.isna().any() or lstm.isna().any():
        return ev, "needs author review"
    verdict = "supported" if (gru < lstm.reindex(gru.index)).all() else "not supported"
    return ev, verdict


def claim_recurrence():
    acc = read("accuracy.csv")
    if acc is None:
        return "n/a", "needs author review"
    p = acc[acc["block"] == "pooled"].groupby("model")["rMAE"].mean()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze import ensemble, load
    ens = ensemble(load())
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    sig = 0
    for h in C.HORIZONS:
        g = test[test["h"] == h]
        a = g[g["label"] == "DNN"].set_index(["block", "origin_date"])
        b = g[g["label"] == "GRU"].set_index(["block", "origin_date"])
        if a.empty or b.empty:
            return "n/a", "needs author review"
        idx = a.index.intersection(b.index).sort_values()          # time order
        md, st, pv = dm_test((a.loc[idx, "price_true"] - a.loc[idx, "price_pred"]) ** 2,
                             (b.loc[idx, "price_true"] - b.loc[idx, "price_pred"]) ** 2, h)
        sig += int(pv < 0.05 and md > 0)
    ev = (f"pooled rMAE (mean over h): DNN {fmt(p.get('DNN', np.nan), 3)}, GRU {fmt(p.get('GRU', np.nan), 3)}; "
          f"DNN significantly worse than GRU (SE, raw p<0.05) in {sig}/{len(C.HORIZONS)} horizons")
    # a forecast comparison cannot establish a mechanism; the rule only decides when the gap is absent
    verdict = "not supported" if sig == 0 else "needs author review"
    return ev, verdict


def claim_bagging():
    acc = read("accuracy.csv")
    if acc is None:
        return "n/a", "needs author review"
    p = acc[acc["block"] == "pooled"].groupby("model")["rMAE"].mean()
    rf, xgb = p.get("RF", np.nan), p.get("XGB", np.nan)
    ev = f"pooled rMAE (mean over h): RF {fmt(rf, 3)}, XGB {fmt(xgb, 3)}"
    verdict = "not supported" if not rf < xgb else "needs author review"
    return ev, verdict


def claim_heuristic_band():
    iv = read("intervals_pooled.csv")
    if iv is None:
        return "n/a", "needs author review"
    g = iv[(iv["model"] == "RW") & (iv["method"] == "heur") & (iv["level"] == 0.95)]
    cov = g["coverage"].mean()
    ev = (f"RW, 95% √h rule calibrated on validation: coverage {fmt(cov, 3)} (mean over h; "
          f"range {fmt(g['coverage'].min(), 3)}–{fmt(g['coverage'].max(), 3)}); Kupiec p<0.05 in "
          f"{int((g['kupiec_p'] < 0.05).sum())}/{len(g)} horizons")
    verdict = "supported" if 0.93 <= cov <= 0.97 and (g["kupiec_p"] >= 0.05).all() else "not supported"
    return ev, verdict


def claim_sell_forward():
    d, dm = read("v2_decisions.csv"), read("v2_decisions_dm.csv")
    if d is None or dm is None:
        return "n/a", "needs author review"
    p = d[(d["block"] == "pooled") & (d["storage_mult"] == 1) & (d["side"] == "seller") & (d["basis_sd"] == 0)]
    g = p[p["model"] == "GRU"]
    r = dm[(dm["model"] == "GRU") & (dm["storage_mult"] == 1) & (dm["side"] == "seller") &
           (dm["basis_sd"] == 0) & (dm["loss"] == "regret")]
    better = int(((r["p_holm"] < 0.05) & (r["mean_diff_vs_RW"] < 0)).sum())
    ev = (f"GRU point rule, seller, sourced cost: mean payoff vs selling now {fmt(g['mean_payoff'].mean())} USD/t "
          f"(mean over h); regret lower than RW at Holm 5% in {better}/{len(r)} horizons")
    verdict = "supported" if better == len(r) and g["mean_payoff"].mean() > 0 else "not supported"
    return ev, verdict


CLAIMS = [
    ("GRU is the best model", "GRU MAE 96.55, RMSE 125.91 (300-day v1 test, seed 42)", claim_gru_best),
    ("GRU beats all five other models, DM p < 0.01", "5/5 comparisons p < 0.01 (seed 42)", claim_gru_dm),
    ("GRU is more stable than LSTM across seeds", "CV(MAE) GRU 24.8%, LSTM 39.5%", claim_stability),
    ("Recurrence, not depth, explains DL gains (DNN fails)", "DNN R² −1.665, MAE 1,081.47", claim_recurrence),
    ("Bagging beats boosting when extrapolating (RF > XGB)", "RF R² 0.888 vs XGB 0.465", claim_bagging),
    ("7-day risk band ±1.96·RMSE·√h is a usable 95% range", "band ±246.79 → ±652.95 USD/t", claim_heuristic_band),
    ("Exporters should sell forward early", "recommendation from the GRU 7-day path", claim_sell_forward),
]


def main():
    rows = ["| # | v1 claim | v1 number | new evidence | verdict |", "|---|---|---|---|---|"]
    for k, (claim, v1num, fn) in enumerate(CLAIMS, 1):
        try:
            ev, verdict = fn()
        except Exception as e:                                    # keep the table complete
            ev, verdict = f"error: {type(e).__name__}: {e}", "needs author review"
        rows.append(f"| {k} | {claim} | {v1num} | {ev} | {verdict} |")
    text = ("# v1 claims vs new evidence (auto-generated by scripts/v1_contrast.py)\n\n"
            "Verdict rules are in the script. 'needs author review' = the rule cannot decide.\n\n" + "\n".join(rows) + "\n")
    T.mkdir(parents=True, exist_ok=True)
    (T / "v1_contrast.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
