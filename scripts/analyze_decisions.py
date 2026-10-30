"""RQ1-RQ4 tables (DECISIONS D-20 .. D-24). Run after analyze.py inputs exist.

RQ1  decision fidelity of every point forecaster; MAE-vs-regret rank agreement
RQ2  decision-aware XGB/GRU vs their MSE versions (seller, sourced cost)
All rules use the sourced carry cost of D-29 (D-37); the kappa grid is a sensitivity in RQ1 only.
RQ3  payoff of each rule minus the always-act-now rule, HAC test + Holm
RQ4  uncertainty-triggered routing RW -> expensive model at query budgets 10/25/50/100%
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src import costs as K, decision as D                         # noqa: E402
from src.stats_tests import dm_test, holm                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, ewma_sigma, load                    # noqa: E402

OUT = C.RESULTS / "tables"
BUDGETS = [0.10, 0.25, 0.50, 1.00]


def pooled_and_blocks(d, fn, keys):
    a = d.groupby(keys + ["block", "h"]).apply(fn, include_groups=False).reset_index()
    b = d.groupby(keys + ["h"]).apply(fn, include_groups=False).reset_index()
    b["block"] = "pooled"
    return pd.concat([a, b], ignore_index=True)


def rq1(test):
    rows = []
    priced = K.add_cost(test)
    for side in ["seller", "buyer"]:
        for spec in ["sourced"] + D.KAPPAS:
            if spec == "sourced":
                s = pooled_and_blocks(K.evaluate_rule(priced, side), K.summary, ["label"])
            else:
                s = pooled_and_blocks(D.evaluate(test, spec, side), D.summary, ["label"])
            s["side"], s["cost_spec"] = side, str(spec)
            rows.append(s)
    dec = pd.concat(rows, ignore_index=True)
    # MAE per model/block/h for rank agreement
    e = test.assign(ae=(test["price_true"] - test["price_pred"]).abs())
    mae = pd.concat([e.groupby(["label", "block", "h"])["ae"].mean().reset_index(),
                     e.groupby(["label", "h"])["ae"].mean().reset_index().assign(block="pooled")])
    prim = dec[(dec["side"] == "seller") & (dec["cost_spec"] == "sourced")].merge(mae, on=["label", "block", "h"])
    rank = []
    for (blk, h), g in prim.groupby(["block", "h"]):
        rho, p = stats.spearmanr(g["ae"], g["mean_regret"])
        rank.append({"block": blk, "h": h, "n_models": len(g), "spearman_MAE_regret": rho, "p": p})
    return dec, pd.DataFrame(rank)


def da_ensemble():
    files = sorted((C.RESULTS / "decision_aware").glob("*.csv"))
    if not files:
        return None
    da = pd.concat([pd.read_csv(f, parse_dates=["origin_date"]) for f in files])
    return da.groupby(["model", "block", "split", "origin_date", "h"], as_index=False)["p_wait"].mean()


def rq2(test, da):
    if da is None:
        return None, None
    base = test[test["label"] == "RW"][["block", "origin_date", "h", "price_origin", "r_true", "target_date"]]
    d = da[da["split"] == "test"].merge(base, on=["block", "origin_date", "h"]).assign(r_pred=0.0)
    d = K.evaluate_rule(d.rename(columns={"model": "label"}), "seller", score="p_wait")
    da_tab = pooled_and_blocks(d, K.summary, ["label"])
    # paired regret comparison DA vs MSE version
    mse = K.evaluate_rule(test[test["label"].isin(["XGB", "GRU"])], "seller")
    rows = []
    for algo in ["XGB", "GRU"]:
        a = d[d["label"] == f"{algo}-DA"].set_index(["block", "origin_date", "h"])
        b = mse[mse["label"] == algo].set_index(["block", "origin_date", "h"])
        if a.empty or b.empty:
            continue
        idx = a.index.intersection(b.index)
        part = []
        for h in C.HORIZONS:
            ia = idx[idx.get_level_values("h") == h].sort_values()          # time order for the HAC variance
            md, st, p = dm_test(a.loc[ia, "regret"], b.loc[ia, "regret"], h)
            # D-52: DA payoff vs selling now (payoff of acting now = 0), as in RQ3
            pm, pst, pp = dm_test(-a.loc[ia, "payoff"].to_numpy(), np.zeros(len(ia)), h)
            part.append({"algo": algo, "h": h, "mean_regret_DA_minus_MSE": md, "stat": st, "p_raw": p,
                         "acc_DA": a.loc[ia, "correct"].mean(), "acc_MSE": b.loc[ia, "correct"].mean(),
                         "mean_payoff_DA_vs_act_now": -pm, "stat_payoff": -pst, "p_payoff_raw": pp})
        for r, adj in zip(part, holm([r["p_payoff_raw"] for r in part])):      # Holm over 7 h within algorithm
            r["p_payoff_holm"] = adj
        rows += part
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_raw"].to_numpy())                          # D-52: Holm over all 14 tests
    return da_tab, out


def rq3(test, da):
    d = K.evaluate_rule(test, "seller")
    if da is not None:
        base = test[test["label"] == "RW"][["block", "origin_date", "h", "price_origin", "r_true", "target_date"]]
        x = da[da["split"] == "test"].merge(base, on=["block", "origin_date", "h"]).rename(columns={"model": "label"})
        d = pd.concat([d, K.evaluate_rule(x.assign(r_pred=0.0), "seller", score="p_wait")], ignore_index=True)
    rows = []
    for h in C.HORIZONS:
        g = d[d["h"] == h]
        part = []
        for lab, gl in g.groupby("label"):
            if lab == "RW":
                continue
            # RW rule never waits when the cost is > 0, so its payoff is 0: test mean payoff against 0
            md, st, p = dm_test(-gl["payoff"].to_numpy(), np.zeros(len(gl)), h)
            part.append({"h": h, "model": lab, "mean_payoff_vs_act_now": -md, "stat": -st, "p_raw": p,
                         "share_wait": gl["wait"].mean(), "hit_rate_when_wait": gl.loc[gl["wait"], "wait_correct"].mean()})
        for r, a in zip(part, holm([r["p_raw"] for r in part])):
            r["p_holm"] = a
        rows += part
    return pd.DataFrame(rows)


def rq4(ens, expensive_labels):
    """Uncertainty-triggered routing RW -> expensive model (D-24), sourced cost (D-37), DM vs always-RW (D-39).

    Returns (route_table, route_dm)."""
    sigma = ewma_sigma()
    rows, routed_all = [], []
    timing = pd.read_csv(C.RESULTS / "chronos_timing.csv") if (C.RESULTS / "chronos_timing.csv").exists() else None
    sec_per_origin = (timing["seconds"].sum() / timing["n_origins"].sum()) if timing is not None else np.nan
    for exp in expensive_labels:
        sub = ens[ens["label"].isin(["RW", exp])].join(sigma, on="origin_date")
        for blk, gb in sub.groupby("block"):
            cal_sig = gb[(gb["split"] == "cal") & (gb["label"] == "RW") & (gb["h"] == 1)]["sigma"]
            test = gb[gb["split"] == "test"].dropna(subset=["r_true"])
            rw = test[test["label"] == "RW"].set_index(["origin_date", "h"])
            ex = test[test["label"] == exp].set_index(["origin_date", "h"]).loc[rw.index]
            for bud in BUDGETS:
                tau = -np.inf if bud >= 1 else cal_sig.quantile(1 - bud)
                use = rw["sigma"] > tau
                routed = rw.copy()
                routed.loc[use, "r_pred"] = ex.loc[use, "r_pred"]
                routed["price_pred"] = routed["price_origin"] * np.exp(routed["r_pred"])
                r = routed.reset_index().assign(label=f"route-{exp}-{int(bud * 100)}", block=blk)
                dsum = K.evaluate_rule(r, "seller")
                dsum["ae"] = (dsum["price_true"] - dsum["price_pred"]).abs()
                routed_all.append(dsum.assign(expensive=exp, budget=bud))
                for h, gh in dsum.groupby("h"):
                    rows.append({"expensive": exp, "budget": bud, "block": blk, "h": h,
                                 "share_queried": use.xs(h, level="h").mean(),
                                 "MAE": gh["ae"].mean(), "mean_regret": gh["regret"].mean(),
                                 "accuracy": gh["correct"].mean(), "mean_payoff": gh["payoff"].mean()})
    out = pd.DataFrame(rows)
    if not len(out):
        return out, pd.DataFrame()
    pooled = out.groupby(["expensive", "budget", "h"]).apply(
        lambda g: pd.Series({c: np.average(g[c], weights=None) for c in
                             ["share_queried", "MAE", "mean_regret", "accuracy", "mean_payoff"]}),
        include_groups=False).reset_index()
    pooled["block"] = "pooled(mean of blocks)"
    out = pd.concat([out, pooled], ignore_index=True)
    out["chronos_sec_per_origin"] = sec_per_origin

    # D-39: each routed rule vs always-RW on pooled test origins; Holm across budgets
    allr = pd.concat(routed_all, ignore_index=True)
    rw_rule = K.evaluate_rule(ens[(ens["label"] == "RW") & (ens["split"] == "test")].dropna(subset=["r_true"]), "seller")
    rw_rule["ae"] = (rw_rule["price_true"] - rw_rule["price_pred"]).abs()
    rw_rule = rw_rule.set_index(["block", "origin_date", "h"])
    dm_rows = []
    for (exp, h), g in allr.groupby(["expensive", "h"]):
        for loss in ["regret", "ae"]:
            part = []
            for bud, gb in g.groupby("budget"):
                a = gb.set_index(["block", "origin_date", "h"]).sort_index()
                b = rw_rule.loc[a.index]
                md, st, p = dm_test(a[loss].to_numpy(), b[loss].to_numpy(), h)
                part.append({"expensive": exp, "budget": bud, "h": h, "loss": loss, "n": len(a),
                             "mean_diff_vs_RW": md, "DM_HLN": st, "p_raw": p})
            for x, adj in zip(part, holm([x["p_raw"] for x in part])):
                x["p_holm"] = adj
            dm_rows += part
    return out, pd.DataFrame(dm_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ens = ensemble(load())
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    dec, rank = rq1(test)
    dec.to_csv(OUT / "rq1_decisions.csv", index=False)
    rank.to_csv(OUT / "rq1_rank_agreement.csv", index=False)
    da = da_ensemble()
    da_tab, da_cmp = rq2(test, da)
    if da_tab is not None:
        da_tab.to_csv(OUT / "rq2_da_decisions.csv", index=False)
        da_cmp.to_csv(OUT / "rq2_da_vs_mse.csv", index=False)
    rq3(test, da).to_csv(OUT / "rq3_payoff.csv", index=False)
    cal = ens[ens["split"] == "cal"].dropna(subset=["r_true"])
    cal_mae = cal.assign(ae=(cal["price_true"] - cal["price_pred"]).abs()).groupby("label")["ae"].mean()
    # D-46: COMB and SHRINK-* are excluded (their validation MAE is not a fair out-of-sample choice)
    learned = [l for l in cal_mae.index if l not in ("RW", "Drift", "ARIMA", "Chronos", "COMB")
               and not l.startswith("SHRINK-")]
    exps = (["Chronos"] if "Chronos" in cal_mae.index else []) + ([cal_mae.loc[learned].idxmin()] if learned else [])
    route, route_dm = rq4(ens, exps)
    route.to_csv(OUT / "rq4_routing.csv", index=False)
    route_dm.to_csv(OUT / "rq4_routing_dm.csv", index=False)
    print("expensive models for RQ4:", exps)


if __name__ == "__main__":
    main()
