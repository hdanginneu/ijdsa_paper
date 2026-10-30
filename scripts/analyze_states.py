"""When to use which model (DECISIONS D-25 .. D-28). Uses saved forecasts only.

All decision rules use the sourced carry cost of D-29 (D-37)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src import costs as K                                        # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.stats_tests import dm_test, holm                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, ewma_sigma, load                    # noqa: E402

OUT = C.RESULTS / "tables"
GW_MODELS = ["RF", "XGB", "Chronos", "GRU"]
CANDIDATES = ["RW", "RF", "XGB", "Chronos", "GRU"]
P_STARS = [0.5, 0.6, 0.7]


def states(first_b1):
    t = build_table()
    trend = t["logp"] - t["logp"].shift(60)
    sig = ewma_sigma()
    pre = t.index < first_b1
    out = pd.DataFrame({"trend": trend, "vol": sig})
    for c in ["trend", "vol"]:
        cuts = out.loc[pre, c].quantile([1 / 3, 2 / 3]).to_numpy()
        out[c + "_state"] = pd.cut(out[c], [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"])
    return out


def gw_test(d, Z, h):
    """Giacomini-White: Wald test of E[d | Z] = 0 using OLS of d on [1, Z] with HAC lags h-1."""
    X = sm.add_constant(Z)
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 0)})
    w = res.wald_test(np.eye(X.shape[1]), scalar=True)
    return float(w.statistic), float(w.pvalue), res.params.to_dict()


def main():
    ens = ensemble(load())
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    first_b1 = test.loc[test["block"] == "B1", "origin_date"].min()
    st = states(first_b1)
    dec = K.evaluate_rule(test, "seller").join(st, on="origin_date")
    dec["se"] = (dec["price_true"] - dec["price_pred"]) ** 2
    key = ["block", "origin_date", "h"]
    rw = dec[dec["label"] == "RW"].set_index(key)

    # D-26 Giacomini-White
    rows = []
    for h in C.HORIZONS:
        part = []
        for m in GW_MODELS:
            a = dec[(dec["label"] == m) & (dec["h"] == h)].set_index(key)
            b = rw.loc[a.index]
            Z = a[["trend", "vol"]].copy()
            Z = (Z - Z.mean()) / Z.std()
            for loss in ["regret", "se"]:
                stat, p, coef = gw_test((a[loss] - b[loss]).to_numpy(), Z, h)
                part.append({"h": h, "model": m, "loss": loss, "GW_wald": stat, "p_raw": p,
                             "coef_const": coef["const"], "coef_trend": coef["trend"], "coef_vol": coef["vol"]})
        for loss in ["regret", "se"]:
            sub = [r for r in part if r["loss"] == loss]
            for r, adj in zip(sub, holm([r["p_raw"] for r in sub])):
                r["p_holm"] = adj
        rows += part
    pd.DataFrame(rows).to_csv(OUT / "gw_conditional.csv", index=False)

    # mean regret differential vs RW by trend state (descriptive)
    cell = dec[dec["label"].isin(CANDIDATES)].groupby(["label", "trend_state", "h"], observed=True)["regret"].mean().unstack("label")
    cell.to_csv(OUT / "regret_by_trend_state.csv")

    # D-27 state-switching rule, out of sample
    order = ["B1", "B2", "B3", "B4", "B5", "B6"]
    cand = dec[dec["label"].isin(CANDIDATES)]
    picks, chosen_rows = [], []
    for k in range(1, len(order)):
        past = cand[cand["block"].isin(order[:k])]
        now = cand[cand["block"] == order[k]]
        for h in C.HORIZONS:
            for s in ["low", "mid", "high"]:
                p = past[(past["h"] == h) & (past["trend_state"] == s)]
                n_obs = p[p["label"] == "RW"].shape[0]
                if n_obs < 100:
                    best = "RW"
                else:
                    best = p.groupby("label")["regret"].mean().idxmin()
                picks.append({"block": order[k], "h": h, "trend_state": s, "chosen": best, "n_past": n_obs})
                sel = now[(now["h"] == h) & (now["trend_state"] == s) & (now["label"] == best)]
                chosen_rows.append(sel.assign(rule="switch"))
    sw = pd.concat(chosen_rows, ignore_index=True)
    pd.DataFrame(picks).to_csv(OUT / "switch_picks.csv", index=False)
    eval_blocks = order[1:]
    res = []
    for h in C.HORIZONS:
        base = {lab: cand[(cand["label"] == lab) & (cand["h"] == h) & (cand["block"].isin(eval_blocks))].set_index(key)
                for lab in ["RW", "RF"]}
        s = sw[sw["h"] == h].set_index(key)
        for name, ref in base.items():
            idx = s.index.intersection(ref.index).sort_values()     # D-58: time order (block, origin, h fixed)
            md, stt, p = dm_test(s.loc[idx, "regret"], ref.loc[idx, "regret"], h)
            res.append({"h": h, "compare": f"switch vs always-{name}", "n": len(idx),
                        "regret_switch": s.loc[idx, "regret"].mean(), "regret_ref": ref.loc[idx, "regret"].mean(),
                        "payoff_switch": s.loc[idx, "payoff"].mean(), "payoff_ref": ref.loc[idx, "payoff"].mean(),
                        "diff": md, "stat": stt, "p": p})
    pd.DataFrame(res).to_csv(OUT / "switch_rule.csv", index=False)

    # D-28 risk-averse probability rule, threshold from the sourced cost of each origin (D-37)
    sig = ewma_sigma()
    e2 = K.add_cost(ens.dropna(subset=["target_date"]), 1.0).join(sig, on="origin_date")
    rows = []
    for lab in ["RW", "RF", "XGB", "Chronos"]:
        sub = e2[e2["label"] == lab]
        for (blk, h), g in sub.groupby(["block", "h"]):
            cal = g[(g["split"] == "cal")].dropna(subset=["r_true"])
            tst = g[(g["split"] == "test")].dropna(subset=["r_true"]).copy()
            if tst.empty:
                continue
            z = ((cal["r_true"] - cal["r_pred"]) / (cal["sigma"] * np.sqrt(h))).to_numpy()
            thr = np.log1p(tst["cost"].to_numpy() / tst["price_origin"].to_numpy())
            need = (thr - tst["r_pred"].to_numpy()) / (tst["sigma"].to_numpy() * np.sqrt(h))
            prob = (z[None, :] > need[:, None]).mean(1)                 # P(waiting beats the cost)
            for ps in P_STARS:
                t = tst.assign(label=f"{lab}-p{int(ps * 100)}")
                t["p_wait"] = (prob > ps).astype(float)
                rows.append(K.evaluate_rule(t, "seller", score="p_wait").assign(base=lab, p_star=ps))
    ra = pd.concat(rows, ignore_index=True)
    out = []
    for (base, ps, h), g in ra.groupby(["base", "p_star", "h"]):
        s = K.summary(g)
        pay = g["payoff"].to_numpy()
        lo = np.percentile(pay[g["wait"].to_numpy()], 5) if g["wait"].any() else 0.0
        out.append({"model": base, "p_star": ps, "h": h, **s.to_dict(), "p5_payoff_when_wait": lo})
    pd.DataFrame(out).to_csv(OUT / "risk_averse_rule.csv", index=False)
    print("written gw_conditional, regret_by_trend_state, switch_picks, switch_rule, risk_averse_rule")


if __name__ == "__main__":
    main()
