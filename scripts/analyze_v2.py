"""Sourced costs, risk, states, mechanisms and the firm-facing guide (DECISIONS D-29 .. D-34).

Uses saved forecasts only (results/forecasts). Outputs in results/tables/v2_*.csv.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, costs as K, intervals as IV          # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.stats_tests import dm_test, holm, kupiec                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, ewma_sigma, load                    # noqa: E402

OUT = C.RESULTS / "tables"
KEY = ["block", "origin_date", "h"]
BLOCKS = ["B1", "B2", "B3", "B4", "B5", "B6"]
# models with a predictive distribution (D-31). COMB added (D-42/D-46). SHRINK-* are deliberately NOT here:
# their cal rows are in-sample for w (D-43), so they must not calibrate a distribution or an interval.
DIST = ["RW", "ARIMA", "RF", "XGB", "GRU", "Chronos", "COMB"]
LAMBDAS = [0.0, 0.1, 0.25, 0.5]
STATE_VARS = ["trend", "vol", "volratio"]
MIN_OBS = 20


# ------------------------------------------------------------------ helpers
def present(labels, wanted):
    return [m for m in wanted if m in labels]


def states(first_b1):
    t = build_table()
    sig = ewma_sigma()
    out = pd.DataFrame({"trend": t["logp"] - t["logp"].shift(60), "vol": sig,
                        "volratio": sig / t["ret"].rolling(250).std()})
    pre = out.index < first_b1
    for c in STATE_VARS:
        cuts = out.loc[pre, c].quantile([1 / 3, 2 / 3]).to_numpy()
        out[c + "_state"] = pd.cut(out[c], [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"]).astype(str)
    return out


def write(df, name):
    df.to_csv(OUT / f"v2_{name}.csv", index=False)
    print("written", f"v2_{name}.csv", len(df), "rows", flush=True)


def dm_vs(a, b, h):
    if len(a) < 10:
        return np.nan, np.nan, np.nan
    return dm_test(np.asarray(a), np.asarray(b), h)


# ------------------------------------------------------------------ D-29, D-30: decisions with sourced costs
def decisions(test, labels):
    rows, dm_rows = [], []
    base = {}
    for sm_ in K.STORAGE_MULT:
        priced = K.add_cost(test, sm_)
        for side in ["seller", "buyer"]:
            for b in K.BASIS_SD:
                if (sm_ != 1.0 or side == "buyer") and b != 0:      # basis grid only around the primary case
                    continue
                ev = {lab: K.evaluate(priced[priced["label"] == lab], side, basis_sd=b) for lab in labels}
                if sm_ == 1.0 and side == "seller" and b == 0:
                    base = ev
                for lab, d in ev.items():
                    for (blk, h), g in d.groupby(["block", "h"]):
                        rows.append({"model": lab, "block": blk, "h": h, "storage_mult": sm_, "side": side,
                                     "basis_sd": b, **K.summary(g).to_dict()})
                    for h, g in d.groupby("h"):
                        rows.append({"model": lab, "block": "pooled", "h": h, "storage_mult": sm_, "side": side,
                                     "basis_sd": b, **K.summary(g).to_dict()})
                # regret DM vs RW, Holm within horizon
                rw = ev["RW"].set_index(KEY)
                for h in C.HORIZONS:
                    part = []
                    for lab in labels:
                        if lab == "RW":
                            continue
                        a = ev[lab][ev[lab]["h"] == h].set_index(KEY)
                        r = rw.loc[a.index]
                        for loss in ["regret", "payoff"]:
                            md, st, p = dm_vs(a[loss], r[loss], h)
                            part.append({"model": lab, "h": h, "storage_mult": sm_, "side": side, "basis_sd": b,
                                         "loss": loss, "mean_diff_vs_RW": md, "DM_HLN": st, "p_raw": p})
                    for loss in ["regret", "payoff"]:
                        sub = [x for x in part if x["loss"] == loss]
                        for x, adj in zip(sub, holm([x["p_raw"] for x in sub])):
                            x["p_holm"] = adj
                    dm_rows += part
    write(pd.DataFrame(rows), "decisions")
    write(pd.DataFrame(dm_rows), "decisions_dm")
    return base


# ------------------------------------------------------------------ D-31: predictive distribution of the waiting payoff
def risk(ens, sigma, labels):
    models = present(labels, DIST)
    e2 = ens[ens["label"].isin(models)].join(sigma, on="origin_date")
    e2 = K.add_cost(e2, 1.0)
    per_origin = []
    for (lab, blk, h), g in e2.groupby(["label", "block", "h"]):
        cal = g[g["split"] == "cal"].dropna(subset=["r_true"])
        tst = g[g["split"] == "test"].dropna(subset=["r_true"]).copy()
        if tst.empty or len(cal) < 20:
            continue
        z = ((cal["r_true"] - cal["r_pred"]) / (cal["sigma"] * np.sqrt(h))).to_numpy()
        r = tst["r_pred"].to_numpy()[:, None] + z[None, :] * (tst["sigma"].to_numpy()[:, None] * np.sqrt(h))
        pi = tst["price_origin"].to_numpy()[:, None] * (np.exp(r) - 1) - tst["cost"].to_numpy()[:, None]
        q5 = np.quantile(pi, 0.05, axis=1)
        cvar_loss = -np.array([row[row <= q].mean() for row, q in zip(pi, q5)])
        tst["E_pi"] = pi.mean(1)
        tst["VaR05"] = q5
        tst["CVaR05_loss"] = cvar_loss
        per_origin.append(tst)
    po = pd.concat(per_origin, ignore_index=True)
    po["pi_true"] = K.payoff_wait(po)
    po["exceed"] = po["pi_true"] < po["VaR05"]

    # reliability of the 5% VaR of the waiting payoff
    rel = []
    for keys, g in list(po.groupby(["label", "h"])) + list(po.groupby(["label", "block"])):
        hit = (~g["exceed"]).astype(float)
        lr, p = kupiec(hit, 0.95)
        lab, k2 = keys
        rel.append({"model": lab, "by": "h" if isinstance(k2, (int, np.integer)) else "block", "key": k2,
                    "n": len(g), "exceed_rate": g["exceed"].mean(), "kupiec_p": p})
    write(pd.DataFrame(rel), "var_reliability")

    # mean-CVaR rule
    rows, dm_rows = [], []
    evs = {}
    for lam in LAMBDAS:
        for lab, g in po.groupby("label"):
            wait = (g["E_pi"] - lam * g["CVaR05_loss"]) > 0
            evs[(lab, lam)] = K.evaluate(g, "seller", wait=wait.to_numpy())
    point = {lab: K.evaluate(po[po["label"] == lab], "seller") for lab in po["label"].unique()}
    for (lab, lam), d in evs.items():
        for h, g in d.groupby("h"):
            rows.append({"model": lab, "rule": f"meanCVaR_l{lam}", "lambda": lam, "h": h, **K.summary(g).to_dict()})
    for lab, d in point.items():
        for h, g in d.groupby("h"):
            rows.append({"model": lab, "rule": "point", "lambda": np.nan, "h": h, **K.summary(g).to_dict()})
    write(pd.DataFrame(rows), "mean_cvar")
    # each mean-CVaR rule vs the same model's point rule (regret and payoff)
    for (lab, lam), d in evs.items():
        p0 = point[lab].set_index(KEY)
        d = d.set_index(KEY)
        for h in C.HORIZONS:
            a = d.xs(h, level="h", drop_level=False)
            b = p0.loc[a.index]
            for loss in ["regret", "payoff"]:
                md, st, p = dm_vs(a[loss], b[loss], h)
                dm_rows.append({"model": lab, "lambda": lam, "h": h, "loss": loss,
                                "mean_diff_vs_point": md, "DM_HLN": st, "p_raw": p})
    write(pd.DataFrame(dm_rows), "mean_cvar_dm")
    return po


# ------------------------------------------------------------------ D-32: five indicators by state
def run_costs(labels):
    """Compute cost per model. D-54: if the re-measured table exists (main run, scripts/time_models.py) it is
    used for every run; otherwise fall back to wall seconds per 1,000 origins from the estimation logs."""
    f = C.MAIN_RESULTS / "tables" / "v3_compute_cost.csv"
    if f.exists():
        t = pd.read_csv(f)
        t["source"] = "time_models (D-54)"
        return t[t["model"].isin(labels)]
    rows = []
    logs = list((C.ROOT / "logs").glob("*.log")) + list((C.RESULTS / "logs").glob("*.log"))
    n_per_block = 550                                                 # 250 validation + ~300 test origins
    pat = re.compile(r"^(SVR|RF|XGB|DNN|LSTM|GRU)\s+(full|price)\s+seed\s+\d+\s+B\d\s+([\d.]+)s")
    acc = {}
    for f in logs:
        for line in f.read_text(errors="ignore").splitlines():
            m = pat.match(line.strip())
            if m:
                lab = m.group(1) if m.group(2) == "full" else m.group(1) + "-price"
                acc.setdefault(lab, []).append(float(m.group(3)))
    for lab, secs in acc.items():
        rows.append({"model": lab, "sec_per_1000_origins": 1000 * np.mean(secs) / n_per_block, "source": "logs"})
    tf = C.RESULTS / "chronos_timing.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        rows.append({"model": "Chronos", "sec_per_1000_origins": 1000 * t["seconds"].sum() / t["n_origins"].sum(),
                     "source": "chronos_timing"})
    # benchmarks: time them directly on one block
    from src import models as M
    from src.folds import make_blocks
    tab = build_table()
    lp = tab["logp"].to_numpy()
    b = make_blocks(tab)[0]
    t0 = time.time()
    _, res = M.arima_select(lp[: b.fit[-1] + C.GAP + 1])
    M.arima_forecasts(res, lp, b.origins)
    rows.append({"model": "ARIMA", "sec_per_1000_origins": 1000 * (time.time() - t0) / len(b.origins), "source": "timed"})
    rows += [{"model": m, "sec_per_1000_origins": 0.0, "source": "closed form"} for m in ["RW", "Drift"]]
    out = pd.DataFrame(rows)
    return out[out["model"].isin(labels)]


def indicators(fc, ens, base, st, sigma, labels):
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    d = pd.concat(base.values(), ignore_index=True).join(st, on="origin_date")
    d["ae"] = (d["price_true"] - d["price_pred"]).abs()
    rows = []
    # uncertainty: 90% intervals for the models with a distribution
    ivs = []
    for lab in present(labels, DIST):
        sub = ens[ens["label"] == lab].assign(model=lab).join(sigma, on="origin_date")
        ivs.append(IV.build(sub, [0.90]))
    iv = pd.concat(ivs, ignore_index=True).join(st, on="origin_date")
    for m in ["vol", "conf"]:
        iv[f"{m}_hit"] = ((iv["price_true"] >= iv[f"{m}_lo"]) & (iv["price_true"] <= iv[f"{m}_hi"])).astype(float)
        iv[f"{m}_width"] = iv[f"{m}_hi"] - iv[f"{m}_lo"]
    # stability: per-seed MAE and regret
    t = fc[(fc["split"] == "test")].dropna(subset=["r_true"])
    t = t[t["label"].isin(labels)]
    ps = K.evaluate(K.add_cost(t, 1.0), "seller").join(st, on="origin_date")
    ps["ae"] = (ps["price_true"] - ps["price_pred"]).abs()
    for sv in STATE_VARS + ["all"]:
        col = f"{sv}_state"
        groups = d.assign(all_state="all") if sv == "all" else d
        ivg = iv.assign(all_state="all") if sv == "all" else iv
        psg = ps.assign(all_state="all") if sv == "all" else ps
        rw = groups[groups["label"] == "RW"].groupby([col, "h"])["ae"].mean()
        a = groups.groupby(["label", col, "h"]).agg(n=("ae", "size"), MAE=("ae", "mean"),
                                                     regret=("regret", "mean"), payoff=("payoff", "mean"))
        a["rMAE"] = a["MAE"] / rw.reindex(a.index.droplevel(0)).to_numpy()
        u = ivg.groupby(["model", col, "h"]).agg(cov90_vol=("vol_hit", "mean"), cov90_conf=("conf_hit", "mean"),
                                                  width90_vol=("vol_width", "mean"))
        u.index = u.index.set_names(["label", col, "h"])
        s = psg.groupby(["label", "seed", col, "h"]).agg(ae=("ae", "mean"), rg=("regret", "mean"))
        s = s.groupby(["label", col, "h"]).agg(seed_sd_MAE=("ae", "std"), seed_sd_regret=("rg", "std")).fillna(0.0)
        tab = a.join(u, how="left").join(s, how="left").reset_index().rename(columns={"label": "model", col: "state"})
        tab.insert(1, "state_var", sv)
        rows.append(tab)
    out = pd.concat(rows, ignore_index=True)
    cost = run_costs(labels)
    cols = [c for c in ["predict_ms_per_origin", "fit_seconds", "sec_per_1000_origins"] if c in cost.columns]
    out = out.merge(cost[["model"] + cols], on="model", how="left")
    write(out, "indicators_by_state")
    write(cost, "compute_cost")
    return d, iv


# ------------------------------------------------------------------ D-33: guide with evidence grades
def grade(p_holm, n_consistent, b6_same):
    if p_holm < 0.05 and n_consistent >= 4 and b6_same:
        return "Strong"
    if n_consistent >= 4:
        return "Moderate"
    return "Weak"


def guide(d, iv, labels):
    cells = []
    cands = [m for m in labels if m != "RW"]
    for sv in STATE_VARS + ["all"]:
        col = f"{sv}_state"
        dd = d.assign(all_state="all") if sv == "all" else d
        for state in sorted(dd[col].dropna().unique()):
            if state == "nan":
                continue
            ds = dd[dd[col] == state]
            for h in C.HORIZONS:
                dh = ds[ds["h"] == h]
                rw = dh[dh["label"] == "RW"].set_index(KEY)
                for loss in ["ae", "regret"]:
                    part = []
                    for m in cands:
                        a = dh[dh["label"] == m].set_index(KEY)
                        idx = a.index.intersection(rw.index).sort_values()   # D-58 addendum: time order
                        if len(idx) < MIN_OBS:
                            continue
                        diff = a.loc[idx, loss] - rw.loc[idx, loss]
                        md, stt, p = dm_vs(a.loc[idx, loss], rw.loc[idx, loss], h)
                        by_blk = diff.groupby(level="block").agg(["mean", "size"])
                        by_blk = by_blk[by_blk["size"] >= MIN_OBS]
                        sign = np.sign(md)
                        n_cons = int((np.sign(by_blk["mean"]) == sign).sum())
                        b6 = bool("B6" in by_blk.index and np.sign(by_blk.loc["B6", "mean"]) == sign)
                        part.append({"state_var": sv, "state": state, "h": h, "loss": loss, "model": m,
                                     "n": len(idx), "mean_model": a.loc[idx, loss].mean(), "mean_RW": rw.loc[idx, loss].mean(),
                                     "mean_diff": md, "DM_HLN": stt, "p_raw": p, "n_blocks": len(by_blk),
                                     "n_consistent": n_cons, "B6_same_sign": b6})
                    for x, adj in zip(part, holm([x["p_raw"] for x in part])):
                        x["p_holm"] = adj
                        x["grade"] = grade(adj, x["n_consistent"], x["B6_same_sign"])
                        x["better_than_RW"] = x["mean_diff"] < 0
                    cells += part
    cells = pd.DataFrame(cells)
    write(cells, "guide_cells")

    # one recommendation per state x h x indicator
    rec = []
    for (sv, state, h, loss), g in cells.groupby(["state_var", "state", "h", "loss"]):
        ok = g[g["better_than_RW"] & (g["grade"] == "Strong")]          # D-40: only Strong cells recommend
        if ok.empty:
            mod = g[g["better_than_RW"] & (g["grade"] == "Moderate")].sort_values("mean_diff")
            note = "; ".join(f"{r['model']} {r['mean_diff']:+.2f} (p_holm {r['p_holm']:.3f})" for _, r in mod.iterrows())
            rec.append({"state_var": sv, "state": state, "h": h, "indicator": "accuracy" if loss == "ae" else "decision",
                        "recommend": "RW", "grade": "default",
                        "note": f"Moderate (sign-consistent, not significant): {note}" if note else "none"})
        else:
            best = ok.sort_values("mean_model").iloc[0]
            rec.append({"state_var": sv, "state": state, "h": h, "indicator": "accuracy" if loss == "ae" else "decision",
                        "recommend": best["model"], "grade": best["grade"],
                        "note": f"diff vs RW {best['mean_diff']:+.2f} USD/t, Holm p {best['p_holm']:.3f}, "
                                f"{best['n_consistent']}/{best['n_blocks']} blocks, B6 same sign: {best['B6_same_sign']}"})
    # interval method by state (base model RW), uncertainty indicator
    for sv in STATE_VARS + ["all"]:
        col = f"{sv}_state"
        ivs = iv[iv["model"] == "RW"]
        ivs = ivs.assign(all_state="all") if sv == "all" else ivs
        for (state, h), g in ivs.groupby([col, "h"]):
            if state == "nan":
                continue
            best, best_err, info = None, np.inf, ""
            for m in ["vol", "conf", "heur"]:
                hit = ((g["price_true"] >= g[f"{m}_lo"]) & (g["price_true"] <= g[f"{m}_hi"])).astype(float)
                err = abs(hit.mean() - 0.90)
                okb = 0
                b6 = False
                for blk, gb in hit.groupby(g["block"]):
                    if len(gb) < MIN_OBS:
                        continue
                    pk = kupiec(gb, 0.90)[1]
                    okb += pk > 0.05
                    b6 = b6 or (blk == "B6" and pk > 0.05)
                if err < best_err:
                    best, best_err = m, err
                    gr = "Strong" if okb >= 4 and b6 else ("Moderate" if okb >= 4 else "Weak")
                    info = f"coverage {hit.mean():.3f}; Kupiec p>0.05 in {okb} blocks; B6 ok: {b6}"
            rec.append({"state_var": sv, "state": state, "h": h, "indicator": "uncertainty",
                        "recommend": {"vol": "vol-scaled conformal", "conf": "static conformal", "heur": "sqrt(h) rule"}[best],
                        "grade": gr, "note": info})
    write(pd.DataFrame(rec), "guide")


# ------------------------------------------------------------------ D-34: mechanisms
def mechanisms(test, st, first_b1):
    t = build_table()
    ret = t["ret"]
    rows = []
    # (a) time-series momentum: r(t,h) on the 60-day trend
    trend = st["trend"]
    for h in C.HORIZONS:
        fwd = t["logp"].shift(-h) - t["logp"]
        df = pd.DataFrame({"y": fwd, "x": trend}).dropna()
        pre = df[df.index.to_numpy() < np.datetime64(first_b1)]
        pre = pre.iloc[: max(len(pre) - h, 0)]                       # targets must fall before B1
        tst = df.loc[df.index.isin(test["origin_date"].unique())]
        for name, s in [("pre_B1", pre), ("test_blocks", tst)]:
            r = sm.OLS(s["y"], sm.add_constant(s["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 1)})
            rows.append({"test": "momentum_regression", "period": name, "h": h, "n": len(s),
                         "coef": r.params["x"], "t": r.tvalues["x"], "p": r.pvalues["x"], "r2": r.rsquared})
    # (b) autocorrelation of daily returns by trend tercile (state at day t, return pairs t, t+k)
    s_state = st["trend_state"]
    for name, mask in [("pre_B1", ret.index < first_b1), ("test_blocks", ret.index >= first_b1)]:
        for state in ["low", "mid", "high"]:
            for k in range(1, 6):
                x = ret.where(mask & (s_state == state))
                y = ret.shift(-k)
                pair = pd.concat([x, y], axis=1).dropna()
                rows.append({"test": "autocorr_by_trend_state", "period": name, "state": state, "lag": k,
                             "n": len(pair), "coef": pair.iloc[:, 0].corr(pair.iloc[:, 1])})
    # Ljung-Box(10) and ARCH-LM(5) per period
    periods = [("pre_B1", ret[ret.index < first_b1].iloc[1:])]
    for blk in BLOCKS:
        dts = test.loc[test["block"] == blk, "origin_date"]
        if len(dts):
            periods.append((blk, ret[(ret.index >= dts.min()) & (ret.index <= dts.max())]))
    for name, r in periods:
        lb = acorr_ljungbox(r, lags=[10], return_df=True)
        lm, lm_p, _, _ = het_arch(r, nlags=5)
        rows.append({"test": "ljung_box_10", "period": name, "n": len(r), "coef": float(lb["lb_stat"].iloc[0]),
                     "p": float(lb["lb_pvalue"].iloc[0])})
        rows.append({"test": "arch_lm_5", "period": name, "n": len(r), "coef": lm, "p": lm_p})
    # (d) are tree models trend followers?
    tt = test.join(st[["trend"]], on="origin_date")
    for m in ["RF", "XGB", "GRU", "Chronos"]:
        g = tt[tt["label"] == m]
        if g.empty:
            continue
        for (blk, h), gg in g.groupby(["block", "h"]):
            rows.append({"test": "forecast_trend_corr", "model": m, "period": blk, "h": h, "n": len(gg),
                         "coef": gg["r_pred"].corr(gg["trend"])})
    write(pd.DataFrame(rows), "mechanisms")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc = load()
    ens = ensemble(fc)
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    labels = sorted(test["label"].unique())
    first_b1 = test.loc[test["block"] == "B1", "origin_date"].min()
    st = states(first_b1)
    sigma = ewma_sigma()
    base = decisions(test, labels)
    risk(ens, sigma, labels)
    d, iv = indicators(fc, ens, base, st, sigma, labels)
    guide(d, iv, labels)
    mechanisms(test, st, first_b1)


if __name__ == "__main__":
    main()
