"""Tables for the paper: accuracy, skill vs RW, DM/Holm, MCS, seeds, macro value, regimes, intervals.

Reads every file in results/forecasts. The primary forecast of a stochastic model is the
mean of its seed forecasts (seed ensemble). Pooled results use blocks B1..B5 (and B6 when present).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, intervals as IV                      # noqa: E402
from src.evaluate import metrics, with_skill                      # noqa: E402
from src.stats_tests import dm_test, holm, mcs                    # noqa: E402

OUT = C.RESULTS / "tables"
KEY = ["block", "split", "origin_date", "h"]


def load():
    fc = pd.concat([pd.read_csv(f, parse_dates=["origin_date", "target_date"])
                    for f in sorted(C.FORECASTS.glob("*.csv"))], ignore_index=True)
    fc["label"] = np.where(fc["variant"] == "full", fc["model"], fc["model"] + "-price")
    return fc


def ensemble(fc):
    g = fc.groupby(["label", "model", "variant"] + KEY, as_index=False)
    ens = g.agg(r_pred=("r_pred", "mean"), r_true=("r_true", "first"),
                price_origin=("price_origin", "first"), price_true=("price_true", "first"),
                 target_date=("target_date", "first"))
    ens["price_pred"] = ens["price_origin"] * np.exp(ens["r_pred"])
    return ens


def accuracy(test):
    by_block = test.groupby(["label", "block", "h"]).apply(metrics, include_groups=False).reset_index()
    pooled = test.groupby(["label", "h"]).apply(metrics, include_groups=False).reset_index()
    pooled["block"] = "pooled"
    tab = pd.concat([by_block, pooled]).rename(columns={"label": "model"})
    return with_skill(tab)


def dm_tables(test, labels):
    rows = []
    for h in C.HORIZONS:
        g = test[test["h"] == h]
        piv_ae = g.pivot_table(index=["block", "origin_date"], columns="label",
                               values="price_pred", aggfunc="first")
        y = g.groupby(["block", "origin_date"])["price_true"].first().loc[piv_ae.index]
        rw = piv_ae["RW"]
        for loss in ["SE", "AE"]:
            f = (lambda e: e ** 2) if loss == "SE" else np.abs
            for bw_name, bw in [("h-1", None), ("5", 5), ("auto", int(np.floor(4 * (len(y) / 100) ** (2 / 9))))]:
                part = []
                for lab in labels:
                    if lab == "RW":
                        continue
                    md, st, p = dm_test(f(y - piv_ae[lab]), f(y - rw), h, lags=bw)
                    part.append({"h": h, "loss": loss, "bandwidth": bw_name, "model": lab,
                                 "mean_diff_vs_RW": md, "DM_HLN": st, "p_raw": p})
                adj = holm([r["p_raw"] for r in part])
                for r, a in zip(part, adj):
                    r["p_holm"] = a
                rows += part
    return pd.DataFrame(rows)


def mcs_tables(test, labels):
    rows = []
    for h in C.HORIZONS:
        g = test[test["h"] == h]
        piv = g.pivot_table(index=["block", "origin_date"], columns="label", values="price_pred", aggfunc="first")
        y = g.groupby(["block", "origin_date"])["price_true"].first().loc[piv.index]
        for loss in ["SE", "AE"]:
            L = (piv[labels].sub(y, axis=0)) ** 2 if loss == "SE" else piv[labels].sub(y, axis=0).abs()
            # models with identical losses (e.g. SHRINK-m with w = 0 equals RW) make the MCS statistic
            # degenerate; test each distinct loss series once and copy the result to its duplicates
            twin = {}
            for c in L.columns:
                same = [u for u in dict.fromkeys(twin.values()) if np.allclose(L[c], L[u], equal_nan=True)]
                twin[c] = same[0] if same else c
            uniq = list(dict.fromkeys(twin.values()))
            r = mcs(L[uniq].reset_index(drop=True), alpha=C.MCS_ALPHA, block_size=max(10, h))
            r = r.reindex([twin[c] for c in L.columns])
            r.index = L.columns
            r["identical_to"] = [twin[c] if twin[c] != c else "" for c in L.columns]
            r["h"], r["loss"] = h, loss
            r = r.reset_index()
            rows.append(r.rename(columns={r.columns[0]: "model"}))
    return pd.concat(rows, ignore_index=True)


def seed_spread(fc):
    t = fc[fc["split"] == "test"].dropna(subset=["r_true"])
    m = t.groupby(["label", "seed", "h"]).apply(metrics, include_groups=False).reset_index()
    s = m.groupby(["label", "h"])["MAE"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    return m, s


def dm_per_seed(fc, test):
    """For each stochastic model and seed: is its pooled DM vs RW (SE, h=1..7) significant at 5%?"""
    rows = []
    t = fc[(fc["split"] == "test")].dropna(subset=["r_true"])
    rw = test[test["label"] == "RW"].set_index(["block", "origin_date", "h"])["price_pred"]
    for (lab, seed), g in t.groupby(["label", "seed"]):
        if g["seed"].nunique() == 0 or lab in ("RW", "Drift", "ARIMA", "SVR"):
            continue
        g = g.set_index(["block", "origin_date", "h"])
        for h in C.HORIZONS:
            gh = g.xs(h, level="h")
            r = rw.xs(h, level="h").loc[gh.index]
            md, st, p = dm_test((gh["price_true"] - gh["price_pred"]) ** 2, (gh["price_true"] - r) ** 2, h)
            rows.append({"model": lab, "seed": seed, "h": h, "mean_diff_vs_RW": md, "DM_HLN": st, "p_raw": p})
    return pd.DataFrame(rows)


def regimes(test):
    """Volatility terciles from 20-day realised volatility at the origin; cut-offs from data before B1."""
    from src.data import build_table
    tab = build_table()
    vol = tab["ret"].rolling(20).std()
    first_b1 = test.loc[test["block"] == "B1", "origin_date"].min()
    cuts = vol[vol.index < first_b1].quantile([1 / 3, 2 / 3]).to_numpy()
    lab = pd.cut(vol, [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "mid", "high"])
    t = test.join(lab.rename("regime"), on="origin_date")
    out = t.groupby(["label", "regime", "h"], observed=True).apply(metrics, include_groups=False).reset_index()
    out = out.rename(columns={"label": "model", "regime": "block"})
    return with_skill(out).rename(columns={"block": "regime"})


def ewma_sigma(lam=0.94):
    """RiskMetrics EWMA volatility of daily log returns; value at date i uses returns up to i."""
    from src.data import build_table
    ret = build_table()["ret"]
    var = (ret ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).rename("sigma")


DIAG_MODELS = ["ARIMA", "RF", "XGB", "GRU", "Chronos", "COMB", "SHRINK-COMB"]


def loss_diff_acf(test, labels):
    """ACF (lags 1-10) and Ljung-Box(10) of d_t = loss_model - loss_RW, per block, sorted by origin."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf
    rows = []
    rw = test[test["label"] == "RW"].set_index(["block", "origin_date", "h"])
    for lab in [m for m in DIAG_MODELS if m in labels]:
        g = test[test["label"] == lab].set_index(["block", "origin_date", "h"])
        for (blk, h), gb in g.groupby(level=["block", "h"]):
            gb = gb.sort_index()
            b = rw.loc[gb.index]
            e_m, e_r = gb["price_true"] - gb["price_pred"], b["price_true"] - b["price_pred"]
            for loss, f in [("SE", np.square), ("AE", np.abs)]:
                d = (f(e_m) - f(e_r)).to_numpy()
                if len(d) < 30:
                    continue
                a = acf(d, nlags=10, fft=True)
                lb = acorr_ljungbox(d, lags=[10], return_df=True)
                rows.append({"model": lab, "block": blk, "h": h, "loss": loss, "n": len(d),
                             **{f"acf{k}": a[k] for k in range(1, 11)},
                             "ljung_box10_p": float(lb["lb_pvalue"].iloc[0])})
    out = pd.DataFrame(rows)
    if len(out):
        cols = [f"acf{k}" for k in range(1, 11)] + ["ljung_box10_p"]
        mean = out.groupby(["model", "h", "loss"])[cols].mean().reset_index().assign(block="mean")
        out = pd.concat([out, mean], ignore_index=True)
    return out


def dm_by_block(test, labels):
    """DM-HLN vs RW within each block (HAC lags h-1); Holm within block x h x loss."""
    rows = []
    for (blk, h), g in test.groupby(["block", "h"]):
        piv = g.pivot_table(index="origin_date", columns="label", values="price_pred", aggfunc="first")
        y = g.groupby("origin_date")["price_true"].first().loc[piv.index]
        for loss, f in [("SE", np.square), ("AE", np.abs)]:
            part = []
            for lab in labels:
                if lab == "RW" or lab not in piv.columns:
                    continue
                md, st, p = dm_test(f(y - piv[lab]), f(y - piv["RW"]), h)
                part.append({"block": blk, "h": h, "loss": loss, "model": lab, "n": len(y),
                             "mean_diff_vs_RW": md, "DM_HLN": st, "p_raw": p})
            for r, a in zip(part, holm([r["p_raw"] for r in part])):
                r["p_holm"] = a
            rows += part
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc = load()
    ens = ensemble(fc)
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    labels = sorted(test["label"].unique(), key=lambda s: (s not in ("RW", "Drift", "ARIMA"), s))

    acc = accuracy(test)
    acc.to_csv(OUT / "accuracy.csv", index=False)
    dm = dm_tables(test, labels)
    dm.to_csv(OUT / "dm_vs_rw.csv", index=False)
    mcs_tables(test, labels).to_csv(OUT / "mcs.csv", index=False)
    per_seed, spread = seed_spread(fc)
    per_seed.to_csv(OUT / "seed_metrics.csv", index=False)
    spread.to_csv(OUT / "seed_spread.csv", index=False)
    dm_per_seed(fc, test).to_csv(OUT / "dm_per_seed.csv", index=False)
    loss_diff_acf(test, labels).to_csv(OUT / "loss_diff_acf.csv", index=False)
    dm_by_block(test, labels).to_csv(OUT / "dm_by_block.csv", index=False)
    regimes(test).to_csv(OUT / "regimes.csv", index=False)

    # macro value: full vs price-only for the same algorithm
    rows = []
    for algo in ["XGB", "GRU"]:
        if f"{algo}-price" not in labels:
            continue
        for h in C.HORIZONS:
            g = test[test["h"] == h]
            a = g[g["label"] == algo].set_index(["block", "origin_date"])
            b = g[g["label"] == f"{algo}-price"].set_index(["block", "origin_date"]).loc[a.index]
            md, st, p = dm_test((a["price_true"] - a["price_pred"]) ** 2, (b["price_true"] - b["price_pred"]) ** 2, h)
            rows.append({"algo": algo, "h": h, "mean_SE_full_minus_price": md, "DM_HLN": st, "p_raw": p})
    pd.DataFrame(rows).to_csv(OUT / "macro_value.csv", index=False)

    # v1 comparison: h = 1 on the v1 test window (B5)
    v1 = acc[(acc["block"] == "B5") & (acc["h"] == 1)].sort_values("MAE")
    v1.to_csv(OUT / "v1_window_h1.csv", index=False)

    # prediction intervals: RW, ARIMA and the learned model with the lowest calibration MAE
    cal = ens[(ens["split"] == "cal")].dropna(subset=["r_true"])
    cal_mae = cal.assign(ae=(cal["price_true"] - cal["price_pred"]).abs()).groupby("label")["ae"].mean()
    # COMB and SHRINK-* are not eligible as "best learned model by validation MAE"
    learned = [l for l in cal_mae.index if l not in ("RW", "Drift", "ARIMA", "COMB") and not l.startswith("SHRINK-")]
    chosen = cal_mae.loc[learned].idxmin() if learned else None
    iv_models = [m for m in ["RW", "ARIMA", chosen, "COMB"] if m and m in cal_mae.index]
    sigma = ewma_sigma()
    ivs = []
    for lab in iv_models:
        sub = ens[ens["label"] == lab].assign(model=lab).join(sigma, on="origin_date")
        ivs.append(IV.build(sub, C.INTERVAL_LEVELS))
    iv = pd.concat(ivs, ignore_index=True)
    iv.to_csv(OUT / "intervals_rows.csv", index=False)
    IV.evaluate(iv, ["model", "level", "h"]).to_csv(OUT / "intervals_pooled.csv", index=False)
    IV.evaluate(iv, ["model", "level", "block"]).to_csv(OUT / "intervals_by_block.csv", index=False)
    IV.evaluate(iv, ["model", "level", "block", "h"]).to_csv(OUT / "intervals_by_block_h.csv", index=False)
    (OUT / "README_intervals.md").write_text(
        "kupiec_p in intervals_*.csv: Kupiec assumes independent hits; for h > 1 overlapping forecasts "
        "induce dependence, so p-values are indicative only.\n")
    cal_mae.rename("cal_MAE").to_csv(OUT / "calibration_mae.csv")
    print("interval models:", iv_models)
    print("done:", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
