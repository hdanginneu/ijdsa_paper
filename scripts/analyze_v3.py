"""Round-3 additional analyses (DECISIONS D-47 .. D-53). Saved forecasts only; tables in <results>/tables/v3_*.csv.

C1 Clark-West vs RW (D-47)            -> v3_clark_west.csv
C2 MAE ratio CI + equivalence (D-48)  -> v3_mae_ratio_ci.csv, v3_equivalence_summary.csv
C3 roll robustness (D-49)             -> v3_roll_robustness.csv
C4 Chronos on B5-B6 (D-50)            -> v3_chronos_b56.csv
C5 economic magnitude (D-51)          -> v3_economic_magnitude.csv, v3_regret_leave_one_block_out.csv
C7 Christoffersen h = 1 (D-53)        -> v3_christoffersen_h1.csv
Parts that do not apply to a run (e.g. no Chronos) are skipped and listed in v3_notes.md.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, costs as K                           # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.stats_tests import christoffersen, clark_west, dm_test, holm   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import ensemble, ewma_sigma, load                    # noqa: E402

OUT = C.RESULTS / "tables"
MAIN_SET = ["ARIMA", "RF", "XGB", "GRU", "Chronos", "COMB", "SHRINK-COMB"]
BLOCK_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6"]
B_REPS, BOOT_SEED = 2000, 2026
LOT_T = 10
BUDGETS = [0.10, 0.25, 0.50, 1.00]
NOTES = []


# ------------------------------------------------------------------ helpers
def ordered(df):
    """Sort by block (B1..B6) then origin date: the time order used by every HAC variance."""
    rank = {b: i for i, b in enumerate(BLOCK_ORDER)}
    return df.assign(_b=df["block"].map(rank)).sort_values(["_b", "origin_date"]).drop(columns="_b")


def paired(test, lab, h, ref="RW"):
    """Rows of model `lab` and the reference at horizon h on common origins, both in time order."""
    a = ordered(test[(test["label"] == lab) & (test["h"] == h)]).set_index(["block", "origin_date"])
    b = test[(test["label"] == ref) & (test["h"] == h)].set_index(["block", "origin_date"])
    idx = a.index.intersection(b.index, sort=False)
    return a.loc[idx], b.loc[idx]


def regret(dec, lab, h, idx):
    """Regret (sourced cost, seller) of model `lab` at horizon h for (block, origin_date) pairs idx."""
    return dec[lab].xs(h, level="h").loc[idx, "regret"].to_numpy()


def mean_block_len(h):
    return 10 if h <= 3 else 20


def stationary_indices(sizes, L, rng):
    """Politis-Romano stationary bootstrap, resampled within each block (circular), blocks kept in place.

    sizes: number of origins per block, in order. Returns an index array of shape (B_REPS, sum(sizes))."""
    parts, start = [], 0
    for m in sizes:
        new = rng.random((B_REPS, m)) < 1.0 / L
        new[:, 0] = True
        starts = rng.integers(0, m, size=(B_REPS, m))
        pos = np.arange(m)
        last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        first_val = np.take_along_axis(starts, last, axis=1)
        parts.append(start + (first_val + (pos - last)) % m)
        start += m
    return np.concatenate(parts, axis=1)


def boot_idx(a, h):
    sizes = a.groupby(level="block", sort=False).size().to_numpy()
    return stationary_indices(sizes, mean_block_len(h), np.random.default_rng(BOOT_SEED))


def sets(labels):
    main = [m for m in MAIN_SET if m in labels]
    return [("main", main), ("all", [m for m in labels if m != "RW"])]


# ------------------------------------------------------------------ C1
def c1_clark_west(test, labels):
    rows = []
    for set_name, models in sets(labels):
        for h in C.HORIZONS:
            part = []
            for m in models:
                a, b = paired(test, m, h)
                mf, t, p = clark_west(a["price_true"], b["price_pred"], a["price_pred"], h)
                part.append({"set": set_name, "model": m, "h": h, "n": len(a), "mean_f": mf, "CW_t": t, "p_one_sided": p})
            for r, adj in zip(part, holm([r["p_one_sided"] for r in part])):
                r["p_holm"] = adj
            rows += part
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "v3_clark_west.csv", index=False)
    return out


# ------------------------------------------------------------------ C2
def c2_mae_ratio(test, labels):
    rows = []
    for set_name, models in sets(labels):
        for m in (["RW"] if set_name == "all" else []) + models:
            for h in C.HORIZONS:
                a, b = paired(test, m, h)
                ea, eb = (a["price_true"] - a["price_pred"]).abs().to_numpy(), (b["price_true"] - b["price_pred"]).abs().to_numpy()
                idx = boot_idx(a, h)
                boot = ea[idx].mean(1) / eb[idx].mean(1)
                ratio = ea.mean() / eb.mean()
                lo90, hi90, lo95, hi95 = np.percentile(boot, [5, 95, 2.5, 97.5])
                lo90, lo95 = min(lo90, ratio), min(lo95, ratio)          # guard: point estimate inside the CI
                hi90, hi95 = max(hi90, ratio), max(hi95, ratio)
                rows.append({"set": set_name, "model": m, "h": h, "n": len(a), "ratio": ratio,
                             "ci90_lo": lo90, "ci90_hi": hi90, "ci95_lo": lo95, "ci95_hi": hi95,
                             "equiv_1pct": bool(lo90 >= 0.99 and hi90 <= 1.01),
                             "equiv_2pct": bool(lo90 >= 0.98 and hi90 <= 1.02),
                             "worse_1pct": bool(lo90 > 1.01), "better_1pct": bool(hi90 < 0.99)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "v3_mae_ratio_ci.csv", index=False)
    cat = np.select([out["equiv_1pct"], out["better_1pct"], out["worse_1pct"]],
                    ["equivalent", "better", "worse"], "inconclusive")
    summ = (out.assign(category=cat).groupby(["set", "model", "category"]).size()
            .unstack(fill_value=0).reset_index())
    for c in ["equivalent", "inconclusive", "worse", "better"]:
        if c not in summ.columns:
            summ[c] = 0
    summ[["set", "model", "equivalent", "inconclusive", "worse", "better"]].to_csv(
        OUT / "v3_equivalence_summary.csv", index=False)
    return out


# ------------------------------------------------------------------ C3
def roll_file():
    name = "arabica" if C.PRICE_UNIT == "cents_lb" else "robusta"
    return C.MAIN_RESULTS / "tables" / f"roll_calendar_{name}.csv"


def excluded_dates(cal, kind, version):
    c = cal[cal["anchor_type"] == kind]
    if version == "flagged":
        return set(pd.to_datetime(c["flagged_date"].dropna()))
    out = set()
    for s, e in zip(pd.to_datetime(c["window_start"]), pd.to_datetime(c["window_end"])):
        out |= set(pd.bdate_range(s, e))                 # calendar business days cover every trading day in range
    return out


def c3_roll(test, labels, dec):
    f = roll_file()
    if not f.exists():
        NOTES.append(f"C3 step 3 skipped: {f.name} not found.")
        return None
    cal = pd.read_csv(f)
    dates = build_table().index
    models = [m for m in MAIN_SET if m in labels]
    rows = []
    for kind, suffix in [("fnd", ""), ("ltd", "_ltd")]:
        for version in ["flagged", "window"]:
            ex = np.isin(dates, list(excluded_dates(cal, kind, version))).astype(int)
            cs = np.concatenate([[0], np.cumsum(ex)])
            for h in C.HORIZONS:
                part = []
                for m in models:
                    a, b = paired(test, m, h)
                    pos = dates.get_indexer(pd.DatetimeIndex(a.index.get_level_values("origin_date")))
                    keep = (cs[np.minimum(pos + h, len(dates) - 1) + 1] - cs[pos + 1]) == 0   # no excluded day in (t, t+h]
                    ak, bk = a[keep], b[keep]
                    ea, eb = ak["price_true"] - ak["price_pred"], bk["price_true"] - bk["price_pred"]
                    _, _, p_dm = dm_test(ea ** 2, eb ** 2, h)
                    _, _, p_cw = clark_west(ak["price_true"], bk["price_pred"], ak["price_pred"], h)
                    md, _, p_rg = dm_test(regret(dec, m, h, ak.index), regret(dec, "RW", h, bk.index), h)
                    part.append({"version": version + suffix, "model": m, "h": h, "n_kept": int(keep.sum()),
                                 "share_dropped": 1 - keep.mean(), "rMAE": ea.abs().mean() / eb.abs().mean(),
                                 "dm_p": p_dm, "cw_p": p_cw, "regret_diff_vs_RW": md, "regret_p": p_rg})
                for key in ["dm_p", "cw_p", "regret_p"]:
                    for r, adj in zip(part, holm([r[key] for r in part])):
                        r[key + "_holm"] = adj
                rows += part
    out = pd.DataFrame(rows).drop(columns=["dm_p", "cw_p", "regret_p"])
    out.to_csv(OUT / "v3_roll_robustness.csv", index=False)
    NOTES.append("C3: evaluation-only exclusion; training data still contain the roll days (contract-level prices "
                 "are not available, so no back-adjusted series). Versions *_ltd use the supplementary LTD anchor.")
    return out


# ------------------------------------------------------------------ C4
def c4_chronos(ens, test, labels, dec):
    if "Chronos" not in labels:
        NOTES.append("C4 skipped: no Chronos forecasts in this run.")
        return None
    t56 = test[test["block"].isin(["B5", "B6"])]
    models = [m for m in ["Chronos", "SHRINK-Chronos"] if m in labels]
    rows = []
    for h in C.HORIZONS:
        part = []
        for m in models:
            a, b = paired(t56, m, h)
            ea, eb = a["price_true"] - a["price_pred"], b["price_true"] - b["price_pred"]
            _, _, p_se = dm_test(ea ** 2, eb ** 2, h)
            _, _, p_ae = dm_test(ea.abs(), eb.abs(), h)
            _, t_cw, p_cw = clark_west(a["price_true"], b["price_pred"], a["price_pred"], h)
            md, _, p_rg = dm_test(regret(dec, m, h, a.index), regret(dec, "RW", h, b.index), h)
            part.append({"model": m, "h": h, "n": len(a), "rMAE": ea.abs().mean() / eb.abs().mean(),
                         "dm_se_p": p_se, "dm_ae_p": p_ae, "cw_t": t_cw, "cw_p": p_cw,
                         "regret_diff_vs_RW": md, "regret_p": p_rg})
        for key in ["dm_se_p", "dm_ae_p", "cw_p", "regret_p"]:
            for r, adj in zip(part, holm([r[key] for r in part])):
                r[key + "_holm"] = adj
        rows += part
    # RQ4 routing RW -> Chronos on B5-B6 (same rule as analyze_decisions.rq4)
    sigma = ewma_sigma()
    sub = ens[ens["label"].isin(["RW", "Chronos"]) & ens["block"].isin(["B5", "B6"])].join(sigma, on="origin_date")
    routed = []
    for blk, gb in sub.groupby("block"):
        cal_sig = gb[(gb["split"] == "cal") & (gb["label"] == "RW") & (gb["h"] == 1)]["sigma"]
        tst = gb[gb["split"] == "test"].dropna(subset=["r_true"])
        rw = tst[tst["label"] == "RW"].set_index(["origin_date", "h"])
        ex = tst[tst["label"] == "Chronos"].set_index(["origin_date", "h"]).loc[rw.index]
        for bud in BUDGETS:
            tau = -np.inf if bud >= 1 else cal_sig.quantile(1 - bud)
            use = rw["sigma"] > tau
            r = rw.copy()
            r.loc[use, "r_pred"] = ex.loc[use, "r_pred"]
            r["price_pred"] = r["price_origin"] * np.exp(r["r_pred"])
            routed.append(K.evaluate_rule(r.reset_index().assign(block=blk), "seller").assign(budget=bud))
    routed = pd.concat(routed, ignore_index=True)
    for h in C.HORIZONS:
        part = []
        for bud, g in routed[routed["h"] == h].groupby("budget"):
            a = ordered(g).set_index(["block", "origin_date"])
            b = dec["RW"].xs(h, level="h").loc[a.index]                  # always-RW rule on the same origins
            ea, eb = a["price_true"] - a["price_pred"], b["price_true"] - b["price_pred"]
            _, _, p_ae = dm_test(ea.abs(), eb.abs(), h)
            md, _, p_rg = dm_test(a["regret"], b["regret"], h)
            part.append({"model": f"route-Chronos-{int(bud * 100)}", "h": h, "n": len(a),
                         "rMAE": ea.abs().mean() / eb.abs().mean(), "dm_ae_p": p_ae,
                         "regret_diff_vs_RW": md, "regret_p": p_rg})
        for key in ["dm_ae_p", "regret_p"]:
            for r, adj in zip(part, holm([r[key] for r in part])):
                r[key + "_holm"] = adj
        rows += part
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "v3_chronos_b56.csv", index=False)
    return out


# ------------------------------------------------------------------ C5
def c5_economic(test, labels, dec):
    models = [m for m in MAIN_SET + ["RF", "SHRINK-RF"] if m in labels]
    models = list(dict.fromkeys(models))
    unit_note = ("Arabica converted to USD/t; the 10-t lot is a comparison convention, not the ICE contract size"
                 if C.PRICE_UNIT == "cents_lb" else "ICE Robusta, 10-t contract")
    rows, lobo = [], []
    for m in models:
        for h in C.HORIZONS:
            a, b = paired(test, m, h)
            d = regret(dec, m, h, a.index) - regret(dec, "RW", h, b.index)
            blocks = a.index.get_level_values("block")
            idx = boot_idx(a, h)
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            mean_price = a["price_origin"].mean()
            by_blk = pd.Series(d, index=blocks).groupby(level=0, sort=False).mean()
            rows.append({"model": m, "h": h, "n": len(d), "regret_diff_usd_t": d.mean(),
                         "ci95_lo_usd_t": lo, "ci95_hi_usd_t": hi,
                         "regret_diff_pct_of_price": 100 * d.mean() / mean_price,
                         "regret_diff_usd_per_10t_lot": LOT_T * d.mean(), "median_diff_usd_t": np.median(d),
                         "n_blocks": len(by_blk), "n_blocks_lower_regret": int((by_blk < 0).sum()),
                         "mean_origin_price_usd_t": mean_price, "unit_note": unit_note})
            for blk in by_blk.index:
                keep = np.asarray(blocks != blk)
                lobo.append({"model": m, "h": h, "left_out_block": blk, "n": int(keep.sum()),
                             "regret_diff_usd_t": d[keep].mean()})
    pd.DataFrame(rows).to_csv(OUT / "v3_economic_magnitude.csv", index=False)
    pd.DataFrame(lobo).to_csv(OUT / "v3_regret_leave_one_block_out.csv", index=False)


# ------------------------------------------------------------------ C7
def c7_christoffersen():
    f = OUT / "intervals_rows.csv"
    if not f.exists():
        NOTES.append("C7 skipped: intervals_rows.csv not found.")
        return
    iv = pd.read_csv(f, parse_dates=["origin_date"])
    iv = ordered(iv[(iv["h"] == 1) & iv["model"].isin(["RW", "ARIMA", "COMB"])])
    rows = []
    for (m, level), g in iv.groupby(["model", "level"]):
        for meth in ["conf", "vol", "heur"]:
            hits = ((g["price_true"] >= g[f"{meth}_lo"]) & (g["price_true"] <= g[f"{meth}_hi"])).astype(float)
            segs = []
            for blk in [b for b in BLOCK_ORDER if b in set(g["block"])]:
                hb = hits[g["block"] == blk].to_numpy()
                segs.append(hb)
                rows.append({"model": m, "method": meth, "level": level, "block": blk, **christoffersen(hb, level)})
            rows.append({"model": m, "method": meth, "level": level, "block": "pooled", **christoffersen(segs, level)})
    pd.DataFrame(rows).to_csv(OUT / "v3_christoffersen_h1.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ens = ensemble(load())
    test = ens[ens["split"] == "test"].dropna(subset=["r_true"])
    labels = sorted(test["label"].unique())
    ev = K.evaluate_rule(test, "seller")                      # sourced cost, seller, storage x1, b = 0
    dec = {lab: g.set_index(["block", "origin_date", "h"]).sort_index() for lab, g in ev.groupby("label")}
    c1_clark_west(test, labels)
    c2_mae_ratio(test, labels)
    c3_roll(test, labels, dec)
    c4_chronos(ens, test, labels, dec)
    c5_economic(test, labels, dec)
    c7_christoffersen()
    if not C.TAG:                                             # documentation that travels with the main results
        for src in [C.ROOT / "CHRONOS_CORPUS.md", C.RAW / "update_2026-09" / "ROLL_NOTES.md"]:
            if src.exists():
                shutil.copy2(src, C.RESULTS / src.name)
    (OUT / "v3_notes.md").write_text("\n".join(["# analyze_v3 notes"] + [f"- {n}" for n in NOTES]) + "\n")
    print("analyze_v3 done", NOTES)


if __name__ == "__main__":
    main()
