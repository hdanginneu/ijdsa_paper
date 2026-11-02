"""Main-text tables and figures. Main run only; reads existing tables, computes nothing new.

Writes results/paper/: T1..T6 (.csv and .md), F1..F4 (.png) and README.md. Every other table in tables/ is
appendix material.
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402

T = C.MAIN_RESULTS / "tables"
P = C.MAIN_RESULTS / "paper"
MAIN_SET = ["ARIMA", "RF", "XGB", "GRU", "Chronos", "COMB", "SHRINK-COMB"]
H4 = [1, 3, 5, 7]


def rd(path):
    return pd.read_csv(path) if Path(path).exists() and Path(path).stat().st_size > 5 else None


def save(df, name, nd=3):
    df.to_csv(P / f"{name}.csv", index=False)
    (P / f"{name}.md").write_text(df.round(nd).to_markdown(index=False) + "\n")


def t1():
    b = rd(C.MAIN_RESULTS / "blocks.csv")
    tab = build_table()
    vr = rd(T / "v4_variance_ratio.csv")
    rows = []
    for _, r in b.iterrows():
        s = tab.loc[(tab.index >= r["first_origin"]) & (tab.index <= r["last_origin"])]
        row = {"block": r["block"], "first_origin": r["first_origin"], "last_origin": r["last_origin"],
               "n_origins": r["n_origins"], "mean_price_usd_t": s["price"].mean(), "sd_daily_return_pct": 100 * s["ret"].std()}
        if vr is not None:
            v = vr[(vr["period"] == r["block"]) & (vr["q"] == 5)]
            row["VR_q5"] = v["VR"].iloc[0] if len(v) else np.nan
            row["VR_q5_p_holm"] = v["p_holm"].iloc[0] if len(v) else np.nan
        rows.append(row)
    save(pd.DataFrame(rows), "T1_data_blocks")


def t2():
    ci, dm, cw = rd(T / "v3_mae_ratio_ci.csv"), rd(T / "dm_vs_rw.csv"), rd(T / "v3_clark_west.csv")
    models = [m for m in MAIN_SET + ["Drift"] if m in set(ci["model"])]
    rows = []
    for m in models:
        for h in H4:
            c = ci[(ci["model"] == m) & (ci["h"] == h)]
            c = c[c["set"] == "main"] if (c["set"] == "main").any() else c
            d = dm[(dm["model"] == m) & (dm["h"] == h) & (dm["loss"] == "SE") & (dm["bandwidth"] == "h-1")]
            w = cw[(cw["model"] == m) & (cw["h"] == h)]
            w = w[w["set"] == "main"] if (w["set"] == "main").any() else w
            c = c.iloc[0]
            cat = ("equivalent" if c["equiv_1pct"] else "better" if c["better_1pct"]
                   else "worse" if c["worse_1pct"] else "inconclusive")
            rows.append({"model": m, "h": h, "rMAE": c["ratio"], "ci90_lo": c["ci90_lo"], "ci90_hi": c["ci90_hi"],
                         "DM_SE_p_holm": d["p_holm"].iloc[0] if len(d) else np.nan,
                         "CW_p_holm": w["p_holm"].iloc[0] if len(w) else np.nan, "equivalence_1pct": cat,
                         "CW_holm_set": w["set"].iloc[0] if len(w) else ""})
    save(pd.DataFrame(rows), "T2_accuracy")


def t3():
    iv, ch, rl, z = (rd(T / f) for f in ["intervals_pooled.csv", "v3_christoffersen_h1.csv",
                                         "v4_risk_limit.csv", "v4_risk_limit_zones.csv"])
    rows = []
    for meth in ["conf", "vol", "heur"]:
        for h in [1, 5]:
            row = {"method": meth, "h": h}
            for lev in [0.8, 0.9, 0.95]:
                g = iv[(iv["model"] == "RW") & (iv["method"] == meth) & (iv["h"] == h) & (iv["level"] == lev)]
                row[f"coverage_{int(lev * 100)}"] = g["coverage"].iloc[0] if len(g) else np.nan
            if h == 1 and ch is not None:
                c = ch[(ch["model"] == "RW") & (ch["method"] == meth) & (ch["level"] == 0.9) & (ch["block"] == "pooled")]
                row["christoffersen_p_cc_90"] = c["p_cc"].iloc[0] if len(c) else np.nan
            r = rl[(rl["model"] == "RW") & (rl["method"] == meth) & (rl["h"] == h) & (rl["block"] == "pooled")]
            for k in ["breach_rate", "mean_shortfall", "mean_capital"]:
                row[k] = r[k].iloc[0] if len(r) else np.nan
            zz = z[(z["model"] == "RW") & (z["method"] == meth) & (z["h"].astype(str) == str(h))]
            row["n_red_blocks"], row["n_yellow_blocks"] = int((zz["zone"] == "red").sum()), int((zz["zone"] == "yellow").sum())
            rows.append(row)
    save(pd.DataFrame(rows), "T3_uncertainty")


def t4():
    em, lobo = rd(T / "v3_economic_magnitude.csv"), rd(T / "v3_regret_leave_one_block_out.csv")
    vs, oracle = rd(T / "v4_value_share.csv"), rd(T / "v4_oracle_value.csv")
    orc = oracle[oracle["block"] == "pooled"].set_index("h")
    rows = []
    for m in [x for x in MAIN_SET if x in set(em["model"])]:
        for h in H4:
            e = em[(em["model"] == m) & (em["h"] == h)].iloc[0]
            l4 = lobo[(lobo["model"] == m) & (lobo["h"] == h) & (lobo["left_out_block"] == "B4")]
            v = vs[(vs["model"] == m) & (vs["h"] == h)].iloc[0]
            rows.append({"model": m, "h": h, "regret_diff_usd_t": e["regret_diff_usd_t"], "ci95_lo": e["ci95_lo_usd_t"],
                         "ci95_hi": e["ci95_hi_usd_t"], "regret_diff_pct_price": e["regret_diff_pct_of_price"],
                         "regret_diff_without_B4": l4["regret_diff_usd_t"].iloc[0] if len(l4) else np.nan,
                         "share_of_VPI": v["share"], "share_ci95_lo": v["ci95_lo"], "share_ci95_hi": v["ci95_hi"],
                         "accuracy": v["accuracy"], "q_breakeven": orc.loc[h, "q_breakeven"]})
    for h in H4:
        rows.append({"model": "VPI (oracle, USD/t)", "h": h, "regret_diff_usd_t": orc.loc[h, "VPI_usd_t"],
                     "regret_diff_pct_price": orc.loc[h, "VPI_pct_price"], "q_breakeven": orc.loc[h, "q_breakeven"]})
    save(pd.DataFrame(rows), "T4_decision")


def robust_runs():
    root = C.MAIN_RESULTS / "robust"
    return [("main", C.MAIN_RESULTS)] + ([(d.name, d) for d in sorted(root.glob("*")) if d.is_dir()] if root.exists() else [])


def t5():
    main = rd(T / "v4_selection_risk.csv")
    runs = robust_runs()[1:]
    same = []
    for _, r in main.iterrows():
        n_same, n_runs = 0, 0
        for _, d in runs:
            s = rd(d / "tables" / "v4_selection_risk.csv")
            if s is None or r["procedure"] not in set(s["procedure"]):
                continue
            n_runs += 1
            n_same += int(bool(s.set_index("procedure").loc[r["procedure"], "conclusion_worse_majority"])
                          == bool(r["conclusion_worse_majority"]))
        same.append((n_same, n_runs))
    main["n_robust_runs_same_conclusion"] = [a for a, _ in same]
    main["n_robust_runs_with_procedure"] = [b for _, b in same]
    rk = rd(T / "v4_selection_rankcorr.csv")
    if rk is not None:
        sm = rk[rk["block"] == "summary"].set_index("procedure")
        main["spearman_mean_rho"] = main["procedure"].map(sm["spearman_rho"])
        main["spearman_rank_by"] = main["procedure"].map(sm["rank_by"])
    save(main, "T5_selection")


def t6():
    rows = []
    for name, d in robust_runs():
        t = d / "tables"
        dm, cw, ci, iv, dec, sel = (rd(t / f) for f in ["dm_vs_rw.csv", "v3_clark_west.csv", "v3_mae_ratio_ci.csv",
                                                       "intervals_pooled.csv", "v2_decisions.csv", "v4_selection_risk.csv"])
        row = {"run": name}
        if dm is not None:
            g = dm[(dm["loss"] == "SE") & (dm["bandwidth"] == "h-1")]
            row["n_better_than_RW_DM"] = int(((g["p_holm"] < 0.05) & (g["mean_diff_vs_RW"] < 0)).sum())
        if cw is not None:
            row["n_better_than_RW_CW"] = int((cw[cw["set"] == "all"]["p_holm"] < 0.05).sum())
        if ci is not None:
            g = ci[(ci["set"] == "all") & (ci["model"] != "RW")]
            row["n_equivalent_1pct"], row["n_model_h_cells"] = int(g["equiv_1pct"].sum()), len(g)
        if iv is not None:
            for meth in ["vol", "conf", "heur"]:
                g = iv[(iv["model"] == "RW") & (iv["method"] == meth) & (iv["h"] == 1) & (iv["level"] == 0.9)]
                row[f"coverage90_h1_{meth}"] = g["coverage"].iloc[0] if len(g) else np.nan
        if dec is not None:
            g = dec[(dec["block"] == "pooled") & (dec["storage_mult"] == 1) & (dec["side"] == "seller") &
                    (dec["basis_sd"] == 0) & (dec["h"] == 5)]
            row["lowest_regret_model_h5"] = g.sort_values("mean_regret")["model"].iloc[0] if len(g) else ""
        if sel is not None:
            for _, s in sel.iterrows():
                row[f"{s['procedure']}_share_worse_than_RW"] = s["share_worse_than_RW"]
        rows.append(row)
    save(pd.DataFrame(rows), "T6_robustness")


def figures():
    F = C.MAIN_RESULTS / "figures"
    for src, dst in [("fig1_price_blocks.png", "F1_price_blocks.png"), ("fig5_coverage_block_h.png", "F3_coverage_block_h.png"),
                     ("fig1_price_blocks.pdf", "F1_price_blocks.pdf"), ("fig5_coverage_block_h.pdf", "F3_coverage_block_h.pdf")]:
        if (F / src).exists():
            shutil.copy2(F / src, P / dst)
    ci = rd(T / "v3_mae_ratio_ci.csv")
    if ci is not None:
        g = ci[ci["set"] == "main"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        models = [m for m in MAIN_SET if m in set(g["model"])]
        for k, m in enumerate(models):
            x = g[g["model"] == m].sort_values("h")
            off = (k - (len(models) - 1) / 2) * 0.08
            ax.errorbar(x["h"] + off, x["ratio"], yerr=[x["ratio"] - x["ci90_lo"], x["ci90_hi"] - x["ratio"]],
                        fmt="o", ms=3, capsize=2, lw=0.8, label=m)
        for yv in [0.99, 1.01]:
            ax.axhline(yv, color="grey", ls="--", lw=0.7)
        ax.axhline(1, color="black", lw=0.6)
        ax.set_xlabel("horizon h (trading days)")
        ax.set_ylabel("MAE / MAE_RW (90% CI)")
        ax.legend(fontsize=6, ncol=4)
        fig.tight_layout()
        fig.savefig(P / "F2_rmae_ci.png", dpi=300)
        fig.savefig(P / "F2_rmae_ci.pdf")
        plt.close(fig)
    dec = rd(T / "v2_decisions.csv")
    if dec is not None:
        d = dec[(dec["storage_mult"] == 1) & (dec["side"] == "seller") & (dec["basis_sd"] == 0) & (dec["block"] != "pooled")]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
        for ax, h in zip(axes, [1, 5]):
            g = d[d["h"] == h].pivot(index="block", columns="model", values="mean_regret")
            for m in ["RF", "COMB", "Chronos"]:
                if m in g.columns:
                    ax.plot(g.index, g[m] - g["RW"], marker="o", ms=3, lw=0.9, label=m)
            ax.axhline(0, color="black", lw=0.6)
            if "B4" in g.index:
                k = list(g.index).index("B4")
                ax.axvspan(k - 0.4, k + 0.4, color="orange", alpha=0.15, label="B4")
            ax.set_title(f"h = {h}", fontsize=9)
            ax.set_xlabel("block")
        axes[0].set_ylabel("regret − regret_RW (USD/t)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(P / "F4_regret_by_block.png", dpi=300)
        fig.savefig(P / "F4_regret_by_block.pdf")
        plt.close(fig)


README = """# Main-text tables and figures

Generated by scripts/paper_tables.py from existing tables (nothing recomputed).

| File | Content | Answers |
|---|---|---|
| T1_data_blocks | Six evaluation blocks: dates, origins, mean price, daily return sd, variance ratio q = 5 (Holm) | data; RQ1 mechanism |
| T2_accuracy | rMAE with 90% CI, DM-HLN (SE) Holm p, Clark–West Holm p, ±1% equivalence; MAIN_SET + Drift, h = 1, 3, 5, 7 | RQ1 |
| T3_uncertainty | RW intervals (conf / vol / heur), h = 1, 5: coverage 80/90/95%, Christoffersen p_cc (h = 1), breach rate, shortfall, risk capital, red/yellow blocks | RQ2 |
| T4_decision | Regret difference vs RW (USD/t, % price, 95% CI, without B4), share of VPI with CI, accuracy, q*; VPI rows | RQ3 |
| T5_selection | Model-selection procedures P1–P4: worse/better than RW, test ratio, selection regret, Spearman (MAE ranks for P1, regret ranks for P3; descriptive), robustness runs with the same conclusion | RQ4 |
| T6_robustness | Main run and every robustness run: models better than RW (DM, CW), ±1% equivalent cells, 90% coverage at h = 1, lowest-regret model at h = 5, share of selection cells worse than RW per procedure | robustness |
| F1_price_blocks | Price and evaluation blocks | data |
| F2_rmae_ci | rMAE with 90% CI by h, MAIN_SET, lines at 1 ± 0.01 | RQ1 |
| F3_coverage_block_h | Coverage of the RW 90% interval by block × h, three methods | RQ2 |
| F4_regret_by_block | Regret difference vs RW by block (RF, COMB, Chronos), B4 marked | RQ3 |

"Same conclusion" (T5): a run's conclusion for a procedure is whether more than 50% of its block × h cells are worse
than RW on test (P4: selected w worse than w = 0); a robustness run agrees if this indicator is the same as in
the main run.

Appendix: every other file in `tables/` (and `robust/*/tables/`).
"""


def main():
    P.mkdir(parents=True, exist_ok=True)
    for fn in [t1, t2, t3, t4, t5, t6, figures]:
        try:
            fn()
            print("done", fn.__name__, flush=True)
        except Exception as e:
            print("FAILED", fn.__name__, type(e).__name__, e, flush=True)
            raise
    (P / "README.md").write_text(README)


if __name__ == "__main__":
    main()
