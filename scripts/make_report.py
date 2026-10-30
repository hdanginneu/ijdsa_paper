"""Write results/RESULTS.md from the tables (numbers only; no interpretation beyond what the data show)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402

T = C.RESULTS / "tables"


def md(df, floatfmt=3):
    return df.round(floatfmt).to_markdown(index=False)


def v2_sections():
    """Sourced costs, risk, guide and robustness (D-29 .. D-35). Skipped silently if a table is missing."""
    L = []
    f = T / "v2_decisions.csv"
    if f.exists():
        d = pd.read_csv(f)
        p = d[(d["block"] == "pooled") & (d["storage_mult"] == 1) & (d["side"] == "seller") & (d["basis_sd"] == 0)]
        L += ["## V2-1. Decisions with sourced carry cost (ICE storage 14.42 USD/t/month + EFFR financing; seller; pooled)",
              "Mean regret (USD/t):", md(p.pivot(index="model", columns="h", values="mean_regret").reset_index(), 2), "",
              "Mean payoff vs selling now (USD/t):", md(p.pivot(index="model", columns="h", values="mean_payoff").reset_index(), 2), ""]
        b = d[(d["block"] == "pooled") & (d["storage_mult"] == 1) & (d["side"] == "seller") & (d["h"] == 5)]
        L += ["Basis-noise sensitivity, h = 5, mean regret:",
              md(b.pivot(index="model", columns="basis_sd", values="mean_regret").reset_index(), 2), ""]
        dm = pd.read_csv(T / "v2_decisions_dm.csv")
        dm = dm[(dm["storage_mult"] == 1) & (dm["side"] == "seller") & (dm["basis_sd"] == 0) & (dm["loss"] == "regret")]
        L += ["Regret vs RW, Holm p (within horizon):", md(dm.pivot(index="model", columns="h", values="p_holm").reset_index(), 3), ""]
    f = T / "v2_var_reliability.csv"
    if f.exists():
        v = pd.read_csv(f)
        v = v[v["by"] == "h"]
        L += ["## V2-2. 5% VaR of the waiting payoff: exceedance rate (target 0.05)",
              md(v.pivot(index="model", columns="key", values="exceed_rate").reset_index(), 3), ""]
        m = pd.read_csv(T / "v2_mean_cvar.csv")
        m = m[m["h"] == 5]
        L += ["## V2-3. Mean–CVaR rule, h = 5", md(m[["model", "rule", "share_wait", "mean_regret", "mean_payoff",
                                                      "payoff_q05", "payoff_cvar05"]], 2), ""]
    f = T / "v2_guide.csv"
    if f.exists():
        g = pd.read_csv(f)
        L += ["## V2-4. Guide: recommended model / interval method by state and horizon (evidence grade, D-33)",
              md(g[g["h"].isin([1, 3, 5, 7])][["indicator", "state_var", "state", "h", "recommend", "grade", "note"]]), "",
              "Strong: Holm-adjusted p < 0.05, same sign in ≥ 4 blocks and in B6. Moderate: same sign as the pooled "
              "difference in ≥ 4 blocks; no significance requirement (sign consistency only). Weak: otherwise. "
              "A model is recommended only when its cell is Strong; otherwise the default is RW.", ""]
        c = pd.read_csv(T / "v2_compute_cost.csv")
        L += ["Compute cost (D-54 when available: B5, seed 0, sequential, 2 threads, CPU, median of 3):", md(c, 4), ""]
    f = T / "v2_mechanisms.csv"
    if f.exists():
        k = pd.read_csv(f)
        L += ["## V2-5. Mechanism tests", md(k[k["test"].isin(["momentum_regression", "arch_lm_5", "ljung_box_10"])], 4), ""]
    f = C.MAIN_RESULTS / "tables" / "robustness_claims.csv"
    if f.exists():
        L += ["## V2-6. Robustness runs (D-35)", md(pd.read_csv(f)), "",
              "Full comparison: tables/robustness_summary.csv", ""]
    return L


def v3_sections():
    """Round-3 tables (D-47 .. D-54). Each section: table(s) + one factual line. Missing tables are skipped."""
    L = []

    def rd(name):
        f = T / name
        return pd.read_csv(f) if f.exists() and f.stat().st_size > 5 else None

    cw = rd("v3_clark_west.csv")
    if cw is not None:
        m = cw[cw["set"] == "main"]
        L += ["## V3-1. Clark–West vs RW (one-sided, HAC h−1), Holm within h over MAIN_SET (D-47)",
              md(m.pivot(index="model", columns="h", values="p_holm").reset_index(), 3), "",
              f"Data: {int((m['p_holm'] < 0.05).sum())} of {len(m)} model×horizon cells have Holm p < 0.05.", ""]
    r = rd("v3_mae_ratio_ci.csv")
    if r is not None:
        m = r[r["set"] == "main"].copy()
        m["ratio [90% CI]"] = m.apply(lambda x: f"{x['ratio']:.3f} [{x['ci90_lo']:.3f}, {x['ci90_hi']:.3f}]", axis=1)
        summ = rd("v3_equivalence_summary.csv")
        L += ["## V3-2. MAE ratio vs RW with stationary-bootstrap CI; equivalence margin ±1% (D-48)",
              m.pivot(index="model", columns="h", values="ratio [90% CI]").reset_index().to_markdown(index=False), "",
              "Number of horizons by category (90% CI vs [0.99, 1.01]):",
              md(summ[summ["set"] == "main"]) if summ is not None else "", "",
              f"Data: {int(m['equiv_1pct'].sum())} of {len(m)} MAIN_SET cells are equivalent within ±1%; "
              f"{int(m['equiv_2pct'].sum())} within ±2%; {int(m['better_1pct'].sum())} better, "
              f"{int(m['worse_1pct'].sum())} worse by more than 1%.", ""]
    rr = rd("v3_roll_robustness.csv")
    if rr is not None:
        g = rr.groupby(["version", "model"]).agg(share_dropped=("share_dropped", "mean"), rMAE=("rMAE", "mean"),
                                                  n_dm_sig=("dm_p_holm", lambda x: int((x < 0.05).sum())),
                                                  n_cw_sig=("cw_p_holm", lambda x: int((x < 0.05).sum()))).reset_index()
        L += ["## V3-3. Roll robustness: origins whose (t, t+h] contains a suspected roll day removed (D-49); means over h",
              md(g, 3), "",
              f"Data: across versions, {int(g['n_dm_sig'].sum())} DM and {int(g['n_cw_sig'].sum())} CW cells "
              "have Holm p < 0.05; versions *_ltd use the supplementary LTD anchor.", ""]
    cb = rd("v3_chronos_b56.csv")
    if cb is not None:
        L += ["## V3-4. Chronos and SHRINK-Chronos on B5–B6 only (after the Chronos-Bolt release) (D-50)", md(cb, 4), "",
              f"Data: Holm p < 0.05 in {int((cb.filter(like='_holm') < 0.05).any(axis=1).sum())} of {len(cb)} rows.", ""]
    em = rd("v3_economic_magnitude.csv")
    if em is not None:
        lobo = rd("v3_regret_leave_one_block_out.csv")
        L += ["## V3-5. Economic magnitude of regret differences vs RW (seller, sourced cost) (D-51)",
              md(em[em["h"].isin([1, 3, 5, 7])].drop(columns=["unit_note"]), 3), "", f"Units: {em['unit_note'].iloc[0]}.", ""]
        if lobo is not None:
            lb = lobo.groupby(["model", "h"])["regret_diff_usd_t"].agg(["min", "max"]).reset_index()
            L += ["Leave-one-block-out range of the mean regret difference (USD/t):", md(lb[lb["h"].isin([1, 3, 5, 7])], 3), ""]
        L += [f"Data: largest absolute mean difference {em['regret_diff_usd_t'].abs().max():.2f} USD/t "
              f"({em['regret_diff_pct_of_price'].abs().max():.3f}% of price); 95% CI excludes 0 in "
              f"{int(((em['ci95_lo_usd_t'] > 0) | (em['ci95_hi_usd_t'] < 0)).sum())} of {len(em)} cells.", ""]
    q2 = rd("rq2_da_vs_mse.csv")
    if q2 is not None and "p_holm" in q2.columns:
        L += ["## V3-6. RQ2 with Holm over 14 tests; DA payoff vs selling now (D-52)", md(q2, 4), "",
              f"Data: {int((q2['p_holm'] < 0.05).sum())} of {len(q2)} DA-vs-MSE tests have Holm p < 0.05; "
              f"{int((q2['p_payoff_holm'] < 0.05).sum())} DA payoff tests have Holm p < 0.05.", ""]
    ch = rd("v3_christoffersen_h1.csv")
    if ch is not None:
        pooled = ch[ch["block"] == "pooled"][["model", "method", "level", "n", "misses", "p_uc", "p_ind", "p_cc"]]
        L += ["## V3-7. Christoffersen (1998) at h = 1, pooled over blocks (no cross-block transitions) (D-53)",
              md(pooled, 3), "", f"Data: p_cc < 0.05 in {int((pooled['p_cc'] < 0.05).sum())} of {len(pooled)} pooled rows; "
              "per-block rows in v3_christoffersen_h1.csv.", ""]
    cc = rd("v3_compute_cost.csv")
    if cc is not None:
        envf = T / "v3_compute_env.json"
        L += ["## V3-8. Compute cost re-measured: B5, seed 0, sequential, 2 threads, CPU, median of 3 (D-54)", md(cc, 4), "",
              "Environment: " + (envf.read_text().replace("\n", " ") if envf.exists() else "n/a"), ""]
    return L


def v4_sections():
    """Round-4 tables (D-59 .. D-63): table + one factual line each; main-text tables are in results/paper/."""
    L = ["## Main-text tables and figures: see results/paper/ (README.md lists which RQ each answers)", ""]

    def rd(name):
        f = T / name
        return pd.read_csv(f) if f.exists() and f.stat().st_size > 5 else None

    rl, z = rd("v4_risk_limit.csv"), rd("v4_risk_limit_zones.csv")
    if rl is not None:
        p = rl[rl["block"] == "pooled"]
        L += ["## V4-1. Inventory risk limit = lower bound of the 90% interval (D-59), pooled",
              md(p.pivot_table(index=["model", "method"], columns="h", values="breach_rate").reset_index(), 3), "",
              "Zone counts (block × h):", md(z[z["block"] == "count"][["model", "method", "green", "yellow", "red"]]) if z is not None else "", "",
              f"Data: pooled breach rate ranges {p['breach_rate'].min():.3f}–{p['breach_rate'].max():.3f} (target 0.05).", ""]
    o, v = rd("v4_oracle_value.csv"), rd("v4_value_share.csv")
    if o is not None and v is not None:
        L += ["## V4-2. Oracle value (VPI) and break-even accuracy (D-60)", md(o[o["block"] == "pooled"], 3), "",
              md(v[v["h"].isin([1, 3, 5, 7])], 3), "",
              f"Data: pooled VPI {o[o['block'] == 'pooled']['VPI_usd_t'].min():.1f}–{o[o['block'] == 'pooled']['VPI_usd_t'].max():.1f} USD/t; "
              f"largest share of VPI captured {v['share'].max():.3f}.", ""]
    sr = rd("v4_selection_risk.csv")
    if sr is not None:
        allr = rd("v4_selection_risk_all_runs.csv") if (T / "v4_selection_risk_all_runs.csv").exists() else None
        L += ["## V4-3. Reliability of model selection (D-61)", md(sr.drop(columns=["selected_frequency"]), 3), "",
              "Selected models: " + "; ".join(f"{r['procedure']}: {r['selected_frequency']}" for _, r in sr.iterrows()), ""]
        if allr is not None:
            L += ["Across runs (share of cells worse than RW):",
                  md(allr.pivot_table(index="run", columns="procedure", values="share_worse_than_RW").reset_index(), 3), ""]
        L += [f"Data: share of cells worse than RW on test: " +
              ", ".join(f"{r['procedure']} {r['share_worse_than_RW']:.2f}" for _, r in sr.iterrows()) + ".", ""]
    vr = rd("v4_variance_ratio.csv")
    if vr is not None:
        L += ["## V4-4. Variance-ratio tests, Lo–MacKinlay z*, Holm over q (D-62)", md(vr, 4), "",
              f"Data: Holm p < 0.05 in {int((vr['p_holm'] < 0.05).sum())} of {len(vr)} period × q cells.", ""]
    return L


def main():
    acc = pd.read_csv(T / "accuracy.csv")
    dm = pd.read_csv(T / "dm_vs_rw.csv")
    mcs = pd.read_csv(T / "mcs.csv")
    iv = pd.read_csv(T / "intervals_pooled.csv")
    ivb = pd.read_csv(T / "intervals_by_block.csv")
    spread = pd.read_csv(T / "seed_spread.csv")
    v1 = pd.read_csv(T / "v1_window_h1.csv")
    blocks = pd.read_csv(C.RESULTS / "blocks.csv")
    seg = pd.read_csv(T / "pelt_segments.csv")
    ur = pd.read_csv(T / "unit_root.csv")
    macro = pd.read_csv(T / "macro_value.csv") if (T / "macro_value.csv").stat().st_size > 5 else None
    dps = pd.read_csv(T / "dm_per_seed.csv") if (T / "dm_per_seed.csv").stat().st_size > 5 else None
    reg = pd.read_csv(T / "regimes.csv")

    L = ["# RESULTS — IJDSA revision (auto-generated by scripts/make_report.py)", ""]
    L += ["## Blocks (forecast origins)", md(blocks), ""]

    L += ["## 1. v1 test window, h = 1 (B5, 300 origins) — comparable with v1 Table 4.2",
          md(v1[["model", "n", "MAE", "RMSE", "MAPE", "rMAE", "R2_OOS"]]), "",
          "v1 reported GRU MAE 96.55 / RMSE 125.91 on the same 300 days.", ""]

    p = acc[acc["block"] == "pooled"]
    L += ["## 2. Pooled B1–B5: MAE by horizon (USD/t)",
          md(p.pivot(index="model", columns="h", values="MAE").reset_index()), "",
          "## 3. Pooled skill vs RW: rMAE (MAE / MAE_RW) by horizon",
          md(p.pivot(index="model", columns="h", values="rMAE").reset_index(), 4), "",
          "## 4. Pooled R2_OOS = 1 − MSE/MSE_RW by horizon",
          md(p.pivot(index="model", columns="h", values="R2_OOS").reset_index(), 4), ""]

    d = dm[(dm["loss"] == "SE") & (dm["bandwidth"] == "h-1")]
    L += ["## 5. DM-HLN vs RW, squared error, HAC lags h−1, Holm within horizon (negative = better than RW)",
          md(d.pivot(index="model", columns="h", values="p_holm").reset_index(), 4), "",
          "Mean loss differential (model − RW), USD²:",
          md(d.pivot(index="model", columns="h", values="mean_diff_vs_RW").reset_index(), 1), ""]
    sig = dm[dm["p_holm"] < 0.05]
    L += [f"Comparisons with Holm p < 0.05 across all losses/bandwidths: {len(sig)}", ""]
    if len(sig):
        L += [md(sig), ""]

    m = mcs[mcs["loss"] == "SE"]
    L += ["## 6. Model Confidence Set (90%), squared error: MCS p-value by horizon",
          md(m.pivot(index="model", columns="h", values="mcs_pvalue").reset_index(), 3), ""]

    if len(spread):
        L += ["## 7. Seed spread of MAE (pooled, by horizon)", md(spread[spread["h"].isin([1, 4, 7])], 2), ""]
    if dps is not None and len(dps):
        cnt = dps.assign(sig=dps["p_raw"] < 0.05).groupby(["model", "h"])["sig"].sum().unstack()
        L += ["Number of seeds (of 5) whose own DM vs RW has raw p < 0.05:", cnt.reset_index().to_markdown(index=False), ""]
    if macro is not None and len(macro):
        L += ["## 8. Do macro/volume inputs help? DM full vs price-only (negative = full better)", md(macro, 4), ""]

    r = reg[reg["h"].isin([1, 7])]
    L += ["## 9. Volatility regimes (cut-offs from data before B1): rMAE",
          md(r.pivot_table(index=["model"], columns=["regime", "h"], values="rMAE").reset_index(), 4), ""]

    L += ["## 10. Prediction intervals, pooled, by horizon (coverage target = level)",
          md(iv[iv["level"] == 0.9][["model", "h", "method", "coverage", "mean_width", "interval_score", "kupiec_p"]]),
          "", md(iv[iv["level"] == 0.8][["model", "h", "method", "coverage", "mean_width", "interval_score", "kupiec_p"]]),
          ""]
    ivbh = pd.read_csv(T / "intervals_by_block_h.csv")
    ivbh = ivbh[(ivbh["level"] == 0.9) & (ivbh["model"] == "RW")]
    for meth, name in [("conf", "static conformal"), ("vol", "vol-scaled conformal"), ("heur", "sqrt(h) rule")]:
        g = ivbh[ivbh["method"] == meth]
        L += [f"RW, 90% {name}: coverage by block x h", md(g.pivot(index="block", columns="h", values="coverage").reset_index(), 3),
              "Kupiec p:", md(g.pivot(index="block", columns="h", values="kupiec_p").reset_index(), 3), ""]
    L += ["Kupiec assumes independent hits; for h > 1 overlapping forecasts induce dependence, so p-values are "
          "indicative only (D-41).", ""]

    dec = pd.read_csv(T / "rq1_decisions.csv")
    p1 = dec[(dec["block"] == "pooled") & (dec["side"] == "seller") & (dec["cost_spec"].astype(str) == "sourced")]
    rank = pd.read_csv(T / "rq1_rank_agreement.csv")
    L += ["## RQ1. Decision fidelity (seller, sourced carry cost, pooled B1–B6)",
          "Mean regret (USD/t):", md(p1.pivot(index="label", columns="h", values="mean_regret").reset_index(), 2), "",
          "Balanced accuracy:", md(p1.pivot(index="label", columns="h", values="balanced_accuracy").reset_index(), 3), "",
          "Spearman rank correlation between MAE and regret across models:", md(rank, 3), "",
          "Note: the buyer's problem is the exact mirror image and gives identical regret.", ""]
    if (T / "rq2_da_vs_mse.csv").exists():
        L += ["## RQ2. Decision-aware (cost-sensitive) training vs MSE: regret difference (negative = DA better)",
              md(pd.read_csv(T / "rq2_da_vs_mse.csv"), 4), ""]
    q3 = pd.read_csv(T / "rq3_payoff.csv")
    L += ["## RQ3. Mean payoff of each rule vs acting now (USD/t per decision)",
          md(q3.pivot(index="model", columns="h", values="mean_payoff_vs_act_now").reset_index(), 2), "",
          "Holm-adjusted p-values:", md(q3.pivot(index="model", columns="h", values="p_holm").reset_index(), 3), ""]
    q4 = pd.read_csv(T / "rq4_routing.csv")
    q4 = q4[q4["block"] == "pooled(mean of blocks)"]
    L += ["## RQ4. Uncertainty-triggered routing RW → expensive model (mean over blocks)",
          md(q4[["expensive", "budget", "h", "share_queried", "MAE", "mean_regret", "accuracy", "mean_payoff"]], 3), ""]
    if (T / "rq4_routing_dm.csv").exists():
        q4d = pd.read_csv(T / "rq4_routing_dm.csv")
        L += ["RQ4 routed rule vs always-RW, DM-HLN (HAC h−1), Holm across budgets within (model, h, loss) (D-39):",
              md(q4d, 4), ""]
    if (T / "shrink_weights.csv").exists():
        w = pd.read_csv(T / "shrink_weights.csv")
        L += ["## COMB / SHRINK (D-42, D-43): shrinkage weight w by block (mean over h)",
              md(w.pivot_table(index="model", columns="block", values="w", aggfunc="mean").reset_index(), 2), ""]
    if (T / "v1_contrast.md").exists():
        L += [(T / "v1_contrast.md").read_text(), ""]

    L += ["## 11. Unit-root tests (data before the v1 test window)", md(ur, 4), "",
          "## 12. PELT segments of daily log returns", md(seg), ""]
    L += v2_sections()
    L += v3_sections()
    L += v4_sections()
    (C.RESULTS / "RESULTS.md").write_text("\n".join(L))
    print("written", C.RESULTS / "RESULTS.md")


if __name__ == "__main__":
    main()
