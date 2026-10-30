"""Paper figures (PNG, 300 dpi) from saved tables. Output: <results>/figures/."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402

T = C.RESULTS / "tables"
F = C.RESULTS / "figures"
MAIN = ["RW", "ARIMA", "RF", "XGB", "GRU", "Chronos"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=300)
    plt.close(fig)
    print("figure", name, flush=True)


def fig_price_blocks():
    tab = build_table()
    blocks = pd.read_csv(C.RESULTS / "blocks.csv", parse_dates=["first_origin", "last_origin"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tab.index, tab["price"], lw=0.7, color="black")
    for k, b in blocks.iterrows():
        ax.axvspan(b["first_origin"], b["last_origin"], color="C0" if k % 2 == 0 else "C1", alpha=0.15)
        ax.text(b["first_origin"], ax.get_ylim()[1] * 0.95, b["block"], fontsize=7)
    ax.set_ylabel("USD/t")
    ax.set_title("Coffee futures close and evaluation blocks")
    save(fig, "fig1_price_blocks.png")


def fig_rmae():
    acc = pd.read_csv(T / "accuracy.csv")
    acc = acc[(acc["block"] == "pooled") & acc["model"].isin(MAIN)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m, g in acc.groupby("model"):
        ax.plot(g["h"], g["rMAE"], marker="o", ms=3, label=m)
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("horizon h (trading days)")
    ax.set_ylabel("MAE relative to RW")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig2_rmae_by_h.png")


def fig_coverage():
    iv = pd.read_csv(T / "intervals_by_block.csv")
    iv = iv[(iv["model"] == "RW") & (iv["level"] == 0.9)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for m, g in iv.groupby("method"):
        ax.plot(g["block"], g["coverage"], marker="o", label={"vol": "vol-scaled", "conf": "static conformal",
                                                               "heur": "sqrt(h) rule"}[m])
    ax.axhline(0.9, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("coverage of 90% interval (RW, all h)")
    ax.legend(fontsize=7)
    save(fig, "fig3_coverage_by_block.png")


def fig_regret_states():
    ind = pd.read_csv(T / "v2_indicators_by_state.csv")
    ind = ind[(ind["state_var"] == "trend") & (ind["h"] == 5) & ind["model"].isin(MAIN)]
    piv = ind.pivot(index="state", columns="model", values="regret").reindex(["low", "mid", "high"])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    piv[[m for m in MAIN if m in piv.columns]].plot.bar(ax=ax, width=0.8)
    ax.set_xlabel("60-day trend tercile at the origin")
    ax.set_ylabel("mean regret, h = 5 (USD/t)")
    ax.legend(fontsize=7, ncol=3)
    save(fig, "fig4_regret_by_trend_state.png")


def fig_coverage_block_h():
    """D-57: coverage of the 90% interval (RW) by block x horizon, one panel per method."""
    iv = pd.read_csv(T / "intervals_by_block_h.csv")
    iv = iv[(iv["model"] == "RW") & (iv["level"] == 0.9)]
    names = {"conf": "static conformal", "vol": "vol-scaled conformal", "heur": "sqrt(h) rule"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, meth in zip(axes, ["conf", "vol", "heur"]):
        piv = iv[iv["method"] == meth].pivot(index="block", columns="h", values="coverage")
        im = ax.imshow(piv.to_numpy(float), cmap="RdBu", vmin=0.70, vmax=1.10, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if (v < 0.76 or v > 1.04) else "black")
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel("h")
        ax.set_title(names[meth], fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, label="coverage (target 0.90)")
    fig.suptitle("RW, 90% prediction interval: coverage by block and horizon", fontsize=9)
    fig.savefig(F / "fig5_coverage_block_h.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("figure fig5_coverage_block_h.png", flush=True)


def fig_b5_forecasts():
    """B5, h = 1 and h = 5: actual price, RW / RF / COMB forecasts (distinct styles), RW 90% vol-scaled interval."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze import ensemble, load
    ens = ensemble(load())
    ens = ens[(ens["block"] == "B5") & (ens["split"] == "test")].dropna(subset=["r_true"])
    iv = pd.read_csv(T / "intervals_rows.csv", parse_dates=["origin_date"])
    iv = iv[(iv["model"] == "RW") & (iv["block"] == "B5") & (iv["level"] == 0.9)]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    for ax, h in zip(axes, [1, 5]):
        e = ens[ens["h"] == h]
        rw = e[e["label"] == "RW"].sort_values("target_date")
        band = rw.merge(iv[iv["h"] == h][["origin_date", "vol_lo", "vol_hi"]], on="origin_date")
        ax.fill_between(band["target_date"], band["vol_lo"], band["vol_hi"], color="grey", alpha=0.25,
                        label="RW 90% interval (vol-scaled)")
        ax.plot(rw["target_date"], rw["price_true"], color="black", lw=0.9, label="actual")
        for lab, c, ls, lw in [("RW", "C0", "-", 1.1), ("RF", "C3", "--", 0.8), ("COMB", "C2", ":", 1.3)]:
            g = e[e["label"] == lab].sort_values("target_date")
            if len(g):
                ax.plot(g["target_date"], g["price_pred"], lw=lw, ls=ls, color=c, label=lab)
        ax.set_ylabel("USD/t")
        ax.set_title(f"B5 (v1 test window), h = {h}", fontsize=9)
    axes[0].legend(fontsize=7, ncol=3)
    save(fig, "fig6_b5_forecast_vs_actual.png")


def fig_returns_breaks():
    """Daily returns before B5 with PELT segment boundaries and BIC-selected mean breaks."""
    tab = build_table()
    pre = tab[tab.index < C.B5_FIRST_TARGET]
    seg = pd.read_csv(T / "pelt_segments.csv", parse_dates=["start"])
    mb = pd.read_csv(T / "mean_breaks.csv", parse_dates=["break_date_after"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(pre.index, pre["ret"] * 100, lw=0.4, color="black")
    for i, d in enumerate(seg["start"].iloc[1:]):
        ax.axvline(d, color="C3", lw=0.8, ls="--", label="PELT boundary (mean/variance)" if i == 0 else None)
    for i, d in enumerate(mb["break_date_after"].dropna()):
        ax.axvline(d, color="C0", lw=1.0, label="mean break (DP + BIC)" if i == 0 else None)
    ax.set_ylabel("daily log return, %")
    if mb["break_date_after"].dropna().empty:
        ax.plot([], [], color="C0", lw=1.0, label="No mean break selected (DP + BIC, k = 0)")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "fig7_returns_breaks.png")


def main():
    F.mkdir(parents=True, exist_ok=True)
    for f in [fig_price_blocks, fig_rmae, fig_coverage, fig_regret_states, fig_coverage_block_h, fig_b5_forecasts,
              fig_returns_breaks]:
        try:
            f()
        except Exception as e:                                    # a missing table must not stop the run
            print("figure skipped:", f.__name__, type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main()
