"""Compare the main run with every robustness run in results/robust/<tag>/."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402


def one(root, tag):
    t = root / "tables"
    if not (t / "accuracy.csv").exists():
        return None
    acc = pd.read_csv(t / "accuracy.csv")
    acc = acc[acc["block"] == "pooled"][["model", "h", "MAE", "rMAE"]]
    dm = pd.read_csv(t / "dm_vs_rw.csv")
    dm = dm[(dm["loss"] == "SE") & (dm["bandwidth"] == "h-1")][["model", "h", "mean_diff_vs_RW", "p_holm"]]
    out = acc.merge(dm, on=["model", "h"], how="left")
    dec = t / "v2_decisions.csv"
    if dec.exists():
        d = pd.read_csv(dec)
        d = d[(d["block"] == "pooled") & (d["storage_mult"] == 1) & (d["side"] == "seller") & (d["basis_sd"] == 0)]
        out = out.merge(d[["model", "h", "mean_regret", "mean_payoff"]], on=["model", "h"], how="left")
        out["regret_rank"] = out.groupby("h")["mean_regret"].rank()
    out.insert(0, "run", tag)
    return out


def main():
    parts = [one(C.MAIN_RESULTS, "main")]
    for d in sorted((C.MAIN_RESULTS / "robust").glob("*")) if (C.MAIN_RESULTS / "robust").exists() else []:
        parts.append(one(d, d.name))
    tab = pd.concat([p for p in parts if p is not None], ignore_index=True)
    out = C.MAIN_RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "robustness_summary.csv", index=False)
    claims = tab.groupby("run").apply(lambda g: pd.Series({
        "models_sig_better_than_RW (Holm<0.05)": int(((g["mean_diff_vs_RW"] < 0) & (g["p_holm"] < 0.05)).sum()),
        "models_sig_worse_than_RW (Holm<0.05)": int(((g["mean_diff_vs_RW"] > 0) & (g["p_holm"] < 0.05)).sum()),
        "lowest_regret_model_by_h": ",".join(g.sort_values("mean_regret").groupby("h").head(1).sort_values("h")["model"])
        if "mean_regret" in g else "",
    }), include_groups=False).reset_index()
    claims.to_csv(out / "robustness_claims.csv", index=False)
    # model-selection reliability of every run (candidates = the models present in that run)
    sel = []
    for d, name in [(C.MAIN_RESULTS, "main")] + [(d, d.name) for d in sorted((C.MAIN_RESULTS / "robust").glob("*"))
                                                 if (C.MAIN_RESULTS / "robust").exists()]:
        f = d / "tables" / "v4_selection_risk.csv"
        if f.exists():
            sel.append(pd.read_csv(f).assign(run=name))
    if sel:
        pd.concat(sel, ignore_index=True).to_csv(out / "v4_selection_risk_all_runs.csv", index=False)
    print(claims.to_string(index=False))


if __name__ == "__main__":
    main()
