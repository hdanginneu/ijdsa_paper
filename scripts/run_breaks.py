"""Unit-root tests and change-point detection on data BEFORE the v1 test window (R3.3).

ADF (constant, and constant + trend; lag by AIC) and KPSS on log price and daily log return.
PELT (ruptures, 'normal' cost = changes in mean and/or variance) on daily log returns,
penalty = 3 log(n) (BIC-type, 2 parameters per segment), minimum segment 60 days.

Mean breaks: least-squares dynamic programming (Bai & Perron 2003 estimator) with BIC;
no sup-F tests. ruptures.Dynp(model="l2", min_size=60, jump=5) on the same sample as PELT;
k = 0..5 breaks, BIC(k) = n log(RSS_k / n) + (2k + 1) log(n) (k + 1 means, k break dates).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.stats_tests import adf_kpss                              # noqa: E402


def mean_breaks(pre, out, max_k=5):
    r = pre["ret"].to_numpy()
    n = len(r)
    algo = rpt.Dynp(model="l2", min_size=60, jump=5).fit(r.reshape(-1, 1))
    bic, fits = [], {}
    for k in range(0, max_k + 1):
        bk = [n] if k == 0 else algo.predict(n_bkps=k)
        edges = [0] + bk
        rss = sum(((r[a:b] - r[a:b].mean()) ** 2).sum() for a, b in zip(edges[:-1], edges[1:]))
        bic.append({"k": k, "RSS": rss, "BIC": n * np.log(rss / n) + (2 * k + 1) * np.log(n),
                    "break_dates": ";".join(str(pre.index[b].date()) for b in bk[:-1])})
        fits[k] = edges
    bic = pd.DataFrame(bic)
    k_best = int(bic.loc[bic["BIC"].idxmin(), "k"])
    bic["chosen"] = bic["k"] == k_best
    bic.to_csv(out / "mean_breaks_bic.csv", index=False)
    edges = fits[k_best]
    seg = [{"segment": i + 1, "start": pre.index[a].date(), "end": pre.index[b - 1].date(), "n": b - a,
            "mean_ret_bp": r[a:b].mean() * 1e4, "break_date_after": pre.index[b].date() if b < n else None}
           for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    pd.DataFrame(seg).to_csv(out / "mean_breaks.csv", index=False)
    print("mean breaks (BIC):", k_best, [s_["break_date_after"] for s_ in seg[:-1]])


def main():
    t = build_table()
    pre = t[t.index < C.B5_FIRST_TARGET]
    rows = adf_kpss(pre["logp"], "log price") + adf_kpss(pre["ret"], "log return")
    out = C.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "unit_root.csv", index=False)

    x = pre["ret"].to_numpy().reshape(-1, 1)
    n = len(x)
    bk = rpt.Pelt(model="normal", min_size=60).fit(x).predict(pen=3 * np.log(n))
    edges = [0] + bk
    seg = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = pre["ret"].iloc[a:b]
        seg.append({"start": s.index[0].date(), "end": s.index[-1].date(), "n": len(s),
                    "mean_ret_bp": s.mean() * 1e4, "sd_ret_pct": s.std() * 100,
                    "mean_price": pre["price"].iloc[a:b].mean()})
    pd.DataFrame(seg).to_csv(out / "pelt_segments.csv", index=False)
    mean_breaks(pre, out)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(seg).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
