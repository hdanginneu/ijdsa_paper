"""Forecast-comparison tests and series diagnostics."""
import numpy as np
import pandas as pd
from scipy import stats


def _hac_var(d, lags):
    """Newey-West (Bartlett kernel) long-run variance of the mean of d."""
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return max(v, 1e-12) / n


def dm_test(loss_a, loss_b, h, lags=None):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    d = loss_a - loss_b ; negative statistic favours model A.
    Default HAC lags = h - 1 (standard for h-step forecasts).
    Returns (mean_d, statistic, two-sided p-value from t(n-1)).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    lags = h - 1 if lags is None else lags
    dm = d.mean() / np.sqrt(_hac_var(d, lags))
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return d.mean(), stat, p


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def mcs(loss_table, alpha=0.10, block_size=10, reps=2000, seed=42):
    """Model Confidence Set (Hansen, Lunde & Nason 2011) via the arch package.

    loss_table: DataFrame, one column per model, rows aligned in time.
    Returns DataFrame with MCS p-value and membership for each model.
    """
    from arch.bootstrap import MCS
    m = MCS(loss_table, size=alpha, block_size=block_size, reps=reps, method="R",
            bootstrap="stationary", seed=seed)
    m.compute()
    pv = m.pvalues.iloc[:, 0]
    return pd.DataFrame({"mcs_pvalue": pv, "in_mcs": pv > alpha})


def kupiec(hits, level):
    """Unconditional coverage test. hits = 1 if the outcome is inside the interval."""
    hits = np.asarray(hits, float)
    n, x = len(hits), int((1 - hits).sum())          # x = number of misses
    p = 1 - level
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        ph = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return lr, stats.chi2.sf(lr, 1)


def adf_kpss(series, name):
    """ADF (constant, lag by AIC) and KPSS (level) with their specifications written out."""
    from statsmodels.tsa.stattools import adfuller, kpss
    import warnings
    s = pd.Series(series).dropna()
    rows = []
    for reg in ["c", "ct"]:
        a = adfuller(s, regression=reg, autolag="AIC")
        rows.append({"series": name, "test": "ADF", "spec": f"regression={reg}, autolag=AIC",
                     "lags": a[2], "stat": a[0], "p": a[1], "cv5": a[4]["5%"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = kpss(s, regression="c", nlags="auto")
    rows.append({"series": name, "test": "KPSS", "spec": "regression=c, nlags=auto",
                 "lags": k[2], "stat": k[0], "p": k[1], "cv5": k[3]["5%"]})
    return rows


def clark_west(y, f_bench, f_model, h, lags=None):
    """Clark-West (2007) MSPE-adjusted test, H1: model beats the nested benchmark (one-sided).

    f_t = (y - f_bench)^2 - [(y - f_model)^2 - (f_bench - f_model)^2]; HAC (Bartlett) lags h-1.
    The p-value is one-sided from N(0, 1), as in Clark & West (2007).
    Returns (mean_f, t_stat, one-sided p from N(0,1)).
    """
    y, fb, fm = (np.asarray(a, float) for a in (y, f_bench, f_model))
    f = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    lags = h - 1 if lags is None else lags
    t = f.mean() / np.sqrt(_hac_var(f, lags))
    return f.mean(), t, stats.norm.sf(t)


def christoffersen(hits, level):
    """Christoffersen (1998): LR_ind (first-order Markov independence of misses) and
    LR_cc = LR_uc + LR_ind. Returns dict with LR_uc, p_uc, LR_ind, p_ind, LR_cc, p_cc.
    Handle zero counts as in the standard implementation (0 * log 0 = 0).

    hits = 1 if the outcome is inside the interval; pass a list of arrays to pool segments
    without counting transitions across segment boundaries (blocks).
    """
    segs = hits if isinstance(hits, list) else [hits]
    miss = [1 - np.asarray(s, float) for s in segs]
    all_miss = np.concatenate(miss)
    lr_uc, p_uc = kupiec(1 - all_miss, level)
    n = np.zeros((2, 2))
    for m in miss:
        m = m.astype(int)
        for a, b in zip(m[:-1], m[1:]):
            n[a, b] += 1

    def xlogy(x, p):
        return x * np.log(p) if x > 0 else 0.0

    p01 = n[0, 1] / max(n[0, 0] + n[0, 1], 1)
    p11 = n[1, 1] / max(n[1, 0] + n[1, 1], 1)
    p1 = (n[0, 1] + n[1, 1]) / max(n.sum(), 1)
    ll0 = xlogy(n[0, 0] + n[1, 0], 1 - p1) + xlogy(n[0, 1] + n[1, 1], p1)
    ll1 = xlogy(n[0, 0], 1 - p01) + xlogy(n[0, 1], p01) + xlogy(n[1, 0], 1 - p11) + xlogy(n[1, 1], p11)
    lr_ind = max(-2 * (ll0 - ll1), 0.0)
    lr_cc = lr_uc + lr_ind
    return {"LR_uc": lr_uc, "p_uc": p_uc, "LR_ind": lr_ind, "p_ind": stats.chi2.sf(lr_ind, 1),
            "LR_cc": lr_cc, "p_cc": stats.chi2.sf(lr_cc, 2), "n": len(all_miss), "misses": int(all_miss.sum()),
            "n01": int(n[0, 1]), "n11": int(n[1, 1])}
