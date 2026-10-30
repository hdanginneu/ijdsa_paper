"""Horizon-specific prediction intervals and their evaluation.

Conformal (empirical) interval for block b, horizon h, level L:
    errors on the block's calibration split: e = r_true - r_pred (log-return units)
    lower/upper = finite-sample quantiles of e at (1-L)/2 and (1+L)/2
    interval in price = P_origin * exp(r_pred + [lower, upper])
The calibration forecasts come from parameters fitted before the validation window,
so no test outcome is used.

Volatility-scaled conformal interval (adapts to the volatility level at the origin):
    s_i = EWMA volatility of daily log returns at origin i (lambda = 0.94, data <= i only)
    scores on calibration: e / (s_i * sqrt(h)); interval = r_pred + s_i * sqrt(h) * [q_lo, q_hi]

Heuristic interval (the rule used in v1, but calibrated on validation instead of test):
    price_pred +/- z * RMSE_cal(h=1) * sqrt(h)
"""
import numpy as np
import pandas as pd
from scipy import stats

from .stats_tests import kupiec


def _q(e, prob):
    e = np.sort(e)
    n = len(e)
    if prob >= 0.5:
        k = int(np.ceil((n + 1) * prob)) - 1
    else:
        k = int(np.floor((n + 1) * prob)) - 1
    return e[min(max(k, 0), n - 1)]


def build(fc, levels):
    """fc: forecasts of ONE model (cal + test rows). Returns test rows with interval bounds."""
    out = []
    for (blk, h), g in fc.groupby(["block", "h"]):
        cal = g[(g["split"] == "cal")].dropna(subset=["r_true"])
        test = g[(g["split"] == "test")].dropna(subset=["r_true"]).copy()
        if test.empty:
            continue
        e = (cal["r_true"] - cal["r_pred"]).to_numpy()
        scale_cal = cal["sigma"].to_numpy() * np.sqrt(h)
        scale_test = test["sigma"].to_numpy() * np.sqrt(h)
        cal_h1 = fc[(fc["block"] == blk) & (fc["split"] == "cal") & (fc["h"] == 1)].dropna(subset=["r_true"])
        rmse1 = np.sqrt(((cal_h1["price_true"] - cal_h1["price_pred"]) ** 2).mean())
        for L in levels:
            lo, hi = _q(e, (1 - L) / 2), _q(e, (1 + L) / 2)
            vlo, vhi = _q(e / scale_cal, (1 - L) / 2), _q(e / scale_cal, (1 + L) / 2)
            t = test[["model", "block", "h", "origin_date", "price_true", "price_pred", "price_origin", "r_pred"]].copy()
            t["level"] = L
            t["conf_lo"] = t["price_origin"] * np.exp(t["r_pred"] + lo)
            t["conf_hi"] = t["price_origin"] * np.exp(t["r_pred"] + hi)
            t["vol_lo"] = t["price_origin"] * np.exp(t["r_pred"] + vlo * scale_test)
            t["vol_hi"] = t["price_origin"] * np.exp(t["r_pred"] + vhi * scale_test)
            z = stats.norm.ppf((1 + L) / 2)
            t["heur_lo"] = t["price_pred"] - z * rmse1 * np.sqrt(h)
            t["heur_hi"] = t["price_pred"] + z * rmse1 * np.sqrt(h)
            out.append(t)
    return pd.concat(out, ignore_index=True)


def interval_score(y, lo, hi, level):
    a = 1 - level
    return (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)


def evaluate(iv, by):
    rows = []
    for key, g in iv.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        for method in ["conf", "vol", "heur"]:
            lo, hi, y = g[f"{method}_lo"], g[f"{method}_hi"], g["price_true"]
            hit = ((y >= lo) & (y <= hi)).astype(float)
            L = g["level"].iloc[0]
            lr, p = kupiec(hit, L)
            rows.append({**dict(zip(by, key)), "method": method, "n": len(g),
                         "coverage": hit.mean(), "mean_width": (hi - lo).mean(),
                         "interval_score": interval_score(y, lo, hi, L).mean(),
                         "kupiec_LR": lr, "kupiec_p": p})
    return pd.DataFrame(rows)
