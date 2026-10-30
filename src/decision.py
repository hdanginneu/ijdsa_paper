"""Stylised sell-now-or-wait decision.

Seller holding physical Robusta at origin t:
    sell now        -> receives P_t
    wait h days     -> receives P_{t+h} and pays carry cost kappa * h * P_t
Payoff of waiting relative to selling now:  P_t * (exp(r) - 1 - kappa*h)
Wait is the correct action iff r > log(1 + kappa*h).

Buyer (mirror): wait saves the carry cost; payoff of waiting = P_t * (1 + kappa*h - exp(r)),
wait is correct iff r < log(1 + kappa*h).
"""
import numpy as np
import pandas as pd

KAPPA = 0.0002          # primary: 0.02% per day
KAPPAS = [0.0, 0.0002, 0.0004]


def threshold(h, kappa):
    return np.log1p(kappa * np.asarray(h, float))


def payoff_wait(price_origin, r, h, kappa, side="seller"):
    if side == "seller":
        return price_origin * (np.exp(r) - 1 - kappa * h)
    return price_origin * (1 + kappa * h - np.exp(r))


def evaluate(df, kappa=KAPPA, side="seller", score="r_pred"):
    """Add action, correct action, payoff and regret columns.

    score = "r_pred": rule from a point forecast (wait if r_pred beyond the threshold)
    score = "p_wait": rule from a decision-aware classifier (wait if p_wait > 0.5)
    """
    d = df.copy()
    thr = threshold(d["h"], kappa)
    if score == "p_wait":
        d["wait"] = d["p_wait"] > 0.5
    elif side == "seller":
        d["wait"] = d["r_pred"] > thr
    else:
        d["wait"] = d["r_pred"] < thr
    d["wait_correct"] = (d["r_true"] > thr) if side == "seller" else (d["r_true"] < thr)
    pw = payoff_wait(d["price_origin"], d["r_true"], d["h"], kappa, side)
    d["payoff"] = np.where(d["wait"], pw, 0.0)                  # relative to acting now
    d["regret"] = np.maximum(pw, 0.0) - d["payoff"]
    d["correct"] = d["wait"] == d["wait_correct"]
    return d


def summary(d):
    pos = d["wait_correct"]
    tpr = d.loc[pos, "correct"].mean() if pos.any() else np.nan
    tnr = d.loc[~pos, "correct"].mean() if (~pos).any() else np.nan
    return pd.Series({
        "n": len(d),
        "accuracy": d["correct"].mean(),
        "balanced_accuracy": np.nanmean([tpr, tnr]),
        "share_wait": d["wait"].mean(),
        "mean_regret": d["regret"].mean(),
        "mean_payoff": d["payoff"].mean(),
    })
