"""Sourced carry cost and decision evaluation with costs in USD/t (DECISIONS D-29, D-30).

Seller holding physical coffee at origin t, horizon h:
    sell now      -> 0 (reference)
    wait h days   -> payoff = P_t * (exp(r) - 1) - cost + basis noise
    cost          = storage_per_day * days + P_t * EFFR/100 * days / 360
Buyer (mirror): waiting saves the cost, payoff = cost - P_t * (exp(r) - 1) + basis noise.
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import load_fred

# ICE, "Rent Allowances - ICE Nominated Warehouse Keepers for Storage, Robusta Coffee",
# 01/05/2022-31/05/2023, Global Average Warehousekeeper Rent, bags: 14.42 USD per tonne per month.
STORAGE_USD_T_MONTH = 14.42
STORAGE_PER_DAY = STORAGE_USD_T_MONTH * 12 / 365
STORAGE_MULT = [0.0, 1.0, 2.0]          # sensitivity
BASIS_SD = [0.0, 25.0, 50.0]            # USD/t, sensitivity (D-30)


def effr_at(dates):
    """EFFR (%) known at each date: forward-filled, then lagged MACRO_LAG coffee trading days."""
    from .data import build_table
    cal = build_table().index                     # coffee trading calendar
    e = load_fred("EFFR")
    e = e.reindex(e.index.union(cal)).ffill().reindex(cal).shift(C.MACRO_LAG)
    return e.reindex(pd.DatetimeIndex(dates))


def add_cost(df, storage_mult=1.0):
    """Add 'cost' (USD/t) of waiting from origin_date to target_date."""
    d = df.copy()
    days = (pd.to_datetime(d["target_date"]) - pd.to_datetime(d["origin_date"])).dt.days.to_numpy()
    rate = effr_at(d["origin_date"].unique()).reindex(pd.to_datetime(d["origin_date"])).to_numpy()
    d["cost"] = storage_mult * STORAGE_PER_DAY * days + d["price_origin"].to_numpy() * rate / 100 * days / 360
    return d


def cost_matrix(table, storage_mult=1.0):
    """Waiting cost (USD/t) for every origin i and horizon h, shape (n, H); NaN when i + h is past the data.

    Uses only what is known at origin i: P_i, EFFR lagged MACRO_LAG days, and the trading calendar
    (target date = date of row i + h).
    """
    dates = table.index
    n, H = len(dates), len(C.HORIZONS)
    rate = effr_at(dates).to_numpy()
    price = table["price"].to_numpy()
    out = np.full((n, H), np.nan)
    for j, h in enumerate(C.HORIZONS):
        days = np.full(n, np.nan)
        days[: n - h] = (dates[h:] - dates[:-h]).days
        out[:, j] = storage_mult * STORAGE_PER_DAY * days + price * rate / 100 * days / 360
    return out


def evaluate_rule(df, side="seller", score="r_pred", storage_mult=1.0):
    """Evaluate any rule with the sourced cost (D-37).

    score = 'r_pred': point-forecast rule; score = 'p_wait': wait iff p_wait > 0.5.
    """
    d = df if "cost" in df.columns else add_cost(df, storage_mult)
    wait = (d["p_wait"] > 0.5).to_numpy() if score == "p_wait" else None
    return evaluate(d, side, wait=wait)


def expected_positive(x, sd):
    """E[max(x + e, 0)] for e ~ N(0, sd^2); equals max(x, 0) when sd = 0."""
    from scipy.stats import norm
    x = np.asarray(x, float)
    if sd == 0:
        return np.maximum(x, 0.0)
    z = x / sd
    return x * norm.cdf(z) + sd * norm.pdf(z)


def payoff_wait(d, side="seller"):
    move = d["price_origin"] * (np.exp(d["r_true"]) - 1)
    return move - d["cost"] if side == "seller" else d["cost"] - move


def evaluate(df, side="seller", wait=None, basis_sd=0.0):
    """df needs price_origin, r_true, r_pred, cost (and origin_date, h).

    wait: boolean array of the rule's action; default = point rule
          (seller waits iff expected price move exceeds cost).
    basis_sd > 0 (D-30): the realised payoff of waiting is pw + e, e ~ N(0, basis_sd^2) independent of the
    forecast. Payoff and regret are their expectation over e (closed form), so the sensitivity has no
    simulation noise: E[payoff] = wait * pw, E[regret] = E[max(pw + e, 0)] - wait * pw.
    """
    d = df.copy()
    exp_move = d["price_origin"] * (np.exp(d["r_pred"]) - 1)
    if wait is None:
        wait = (exp_move > d["cost"]) if side == "seller" else (exp_move < d["cost"])
    d["wait"] = np.asarray(wait, bool)
    pw = payoff_wait(d, side)
    d["payoff_wait"] = pw
    d["payoff"] = np.where(d["wait"], pw, 0.0)
    d["regret"] = expected_positive(pw, basis_sd) - d["payoff"]
    d["wait_correct"] = pw > 0
    d["correct"] = d["wait"] == d["wait_correct"]
    return d


def summary(d):
    pos = d["wait_correct"]
    tpr = d.loc[pos, "correct"].mean() if pos.any() else np.nan
    tnr = d.loc[~pos, "correct"].mean() if (~pos).any() else np.nan
    pay = d["payoff"].to_numpy()
    q5 = np.percentile(pay, 5)
    return pd.Series({
        "n": len(d), "accuracy": d["correct"].mean(), "balanced_accuracy": np.nanmean([tpr, tnr]),
        "share_wait": d["wait"].mean(), "mean_regret": d["regret"].mean(), "mean_payoff": pay.mean(),
        "payoff_q05": q5, "payoff_cvar05": pay[pay <= q5].mean(),
    })
