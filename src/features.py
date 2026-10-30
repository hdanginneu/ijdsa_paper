"""Targets and inputs. Every feature at origin i uses rows <= i only."""
import numpy as np
import pandas as pd

from . import config as C

MACRO = ["d_usd", "d_effr", "d_oil"]
SEQ_FULL = ["ret", "d_usd", "d_effr", "d_oil", "log_vol", "vol_missing"]
SEQ_PRICE = ["ret"]


def targets(table):
    """R[i, h-1] = log P_{i+h} - log P_i  (NaN when i+h is beyond the data)."""
    logp = table["logp"].to_numpy()
    n = len(logp)
    R = np.full((n, len(C.HORIZONS)), np.nan)
    for j, h in enumerate(C.HORIZONS):
        R[: n - h, j] = logp[h:] - logp[: n - h]
    return R


def tabular(table, n_lags, variant="full"):
    """Feature matrix for SVR / RF / XGB.

    ret lags 0..n_lags-1, rolling mean/std of ret (5, 20, 60);
    full variant adds macro changes (lags 0..4 of the already-published series)
    and the volume level and missing flag.
    """
    ret = table["ret"]
    cols = {f"ret_l{k}": ret.shift(k) for k in range(n_lags)}
    for w in (5, 20, 60):
        cols[f"ret_mean{w}"] = ret.rolling(w).mean()
        cols[f"ret_std{w}"] = ret.rolling(w).std()
    if variant == "full":
        for m in MACRO:
            for k in range(5):
                cols[f"{m}_l{k}"] = table[m].shift(k)
        cols["log_vol"] = table["log_vol"]
        cols["vol_missing"] = table["vol_missing"]
    return pd.DataFrame(cols, index=table.index).to_numpy(dtype=np.float32)


def sequences(table, origins, window, variant="full"):
    """3-D array (n_origins, window, channels): rows origin-window+1 .. origin."""
    chans = SEQ_FULL if variant == "full" else SEQ_PRICE
    X = table[chans].to_numpy(dtype=np.float32)
    idx = origins[:, None] + np.arange(-window + 1, 1)[None, :]
    return X[idx]
