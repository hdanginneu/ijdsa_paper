"""Forecast records, price conversion and accuracy metrics."""
import numpy as np
import pandas as pd

from . import config as C


def records(table, R, origins, r_pred, model, variant, seed, block, split):
    """Long table: one row per origin x horizon. Prices: P_hat = P_origin * exp(r_hat)."""
    dates = table.index
    price = table["price"].to_numpy()
    rows = []
    for j, h in enumerate(C.HORIZONS):
        tgt = origins + h
        ok = tgt < len(dates)
        rows.append(pd.DataFrame({
            "model": model, "variant": variant, "seed": seed, "block": block, "split": split,
            "origin_date": dates[origins],
            "h": h,
            "target_date": [dates[i] if o else pd.NaT for i, o in zip(np.minimum(tgt, len(dates) - 1), ok)],
            "price_origin": price[origins],
            "r_true": R[origins, j],
            "r_pred": r_pred[:, j],
        }))
    df = pd.concat(rows, ignore_index=True)
    df["price_true"] = df["price_origin"] * np.exp(df["r_true"])
    df["price_pred"] = df["price_origin"] * np.exp(df["r_pred"])
    return df


def metrics(df):
    e = df["price_true"] - df["price_pred"]
    return pd.Series({
        "n": len(df),
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAPE": (e.abs() / df["price_true"]).mean() * 100,
    })


def with_skill(tab, rw_name="RW"):
    """Add rMAE, rRMSE and R2_OOS = 1 - MSE/MSE_RW within each group (block or pooled, h)."""
    keys = [k for k in ["block", "h"] if k in tab.columns]
    rw = tab[tab["model"] == rw_name].set_index(keys)[["MAE", "RMSE"]].rename(columns=lambda c: c + "_RW")
    out = tab.join(rw, on=keys)
    out["rMAE"] = out["MAE"] / out["MAE_RW"]
    out["rRMSE"] = out["RMSE"] / out["RMSE_RW"]
    out["R2_OOS"] = 1 - (out["RMSE"] ** 2) / (out["RMSE_RW"] ** 2)
    return out.drop(columns=["MAE_RW", "RMSE_RW"])
