"""Load raw files and build one daily table on the ICE coffee trading calendar."""
import numpy as np
import pandas as pd

from . import config as C


def _to_number(text):
    """'4,148.00' -> 4148.0 ; '4.87K' -> 4870 ; '' or '-' -> NaN."""
    s = str(text).strip().replace(",", "")
    if s in ("", "-", "nan", "None"):
        return np.nan
    mult = 1.0
    if s[-1] in "KkMmBb":
        mult = {"k": 1e3, "m": 1e6, "b": 1e9}[s[-1].lower()]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def load_coffee(path=None):
    """Coffee closes and volume.

    Two formats are accepted:
    - the Sep-2026 update file (columns date, close, open, high, low, volume, source)
    - the v1 Investing.com export (Vietnamese column names, dd/mm/YYYY dates)
    """
    path = path or C.COFFEE_FILE
    df = _read_coffee(path)
    if C.PRICE_UNIT == "cents_lb":                   # Arabica (ICE US): US cents/lb -> USD/t
        df["price"] = df["price"] * C.CENTS_LB_TO_USD_T
    return df


def _read_coffee(path):
    head = pd.read_csv(path, nrows=1, encoding="utf-8-sig")
    head.columns = [c.strip() for c in head.columns]
    if "close" in head.columns:
        raw = pd.read_csv(path, parse_dates=["date"])
        df = pd.DataFrame({"date": raw["date"], "price": raw["close"].astype(float),
                           "volume": pd.to_numeric(raw["volume"], errors="coerce")})
        if C.DROP_FLAT_ROWS:
            flat = (raw["open"] == raw["high"]) & (raw["high"] == raw["low"]) & (raw["low"] == raw["close"])
            df = df[~(flat & (raw["date"] >= C.DROP_FLAT_FROM))]
        return df.sort_values("date").reset_index(drop=True)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    date_col, price_col = raw.columns[0], raw.columns[1]
    vol_col = next((c for c in raw.columns if c.lower().startswith(("vol", "kl"))), raw.columns[5])
    fmt = "%m/%d/%Y" if date_col.lower() == "date" else "%d/%m/%Y"    # English vs Vietnamese Investing export
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], format=fmt),
        "price": raw[price_col].map(_to_number),
        "volume": raw[vol_col].map(_to_number),     # blank stays NaN (not zero)
    })
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def load_fred(series):
    d = pd.read_csv(C.FRED_DIR / f"{series}.csv")
    d.columns = ["date", series]
    d["date"] = pd.to_datetime(d["date"])
    d[series] = pd.to_numeric(d[series], errors="coerce")
    return d.set_index("date")[series]


def build_table(coffee_path=None, macro_lag=None):
    """Return the modelling table.

    Columns
    -------
    price, logp, ret        : coffee close, log close, daily log return
    d_usd, d_effr, d_oil     : macro changes, already lagged by `macro_lag` trading days
    log_vol, vol_missing     : log(1+volume) forward-filled, and a 0/1 missing flag
    """
    macro_lag = C.MACRO_LAG if macro_lag is None else macro_lag
    cof = load_coffee(coffee_path)
    cof = cof[cof["date"] >= C.START_DATE]
    if C.END_DATE:
        cof = cof[cof["date"] <= C.END_DATE]
    cof = cof.set_index("date")

    # Align macro to the coffee calendar. Forward fill uses only past values.
    macro = pd.concat([load_fred(s) for s in ["DTWEXBGS", "EFFR", "DCOILWTICO"]], axis=1)
    full_index = macro.index.union(cof.index)
    macro = macro.reindex(full_index).ffill().reindex(cof.index)

    t = pd.DataFrame(index=cof.index)
    t["price"] = cof["price"]
    t["logp"] = np.log(t["price"])
    t["ret"] = t["logp"].diff()

    # WTI was negative on 2020-04-20, so oil uses a plain difference, not a log difference.
    d_usd = np.log(macro["DTWEXBGS"]).diff()
    d_effr = macro["EFFR"].diff()
    d_oil = macro["DCOILWTICO"].diff()
    t["d_usd"] = d_usd.shift(C.USD_LAG if C.USD_LAG is not None else macro_lag)    # D-56
    t["d_effr"] = d_effr.shift(macro_lag)
    t["d_oil"] = d_oil.shift(macro_lag)

    vol = cof["volume"].where(cof["volume"] > 0)        # 0 or blank = not reported
    t["vol_missing"] = vol.isna().astype(float)
    # Before the first reported volume there is nothing to carry forward: use 0 (flag = 1).
    t["log_vol"] = np.log1p(vol).ffill().fillna(0.0)

    t = t.dropna().copy()                                 # drops the first rows only
    t.index.name = "date"
    return t
