"""Suspected contract-roll days in the Investing.com continuous series (DECISIONS D-49).

For each delivery month and year: window = ±10 trading days around the anchor; inside the window the day
with the largest overnight gap |log open_t - log close_{t-1}| is flagged if the gap exceeds 3 x the
full-sample median gap. Anchors (ROLL_NOTES.md): First Notice Day (primary, D-49) and Last Trading Day
(supplementary). Business days are approximated by the trading days in the file.
Output: <main results>/tables/roll_calendar_{robusta|arabica}.csv and *_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402

WINDOW, MULT = 10, 3.0
SPECS = {   # delivery months, FND = k-th trading day before the first trading day of the month,
            # LTD = k-th trading day before the last trading day of the month (Arabica: LND-1, LND ~ 7th, [unverified])
    "robusta": {"file": C.COFFEE_FILE, "months": [1, 3, 5, 7, 9, 11], "fnd_k": 4, "ltd_k": 4},
    "arabica": {"file": C.ARABICA_FILE, "months": [3, 5, 7, 9, 12], "fnd_k": 7, "ltd_k": 8},
}


def anchors(dates, months, fnd_k, ltd_k):
    """Return list of (contract_month, anchor_type, anchor_position)."""
    out = []
    ym = dates.to_period("M")
    for per in ym.unique():
        if per.month not in months:
            continue
        pos = np.where(ym == per)[0]
        first, last = pos[0], pos[-1]
        label = str(per)
        if first - fnd_k >= 0:
            out.append((label, "fnd", first - fnd_k))
        if last - ltd_k >= 0 and last < len(dates) - 1:            # month complete in the data
            out.append((label, "ltd", last - ltd_k))
    return out


def calendar(name, spec):
    raw = pd.read_csv(spec["file"], parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    raw = raw[raw["date"] >= C.START_DATE].reset_index(drop=True)
    gap = (np.log(raw["open"]) - np.log(raw["close"].shift(1))).abs()
    ret = np.log(raw["close"]).diff()
    med = gap.median()
    dates = pd.DatetimeIndex(raw["date"])
    rows = []
    for month, kind, a in anchors(dates, spec["months"], spec["fnd_k"], spec["ltd_k"]):
        lo, hi = max(1, a - WINDOW), min(len(raw) - 1, a + WINDOW)
        w = gap.iloc[lo:hi + 1]
        k = int(w.idxmax())
        flagged = w.loc[k] > MULT * med
        rows.append({"anchor_type": kind, "contract_month": month, "anchor_date": dates[a].date(),
                     "window_start": dates[lo].date(), "window_end": dates[hi].date(),
                     "flagged_date": dates[k].date() if flagged else None,
                     "gap": w.loc[k] if flagged else np.nan, "gap_over_median": w.loc[k] / med,
                     "close_return_on_flagged": ret.loc[k] if flagged else np.nan})
    cal = pd.DataFrame(rows)
    summ = []
    for kind, g in cal.groupby("anchor_type"):
        fl = g.dropna(subset=["flagged_date"])
        r_fl = fl["close_return_on_flagged"].abs()
        r_all = ret.abs().dropna()
        summ.append({"anchor_type": kind, "n_windows": len(g), "n_flagged": len(fl), "median_gap_full_sample": med,
                     "gap_over_median_p50": g["gap_over_median"].median(),
                     "gap_over_median_p90": g["gap_over_median"].quantile(0.9),
                     "gap_over_median_max": g["gap_over_median"].max(),
                     "mean_abs_return_flagged": r_fl.mean(), "mean_abs_return_all_days": r_all.mean(),
                     "median_percentile_of_flagged_abs_return": float(np.mean([(r_all <= x).mean() for x in r_fl]))
                     if len(r_fl) else np.nan})
    return cal, pd.DataFrame(summ)


def main():
    out = C.MAIN_RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in SPECS.items():
        if not Path(spec["file"]).exists():
            print("skip", name, "(file not found)")
            continue
        cal, summ = calendar(name, spec)
        cal.to_csv(out / f"roll_calendar_{name}.csv", index=False)
        summ.to_csv(out / f"roll_calendar_{name}_summary.csv", index=False)
        print(name)
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
