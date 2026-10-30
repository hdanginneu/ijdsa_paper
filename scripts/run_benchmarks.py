"""RW, RW+drift and ARIMA for every block.

split = "test": forecasts for the block origins (parameters estimated on data up to the block start)
split = "cal" : forecasts for the block's validation origins from parameters estimated on pre-train only
                (used later to calibrate prediction intervals)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, models as M                         # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.evaluate import records                                  # noqa: E402
from src.features import targets                                  # noqa: E402
from src.folds import make_blocks, describe                       # noqa: E402


def main(variant="full", macro_lag=None):
    table = build_table(macro_lag=macro_lag)
    R = targets(table)
    blocks = make_blocks(table)
    C.FORECASTS.mkdir(parents=True, exist_ok=True)
    if all((C.FORECASTS / f"{m}_{variant}_seed0.csv").exists() for m in ["RW", "Drift", "ARIMA"]):
        print("benchmarks already done, skipped", flush=True)
        return
    pd.DataFrame(describe(blocks, table)).to_csv(C.RESULTS / "blocks.csv", index=False)
    lp, ret = table["logp"].to_numpy(), table["ret"].to_numpy()

    out = {"RW": [], "Drift": [], "ARIMA": []}
    orders = []
    for b in blocks:
        for split, train_start, train_end, origins in [("test", b.fit[0], b.fit[-1] + C.GAP, b.origins),
                                                       ("cal", b.pretrain[0], b.pretrain[-1] + C.GAP, b.val)]:
            # expanding window: train_start = FIRST_USABLE (same as before); rolling window: later start (D-31)
            train_start = 0 if C.TRAIN_WINDOW is None else train_start
            # train_end = last row whose value is known at the first origin of the segment
            out["RW"].append(records(table, R, origins, M.random_walk(len(origins)), "RW", variant, 0, b.name, split))
            out["Drift"].append(records(table, R, origins, M.drift(ret[max(1, train_start):train_end + 1], len(origins)),
                                        "Drift", variant, 0, b.name, split))
            order, res = M.arima_select(lp[train_start: train_end + 1])
            f = M.arima_forecasts(res, lp, origins)
            out["ARIMA"].append(records(table, R, origins, f, "ARIMA", variant, 0, b.name, split))
            orders.append({"block": b.name, "split": split, "order": str(order)})
            print(b.name, split, "ARIMA order", order, flush=True)

    for name, parts in out.items():
        pd.concat(parts).to_csv(C.FORECASTS / f"{name}_{variant}_seed0.csv", index=False)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orders).to_csv(C.RESULTS / "arima_orders.csv", index=False)


if __name__ == "__main__":
    main()
