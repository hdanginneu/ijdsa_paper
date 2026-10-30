"""Zero-shot Chronos-Bolt (small) forecasts for every block (RQ4 expensive model, D-15b, D-24).

Context = the last 512 closing prices up to and including the origin. No training, no tuning.
Point forecast = median; the native 10% and 90% quantiles are saved as well.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.evaluate import records                                  # noqa: E402
from src.features import targets                                  # noqa: E402
from src.folds import make_blocks                                 # noqa: E402

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT = 512
H = len(C.HORIZONS)


def forecast(pipe, price, origins, batch=64):
    med, q10, q90 = [], [], []
    for k in range(0, len(origins), batch):
        ctx = [torch.tensor(price[max(0, i - CONTEXT + 1): i + 1], dtype=torch.float32) for i in origins[k:k + batch]]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
        q = q.numpy()
        q10.append(q[:, :, 0]); med.append(q[:, :, 1]); q90.append(q[:, :, 2])
    return np.vstack(med), np.vstack(q10), np.vstack(q90)


def main():
    torch.set_num_threads(2)
    if (C.FORECASTS / "Chronos_full_seed0.csv").exists():
        print("Chronos already done, skipped", flush=True)
        return
    C.FORECASTS.mkdir(parents=True, exist_ok=True)
    table = build_table()
    R = targets(table)
    price = table["price"].to_numpy()
    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)
    parts, native, timing = [], [], []
    for b in make_blocks(table):
        for split, origins in [("cal", b.val), ("test", b.origins)]:
            t0 = time.time()
            med, q10, q90 = forecast(pipe, price, origins)
            secs = time.time() - t0
            p0 = price[origins][:, None]
            parts.append(records(table, R, origins, np.log(med / p0), "Chronos", "full", 0, b.name, split))
            native.append(pd.DataFrame({"block": b.name, "split": split,
                                        "origin_date": np.repeat(table.index[origins], H),
                                        "h": np.tile(C.HORIZONS, len(origins)),
                                        "q10": q10.ravel(), "q50": med.ravel(), "q90": q90.ravel()}))
            timing.append({"block": b.name, "split": split, "n_origins": len(origins), "seconds": secs})
            print(b.name, split, f"{secs:.1f}s", flush=True)
    pd.concat(parts).to_csv(C.FORECASTS / "Chronos_full_seed0.csv", index=False)
    pd.concat(native).to_csv(C.RESULTS / "chronos_native_quantiles.csv", index=False)
    pd.DataFrame(timing).to_csv(C.RESULTS / "chronos_timing.csv", index=False)


if __name__ == "__main__":
    main()
