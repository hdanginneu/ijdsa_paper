"""Rolling-origin evaluation blocks.

Positions are row numbers in the modelling table. An ORIGIN i means: the forecast is
made after the close of day i, using rows 0..i only, for targets i+1..i+7.

For a block whose first origin is o:
  fit set   : origins s <= o - GAP            (so every target s+h <= o is already observed)
  validation: the last VAL_N origins of the fit set
  pre-train : origins s <= val_start - GAP    (used only to tune / choose epochs / calibrate)
"""
from dataclasses import dataclass

import numpy as np

from . import config as C

FIRST_USABLE = 120      # leaves room for a 60-day window plus 60-day rolling statistics


@dataclass
class Block:
    name: str
    origins: np.ndarray        # evaluation origins
    fit: np.ndarray            # origins for the final refit
    val: np.ndarray            # validation origins
    pretrain: np.ndarray       # origins used before validation


def _block(name, first, last, n_rows):
    refit_end = first - C.GAP
    val_start = refit_end - C.VAL_N + 1
    pre_end = val_start - C.GAP
    fit = np.arange(FIRST_USABLE, refit_end + 1)
    pretrain = np.arange(FIRST_USABLE, pre_end + 1)
    if C.TRAIN_WINDOW:                               # rolling window: keep only the most recent origins
        fit, pretrain = fit[-C.TRAIN_WINDOW:], pretrain[-C.TRAIN_WINDOW:]
    return Block(
        name=name,
        origins=np.arange(first, min(last, n_rows - 2) + 1),
        fit=fit,
        val=np.arange(val_start, refit_end + 1),
        pretrain=pretrain,
    )


def make_blocks(table):
    dates = table.index
    n = len(dates)
    o5 = int(np.searchsorted(dates, np.datetime64(C.B5_FIRST_TARGET))) - 1   # h=1 target = first test day
    blocks = []
    for k in range(C.EARLY_BLOCKS, 0, -1):
        first = o5 - k * C.EARLY_BLOCK_N
        blocks.append(_block(f"B{C.EARLY_BLOCKS - k + 1}", first, first + C.EARLY_BLOCK_N - 1, n))
    blocks.append(_block("B5", o5, o5 + C.B5_N - 1, n))
    o6 = o5 + C.B5_N
    if o6 <= n - 2:                                   # data continue after the v1 test window
        last6 = n - 2 if C.B6_MAX is None else min(n - 2, o6 + C.B6_MAX - 1)
        blocks.append(_block("B6", o6, last6, n))
    return blocks


def describe(blocks, table):
    d = table.index
    rows = []
    for b in blocks:
        rows.append({
            "block": b.name,
            "first_origin": d[b.origins[0]].date(), "last_origin": d[b.origins[-1]].date(),
            "n_origins": len(b.origins),
            "fit_end": d[b.fit[-1]].date(),
            "val": f"{d[b.val[0]].date()}..{d[b.val[-1]].date()}",
            "pretrain_end": d[b.pretrain[-1]].date(),
        })
    return rows
