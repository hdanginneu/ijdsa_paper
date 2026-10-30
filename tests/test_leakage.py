"""Leakage checks. Run: python3 tests/test_leakage.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.features import sequences, tabular, targets              # noqa: E402
from src.folds import make_blocks                                 # noqa: E402


def main():
    t = build_table()
    R = targets(t)
    blocks = make_blocks(t)

    # 1. every training target is observed before the first origin of its segment
    for b in blocks:
        assert b.fit[-1] + max(C.HORIZONS) <= b.origins[0], b.name
        assert b.pretrain[-1] + max(C.HORIZONS) <= b.val[0], b.name
        assert not np.isnan(R[b.fit]).any() and not np.isnan(R[b.pretrain]).any(), b.name
    # blocks do not overlap and are in time order
    for a, c in zip(blocks[:-1], blocks[1:]):
        assert a.origins[-1] < c.origins[0]

    # 2. features at origin i are identical when the table is cut at i (no future rows used)
    rng = np.random.default_rng(0)
    full_tab = tabular(t, 60, "full")
    for i in rng.choice(np.arange(200, len(t) - 10), 25, replace=False):
        cut = t.iloc[: i + 1]
        assert np.allclose(tabular(cut, 60, "full")[i], full_tab[i], equal_nan=True), i
        assert np.allclose(sequences(cut, np.array([i]), 60)[0], sequences(t, np.array([i]), 60)[0]), i

    # 3. macro columns are lagged: changing a macro value at date d cannot change features before d + lag
    t2 = t.copy()
    d = 3000
    t2.iloc[d, t2.columns.get_loc("d_usd")] += 1.0     # perturb the (already lagged) column itself
    assert np.allclose(tabular(t2, 60)[:d], tabular(t, 60)[:d], equal_nan=True)

    # 4. validation targets lie before the block (val[-1] + 7 <= first origin)
    for b in blocks:
        assert b.val[-1] + max(C.HORIZONS) <= b.origins[0], b.name

    # 5. shrinkage weight: w ~ 0 when cal r_true is unrelated to r_pred, w = 1 when r_true = r_pred
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_combos import choose_weight
    n = 400
    noise = pd.DataFrame({"price_origin": 2000.0, "r_pred": rng.normal(0, 0.02, n), "r_true": rng.normal(0, 0.02, n)})
    assert choose_weight(noise)[0] <= 0.1, choose_weight(noise)
    exact = noise.assign(r_true=noise["r_pred"])
    assert choose_weight(exact)[0] == 1.0, choose_weight(exact)

    # 6. carry cost at origin i does not change when the table is cut at i + 7 (no future information)
    from src.costs import cost_matrix
    full_cost = cost_matrix(t)
    for i in rng.choice(np.arange(200, len(t) - 10), 10, replace=False):
        cut = cost_matrix(t.iloc[: i + max(C.HORIZONS) + 1])
        assert np.allclose(cut[i], full_cost[i]), i

    # 7. usdlag7: d_usd = raw Δlog DTWEXBGS lagged exactly 7 coffee days; d_effr unchanged (lag 6)
    import os
    import subprocess
    import tempfile
    from src.data import load_coffee, load_fred
    with tempfile.TemporaryDirectory() as tmp:
        code = ("import sys; sys.path.insert(0, '.'); from src.data import build_table; "
                f"build_table()[['d_usd', 'd_effr']].to_csv(r'{tmp}/t7.csv')")
        subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1],
                       env={**os.environ, "IJDSA_USD_LAG": "7"})
        t7 = pd.read_csv(f"{tmp}/t7.csv", index_col=0, parse_dates=True)
    usd = load_fred("DTWEXBGS")
    cal = load_coffee().set_index("date").index                  # coffee trading calendar (before dropna)
    cal = cal[cal >= C.START_DATE]
    d_raw = np.log(usd.reindex(usd.index.union(cal)).ffill().reindex(cal)).diff()
    common = t.index.intersection(t7.index)
    assert np.allclose(t7.loc[common, "d_usd"], d_raw.shift(7).loc[common], equal_nan=True), "usdlag7 d_usd"
    assert np.allclose(t7.loc[common, "d_effr"], t.loc[common, "d_effr"]), "usdlag7 must not change d_effr"
    assert not np.allclose(t7.loc[common, "d_usd"], t.loc[common, "d_usd"]), "usdlag7 d_usd should differ"

    print("all leakage checks passed:", [b.name for b in blocks])


if __name__ == "__main__":
    main()
