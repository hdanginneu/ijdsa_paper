"""Checks of the Lo-MacKinlay variance ratio in scripts/analyze_v4.py (D-64). Run: python3 tests/test_variance_ratio.py

1. AR(1) returns with rho = 0.3 and large n: VR(q) clearly above 1, z* large, p tiny, for q = 2, 5, 10.
2. i.i.d. returns: the test does not reject systematically (rejection rate at 5% close to 5%).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_v4 import VR_Q, variance_ratio                       # noqa: E402

RHO, N_AR, N_IID, REPS = 0.3, 20_000, 2_000, 400


def main():
    rng = np.random.default_rng(2026)

    # 1. AR(1), rho = 0.3: population VR(2) = 1 + rho = 1.3
    e = rng.standard_normal(N_AR)
    r = np.empty(N_AR)
    r[0] = e[0]
    for t in range(1, N_AR):
        r[t] = RHO * r[t - 1] + e[t]
    for q in VR_Q:
        vr, z, p = variance_ratio(r, q)
        print(f"AR(1) rho={RHO} n={N_AR} q={q}: VR={vr:.3f} z*={z:.2f} p={p:.2e}")
        assert vr > 1.2 and z > 10 and p < 1e-6, (q, vr, z, p)
    vr2 = variance_ratio(r, 2)[0]
    assert abs(vr2 - (1 + RHO)) < 0.03, vr2

    # 2. i.i.d. (Gaussian and Student-t with 4 df): rejection rate at 5% within [2%, 9%] for every q
    for name, draw in [("normal", lambda: rng.standard_normal(N_IID)),
                       ("t4", lambda: rng.standard_t(4, N_IID))]:
        rej = {q: 0 for q in VR_Q}
        zs = {q: [] for q in VR_Q}
        for _ in range(REPS):
            x = draw()
            for q in VR_Q:
                _, z, p = variance_ratio(x, q)
                rej[q] += p < 0.05
                zs[q].append(z)
        for q in VR_Q:
            rate = rej[q] / REPS
            print(f"iid {name} n={N_IID} q={q}: rejection rate {rate:.3f}, mean z* {np.mean(zs[q]):+.3f}, "
                  f"sd z* {np.std(zs[q]):.3f}")
            assert 0.02 <= rate <= 0.09, (name, q, rate)
    print("variance ratio test passed")


if __name__ == "__main__":
    main()
