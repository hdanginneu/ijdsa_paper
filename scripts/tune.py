"""Tune each learned model once, on block B1 only: fit on B1 pre-train, score on B1 validation.

The chosen hyperparameters are frozen for every block. Usage:
    python scripts/tune.py rf full
"""
import json
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import build_table                                  # noqa: E402
from src.features import targets                                  # noqa: E402
from src.folds import make_blocks                                 # noqa: E402
from src.learned import fit_predict, price_mae, space             # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main(kind, variant):
    if (C.TUNING / f"{kind}_{variant}.json").exists():
        print(kind, variant, "already tuned, skipped (hyperparameters are frozen)", flush=True)
        return
    table = build_table()
    R = targets(table)
    b1 = [b for b in make_blocks(table) if b.name == C.TUNE_ON_BLOCK][0]

    def objective(trial):
        p = space(trial, kind)
        pred, ep = fit_predict(table, R, kind, p, variant, 42, b1.pretrain, b1.val, val_o=b1.val)
        if ep is not None:
            trial.set_user_attr("best_epoch", ep)
        return price_mae(table, R, b1.val, pred)

    t0 = time.time()
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=C.N_TRIALS)
    out = {"kind": kind, "variant": variant, "best_params": study.best_params,
           "best_val_mae": study.best_value, "seconds": round(time.time() - t0, 1),
           "trials": [{"params": t.params, "value": t.value, **t.user_attrs} for t in study.trials]}
    d = C.TUNING
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}_{variant}.json").write_text(json.dumps(out, indent=1, default=str))
    print(kind, variant, "best", round(study.best_value, 2), study.best_params, f"{out['seconds']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full")
