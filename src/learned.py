"""Search spaces, tuning objective and the fit/predict routine shared by all learned models."""
import numpy as np

from . import config as C
from .features import sequences, tabular
from .models import TabularModel

TABULAR = {"svr", "rf", "xgb"}
SEQUENCE = {"dnn", "lstm", "gru"}


def space(trial, kind):
    """Optuna search space. Same number of trials for every model (config.N_TRIALS)."""
    if kind in TABULAR:
        p = {"n_lags": trial.suggest_categorical("n_lags", [5, 20, 60])}
        if kind == "svr":
            p |= {"C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                  "epsilon": trial.suggest_float("epsilon", 1e-2, 1.0, log=True),
                  "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True)}
        elif kind == "rf":
            p |= {"max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, 12, None]),
                  "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [5, 20, 50, 100]),
                  "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 1.0])}
        else:
            p |= {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                  "max_depth": trial.suggest_int("max_depth", 2, 6),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                  "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                  "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)}
        return p
    return {"window": trial.suggest_categorical("window", [20, 60]),
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch": trial.suggest_categorical("batch", [32, 64])}


def price_mae(table, R, origins, r_pred):
    """Mean over horizons of price MAE (USD/t)."""
    p0 = table["price"].to_numpy()[origins][:, None]
    return float(np.nanmean(np.abs(p0 * np.exp(R[origins]) - p0 * np.exp(r_pred))))


def inputs(table, kind, params, variant):
    """Return a function origins -> model inputs."""
    if kind in TABULAR:
        X = tabular(table, params["n_lags"], variant)
        return lambda o: X[o]
    return lambda o: sequences(table, o, params["window"], variant)


def fit_predict(table, R, kind, params, variant, seed, train_o, pred_o, val_o=None, epochs=None):
    """Fit on train_o, predict pred_o.

    Sequence models: if val_o is given, early-stop on it (returns best epoch);
    otherwise train for `epochs` epochs.
    Returns (predictions, best_epoch or None).
    """
    X = inputs(table, kind, params, variant)
    if kind in TABULAR:
        m = TabularModel(kind, params, seed).fit(X(train_o), R[train_o])
        return m.predict(X(pred_o)), None
    from .nn_models import SeqModel                 # lazy: tabular jobs never load torch
    m = SeqModel(kind, params, seed)
    if val_o is not None:
        m.fit_early_stop(X(train_o), R[train_o], X(val_o), R[val_o])
        return m.predict(X(pred_o)), m.best_epoch
    m.fit_fixed(X(train_o), R[train_o], epochs)
    return m.predict(X(pred_o)), None
