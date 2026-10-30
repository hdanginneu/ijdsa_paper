"""Forecasting models. Every model maps inputs at origin i to 7 log-return forecasts r(i, h)."""
import itertools
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

from . import config as C

H = len(C.HORIZONS)
HVEC = np.array(C.HORIZONS, dtype=float)


# ---------------------------------------------------------------- benchmarks
def random_walk(n):
    """No-change forecast: predicted log return is zero at every horizon."""
    return np.zeros((n, H))


def drift(train_ret, n):
    """Random walk with drift: mean daily log return on the fit set times h."""
    return np.tile(np.nanmean(train_ret) * HVEC, (n, 1))


def arima_select(logp_train, max_p=3, max_q=3):
    """Choose (p, 1, q) by AIC on the pre-block log price. Returns order and fitted params."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
            try:
                res = ARIMA(logp_train, order=(p, 1, q)).fit()
            except Exception:
                continue
            if best is None or res.aic < best[0]:
                best = (res.aic, (p, 1, q), res)
    return best[1], best[2]


def arima_forecasts(res, logp_all, origins):
    """Keep parameters fixed; at each origin filter the data up to that origin and forecast 7 steps."""
    out = np.empty((len(origins), H))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = res.apply(logp_all[: origins[-1] + 1])        # same parameters, longer data
        for k, i in enumerate(origins):
            # dynamic=True: the path from i+1 onward uses information up to i only
            pred = full.get_prediction(start=i + 1, end=i + H, dynamic=True).predicted_mean
            out[k] = np.asarray(pred) - logp_all[i]
    return out


# ---------------------------------------------------------------- ML models
class TabularModel:
    """Wraps SVR / RF / XGB. Targets are standardised per horizon on the training set."""

    def __init__(self, kind, params, seed=42):
        self.kind, self.params, self.seed = kind, dict(params), seed

    def _make(self):
        p = {k: v for k, v in self.params.items() if k != "n_lags"}
        if self.kind == "svr":
            return MultiOutputRegressor(SVR(kernel="rbf", **p))
        if self.kind == "rf":
            return RandomForestRegressor(n_estimators=300, n_jobs=2, random_state=self.seed, **p)
        if self.kind == "xgb":
            from xgboost import XGBRegressor      # imported here: never in the same process as torch (macOS OpenMP clash)
            return XGBRegressor(tree_method="hist", n_jobs=2, random_state=self.seed, **p)
        raise ValueError(self.kind)

    def fit(self, X, Y):
        self.xs = StandardScaler().fit(X)
        self.ym, self.ysd = Y.mean(0), Y.std(0)
        self.m = self._make().fit(self.xs.transform(X), (Y - self.ym) / self.ysd)
        return self

    def predict(self, X):
        return self.m.predict(self.xs.transform(X)) * self.ysd + self.ym


# ---------------------------------------------------------------- DL models


# Neural models live in src/nn_models.py (torch). They are imported lazily so that a process that fits
# XGBoost never loads torch: on macOS the two bundled OpenMP runtimes crash when both are loaded.
def __getattr__(name):
    if name in ("Net", "SeqModel", "SeqClassifier"):
        from . import nn_models
        return getattr(nn_models, name)
    raise AttributeError(name)
