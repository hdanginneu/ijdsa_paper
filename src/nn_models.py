"""Neural forecasting models (DNN / LSTM / GRU) and the decision-aware classifier. Requires torch."""
import numpy as np
import torch
import torch.nn as nn

from . import config as C

torch.set_num_threads(2)
H = len(C.HORIZONS)


class Net(nn.Module):
    def __init__(self, kind, n_in, window, hidden, layers, dropout):
        super().__init__()
        self.kind = kind
        if kind == "dnn":
            dims = [n_in * window] + [hidden] * layers
            mods = []
            for a, b in zip(dims[:-1], dims[1:]):
                mods += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
            self.body = nn.Sequential(*mods)
        else:
            rnn = nn.LSTM if kind == "lstm" else nn.GRU
            self.body = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        if self.kind == "dnn":
            return self.head(self.body(x.flatten(1)))
        out, _ = self.body(x)
        return self.head(self.drop(out[:, -1]))


def _seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class SeqModel:
    """DNN / LSTM / GRU with 7 outputs. Inputs and targets standardised on the training set."""

    def __init__(self, kind, params, seed=42):
        self.kind, self.p, self.seed = kind, dict(params), seed

    def _scale_fit(self, X, Y):
        flat = X.reshape(-1, X.shape[-1])
        self.xm, self.xsd = flat.mean(0), flat.std(0) + 1e-8
        self.ym, self.ysd = Y.mean(0), Y.std(0) + 1e-12

    def _tx(self, X):
        return torch.tensor((X - self.xm) / self.xsd, dtype=torch.float32)

    def _ty(self, Y):
        return torch.tensor((Y - self.ym) / self.ysd, dtype=torch.float32)

    def _new_net(self, n_in, window):
        _seed_all(self.seed)
        return Net(self.kind, n_in, window, self.p["hidden"], self.p["layers"], self.p["dropout"])

    def _epoch(self, net, opt, X, Y, g):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        for k in range(0, len(X), self.p["batch"]):
            b = perm[k:k + self.p["batch"]]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(X[b]), Y[b])
            loss.backward()
            opt.step()

    def fit_early_stop(self, X, Y, Xv, Yv):
        """Stage 1: train on pre-train, stop on validation loss. Stores best epoch."""
        self._scale_fit(X, Y)
        tx, ty, vx, vy = self._tx(X), self._ty(Y), self._tx(Xv), self._ty(Yv)
        net = self._new_net(X.shape[-1], X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"])
        g = torch.Generator().manual_seed(self.seed)
        best, best_ep, best_state, wait = np.inf, 1, None, 0
        for ep in range(1, C.DL_MAX_EPOCHS + 1):
            self._epoch(net, opt, tx, ty, g)
            net.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(net(vx), vy).item()
            if v < best - 1e-6:
                best, best_ep, wait = v, ep, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= C.DL_PATIENCE:
                    break
        net.load_state_dict(best_state)
        self.net, self.best_epoch = net, best_ep
        return self

    def fit_fixed(self, X, Y, epochs):
        """Stage 2: train from scratch on the full fit set for a fixed number of epochs."""
        self._scale_fit(X, Y)
        tx, ty = self._tx(X), self._ty(Y)
        net = self._new_net(X.shape[-1], X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"])
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            self._epoch(net, opt, tx, ty, g)
        self.net = net
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            z = self.net(self._tx(X)).numpy()
        return z * self.ysd + self.ym


class SeqClassifier(SeqModel):
    """Same network as SeqModel, but trained with a cost-sensitive logistic loss.

    Y = 0/1 labels (wait is correct) for the 7 horizons; W = sample weights |r - threshold|.
    Weights are normalised to mean 1 on the training set.
    """

    def _scale_fit(self, X, Y):
        flat = X.reshape(-1, X.shape[-1])
        self.xm, self.xsd = flat.mean(0), flat.std(0) + 1e-8

    def _ty(self, Y):
        return torch.tensor(Y, dtype=torch.float32)

    def _loss(self, logits, y, w):
        return (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()

    def _run(self, X, Y, W, epochs=None, Xv=None, Yv=None, Wv=None):
        self._scale_fit(X, Y)
        tx, ty = self._tx(X), self._ty(Y)
        tw = torch.tensor(W / W.mean(), dtype=torch.float32)
        net = self._new_net(X.shape[-1], X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"])
        g = torch.Generator().manual_seed(self.seed)
        if Xv is not None:
            vx, vy = self._tx(Xv), self._ty(Yv)
            vw = torch.tensor(Wv / Wv.mean(), dtype=torch.float32)
        best, best_ep, best_state, wait = np.inf, 1, None, 0
        n_ep = C.DL_MAX_EPOCHS if epochs is None else epochs
        for ep in range(1, n_ep + 1):
            net.train()
            perm = torch.randperm(len(tx), generator=g)
            for k in range(0, len(tx), self.p["batch"]):
                b = perm[k:k + self.p["batch"]]
                opt.zero_grad()
                self._loss(net(tx[b]), ty[b], tw[b]).backward()
                opt.step()
            if Xv is None:
                continue
            net.eval()
            with torch.no_grad():
                v = self._loss(net(vx), vy, vw).item()
            if v < best - 1e-6:
                best, best_ep, wait = v, ep, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= C.DL_PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net, self.best_epoch = net, best_ep
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(self._tx(X))).numpy()
