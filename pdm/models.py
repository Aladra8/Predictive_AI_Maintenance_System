"""Model factory.

engine="sklearn" (default) builds the models used in the thesis.
engine="numpy" builds small dependency-free reference models (logistic
regression and a histogram random forest).  They exist so that the whole
pipeline can be smoke-tested without scikit-learn; report sklearn results
in the thesis.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- sklearn
def sklearn_models(cfg: dict, names: list[str], seed: int = 42) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    m = cfg["models"]
    rf = dict(m["rf"]); mlp = dict(m["mlp"])
    mlp["hidden_layer_sizes"] = tuple(mlp["hidden_layer_sizes"])
    out = {
        "rf": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=seed, **rf)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(random_state=seed, **mlp)),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(**m["lr"])),
        "svm": make_pipeline(StandardScaler(), SVC(random_state=seed, **m["svm"])),
    }
    return {k: out[k] for k in names}


# ------------------------------------------------------------------- numpy
class NumpyLogReg:
    """L2-regularised logistic regression, balanced class weights, full-batch gradient descent."""

    def __init__(self, lr=0.5, iters=400, l2=1e-4):
        self.lr, self.iters, self.l2 = lr, iters, l2

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w_pos = 0.5 / max(y.mean(), 1e-12); w_neg = 0.5 / max(1 - y.mean(), 1e-12)
        sw = np.where(y == 1, w_pos, w_neg); sw /= sw.mean()
        self.w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
            g = Z.T @ ((p - y) * sw) / len(y) + self.l2 * np.r_[self.w[:-1], 0]
            self.w -= self.lr * g
        return self

    def predict_proba(self, X):
        Z = np.c_[(np.asarray(X, float) - self.mu) / self.sd, np.ones(len(X))]
        p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
        return np.c_[1 - p, p]


class _Tree:
    def __init__(self, max_depth, min_leaf, n_feat, rng):
        self.max_depth, self.min_leaf, self.n_feat, self.rng = max_depth, min_leaf, n_feat, rng

    def fit(self, B, y, w, n_bins):
        self.nodes = []  # (feature, bin_threshold, left, right, value)
        self._grow(B, y, w, np.arange(len(y)), 0, n_bins)
        return self

    def _grow(self, B, y, w, idx, depth, n_bins):
        wy = w[idx] * y[idx]; tot = w[idx].sum(); pos = wy.sum()
        val = pos / tot if tot > 0 else 0.0
        node = len(self.nodes); self.nodes.append([-1, -1, -1, -1, val])
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf or pos == 0 or pos == tot:
            return node
        best = (0.0, None, None)
        parent = 1 - (val ** 2 + (1 - val) ** 2)
        for f in self.rng.choice(B.shape[1], self.n_feat, replace=False):
            b = B[idx, f]
            cw = np.bincount(b, weights=w[idx], minlength=n_bins)
            cp = np.bincount(b, weights=wy, minlength=n_bins)
            cnt = np.bincount(b, minlength=n_bins)
            Lw, Lp, Lc = np.cumsum(cw)[:-1], np.cumsum(cp)[:-1], np.cumsum(cnt)[:-1]
            Rw, Rp, Rc = tot - Lw, pos - Lp, len(idx) - Lc
            ok = (Lc >= self.min_leaf) & (Rc >= self.min_leaf) & (Lw > 0) & (Rw > 0)
            if not ok.any():
                continue
            pl = np.where(ok, Lp / np.where(Lw > 0, Lw, 1), 0); pr = np.where(ok, Rp / np.where(Rw > 0, Rw, 1), 0)
            gl = 1 - (pl ** 2 + (1 - pl) ** 2); gr = 1 - (pr ** 2 + (1 - pr) ** 2)
            gain = np.where(ok, parent - (Lw * gl + Rw * gr) / tot, -1)
            t = int(np.argmax(gain))
            if gain[t] > best[0] + 1e-12:
                best = (gain[t], f, t)
        if best[1] is None:
            return node
        f, t = best[1], best[2]
        mask = B[idx, f] <= t
        left = self._grow(B, y, w, idx[mask], depth + 1, n_bins)
        right = self._grow(B, y, w, idx[~mask], depth + 1, n_bins)
        self.nodes[node][:4] = [f, t, left, right]
        return node

    def predict(self, B):
        out = np.empty(len(B)); stack = [(0, np.arange(len(B)))]
        while stack:
            n, idx = stack.pop()
            f, t, l, r, v = self.nodes[n]
            if f < 0:
                out[idx] = v; continue
            m = B[idx, f] <= t
            stack.append((l, idx[m])); stack.append((r, idx[~m]))
        return out


class NumpyRandomForest:
    """Histogram random forest (bootstrap rows, sqrt(p) features per split,
    weighted Gini, balanced class weights).  Reference implementation only."""

    def __init__(self, n_estimators=60, max_depth=12, min_leaf=5, n_bins=64, max_samples=40000, seed=42):
        self.n_estimators, self.max_depth, self.min_leaf = n_estimators, max_depth, min_leaf
        self.n_bins, self.max_samples, self.seed = n_bins, max_samples, seed

    def _bin(self, X):
        return np.stack([np.searchsorted(self.edges[j], X[:, j], side="right") for j in range(X.shape[1])], 1).astype(np.int64)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, int)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        B = self._bin(X)
        rng = np.random.default_rng(self.seed)
        w = np.where(y == 1, 0.5 / max(y.mean(), 1e-12), 0.5 / max(1 - y.mean(), 1e-12))
        nf = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for _ in range(self.n_estimators):
            m = min(len(y), self.max_samples)
            idx = rng.integers(0, len(y), m)
            t = _Tree(self.max_depth, self.min_leaf, nf, rng).fit(B[idx], y[idx], w[idx], self.n_bins)
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        B = self._bin(np.asarray(X, float))
        p = np.mean([t.predict(B) for t in self.trees], axis=0)
        return np.c_[1 - p, p]


def numpy_models(names: list[str], seed: int = 42) -> dict:
    avail = {"rf": lambda: NumpyRandomForest(seed=seed), "lr": lambda: NumpyLogReg()}
    return {k: avail[k]() for k in names if k in avail}


def build(cfg: dict, names: list[str] | None = None, engine: str | None = None) -> dict:
    names = names or cfg["models"]["list"]
    engine = engine or cfg["models"]["engine"]
    if engine == "sklearn":
        return sklearn_models(cfg, names, cfg["seed"])
    if engine == "numpy":
        return numpy_models(names, cfg["seed"])
    raise ValueError(engine)


def fit_predict(model, Xtr, ytr, Xte, name: str = "", cfg: dict | None = None, seed: int = 42):
    """Fit and return P(fault) on Xte.  Single-class training sets return a constant."""
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(np.mean(ytr)))
    if name == "svm" and cfg is not None and len(ytr) > cfg["models"]["svm_max_train"]:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(ytr), cfg["models"]["svm_max_train"], replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]
