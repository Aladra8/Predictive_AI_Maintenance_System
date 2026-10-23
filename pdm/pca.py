"""PCA (via SVD) and a sampled silhouette score, NumPy only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca(X: pd.DataFrame, n_components: int = 2):
    Z = (X - X.mean()) / X.std(ddof=0)
    Z = Z.fillna(0.0).to_numpy()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    scores = Z @ Vt[:n_components].T
    loadings = pd.DataFrame(Vt[:n_components].T, index=X.columns,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    # sign convention: largest absolute loading positive
    for c in loadings:
        if loadings[c].abs().idxmax() and loadings[c][loadings[c].abs().idxmax()] < 0:
            loadings[c] *= -1
            scores[:, int(c[2:]) - 1] *= -1
    return scores, var, loadings


def silhouette(Z: np.ndarray, y: np.ndarray, sample: int = 10000, seed: int = 42) -> float:
    """Mean silhouette on a stratified random sample (exact silhouette is O(n^2))."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        k = max(2, int(round(sample * len(ci) / len(y))))
        idx.append(rng.choice(ci, min(k, len(ci)), replace=False))
    idx = np.concatenate(idx); Zs = Z[idx]; ys = y[idx]
    sq = (Zs ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Zs @ Zs.T, 0))
    s = np.zeros(len(ys))
    for i in range(len(ys)):
        same = ys == ys[i]; same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min(D[i, ys == c].mean() for c in np.unique(ys) if c != ys[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())
