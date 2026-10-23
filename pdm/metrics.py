"""Metrics implemented with NumPy only, so results do not depend on library versions."""
from __future__ import annotations

import numpy as np


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC with tie correction."""
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    import pandas as pd
    ranks = pd.Series(s).rank(method="average").to_numpy()   # average ranks for ties
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    # evaluate only at distinct score thresholds
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    prec = tp[last] / (tp[last] + fp[last]); rec = tp[last] / y.sum()
    rec_prev = np.r_[0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def confusion(y, yhat):
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, fp, tn, fn


def classification_metrics(y, score, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(score) >= threshold).astype(int)
    tp, fp, tn, fn = confusion(y, yhat)
    p1 = tp / (tp + fp) if tp + fp else 0.0; r1 = tp / (tp + fn) if tp + fn else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0; r0 = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return {"n": int(len(y)), "positives": int(y.sum()),
            "accuracy": (tp + tn) / len(y), "precision_fault": p1, "recall_fault": r1,
            "f1_fault": f1, "f1_macro": (f1 + f0) / 2,
            "roc_auc": roc_auc(y, score), "avg_precision": average_precision(y, score),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def bootstrap_ci(y, score, metric="f1_macro", n_boot=500, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); score = np.asarray(score); vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        vals.append(classification_metrics(y[i], score[i])[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def event_recall(episodes, score, threshold=0.5, hz=4.0, lookback_s=0.0):
    """Share of episodes warned in advance: at least one alarm in the window
    [start - lookback_s, start).  Also returns the median lead time (seconds
    between the first alarm in that window and the episode start)."""
    score = np.asarray(score); hits = 0; leads = []
    for _, r in episodes.iterrows():
        lo = max(0, int(r.start_row - lookback_s * hz)); hi = int(r.start_row)
        al = np.flatnonzero(score[lo:hi] >= threshold)
        if al.size:
            hits += 1
            leads.append((r.start_row - (lo + al[0])) / hz)
    n = len(episodes)
    return {"episodes": n, "detected": hits, "event_recall": hits / n if n else float("nan"),
            "median_lead_s": float(np.median(leads)) if leads else float("nan")}
