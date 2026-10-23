"""Train/validation/test splitting strategies.

random_stratified reproduces the thesis code (sklearn.train_test_split with
stratify, 70/15/15).  The time-aware alternatives keep neighbouring samples
together, which is required for autocorrelated 4 Hz data.
"""
from __future__ import annotations

import numpy as np


def random_stratified(y: np.ndarray, frac=(0.70, 0.15, 0.15), seed: int = 42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        a = int(round(frac[0] * len(idx)))
        b = a + int(round(frac[1] * len(idx)))
        tr.append(idx[:a]); va.append(idx[a:b]); te.append(idx[b:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


def chronological(n: int, frac=(0.70, 0.15, 0.15)):
    a = int(frac[0] * n)
    b = a + int(frac[1] * n)
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def contiguous_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    return np.array_split(np.arange(n), n_blocks)


def blocked_folds(n: int, n_blocks: int, gap: int = 0):
    """Leave-one-block-out folds.  `gap` rows on each side of the test block are
    dropped from training (purging) to limit leakage through autocorrelation."""
    blocks = contiguous_blocks(n, n_blocks)
    for i, test in enumerate(blocks):
        lo, hi = test[0] - gap, test[-1] + gap
        train = np.r_[np.arange(0, max(lo, 0)), np.arange(min(hi + 1, n), n)]
        yield i, train, test


def positives_per_block(y: np.ndarray, n_blocks: int) -> list[int]:
    return [int(np.asarray(y)[b].sum()) for b in contiguous_blocks(len(y), n_blocks)]


def group_folds(groups: np.ndarray, k: int, order: list | None = None):
    """Leave-groups-out folds: groups (e.g. sessions) are assigned round-robin to k folds
    (in the given order), so every row of a group is either in train or in test."""
    groups = np.asarray(groups)
    uniq = list(order) if order is not None else list(np.unique(groups))
    assign = {g: i % k for i, g in enumerate(uniq)}
    fold_of = np.array([assign[g] for g in groups])
    for f in range(k):
        yield f, np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)
