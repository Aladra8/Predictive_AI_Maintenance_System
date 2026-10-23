"""Unit tests (run with `pytest` or `python tests/test_core.py`)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdm import events, labels, metrics, splits  # noqa: E402
from pdm.audit import r2_affine  # noqa: E402


def test_percentile_vote_counts():
    df = pd.DataFrame({"a": np.arange(100), "b": np.arange(100), "c": np.arange(100)[::-1]})
    y, thr, _ = labels.percentile_vote(df, ["a", "b", "c"], 0.9, 2)
    assert y.sum() == 10                     # rows 90..99 exceed a and b (q90 = 89.1)
    assert thr["a"] == df.a.quantile(0.9)


def test_merge_by_gap():
    assert labels.merge_by_gap(np.array([1, 2, 3, 10, 11, 30]), 5) == [(1, 3), (10, 11), (30, 30)]


def test_vibration_episodes_and_onset():
    v = np.zeros(400); v[100:110] = 10; v[300:305] = 10
    ep, y, thr = labels.vibration_episodes(v, 0.95, merge_gap_s=5, hz=4)
    assert len(ep) == 2 and y.sum() == 15
    on = labels.onset_target(ep, 400, horizon_s=5, hz=4)
    assert on[80:100].all() and not on[100:110].any()


def test_auc_ap():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(metrics.roc_auc(y, s) - 0.75) < 1e-12
    assert abs(metrics.average_precision(y, s) - (0.5 * 1 + 0.5 * 2 / 3)) < 1e-12
    assert metrics.roc_auc(np.array([1, 1]), np.array([0.1, 0.2])) != metrics.roc_auc(np.array([1, 1]), np.array([0.1, 0.2]))  # nan


def test_splits():
    y = np.r_[np.zeros(90), np.ones(10)].astype(int)
    tr, va, te = splits.random_stratified(y, (0.7, 0.15, 0.15), 0)
    assert len(set(tr) | set(va) | set(te)) == 100 and y[te].sum() in (1, 2)
    folds = list(splits.blocked_folds(100, 5, gap=3))
    assert len(folds) == 5 and all(len(np.intersect1d(trn, tst)) == 0 for _, trn, tst in folds)


def test_events():
    t = np.arange(20, dtype=float); f = np.zeros(20, int); f[[1, 2, 3, 10, 11]] = 1
    ev = events.aggregate(t, f, gap_s=2)
    assert len(ev) == 2 and ev.n_rows.tolist() == [3, 2]


def test_r2_affine():
    x = np.arange(50.0); r2, a, b = r2_affine(x, 3 * x + 2)
    assert abs(r2 - 1) < 1e-12 and abs(a - 3) < 1e-9 and abs(b - 2) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
