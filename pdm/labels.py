"""Analytical labelling rules.

* percentile_vote : the thesis rule ("k of n indicators above their q-quantile").
* vibration_episodes : the corrected rule used in the redesigned experiment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_vote(df: pd.DataFrame, cols: list[str], q: float = 0.95, k: int = 2):
    """Return (y, thresholds, flags).  A row is positive if at least k of the
    columns exceed their q-quantile computed on the whole table (as in the thesis)."""
    thr = {c: float(df[c].quantile(q)) for c in cols}
    flags = pd.DataFrame({c: (df[c] > thr[c]).astype(int) for c in cols})
    y = (flags.sum(axis=1) >= k).astype(int).to_numpy()
    return y, thr, flags


def runs_of_ones(y: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive positive rows."""
    idx = np.flatnonzero(np.asarray(y) == 1)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_by_gap(idx: np.ndarray, gap_rows: int) -> list[tuple[int, int]]:
    """Merge sorted positive indices whose spacing is <= gap_rows."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return []
    cut = np.flatnonzero(np.diff(idx) > gap_rows)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def vibration_episodes(vib: np.ndarray, q: float = 0.95, merge_gap_s: float = 60,
                       hz: float = 4.0, min_duration_s: float = 0.0):
    """Episodes of high door vibration.

    Exceedances (vibration > q-quantile, NaNs ignored) closer than merge_gap_s
    are merged into one episode.  Returns (episodes DataFrame, in_episode row label,
    threshold)."""
    vib = np.asarray(vib, dtype=float)
    thr = float(np.nanquantile(vib, q))
    idx = np.flatnonzero(np.nan_to_num(vib, nan=-np.inf) > thr)
    eps = merge_by_gap(idx, int(round(merge_gap_s * hz)))
    rows = []
    y = np.zeros(len(vib), dtype=int)
    for s, e in eps:
        dur = (e - s) / hz
        if dur < min_duration_s:
            continue
        y[s:e + 1] = 1
        rows.append({"start_row": s, "end_row": e, "start_s": s / hz, "end_s": e / hz,
                     "duration_s": dur, "n_exceedances": int(((idx >= s) & (idx <= e)).sum())})
    ep = pd.DataFrame(rows)
    if not ep.empty:
        ep.insert(0, "episode_id", np.arange(1, len(ep) + 1))
    return ep, y, thr


def onset_target(episodes: pd.DataFrame, n: int, horizon_s: float, hz: float = 4.0) -> np.ndarray:
    """1 for rows that precede an episode start by at most horizon_s and are
    not themselves inside an episode (early-warning target)."""
    y = np.zeros(n, dtype=int)
    h = int(round(horizon_s * hz))
    inside = np.zeros(n, dtype=bool)
    for _, r in episodes.iterrows():
        inside[int(r.start_row):int(r.end_row) + 1] = True
    for _, r in episodes.iterrows():
        s = int(r.start_row)
        y[max(0, s - h):s] = 1
    y[inside] = 0
    return y
