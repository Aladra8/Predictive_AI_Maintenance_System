"""Gap-merge aggregation of flagged rows into maintenance events."""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate(t_seconds: np.ndarray, flagged: np.ndarray, gap_s: float) -> pd.DataFrame:
    """Merge flagged rows whose time gap is <= gap_s.  t_seconds must be sorted."""
    t = np.asarray(t_seconds, float)[np.asarray(flagged).astype(bool)]
    if t.size == 0:
        return pd.DataFrame(columns=["event_id", "start_s", "end_s", "duration_s", "n_rows"])
    cut = np.flatnonzero(np.diff(t) > gap_s)
    s_i = np.r_[0, cut + 1]; e_i = np.r_[cut, t.size - 1]
    ev = pd.DataFrame({"start_s": t[s_i], "end_s": t[e_i], "n_rows": e_i - s_i + 1})
    ev["duration_s"] = ev.end_s - ev.start_s
    ev.insert(0, "event_id", np.arange(1, len(ev) + 1))
    return ev


def summary(ev: pd.DataFrame, n_flagged: int) -> dict:
    if ev.empty:
        return {"n_events": 0}
    d = ev.duration_s / 60
    return {"n_events": int(len(ev)), "flagged_rows": int(n_flagged),
            "compression": n_flagged / len(ev),
            "median_min": float(d.median()), "mean_min": float(d.mean()), "max_min": float(d.max()),
            "pct_lt_1min": float((d < 1).mean() * 100), "pct_ge_5min": float((d >= 5).mean() * 100)}


def gap_sensitivity(t_seconds, flagged, gaps) -> pd.DataFrame:
    rows = []
    for g in gaps:
        ev = aggregate(t_seconds, flagged, g)
        rows.append({"gap_s": g, **summary(ev, int(np.sum(flagged)))})
    return pd.DataFrame(rows)
