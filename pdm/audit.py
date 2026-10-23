"""Data provenance and leakage diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import runs_of_ones


def r2_affine(x, y) -> tuple[float, float, float]:
    """R^2, slope, intercept of y ~ a*x + b (NaNs dropped)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    A = np.c_[x[m], np.ones(m.sum())]
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    res = y[m] - A @ coef
    return 1 - res.var() / y[m].var(), float(coef[0]), float(coef[1])


def derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Best simple explanation of every anonymised column by the original channels."""
    base = {"revolutions": df["revolutions"], "revolutions^2": df["revolutions"] ** 2,
            "humidity": df["humidity"], "vibration": df["vibration"]}
    rows = []
    for c in ["x1", "x2", "x3", "x4", "x5"]:
        best = max(((r2_affine(v, df[c]), k) for k, v in base.items()), key=lambda t: t[0][0])
        (r2, a, b), src = best
        rows.append({"column": c, "explained_by": src, "r2": round(r2, 6),
                     "slope": round(a, 4), "intercept": round(b, 4),
                     "relation": f"{c} ≈ {a:.4g}·{src} {'+' if b >= 0 else '−'} {abs(b):.4g}"})
    return pd.DataFrame(rows)


def label_structure(y: np.ndarray, hz: float = 4.0) -> dict:
    runs = runs_of_ones(y)
    lengths = [e - s + 1 for s, e in runs]
    idx = np.flatnonzero(y)
    return {"positives": int(y.sum()), "share": float(np.mean(y)),
            "first_row": int(idx.min()) if idx.size else None,
            "last_row": int(idx.max()) if idx.size else None,
            "n_contiguous_runs": len(runs), "longest_run_rows": max(lengths) if lengths else 0,
            "positives_span_minutes": (idx.max() - idx.min() + 1) / hz / 60 if idx.size else 0.0}


def indicator_overlap(flags: pd.DataFrame) -> pd.DataFrame:
    """Where each indicator fires and how often indicators coincide."""
    rows = []
    for c in flags.columns:
        idx = np.flatnonzero(flags[c].to_numpy())
        rows.append({"indicator": c, "exceedances": int(idx.size),
                     "first_row": int(idx.min()), "last_row": int(idx.max()),
                     "n_runs": len(runs_of_ones(flags[c].to_numpy()))})
    co = flags.T.dot(flags)
    return pd.DataFrame(rows), co


def linear_segment_share(x: np.ndarray, tol: float = 1.5e-3) -> float:
    """Share of rows whose second difference is ~0 (signal locally linear = interpolated)."""
    x = np.asarray(pd.Series(x).interpolate().bfill().ffill(), float)
    return float(np.mean(np.abs(np.diff(x, 2)) <= tol))


def vibration_regularity(vib: np.ndarray, start: int, stop: int, jump: float = 3.0) -> dict:
    """Spacing and amplitude steps of vibration spikes inside a window."""
    seg = np.asarray(vib[start:stop], float)
    base = np.nanmedian(seg)
    sp = np.flatnonzero(np.nan_to_num(seg, nan=base) > base + jump) + start
    if sp.size < 3:
        return {"spikes": int(sp.size)}
    d = np.diff(sp); a = np.diff(np.asarray(vib)[sp])
    return {"window": f"{start}-{stop}", "spikes": int(sp.size),
            "modal_spacing_rows": int(pd.Series(d).mode()[0]),
            "share_spacing_8_rows": float(np.mean(d == 8)),
            "modal_amplitude_step": float(pd.Series(np.round(a, 3)).mode()[0])}


def single_feature_auc(df: pd.DataFrame, y: np.ndarray, cols: list[str]) -> pd.DataFrame:
    from .metrics import roc_auc
    rows = []
    for c in cols:
        a = roc_auc(y, df[c].astype(float).fillna(df[c].median()).to_numpy())
        rows.append({"feature": c, "auc": a, "auc_direction_free": max(a, 1 - a)})
    return pd.DataFrame(rows).sort_values("auc_direction_free", ascending=False)
