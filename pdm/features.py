"""Past-only (causal) feature engineering for the corrected experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_history(df: pd.DataFrame, cols: list[str], windows_s: list[float], hz: float = 4.0) -> pd.DataFrame:
    """Rolling mean / std / max over the PREVIOUS w seconds (current row excluded),
    so that no feature uses information from the future or from the target row."""
    out = {}
    for c in cols:
        s = df[c].astype(float).shift(1)
        for w in windows_s:
            n = max(int(round(w * hz)), 2)
            r = s.rolling(n, min_periods=1)
            out[f"{c}_mean_{int(w)}s"] = r.mean()
            out[f"{c}_std_{int(w)}s"] = r.std()
            out[f"{c}_max_{int(w)}s"] = r.max()
        out[f"{c}_slope_60s"] = (s - s.shift(int(60 * hz))) / 60.0
    X = pd.DataFrame(out, index=df.index)
    return X.bfill().fillna(0.0)


def current_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype(float).ffill().bfill()
