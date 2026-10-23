"""Data loading, cleaning and the real 4 Hz clock.

The Kaggle mirror of the Huawei Munich Research Center dataset contains
ID, revolutions, humidity, vibration and five anonymised columns x1..x5.
The original Zenodo release (doi:10.5281/zenodo.3653909) contains only the
three named channels.  See pdm.audit for how x1..x5 relate to them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_COLUMNS = ["ID", "revolutions", "humidity", "vibration", "x1", "x2", "x3", "x4", "x5"]


def md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kaggle(path: str | Path, expected_md5: str | None = None) -> pd.DataFrame:
    """Load the Kaggle CSV and check its checksum."""
    if expected_md5:
        got = md5(path)
        if got != expected_md5:
            raise ValueError(f"MD5 mismatch for {path}: {got} != {expected_md5}")
    df = pd.read_csv(path)
    if list(df.columns) != KAGGLE_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    return df


def load_zenodo(path: str | Path) -> pd.DataFrame:
    """Load the original Zenodo release (';'-separated, 3 columns).

    The file contains spreadsheet-corrupted cells (e.g. '03. Jun'), which are
    coerced to NaN; the number of such cells is reported by pdm.audit.
    """
    raw = pd.read_csv(path, sep=";", dtype=str)
    out = pd.DataFrame({c: pd.to_numeric(raw[c], errors="coerce") for c in raw.columns})
    out.columns = ["revolutions" if c.startswith("ball") else c for c in out.columns]
    out.attrs["corrupted_cells"] = {c: int(out[c].isna().sum()) for c in out.columns}
    return out


def add_real_clock(df: pd.DataFrame, hz: float = 4.0, start: str = "2020-01-01 16:30:00") -> pd.DataFrame:
    """Attach the documented acquisition clock: row i is sampled at start + i / hz."""
    out = df.copy()
    out["t_s"] = np.arange(len(out)) / hz
    out["timestamp"] = pd.Timestamp(start) + pd.to_timedelta(out["t_s"], unit="s")
    return out


def fill_vibration(df: pd.DataFrame, how: str = "ffill") -> pd.DataFrame:
    out = df.copy()
    if how == "ffill":
        out["vibration"] = out["vibration"].ffill().bfill()
    elif how == "median":
        out["vibration"] = out["vibration"].fillna(out["vibration"].median())
    elif how == "none":
        pass
    else:
        raise ValueError(how)
    return out


def thesis_frame(df: pd.DataFrame, names: dict, vibration_nan: str = "ffill") -> pd.DataFrame:
    """Reproduce the processed table used in the thesis (preprocess_large_dataset.py):
    rename x1..x5, add acceleration = diff(speed) (no division by dt, as in the original code),
    and fill missing vibration values."""
    out = df.rename(columns=names)
    out["acceleration"] = out["speed"].diff().fillna(0.0)
    return fill_vibration(out, vibration_nan)


def synthetic_timestamps(n: int, seed: int | None = 0,
                         start: str = "2023-01-02 06:00:00") -> pd.Series:
    """Re-implementation of the synthetic clock used in the thesis pipeline
    (preprocess_large_dataset.py).  Kept ONLY to demonstrate the artefact it
    creates; the original code was unseeded.  Steps are integer seconds >= 1
    with Poisson means of 1.5 s (07-09h, 17-19h), 3 s (10-16h), 6 s otherwise
    and 10 s at weekends."""
    rng = np.random.default_rng(seed)
    ts = np.empty(n, dtype="datetime64[s]")
    cur = pd.Timestamp(start)
    for i in range(n):
        ts[i] = np.datetime64(cur.to_datetime64(), "s")
        wd, h = cur.weekday(), cur.hour
        if wd < 5:
            lam = 1.5 if (7 <= h <= 9 or 17 <= h <= 19) else (3.0 if 10 <= h <= 16 else 6.0)
        else:
            lam = 10.0
        cur += pd.Timedelta(seconds=max(int(rng.poisson(lam)), 1))
    return pd.Series(pd.to_datetime(ts))
