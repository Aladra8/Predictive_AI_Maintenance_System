"""Semi-synthetic elevator-door benchmark with injected faults and ground truth.

Design (documented in docs/SYNTHETIC_BENCHMARK.md):

* Healthy baseline = the REAL public recording (revolutions, humidity, vibration at 4 Hz),
  cut into sessions ("evenings") of `session_hours`, each augmented with independent
  sensor noise, a gain/offset change and a random crop, so that no two sessions are identical
  and the piecewise-linear interpolation of the public data is broken.
* One simulated, physically coupled channel is added: door-motor current, which depends on
  speed of the door drive (revolutions) and on friction (fault-dependent) plus noise.  It is
  NOT a copy of another column.
* Faults are injected in a subset of sessions with a known onset.  Each fault has a
  precursor phase (severity ramps 0 -> 1) followed by a fault phase (severity 1 -> 1.5):
    - bearing_wear       : broadband vibration rise, slight revolutions drop, current rise
    - misalignment       : periodic vibration impulses every 2 s, current oscillation
    - humidity_friction  : humidity rise (moisture ingress), current rise, mild vibration rise
* Ground truth per row: state (0 normal, 1 precursor, 2 fault), fault_type, seconds to fault.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAULT_TYPES = ["bearing_wear", "misalignment", "humidity_friction"]


def _baseline(raw: pd.DataFrame) -> pd.DataFrame:
    b = raw[["revolutions", "humidity", "vibration"]].astype(float).copy()
    b["vibration"] = b["vibration"].interpolate().bfill().ffill()
    return b.reset_index(drop=True)


def generate(raw: pd.DataFrame, n_sessions: int = 12, session_hours: float = 2.0, hz: float = 4.0,
             fault_share: float = 0.75, precursor_min=(10, 25), fault_min=(5, 15), seed: int = 42,
             strength: float = 1.0):
    """Return (frame, session_table).  Deterministic for a given seed.
    `strength` scales every injected fault effect (1.0 = default severity)."""
    rng = np.random.default_rng(seed)
    base = _baseline(raw)
    L = int(session_hours * 3600 * hz)
    if L >= len(base):
        raise ValueError("session longer than the recording")
    frames, meta = [], []
    n_faulty = 0
    for s in range(n_sessions):
        start = int(rng.integers(0, len(base) - L))
        seg = base.iloc[start:start + L].reset_index(drop=True).copy()
        # augmentation: gain/offset + independent sensor noise
        seg["revolutions"] = seg.revolutions * rng.uniform(0.9, 1.1) + rng.normal(0, 0.15, L)
        seg["humidity"] = seg.humidity + rng.uniform(-1.0, 1.0) + rng.normal(0, 0.03, L)
        seg["vibration"] = np.clip(seg.vibration * rng.uniform(0.9, 1.1) + rng.normal(0, 0.8, L), 0, None)
        t = np.arange(L) / hz
        friction = np.zeros(L)
        state = np.zeros(L, dtype=np.int8)
        ttf = np.full(L, np.nan)
        ftype = "none"; onset = pre_start = end = None
        if rng.random() < fault_share:
            ftype = FAULT_TYPES[n_faulty % len(FAULT_TYPES)]   # balanced fault types
            n_faulty += 1
            pre = rng.uniform(*precursor_min) * 60; flt = rng.uniform(*fault_min) * 60
            pre_start = rng.uniform(0.2, 0.75) * (session_hours * 3600 - pre - flt)
            onset = pre_start + pre; end = onset + flt
            sev = np.zeros(L)
            ip = (t >= pre_start) & (t < onset); iff = (t >= onset) & (t < end)
            sev[ip] = (t[ip] - pre_start) / pre
            sev[iff] = 1 + 0.5 * (t[iff] - onset) / flt
            ev_sev = sev * strength                    # effect size used for the signals
            state[ip] = 1; state[iff] = 2
            ttf[ip] = onset - t[ip]
            if ftype == "bearing_wear":
                seg["vibration"] += ev_sev * 25 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)) + ev_sev * rng.normal(0, 3, L)
                seg["revolutions"] -= ev_sev * 2.0
                friction += ev_sev * 0.4
            elif ftype == "misalignment":
                imp = (np.arange(L) % int(2 * hz) == 0).astype(float)
                seg["vibration"] += ev_sev * 40 * imp
                friction += ev_sev * 0.3 * (1 + np.sin(2 * np.pi * t / 30))
            else:  # humidity_friction
                seg["humidity"] += ev_sev * 2.0
                friction += ev_sev * 0.6
                seg["vibration"] += ev_sev * 10
        seg["motor_current"] = 0.02 * seg.revolutions + 1.0 + friction + rng.normal(0, 0.05, L)
        seg["session"] = s; seg["t_s"] = t
        seg["state"] = state; seg["fault_type"] = ftype if ftype != "none" else "none"
        seg.loc[state == 0, "fault_type"] = "none"
        seg["time_to_fault_s"] = ttf
        frames.append(seg)
        meta.append({"session": s, "source_start_row": start, "fault_type": ftype,
                     "precursor_start_s": pre_start, "fault_onset_s": onset, "fault_end_s": end})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta)
