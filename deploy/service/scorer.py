"""Online scoring and event aggregation (one state machine per elevator/asset).

Train/serve parity: features are computed with the SAME function used in training
(pdm.features.current_values + rolling_history) on a rolling buffer that holds
just enough history for the longest window.
"""
from __future__ import annotations

import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pdm import features


@dataclass
class Event:
    asset_id: str
    start_ts: float
    end_ts: float
    n_alarms: int = 0
    peak_score: float = 0.0

    def as_dict(self):
        return {"asset_id": self.asset_id, "start_ts": self.start_ts, "end_ts": self.end_ts,
                "duration_s": round(self.end_ts - self.start_ts, 3), "n_alarms": self.n_alarms,
                "peak_score": round(self.peak_score, 4)}


@dataclass
class AssetState:
    buffer: pd.DataFrame
    open_event: Event | None = None
    closed: deque = field(default_factory=lambda: deque(maxlen=1000))
    n_scored: int = 0
    last_ts: float | None = None


class Scorer:
    def __init__(self, model_path: str | Path):
        with open(model_path, "rb") as fh:
            b = pickle.load(fh)
        self.model, self.cols, self.ch = b["model"], b["feature_columns"], b["channels"]
        self.win, self.hz = b["window_s"], b["hz"]
        self.threshold, self.gap = b["alarm_threshold"], b["event_gap_s"]
        self.min_alarms = b.get("min_event_alarms", 1)
        self.max_rows = int((max(self.win) + 61) * self.hz) + 1     # longest window + 60 s slope
        self.assets: dict[str, AssetState] = {}
        self.lock = threading.Lock()
        self.meta = {k: b[k] for k in ("channels", "window_s", "hz", "alarm_threshold", "event_gap_s", "engine", "model_name")}

    # ------------------------------------------------------------------ scoring
    def score(self, asset_id: str, readings: list[dict]) -> dict:
        """readings: [{"ts": <unix seconds>, "revolutions": .., "humidity": .., ...}, ...] in time order."""
        if not readings:
            return {"asset_id": asset_id, "scores": [], "closed_events": [], "open_event": None}
        new = pd.DataFrame(readings)
        missing = [c for c in self.ch + ["ts"] if c not in new.columns]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        new = new[["ts"] + self.ch].astype(float)
        with self.lock:
            st = self.assets.get(asset_id)
            if st is None:
                st = self.assets[asset_id] = AssetState(buffer=new.iloc[0:0].copy())
            if st.last_ts is not None and new.ts.iloc[0] <= st.last_ts:
                raise ValueError("timestamps must be increasing per asset")
            buf = pd.concat([st.buffer, new], ignore_index=True)
            X = pd.concat([features.current_values(buf, self.ch),
                           features.rolling_history(buf, self.ch, self.win, self.hz)], axis=1)[self.cols]
            Xn = X.iloc[-len(new):].to_numpy(float)
            p = self.model.predict_proba(Xn)[:, 1]
            st.buffer = buf.iloc[-self.max_rows:].reset_index(drop=True)
            st.n_scored += len(new); st.last_ts = float(new.ts.iloc[-1])
            closed = self._update_events(asset_id, st, new.ts.to_numpy(), p)
            return {"asset_id": asset_id,
                    "scores": [{"ts": float(t), "p_fault": round(float(s), 4), "alarm": bool(s >= self.threshold)}
                               for t, s in zip(new.ts, p)],
                    "closed_events": [e.as_dict() for e in closed],
                    "open_event": st.open_event.as_dict() if st.open_event else None,
                    "warming_up": len(st.buffer) < self.max_rows}

    # ------------------------------------------------------------------ events
    def _update_events(self, asset_id, st: AssetState, ts, p):
        closed = []
        for t, s in zip(ts, p):
            ev = st.open_event
            if ev is not None and t - ev.end_ts > self.gap:          # gap exceeded -> close
                if ev.n_alarms >= self.min_alarms:
                    st.closed.append(ev); closed.append(ev)
                st.open_event = ev = None
            if s >= self.threshold:
                if ev is None:
                    st.open_event = ev = Event(asset_id, float(t), float(t))
                ev.end_ts = float(t); ev.n_alarms += 1; ev.peak_score = max(ev.peak_score, float(s))
        return closed

    def flush(self, asset_id: str):
        """Close the open event (e.g. end of a replay)."""
        with self.lock:
            st = self.assets.get(asset_id)
            if st and st.open_event and st.open_event.n_alarms >= self.min_alarms:
                st.closed.append(st.open_event)
                ev = st.open_event; st.open_event = None
                return [ev.as_dict()]
            if st:
                st.open_event = None
            return []

    def events(self, asset_id: str | None = None):
        with self.lock:
            items = self.assets.items() if asset_id is None else [(asset_id, self.assets.get(asset_id))]
            return [e.as_dict() for a, st in items if st for e in st.closed]

    def status(self):
        with self.lock:
            return {a: {"scored": st.n_scored, "last_ts": st.last_ts, "open_event": st.open_event is not None,
                        "closed_events": len(st.closed)} for a, st in self.assets.items()}
