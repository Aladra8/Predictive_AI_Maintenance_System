"""Service tests: train/serve parity and event aggregation (no network needed).
Run after scripts/09_train_service_model.py has written deploy/model/model.pkl."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "deploy" / "service"))

from pdm import features  # noqa: E402

MODEL = ROOT / "deploy" / "model" / "model.pkl"


def _readings(n=2000, start=1_700_000_000.0):
    rng = np.random.default_rng(0)
    t = start + np.arange(n) / 4
    return pd.DataFrame({"ts": t, "revolutions": 50 + rng.normal(0, 1, n), "humidity": 74 + rng.normal(0, .05, n),
                         "vibration": 20 + rng.normal(0, 1, n), "motor_current": 2 + rng.normal(0, .05, n)})


def test_parity_streaming_equals_batch():
    if not MODEL.exists():
        return
    from scorer import Scorer
    sc = Scorer(MODEL); r = _readings()
    streamed = []
    for i in range(0, len(r), 37):                                 # irregular batch sizes
        res = sc.score("a", r.iloc[i:i + 37].to_dict("records"))
        streamed += [s["p_fault"] for s in res["scores"]]
    X = pd.concat([features.current_values(r, sc.ch), features.rolling_history(r, sc.ch, sc.win, sc.hz)], axis=1)[sc.cols]
    batch = sc.model.predict_proba(X.to_numpy(float))[:, 1]
    tail = slice(sc.max_rows, None)                                # after the buffer is full both must agree
    assert np.allclose(np.array(streamed)[tail], np.round(batch[tail], 4), atol=1e-4)


def test_event_state_machine():
    if not MODEL.exists():
        return
    from scorer import Scorer
    sc = Scorer(MODEL); sc.threshold = 0.5; sc.gap = 10; sc.min_alarms = 2
    st = type("S", (), {})(); st.open_event = None; st.closed = []
    ts = np.arange(0, 100, 1.0); p = np.zeros(100); p[10:20] = 0.9; p[25:27] = 0.9; p[60] = 0.9
    closed = sc._update_events("a", st, ts, p)
    assert len(closed) == 1 and closed[0].start_ts == 10 and closed[0].end_ts == 26   # 10-19 and 25-26 merged (gap 6 s)
    assert st.open_event is None        # the single alarm at 60 s was closed by the gap and dropped (< min_alarms)


if __name__ == "__main__":
    test_parity_streaming_equals_batch(); print("ok parity")
    test_event_state_machine(); print("ok events")
