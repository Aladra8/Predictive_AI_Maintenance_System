"""Replay a recorded session to the scoring service, as if a sensor gateway were streaming it.

    python deploy/replay_client.py --session 0 --speed 200            # from the semi-synthetic benchmark
    python deploy/replay_client.py --csv my_export.csv --asset lift-A  # your own CSV (ts + channels)

--speed is the time-acceleration factor (1 = real time at 4 Hz). Readings are sent in
batches of --batch rows (default 4 = one second of data).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8080")
ap.add_argument("--api-key", default=None)
ap.add_argument("--csv", default=None, help="CSV with columns ts (unix s) + model channels")
ap.add_argument("--session", type=int, default=0, help="session id in data/synthetic/semisynthetic_benchmark.csv.gz")
ap.add_argument("--asset", default=None)
ap.add_argument("--speed", type=float, default=100.0)
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()


def call(path, payload=None):
    req = urllib.request.Request(args.url + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **({"X-API-Key": args.api_key} if args.api_key else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


health = call("/health"); channels = health["model"]["channels"]; hz = health["model"]["hz"]
if args.csv:
    df = pd.read_csv(args.csv); asset = args.asset or Path(args.csv).stem; truth = None
else:
    syn = ROOT / "data" / "synthetic" / "semisynthetic_benchmark.csv.gz"
    if not syn.exists():
        sys.exit("Run python scripts/08_synthetic_benchmark.py first (it writes the benchmark CSV).")
    df = pd.read_csv(syn); df = df[df.session == args.session].reset_index(drop=True)
    df["ts"] = 1_700_000_000 + df.t_s; asset = args.asset or f"session-{args.session}"; truth = df.state.to_numpy()

print(f"replaying {len(df)} rows for {asset} at {args.speed}x; channels {channels}")
n_alarm = 0; t0 = time.time()
for i in range(0, len(df), args.batch):
    chunk = df.iloc[i:i + args.batch]
    res = call("/score", {"asset_id": asset, "readings": chunk[["ts"] + channels].to_dict("records")})
    n_alarm += sum(s["alarm"] for s in res["scores"])
    for e in res["closed_events"]:
        print("EVENT", json.dumps(e))
    time.sleep(len(chunk) / hz / args.speed)
for e in call("/flush", {"asset_id": asset})["closed_events"]:
    print("EVENT", json.dumps(e))
print(f"done in {time.time() - t0:.1f}s; alarms={n_alarm}")
if truth is not None:
    ts = df.ts.to_numpy()
    pre = ts[truth == 1]; flt = ts[truth == 2]
    if len(flt):
        print(f"ground truth: precursor {pre.min() - ts[0]:.0f}-{pre.max() - ts[0]:.0f}s, fault {flt.min() - ts[0]:.0f}-{flt.max() - ts[0]:.0f}s (relative)")
    else:
        print("ground truth: healthy session")
    for e in call(f"/events?asset_id={asset}")["events"]:
        print(f"  event {e['start_ts'] - ts[0]:.0f}-{e['end_ts'] - ts[0]:.0f}s rel., peak {e['peak_score']}")
