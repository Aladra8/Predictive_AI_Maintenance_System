#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path

def first_exists(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

CAND_PROC = [
    "data/processed/processed_large_dataset_v4.csv",
    "data/processed/processed_large_dataset.csv",
]
PROC_CSV = first_exists(CAND_PROC)
if not PROC_CSV:
    sys.exit("Phase3: no processed CSV found in data/processed/. Put your processed file there.")

OUT_ROOT = "outputs/phase3"
TIME_COL  = "timestamp"     # <-- adjust if your column is named differently
EVENT_GAP = "5.0"           # minutes

common = [
    sys.executable, "src/phase2_training.py",
    "--processed-csv", PROC_CSV,
    "--out", OUT_ROOT,
    "--emit-tex",
    "--save-models"
]

cmds = []

# 1) Leakage-guard (drop the four rule variables from features)
cmds.append(common + ["--leakage-guard"])

# 2) Time-blocked (chronological 70/15/15)
cmds.append(common + ["--time-split", "--time-col", TIME_COL])

# 3) Event-grouped (keep incidents in one fold)
cmds.append(common + ["--event-split", "--time-col", TIME_COL, "--event-gap-mins", EVENT_GAP])

for c in cmds:
    print("Running:", " ".join(c))
    ret = subprocess.call(c)
    if ret != 0:
        print("Command failed with code", ret)
