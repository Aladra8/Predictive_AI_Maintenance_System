#!/usr/bin/env python3
import os, sys, subprocess
from pathlib import Path

CAND_PROC = [
    "data/processed/processed_large_dataset_v4.csv",
    "data/processed/processed_large_dataset.csv",
]
CAND_ORIG = [
    "data/raw/predictive-maintenance-dataset.csv",
    "predictive-maintenance-dataset.csv",
]

def first_exists(paths):
    for p in paths:
        if Path(p).exists(): return p
    return None

proc = first_exists(CAND_PROC)
if not proc:
    sys.exit("No processed CSV found under data/processed/. Put your processed file there.")

orig = first_exists(CAND_ORIG)  # optional

# ---- switches (toggle True/False as needed) ----
EMIT_TEX = True
SAVE_MODELS = True

LEAKAGE_GUARD = False     # set True to drop the 2-of-4 variables from features
TIME_SPLIT    = False     # set True for chronological split
EVENT_SPLIT   = False     # set True for event-grouped split
TIME_COL_NAME = "timestamp"  # adjust to your actual column if enabling time/event splits
EVENT_GAP_MINS = "5.0"

cmd = [sys.executable, "src/phase2_training.py",
       "--processed-csv", proc,
       "--out", "outputs/phase2"]

if EMIT_TEX:   cmd.append("--emit-tex")
if SAVE_MODELS: cmd.append("--save-models")
if LEAKAGE_GUARD: cmd.append("--leakage-guard")
if TIME_SPLIT:
    cmd += ["--time-split", "--time-col", TIME_COL_NAME]
if EVENT_SPLIT:
    cmd += ["--event-split", "--time-col", TIME_COL_NAME, "--event-gap-mins", EVENT_GAP_MINS]
if orig:
    cmd += ["--original-csv", orig]

print("Running:", " ".join(cmd))
sys.exit(subprocess.call(cmd))
