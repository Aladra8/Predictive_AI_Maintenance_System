#!/usr/bin/env python3
import os, sys, subprocess

def main():
    csv = "data/processed/processed_large_dataset_v4.csv"
    out = "Images/processed_kaggle"   # figures here for the report

    # You can tweak these from the CLI if you like
    cmd = [
        sys.executable, "src/phase1_noise_fs.py",
        "--csv", csv,
        "--out", out,
        "--seed", "42",
        "--denoise-min", "5",          # time-based smoothing window (minutes) if timestamp exists
        "--perm-topk", "4",            # top-K features by permutation importance
        "--event-gap-min-sec", "60",   # never merge events if gap exceeds 60s
        "--event-gap-max-sec", "900",  # cap the learned merge gap at 15 minutes
        "--base-q", "0.95"             # quantile for the 2-of-4 rule
    ]
    print("Running:", " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(ret)
    print("Done.")

if __name__ == "__main__":
    main()

