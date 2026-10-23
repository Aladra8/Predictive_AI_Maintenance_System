"""Run every experiment in order and collect the key numbers in results/SUMMARY.md.

    python scripts/run_all.py                 # scikit-learn (thesis numbers)
    python scripts/run_all.py --engine numpy  # dependency-free smoke test
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["01_audit_data.py", "02_replicate_thesis.py", "03_time_aware_eval.py",
         "04_vibration_episodes.py", "05_events.py", "06_pca_validation.py",
         "07_processed_dataset.py", "08_synthetic_benchmark.py", "09_train_service_model.py",
         "10_thesis_figures.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default=None, choices=["sklearn", "numpy"])
ap.add_argument("--config", default=None)
args, rest = ap.parse_known_args()
extra = (["--engine", args.engine] if args.engine else []) + (["--config", args.config] if args.config else []) + rest

for s in STEPS:
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.call([sys.executable, str(ROOT / "scripts" / s), *extra])
    if r != 0:
        sys.exit(f"{s} failed with exit code {r}")

sys.path.insert(0, str(ROOT))
from pdm.report import write_summary  # noqa: E402
print(write_summary(args.engine or "sklearn"))
