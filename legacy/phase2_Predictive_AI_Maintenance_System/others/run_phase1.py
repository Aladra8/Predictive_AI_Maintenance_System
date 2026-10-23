# run_phase1.py — thin wrapper to run without flags
import subprocess, sys, os, shutil

# Datasets to try in order
CANDIDATES = [
    "data/processed/processed_large_dataset_v4.csv",
    "data/processed/processed_large_dataset.csv",
    "data/processed/processed_large_dataset_v4_shuffled.csv",
]
csv = next((p for p in CANDIDATES if os.path.exists(p)), None)
if csv is None:
    sys.exit("No processed CSV found under data/processed/. Put your file there and re-run.")

out = "outputs/phase1"
os.makedirs(out, exist_ok=True)

cmd = [
    sys.executable, "src/phase1_artifacts.py",
    "--csv", csv,
    "--out", out,
    "--emit-tex",
]
print("Running:", " ".join(cmd))
sys.exit(subprocess.call(cmd))
