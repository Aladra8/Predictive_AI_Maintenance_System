import pandas as pd
import os

# === Config ===
DATA_PATH = "data/processed/predictive_maintenance_thresholds_labeled_v2.csv"
OUTPUT_PATH = "data/processed/predictive_maintenance_with_combined_label.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Load dataset ===
df = pd.read_csv(DATA_PATH)

# === Compute 95th percentiles for Vibration and Speed ===
vib_thresh = df["Vibration"].quantile(0.95)
speed_thresh = df["Speed"].quantile(0.95)

# === Create combined fault label ===
df["Fault_Combo"] = ((df["Vibration"] >= vib_thresh) & (df["Speed"] >= speed_thresh)).astype(int)

# === Save updated file ===
df.to_csv(OUTPUT_PATH, index=False)
print(f"[✓] Dataset with combined fault label saved to: {OUTPUT_PATH}")
