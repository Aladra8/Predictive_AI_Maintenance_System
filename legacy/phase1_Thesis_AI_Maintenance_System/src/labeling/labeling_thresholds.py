import os
import pandas as pd
import matplotlib.pyplot as plt

# === Setup ===
DATA_PATH = "data/processed/predictive_maintenance_thresholds_labeled_v2.csv"
OUTPUT_CSV = "data/processed/predictive_maintenance_thresholds_labeled_v2.csv"
PLOT_DIR = "outputs/threshold_viz"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(DATA_PATH)

# === Define thresholds ===
thresholds = {
    "Fault_900": df["Vibration"].quantile(0.900),
    "Fault_925": df["Vibration"].quantile(0.925),
    "Fault_950": df["Vibration"].quantile(0.950),
    "Fault_975": df["Vibration"].quantile(0.975),
}

# === Create labels ===
for label, thresh in thresholds.items():
    df[label] = (df["Vibration"] > thresh).astype(int)

# === Plot ===
plt.figure(figsize=(10, 5))
df["Vibration"].hist(bins=50, alpha=0.6, color="skyblue", edgecolor="black")
for label, thresh in thresholds.items():
    plt.axvline(thresh, linestyle="--", label=f"{label} ({thresh:.2f})")
plt.title("Vibration Distribution with Fault Thresholds")
plt.xlabel("Vibration")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "vibration_thresholds_histogram.png"))
plt.show()  # <-- This will display the plot on screen
plt.close()

# === Save updated dataset ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"[✓] Dataset saved to: {OUTPUT_CSV}")
