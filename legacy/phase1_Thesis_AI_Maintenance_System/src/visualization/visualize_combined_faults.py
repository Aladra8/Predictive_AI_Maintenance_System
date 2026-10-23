# src/visualize_combined_faults.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Config ===
DATA_PATH = "data/processed/predictive_maintenance_with_combined_label.csv"
OUTPUT_DIR = "outputs/visualizations/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
df = pd.read_csv(DATA_PATH)

# === Scatter Plot: Vibration vs Speed (colored by Fault_Combo) ===
plt.figure(figsize=(10, 6))
colors = df["Fault_Combo"].map({0: "blue", 1: "red"})
plt.scatter(df["Vibration"], df["Speed"], c=colors, alpha=0.4, label="Data")

plt.xlabel("Vibration")
plt.ylabel("Speed")
plt.title("Combined Faults (Vibration vs Speed)")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "combined_faults_scatter.png"))
plt.close()

print(f"[✓] Scatter plot saved to {OUTPUT_DIR}combined_faults_scatter.png")
