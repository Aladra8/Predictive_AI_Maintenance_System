import os
import pandas as pd
import matplotlib.pyplot as plt

# === Paths ===
DATA_PATH = "data/processed/predictive_maintenance_final_labeled.csv"
OUTPUT_DIR = "outputs/time_fault_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(DATA_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])  # remove bad timestamps
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)

# === 1. Line Plot: Faults Over Time (per minute) ===
faults_per_min = df['Fault'].resample('1min').sum()

plt.figure(figsize=(12, 4))
plt.plot(faults_per_min.index, faults_per_min.values, color='red', linewidth=1.5)
plt.title("Faults Over Time (Per Minute)")
plt.xlabel("Time")
plt.ylabel("Number of Faults")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "faults_over_time_line.png"))
plt.close()

# === 2. Histogram: Fault Frequency by Hour of Day ===
df['hour'] = df.index.hour
faults_by_hour = df[df['Fault'] == 1]['hour'].value_counts().sort_index()

plt.figure(figsize=(8, 4))
plt.bar(faults_by_hour.index, faults_by_hour.values, color='orange')
plt.title("Fault Distribution by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Fault Count")
plt.xticks(range(0, 24))
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "temporal_fault_distribution_hist.png"))
plt.close()

print(f"[✓] Time-based visualizations saved to: {OUTPUT_DIR}")
