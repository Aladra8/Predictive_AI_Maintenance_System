import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = "data/raw/Traction Elevator.csv"
OUTPUT_DIR = "outputs/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and parse datetime
df = pd.read_csv(DATA_PATH)
df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
df = df.dropna(subset=['timestamp'])  # Remove rows with bad timestamps
df = df.sort_values('timestamp')      # Ensure chronological order
df.set_index('timestamp', inplace=True)

# List of sensors
sensors = ['Acceleration', 'Speed', 'Vibration']

# Plot time series for each sensor
for sensor in sensors:
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df[sensor], label=sensor)
    plt.xlabel("Time")
    plt.ylabel(sensor)
    plt.title(f"{sensor} over Time")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{sensor.lower()}_timeseries.png"))
    plt.close()

# Plot histograms
for sensor in sensors:
    plt.figure(figsize=(8, 4))
    df[sensor].hist(bins=30)
    plt.xlabel(sensor)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {sensor}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{sensor.lower()}_histogram.png"))
    plt.close()

print("[✓] Time analysis plots saved to:", OUTPUT_DIR)
