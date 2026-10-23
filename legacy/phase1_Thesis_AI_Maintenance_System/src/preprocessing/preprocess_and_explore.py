import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ------------------------------
# Paths and directories
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "predictive-maintenance-dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------------------
# Load and inspect dataset
# ------------------------------
print(" Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)

# Rename columns for interpretability
df.rename(columns={
    "x1": "sensor_1",
    "x2": "sensor_2",
    "x3": "sensor_3",
    "x4": "sensor_4",
    "x5": "sensor_5"
}, inplace=True)

# Show basic info
print("\n First 5 rows:\n", df.head())
print("\n Dataset structure:")
print(df.info())
print("\n Missing values in each column:")
print(df.isnull().sum())
print("\n Summary statistics:\n", df.describe())

# ------------------------------
# Handle missing values
# ------------------------------
imputer = SimpleImputer(strategy='median')
df['vibration'] = imputer.fit_transform(df[['vibration']])

# ------------------------------
# Create binary fault label
# ------------------------------
threshold = df['vibration'].quantile(0.95)
df['fault'] = df['vibration'] > threshold
print("\n Fault label counts:\n", df['fault'].value_counts())

# ------------------------------
# Visualization
# ------------------------------
# Vibration distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['vibration'], bins=50, kde=True)
plt.axvline(threshold, color='red', linestyle='--', label='95th Percentile Threshold')
plt.title("Distribution of Vibration with Fault Threshold")
plt.xlabel("Vibration")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "vibration_distribution_updated.png"))
plt.close()

# Fault distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='fault', data=df)
plt.title("Fault Label Distribution (0 = Normal, 1 = Fault)")
plt.xlabel("Fault")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fault_distribution_updated.png"))
plt.close()


# Feature scaling and save
features = df.drop(columns=["ID", "fault"])
labels = df["fault"].astype(int)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save processed data
pd.DataFrame(features_scaled, columns=features.columns).to_csv(
    os.path.join(PROCESSED_DIR, "X_processed.csv"), index=False
)
labels.to_csv(os.path.join(PROCESSED_DIR, "y_labels.csv"), index=False)

# Final confirmation
print("\n Feature matrix shape:", features_scaled.shape)
print(" Labels shape:", labels.shape)
print(" Data preprocessing complete. Processed files and plots saved.")
