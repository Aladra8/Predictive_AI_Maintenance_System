# Filename: generate_fault_combo.py

import pandas as pd

# === Load dataset ===
df = pd.read_csv("data/processed/processed_large_dataset.csv")

# === Define thresholds ===
thresholds = {
    'acceleration': 0.9,
    'speed': 0.9,
    'energy': 0.9,
    'vibration': 0.9
}

# Check which rows exceed each threshold
condition_matrix = pd.DataFrame({
    col: (df[col] >= threshold)
    for col, threshold in thresholds.items()
})

# Count how many conditions are True per row
df['Fault_Combo'] = (condition_matrix.sum(axis=1) >= 2).astype(int)

#Save updated dataset
df.to_csv("processed_with_fault_combo.csv", index=False)

#Print sanity check
print("Labeling Complete. Class Distribution:")
print(df['Fault_Combo'].value_counts())
print(f"\nTotal samples: {len(df)}")
print(f"Faults: {df['Fault_Combo'].sum()} ({df['Fault_Combo'].mean() * 100:.2f}%)")
