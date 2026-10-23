#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def find_col(df: pd.DataFrame, aliases):
    """Finds the first matching column name from a list of aliases."""
    m = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    return None

def process_data(input_path: str, output_path: str, quantile: float = 0.95):
    """
    Loads, cleans, labels, and saves the elevator predictive maintenance dataset.

    Args:
        input_path (str): Path to the raw CSV file.
        output_path (str): Path to save the processed CSV file.
        quantile (float): The percentile to use for the fault labeling threshold.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df_initial_shape = df.shape
    print(f"Initial shape: {df_initial_shape[0]} rows, {df_initial_shape[1]} columns.")

    # 1. Handle Timestamp
    print("Processing timestamp...")
    time_col = find_col(df, ["timestamp", "time"])
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
        # Drop rows where timestamp could not be parsed
        df.dropna(subset=[time_col], inplace=True)
        # Sort the entire dataset by time
        df.sort_values(by=time_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Data sorted by '{time_col}'. Time range: {df[time_col].min()} to {df[time_col].max()}")
    else:
        print("Warning: Timestamp column not found. Data is not time-sorted.")

    # 2. Handle Missing Values
    print("Handling missing values...")
    # For environmental sensors, forward-fill is a reasonable strategy
    for col in ["Humidity", "Temperature", "Signal_Strength"]:
        if col in df.columns and df[col].isnull().any():
            initial_nan = df[col].isnull().sum()
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            print(f"- Filled {initial_nan} NaNs in '{col}' using ffill/bfill.")
    # For any other numeric NaNs, use median imputation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            initial_nan = df[col].isnull().sum()
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"- Filled {initial_nan} NaNs in '{col}' with median value ({median_val:.2f}).")


    # 3. Generate Analytical Fault Labels (2-of-4 Rule)
    print(f"Generating fault labels using '2-of-4' rule at the {quantile * 100}th percentile...")
    rule_cols = {
        "Vibration": find_col(df, ["Vibration", "x3"]),
        "Speed": find_col(df, ["Speed", "x2"]),
        "Energy": find_col(df, ["Energy", "x5"]),
        "Revolutions": find_col(df, ["Revolutions", "Motor_Cycles", "x4"])
    }

    # Ensure all required columns were found
    if any(v is None for v in rule_cols.values()):
        missing = [k for k, v in rule_cols.items() if v is None]
        raise ValueError(f"Error: Missing required columns for labeling: {missing}")

    # Calculate thresholds based on the specified quantile
    thresholds = {name: df[col_name].quantile(quantile) for name, col_name in rule_cols.items()}
    print("Calculated thresholds:")
    for name, thr in thresholds.items():
        print(f"- {name}: {thr:.2f}")

    # Apply the 2-of-4 rule
    flags = pd.DataFrame()
    for name, col_name in rule_cols.items():
        flags[name] = (df[col_name] > thresholds[name]).astype(int)

    df['label_fault'] = (flags.sum(axis=1) >= 2).astype(int)
    class_balance = df['label_fault'].value_counts(normalize=True)
    print("Label generation complete.")
    print(f"Class balance: Normal (0) = {class_balance[0]:.3%}, Fault (1) = {class_balance[1]:.3%}")


    # 4. Save Processed Data
    print(f"\nSaving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Processing complete.")
    print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process elevator predictive maintenance data.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/processed_large_dataset_v4.csv",
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/processed_labeled_dataset.csv",
        help="Path to save the processed output CSV file."
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile for the analytical labeling rule threshold."
    )
    args = parser.parse_args()
    process_data(args.input, args.output, args.quantile)