# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# import matplotlib.pyplot as plt
# # Load original dataset
# df = pd.read_csv("data/raw/predictive-maintenance-dataset.csv")

# # Rename columns for clarity
# df.columns = [
#     'ID', 'revolutions', 'humidity', 'vibration',
#     'temperature', 'speed', 'signal_strength', 'energy', 'motor_cycles'
# ]

# # Compute acceleration (Δspeed)
# df['acceleration'] = df['speed'].diff().fillna(0)

# # Simulate realistic timestamps (weekday work hours)
# start_time = datetime(2023, 1, 2, 6, 0, 0)  # Monday 6 AM
# timestamps = []
# current_time = start_time

# while len(timestamps) < len(df):
#     weekday = current_time.weekday()
#     hour = current_time.hour

#     if weekday < 5:
#         if 7 <= hour <= 9 or 17 <= hour <= 19:
#             step = np.random.poisson(1.5)
#         elif 10 <= hour <= 16:
#             step = np.random.poisson(3)
#         else:
#             step = np.random.poisson(6)
#     else:
#         step = np.random.poisson(10)

#     step = max(step, 1)
#     timestamps.append(current_time)
#     current_time += timedelta(seconds=step)

# df = df.iloc[:len(timestamps)].copy()
# df['timestamp'] = timestamps

# # ---------- Fault Labeling ----------
# vib_90 = df['vibration'].quantile(0.90)
# rev_90 = df['revolutions'].quantile(0.90)
# acc_90 = df['acceleration'].quantile(0.90)
# speed_90 = df['speed'].quantile(0.90)

# # Create binary thresholds
# df['vibration_fault'] = (df['vibration'] >= vib_90).astype(int)
# df['revolutions_fault'] = (df['revolutions'] >= rev_90).astype(int)
# df['acceleration_fault'] = (df['acceleration'] >= acc_90).astype(int)
# df['speed_fault'] = (df['speed'] >= speed_90).astype(int)

# # Fault Combo: at least 2 out of 4 conditions are True
# df['Fault_Combo'] = (
#     df[['vibration_fault', 'revolutions_fault', 'acceleration_fault', 'speed_fault']]
#     .sum(axis=1)
#     .ge(2)
# ).astype(int)

# # Optional individual thresholds for comparison
# df['Fault_900'] = df['vibration_fault']
# df['Fault_925'] = (df['vibration'] >= df['vibration'].quantile(0.925)).astype(int)
# df['Fault_950'] = (df['vibration'] >= df['vibration'].quantile(0.950)).astype(int)
# df['Fault_975'] = (df['vibration'] >= df['vibration'].quantile(0.975)).astype(int)

# # Final column ordering
# final_columns = [
#     'timestamp', 'temperature', 'speed', 'acceleration', 'humidity',
#     'vibration', 'revolutions', 'signal_strength', 'energy', 'motor_cycles',
#     'Fault_900', 'Fault_925', 'Fault_950', 'Fault_975', 'Fault_Combo'
# ]
# df = df[final_columns]

# # Save output
# output_path = "data/processed/processed_large_dataset.csv"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# df.to_csv(output_path, index=False)

# # Display summary
# print("Dataset processing complete.")
# print("Final shape:", df.shape)
# print(" Fault_Combo label distribution:\n", df['Fault_Combo'].value_counts())
# plt.figure(figsize=(12, 4))
# plt.plot(df['Fault_Combo'].values, marker='.', linestyle='None', alpha=0.4)
# plt.title("Fault_Combo Values Across Dataset Index")
# plt.xlabel("Index")
# plt.ylabel("Fault_Combo")
# plt.tight_layout()
# plt.show()

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RAW_PATH = "data/raw/predictive-maintenance-dataset.csv"
OUT_MAIN = "data/processed/processed_large_dataset_v4.csv"
OUT_SHUF = "data/processed/processed_large_dataset_v4_shuffled.csv"

# --- combo label tuning ---
AUTO_TUNE = True         # set False to use MANUAL_* percentiles
TARGET_RATE = 0.08       # ~8% positives for Fault_Combo2of3_TUNED (adjust 0.05–0.10)
MANUAL_VIB_P = 0.90      # used only if AUTO_TUNE=False
MANUAL_REV_P = 0.85
MANUAL_SPD_P = 0.85

# --- timestamp simulation config ---
START_DT = datetime(2023, 1, 2, 6, 0, 0)  # Monday 06:00
RUSH_POIS = 1.5     # seconds step (lower = denser)
MID_POIS  = 3.0
QUIET_POIS= 6.0
WEND_POIS = 10.0

def load_raw(path: str) -> pd.DataFrame:
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(path)
    # Expect 9 columns: ID, revolutions, humidity, vibration, x1..x5
    if df.shape[1] != 9:
        raise ValueError(f"Unexpected column count ({df.shape[1]}). Expected 9 columns from Kaggle CSV.")
    df.columns = [
        "ID", "revolutions", "humidity", "vibration",
        "x1", "x2", "x3", "x4", "x5"
    ]
    return df

def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    # Mappings based on prior analysis
    df = df.rename(columns={
        "x1": "temperature",
        "x2": "speed",
        "x3": "signal_strength",
        "x4": "energy",
        "x5": "motor_cycles",
    })
    return df

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    # acceleration as delta of speed
    df["acceleration"] = df["speed"].diff().fillna(0)
    return df

def simulate_timestamps(n_rows: int) -> pd.Series:
    print("[INFO] Generating realistic timestamps (multi-day with rush/quiet periods)...")
    ts = []
    cur = START_DT
    while len(ts) < n_rows:
        wd = cur.weekday()  # 0=Mon .. 6=Sun
        h  = cur.hour
        if wd < 5:  # weekdays
            if 7 <= h <= 9 or 17 <= h <= 19:
                step = np.random.poisson(RUSH_POIS)
            elif 10 <= h <= 16:
                step = np.random.poisson(MID_POIS)
            else:
                step = np.random.poisson(QUIET_POIS)
        else:       # weekends
            step = np.random.poisson(WEND_POIS)
        step = max(int(step), 1)
        ts.append(cur)
        cur += timedelta(seconds=step)
    return pd.Series(ts)

def make_fixed_threshold_labels(df: pd.DataFrame) -> pd.DataFrame:
    # vibration percentile labels
    v90  = df["vibration"].quantile(0.90)
    v925 = df["vibration"].quantile(0.925)
    v95  = df["vibration"].quantile(0.95)
    v975 = df["vibration"].quantile(0.975)

    df["Fault_900"] = (df["vibration"] >= v90 ).astype(int)
    df["Fault_925"] = (df["vibration"] >= v925).astype(int)
    df["Fault_950"] = (df["vibration"] >= v95 ).astype(int)
    df["Fault_975"] = (df["vibration"] >= v975).astype(int)
    return df

def _combo_label(df, vib_p, rev_p, spd_p):
    vib_thr = df["vibration"].quantile(vib_p)
    rev_thr = df["revolutions"].quantile(rev_p)
    spd_thr = df["speed"].abs().quantile(spd_p)

    c_v = df["vibration"] >= vib_thr
    c_r = df["revolutions"] >= rev_thr
    c_s = df["speed"].abs() >= spd_thr

    k = c_v.astype(int) + c_r.astype(int) + c_s.astype(int)
    label = (k >= 2).astype(int)
    rate = float(label.mean())
    thresholds = {"vibration": float(vib_thr), "revolutions": float(rev_thr), "speed_abs": float(spd_thr)}
    return label, rate, thresholds

def auto_tune_combo(df: pd.DataFrame, target=0.08):
    # Wider search grids to find closer match to target rate
    vib_grid = [0.70, 0.75, 0.80, 0.85, 0.90]     # vibration percentiles
    rev_grid = [0.65, 0.70, 0.75, 0.80, 0.85]     # revolutions percentiles
    spd_grid = [0.65, 0.70, 0.75, 0.80, 0.85]     # speed percentiles

    best = None
    for vp in vib_grid:
        for rp in rev_grid:
            for sp in spd_grid:
                lab, rate, thr = _combo_label(df, vp, rp, sp)
                diff = abs(rate - target)
                cand = (diff, vp, rp, sp, rate, thr, lab)
                if best is None or cand[0] < best[0]:
                    best = cand

    diff, vp, rp, sp, rate, thr, lab = best
    return {
        "vib_p": vp,
        "rev_p": rp,
        "spd_p": sp,
        "rate": rate,
        "thr": thr,
        "label": lab
    }

def main():
    os.makedirs(os.path.dirname(OUT_MAIN), exist_ok=True)

    df = load_raw(RAW_PATH)
    df = rename_features(df)
    df = add_engineered(df)

    # timestamps
    df["timestamp"] = simulate_timestamps(len(df))

    # clean NaNs just in case
    nans_before = int(df.isna().sum().sum())
    df = df.ffill().bfill()
    nans_after = int(df.isna().sum().sum())

    # fixed labels
    df = make_fixed_threshold_labels(df)

    # tuned combo label
    if AUTO_TUNE:
        res = auto_tune_combo(df, target=TARGET_RATE)
        df["Fault_Combo2of3_TUNED"] = res["label"]
        print("\n[AUTO-TUNE] Selected percentiles:",
              f"vib={res['vib_p']}, rev={res['rev_p']}, spd={res['spd_p']}")
        print("[AUTO-TUNE] Threshold values:",
              {k: round(v, 4) for k, v in res["thr"].items()})
        print(f"[AUTO-TUNE] Achieved rate: {res['rate']*100:.2f}%")
    else:
        lab, rate, thr = _combo_label(df, MANUAL_VIB_P, MANUAL_REV_P, MANUAL_SPD_P)
        df["Fault_Combo2of3_TUNED"] = lab
        print("\n[MANUAL] Thresholds:",
              f"vib_p={MANUAL_VIB_P}, rev_p={MANUAL_REV_P}, spd_p={MANUAL_SPD_P}")
        print("[MANUAL] Threshold values:",
              {k: round(v, 4) for k, v in thr.items()})
        print(f"[MANUAL] Achieved rate: {rate*100:.2f}%")

    # order columns
    ordered = [
        "timestamp", "temperature", "speed", "acceleration", "humidity", "vibration",
        "revolutions", "signal_strength", "energy", "motor_cycles",
        "Fault_900", "Fault_925", "Fault_950", "Fault_975", "Fault_Combo2of3_TUNED"
    ]
    df = df[ordered]

    # save
    df.to_csv(OUT_MAIN, index=False)
    df.sample(frac=1.0, random_state=42).to_csv(OUT_SHUF, index=False)

    # report
    print("\n Processing complete.")
    print(f"   • Saved main:     {OUT_MAIN}")
    print(f"   • Saved shuffled: {OUT_SHUF}\n")

    # distributions
    total = len(df)
    for col in ["Fault_900", "Fault_925", "Fault_950", "Fault_975", "Fault_Combo2of3_TUNED"]:
        vc = df[col].value_counts().sort_index()
        pos = int(vc.get(1, 0)); neg = int(vc.get(0, 0))
        print(f"🔎 {col}: 1={pos} ({pos/total*100:.2f}%), 0={neg}, total={total}")

    print(f"\n🧹 NaNs before: {nans_before}  →  after: {nans_after}")

if __name__ == "__main__":
    main()
