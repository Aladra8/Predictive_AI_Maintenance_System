#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def find_col(df: pd.DataFrame, aliases):
    m = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in m: return m[a.lower()]
    for a in aliases:
        for k,v in m.items():
            if a.lower() in k: return v
    return None

def standardize(X: pd.DataFrame):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    return Z, scaler

def safe_silhouette(Z, labels):
    y = np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2: return np.nan
    bc = np.bincount(y)
    if len(bc) < 2 or bc.min() < 10: return np.nan
    try:
        return float(silhouette_score(Z, y, metric="euclidean"))
    except Exception:
        return np.nan

def build_labels_2of4(df, vib, spd, eng, rev, base_q=0.95, scale=1.0):
    cols = [vib, spd, eng, rev]
    thr = {c: df[c].quantile(base_q) * scale for c in cols}
    flags = (df[cols] > pd.Series(thr)).astype(int)
    y = (flags.sum(axis=1) >= 2).astype(int).values
    return y, thr

# --- denoising: sample-based fallback or time-based if timestamp present
def time_aware_denoise(series, ts, window_minutes: int):
    if ts is None or window_minutes < 1:
        # simple causal moving average over index
        return series.rolling(window=5, min_periods=1).mean()
    s = pd.DataFrame({"v": series.values}, index=ts)
    sm = s.rolling(f"{window_minutes}T", min_periods=1).mean().reindex(s.index).ffill().bfill()
    return sm["v"].values

def select_topk_by_permutation(X: pd.DataFrame, y: np.ndarray, k=4, seed=42, subsample=6000):
    # light, balanced subsample for PI (avoids bias to majority)
    df = X.copy()
    df["__y__"] = y
    pos = df[df["__y__"]==1]
    neg = df[df["__y__"]==0]
    n = min(len(pos), len(neg), subsample//2)
    if n > 0:
        small = pd.concat([pos.sample(n, random_state=seed),
                           neg.sample(n, random_state=seed)])
    else:
        small = df
    Xs = small.drop(columns="__y__")
    ys = small["__y__"].values

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced")
    rf.fit(Xs, ys)
    pi = permutation_importance(rf, Xs, ys, n_repeats=8, random_state=seed, n_jobs=-1)
    order = np.argsort(-pi.importances_mean)
    cols = list(Xs.columns[order[:min(k, Xs.shape[1])]])
    return cols

# -----------------------------
# Time + event aggregation
# -----------------------------
def detect_time_column(df: pd.DataFrame):
    return find_col(df, ["timestamp","time","datetime","date","ts"])

def parse_timestamp(df: pd.DataFrame, time_col: str):
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    # If tz-naive, pandas will set UTC; if already tz-aware, it is retained
    order = np.argsort(ts.values)
    return ts.iloc[order].reset_index(drop=True), df.iloc[order].reset_index(drop=True)

def learn_gap_seconds(ts_fault: pd.Series, min_sec=60, max_sec=900):
    """Learn a merge gap from consecutive faulty rows, clamp to [min_sec, max_sec]."""
    if ts_fault.empty: return min_sec
    gaps = ts_fault.diff().dropna().dt.total_seconds()
    gaps = gaps[gaps < 24*3600]  # ignore day jumps
    if gaps.empty: return min_sec
    learned = float(np.nanpercentile(gaps, 95))
    return float(np.clip(learned, min_sec, max_sec))

def aggregate_fault_events(df_sorted: pd.DataFrame, ts: pd.Series, label_col="label_fault",
                           gap_sec=120.0):
    y = df_sorted[label_col].astype(int).values
    events = []
    in_evt = False
    start_t = None
    prev_t  = None
    n_rows  = 0

    for t, yy in zip(ts, y):
        if yy == 1:
            if not in_evt:
                in_evt = True
                start_t = t
                prev_t  = t
                n_rows  = 1
            else:
                if (t - prev_t).total_seconds() <= gap_sec:
                    n_rows += 1
                    prev_t = t
                else:
                    events.append({
                        "event_id": len(events)+1,
                        "start": start_t, "end": prev_t,
                        "duration_s": (prev_t - start_t).total_seconds(),
                        "n_rows": n_rows
                    })
                    # new event
                    start_t = t; prev_t = t; n_rows = 1
        else:
            if in_evt:
                events.append({
                    "event_id": len(events)+1,
                    "start": start_t, "end": prev_t,
                    "duration_s": (prev_t - start_t).total_seconds(),
                    "n_rows": n_rows
                })
                in_evt = False
    if in_evt:
        events.append({
            "event_id": len(events)+1,
            "start": start_t, "end": prev_t,
            "duration_s": (prev_t - start_t).total_seconds(),
            "n_rows": n_rows
        })
    ev = pd.DataFrame(events)
    if not ev.empty:
        ev["duration_min"] = ev["duration_s"] / 60.0
        ev["hour"] = ev["start"].dt.hour
        ev["day"]  = ev["start"].dt.floor("D")
        ev["week"] = ev["start"].dt.to_period("W").apply(lambda p: p.start_time)
        ev["month"]= ev["start"].dt.to_period("M").apply(lambda p: p.start_time)
    return ev

# -----------------------------
# Core runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to processed Kaggle CSV")
    ap.add_argument("--out", default="outputs/phase1_noise_fs", help="Output directory for figures")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--denoise-min", type=int, default=5, help="Time-based smoothing window (minutes) if timestamp exists")
    ap.add_argument("--perm-topk", type=int, default=4, help="Top-k features by permutation importance")
    ap.add_argument("--event-gap-min-sec", type=int, default=60)
    ap.add_argument("--event-gap-max-sec", type=int, default=900)
    ap.add_argument("--base-q", type=float, default=0.95, help="Quantile for 2-of-4 analytical labeling")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)

    # Load
    df = pd.read_csv(args.csv)

    # Detect columns
    col_speed     = find_col(df, ["Speed","x2"])
    col_vibration = find_col(df, ["Vibration","x3"])
    col_revs      = find_col(df, ["Revolutions","Motor_Cycles","Ecycles","x4"])
    col_energy    = find_col(df, ["Energy","x5"])
    col_signal    = find_col(df, ["Signal_Strength","Signal"])
    col_humidity  = find_col(df, ["Humidity"])
    col_accel     = find_col(df, ["Acceleration","x1"])
    col_time      = detect_time_column(df)

    for need, nm in [("Speed",col_speed),("Vibration",col_vibration),("Revolutions",col_revs),("Energy",col_energy)]:
        if nm is None:
            print(f"ERROR: required column for {need} not found."); sys.exit(1)

    # Labels (2-of-4 @ quantile)
    y, thr0 = build_labels_2of4(df, col_vibration, col_speed, col_energy, col_revs, base_q=args.base_q, scale=1.0)
    df = df.copy()
    df["label_fault"] = y
    print("Label distribution:", pd.Series(y).value_counts(normalize=True).round(3).to_dict())

    # Parse timestamps (if available)
    ts = None
    df_sorted = df
    if col_time is not None:
        ts, df_sorted = parse_timestamp(df, col_time)

    # Base feature set for PCA
    feats_all = [c for c in [col_accel, col_speed, col_vibration, col_revs, col_energy, col_signal, col_humidity] if c is not None]

    # Build 4 variants
    X_base = df[feats_all].copy()

    # --- denoise (time-aware if ts present)
    X_denoised = X_base.copy()
    if ts is not None:
        for c in [col_vibration, col_speed]:
            if c in X_denoised:
                X_denoised[c] = time_aware_denoise(X_base[c], ts, args.denoise_min)
    else:
        for c in [col_vibration, col_speed]:
            if c in X_denoised:
                X_denoised[c] = X_base[c].rolling(window=5, min_periods=1).mean()

    # --- feature selection on baseline (permutation on balanced subsample)
    X1_clean = X_base.dropna()
    y1 = df.loc[X1_clean.index, "label_fault"].astype(int).values
    topk_cols = select_topk_by_permutation(X1_clean, y1, k=args.perm_topk, seed=args.seed)
    X_selected = X_base[topk_cols].copy()

    # --- denoise + select on common index
    X2_clean = X_denoised.dropna()
    common_idx = X2_clean.index
    X_den_sel = X_denoised.loc[common_idx, topk_cols].copy()

    # helper: PCA + silhouette
    def pca_block(X: pd.DataFrame, y_all: np.ndarray, tag: str):
        Xc = X.dropna()
        idx = Xc.index
        Zs, _ = standardize(Xc)
        pca = PCA(n_components=2, random_state=args.seed)
        Z = pca.fit_transform(Zs)
        var = pca.explained_variance_ratio_
        sil = safe_silhouette(Z, y_all[idx])
        return dict(tag=tag, Z=Z, var=var, sil=sil, n=len(idx), idx=idx)

    y_all = df["label_fault"].astype(int).values
    r1 = pca_block(X_base,     y_all, "baseline")
    r2 = pca_block(X_denoised, y_all, "denoised")
    r3 = pca_block(X_selected, y_all, "selected")
    r4 = pca_block(X_den_sel,  y_all, "denoise+select")

    # 2x2 grid
    plt.figure(figsize=(11.5, 9.0))
    for i, r in enumerate([r1, r2, r3, r4], start=1):
        ax = plt.subplot(2,2,i)
        Z = r["Z"]; idx = r["idx"]
        yplot = df.loc[idx, "label_fault"].values
        ax.scatter(Z[:,0], Z[:,1], c=np.where(yplot==1, "tab:red","tab:blue"), s=6, alpha=0.6)
        ax.set_title(f"{r['tag']}  |  PC1={r['var'][0]:.2f}, PC2={r['var'][1]:.2f}, Sil={0 if np.isnan(r['sil']) else r['sil']:.2f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.text(0.02, 0.02, f"Normal={np.sum(yplot==0)}, Fault={np.sum(yplot==1)}",
                transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", alpha=0.25))
    plt.tight_layout(); plt.savefig(outdir / "pca_denoise_select_grid.png", dpi=180); plt.close()

    # Silhouette bars (with labels)
    sil_names = ["baseline","denoised","selected","denoise+select"]
    sil_vals  = [r1["sil"], r2["sil"], r3["sil"], r4["sil"]]
    plt.figure(figsize=(7.0,4.4))
    bars = plt.bar(sil_names, [0 if v is None or np.isnan(v) else v for v in sil_vals])
    for b, v in zip(bars, sil_vals):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{0 if np.isnan(v) else v:.2f}", ha="center")
    plt.ylabel("Silhouette on PC1 - PC2")
    plt.title("Effect of denoising and feature selection on separation")
    plt.tight_layout(); plt.savefig(outdir / "pca_silhouette_bars.png", dpi=180); plt.close()

    # ----- Event aggregation + plots -----
    if col_time is None:
        print("Event aggregation: no timestamp column found. Skipping time-based plots.")
    else:
        # sort-aligned df
        dfts = df_sorted.copy()
        ts_sorted = ts.copy()

        # learn merge gap and clamp
        gap_learned = learn_gap_seconds(ts_sorted[dfts["label_fault"].values==1],
                                        min_sec=args.event_gap_min_sec,
                                        max_sec=args.event_gap_max_sec)
        ev = aggregate_fault_events(dfts, ts_sorted, label_col="label_fault", gap_sec=gap_learned)

        # summary JSON
        summary = {
            "n_events": int(len(ev)),
            "duration_min_mean": float(ev["duration_min"].mean()) if not ev.empty else 0.0,
            "duration_min_p50": float(ev["duration_min"].median()) if not ev.empty else 0.0,
            "duration_min_p95": float(np.nanpercentile(ev["duration_min"], 95)) if not ev.empty else 0.0,
            "time_min": str(ts_sorted.min()) if len(ts_sorted) else None,
            "time_max": str(ts_sorted.max()) if len(ts_sorted) else None,
            "span_hours": float((ts_sorted.max() - ts_sorted.min()).total_seconds()/3600) if len(ts_sorted) else 0.0,
            "merge_gap_seconds_used": float(gap_learned)
        }
        (outdir / "events_summary.json").write_text(json.dumps(summary, indent=2))
        ev.to_csv(outdir / "events_table.csv", index=False)

        if not ev.empty:
            # durations
            plt.figure(figsize=(7.6,4.4))
            plt.hist(ev["duration_min"].clip(lower=0, upper=np.nanpercentile(ev["duration_min"], 99)), bins=30)
            plt.xlabel("Event duration (minutes)"); plt.ylabel("Count"); plt.title("Fault event duration distribution")
            plt.tight_layout(); plt.savefig(outdir / "events_duration_hist.png", dpi=180); plt.close()

            # choose plots based on coverage
            n_weeks  = ev["week"].nunique()
            n_months = ev["month"].nunique()

            if n_weeks >= 2:
                weekly = ev.groupby("week")["event_id"].nunique().reset_index()
                plt.figure(figsize=(8.8,4.4)); plt.plot(weekly["week"], weekly["event_id"], marker="o")
                plt.xlabel("Week"); plt.ylabel("Events"); plt.title("Events per week")
                plt.tight_layout(); plt.savefig(outdir / "events_per_week.png", dpi=180); plt.close()
            else:
                by_hour = ev.groupby("hour")["event_id"].nunique().reindex(range(24), fill_value=0)
                plt.figure(figsize=(8.8,4.4)); plt.bar(by_hour.index, by_hour.values)
                plt.xlabel("Hour of day"); plt.ylabel("Events"); plt.title("Events by hour (limited span)")
                plt.tight_layout(); plt.savefig(outdir / "events_by_hour.png", dpi=180); plt.close()

            if n_months >= 2:
                monthly = ev.groupby("month")["event_id"].nunique().reset_index()
                plt.figure(figsize=(8.8,4.4)); plt.plot(monthly["month"], monthly["event_id"], marker="o")
                plt.xlabel("Month"); plt.ylabel("Events"); plt.title("Events per month")
                plt.tight_layout(); plt.savefig(outdir / "events_per_month.png", dpi=180); plt.close()
            else:
                by_day = ev.groupby("day")["event_id"].nunique().reset_index()
                plt.figure(figsize=(8.8,4.4)); plt.plot(by_day["day"], by_day["event_id"], marker="o")
                plt.xlabel("Day"); plt.ylabel("Events"); plt.title("Events per day (limited span)")
                plt.tight_layout(); plt.savefig(outdir / "events_per_day.png", dpi=180); plt.close()

    # small readme for the folder
    (outdir / "README_noise_fs.txt").write_text(
        "This folder contains PCA noise/feature-selection checks and event summaries.\n"
        "Figures: pca_denoise_select_grid.png, pca_silhouette_bars.png, events_*.\n"
        "Tables: events_table.csv, events_summary.json.\n"
    )

    print("\n=== Noise/FS add-on done. Outputs in:", outdir, "===")
    print("- pca_denoise_select_grid.png")
    print("- pca_silhouette_bars.png")
    print("- events_*.png + events_table.csv + events_summary.json (if time present)")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# import argparse, sys, json
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import permutation_importance

# # -----------------------------
# # Utilities
# # -----------------------------
# def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# def find_col(df: pd.DataFrame, aliases):
#     m = {c.lower(): c for c in df.columns}
#     for a in aliases:
#         if a.lower() in m: return m[a.lower()]
#     for a in aliases:
#         for k,v in m.items():
#             if a.lower() in k: return v
#     return None

# def standardize(X: pd.DataFrame):
#     scaler = StandardScaler()
#     Z = scaler.fit_transform(X)
#     return Z, scaler

# def safe_silhouette(Z, labels):
#     y = np.asarray(labels).astype(int)
#     if len(np.unique(y)) < 2: return np.nan
#     if min(np.bincount(y)) < 10: return np.nan
#     try:
#         return float(silhouette_score(Z, y, metric="euclidean"))
#     except Exception:
#         return np.nan

# def build_labels_2of4(df, vib, spd, eng, rev, base_q=0.95, scale=1.0):
#     cols = [vib, spd, eng, rev]
#     thr = {c: df[c].quantile(base_q) * scale for c in cols}
#     flags = (df[cols] > pd.Series(thr)).astype(int)
#     y = (flags.sum(axis=1) >= 2).astype(int).values
#     return y, thr

# def simple_denoise(series: pd.Series, window=5):
#     # causal moving average, forward-fill edges
#     if window and window >= 3:
#         rm = series.rolling(window=window, min_periods=1).mean()
#         return rm.ffill().bfill()
#     return series

# def select_topk_by_permutation(X: pd.DataFrame, y: np.ndarray, k=4, seed=42):
#     rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced")
#     rf.fit(X, y)
#     pi = permutation_importance(rf, X, y, n_repeats=8, random_state=seed, n_jobs=-1)
#     order = np.argsort(-pi.importances_mean)
#     cols = list(X.columns[order[:min(k, X.shape[1])]])
#     return cols

# # -----------------------------
# # Timestamp helpers and events
# # -----------------------------
# def detect_time_column(df: pd.DataFrame):
#     candidates = ["timestamp","time","datetime","date","ts","time_stamp"]
#     return find_col(df, candidates)

# def parse_utc(ts: pd.Series):
#     # produce timezone-aware UTC timestamps where possible
#     t = pd.to_datetime(ts, errors="coerce", utc=True)
#     return t

# def seconds_between(a, b):
#     # a and b are pandas Timestamps (aware or naive but aligned)
#     delta = (a - b)
#     return float(delta / pd.Timedelta(seconds=1))

# def learn_global_gap_quantile(fault_ts: pd.Series, q=0.95, min_gaps=10):
#     gaps = fault_ts.sort_values().diff().dropna()
#     # ignore extreme gaps over 1 day
#     gaps = gaps[gaps < pd.Timedelta(days=1)]
#     if len(gaps) >= min_gaps:
#         return float(gaps.quantile(q) / pd.Timedelta(seconds=1))
#     return None

# def learn_daily_gap_quantiles(fault_ts: pd.Series, q=0.95, min_gaps=6):
#     # compute per-day thresholds on consecutive gaps within the day
#     df = pd.DataFrame({"ts": fault_ts.dropna().sort_values()})
#     df["day"] = df["ts"].dt.floor("D")
#     th = {}
#     for day, grp in df.groupby("day"):
#         gaps = grp["ts"].diff().dropna()
#         gaps = gaps[gaps < pd.Timedelta(days=1)]
#         if len(gaps) >= min_gaps:
#             th[day] = float(gaps.quantile(q) / pd.Timedelta(seconds=1))
#     return th  # dict day->seconds

# def aggregate_fault_events(df: pd.DataFrame,
#                            label_col="label_fault",
#                            time_col=None,
#                            mode="fixed",
#                            gap_seconds=60.0,
#                            q=0.95,
#                            fallback_sec=10.0,
#                            min_gaps_global=10,
#                            min_gaps_daily=6):
#     if time_col is None or time_col not in df.columns:
#         return pd.DataFrame(), {"gap_mode_used": "none", "merge_gap_seconds_used": None, "n_events": 0}

#     # parse timestamps
#     ts = parse_utc(df[time_col])
#     d = pd.DataFrame({"ts": ts, "y": df[label_col].astype(int)})
#     d = d.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

#     # pre-compute thresholds
#     daily_map = {}
#     global_q_sec = None
#     if mode in ("quantile", "quantile-per-day"):
#         fault_ts = d.loc[d["y"] == 1, "ts"]
#         if mode == "quantile":
#             global_q_sec = learn_global_gap_quantile(fault_ts, q=q, min_gaps=min_gaps_global)
#         else:
#             daily_map = learn_daily_gap_quantiles(fault_ts, q=q, min_gaps=min_gaps_daily)
#         # compute a reasonable global fallback if needed
#         if global_q_sec is None:
#             global_q_sec = learn_global_gap_quantile(fault_ts, q=min(0.8, q), min_gaps=min_gaps_global//2)
#         if global_q_sec is None:
#             global_q_sec = fallback_sec

#     events = []
#     cur_id = 0
#     in_evt = False
#     start_t = None
#     prev_t = None
#     count = 0

#     def current_threshold_seconds(tstamp):
#         if mode == "fixed":
#             return gap_seconds
#         if mode == "quantile":
#             return global_q_sec
#         if mode == "quantile-per-day":
#             day = pd.Timestamp(tstamp).floor("D")
#             return daily_map.get(day, global_q_sec if global_q_sec is not None else fallback_sec)
#         return gap_seconds

#     for _, r in d.iterrows():
#         t = r["ts"]
#         y = int(r["y"])
#         if y == 1:
#             if not in_evt:
#                 cur_id += 1
#                 in_evt = True
#                 start_t = t
#                 prev_t = t
#                 count = 1
#             else:
#                 gap = seconds_between(t, prev_t)
#                 thr = current_threshold_seconds(t)
#                 if gap <= thr:
#                     count += 1
#                     prev_t = t
#                 else:
#                     dur = seconds_between(prev_t, start_t)
#                     events.append({
#                         "event_id": cur_id,
#                         "start": start_t,
#                         "end": prev_t,
#                         "duration_s": float(max(dur, 0.0)),
#                         "n_rows": count
#                     })
#                     cur_id += 1
#                     start_t = t
#                     prev_t = t
#                     count = 1
#         else:
#             if in_evt:
#                 dur = seconds_between(prev_t, start_t)
#                 events.append({
#                     "event_id": cur_id,
#                     "start": start_t,
#                     "end": prev_t,
#                     "duration_s": float(max(dur, 0.0)),
#                     "n_rows": count
#                 })
#                 in_evt = False
#     if in_evt:
#         dur = seconds_between(prev_t, start_t)
#         events.append({
#             "event_id": cur_id,
#             "start": start_t,
#             "end": prev_t,
#             "duration_s": float(max(dur, 0.0)),
#             "n_rows": count
#         })

#     ev = pd.DataFrame(events)
#     # summary for TeX
#     summary = {
#         "gap_mode_used": mode,
#         "merge_gap_seconds_used": (gap_seconds if mode == "fixed" else None),
#         "quantile_used": (q if mode != "fixed" else None),
#         "global_quantile_gap_sec": (None if mode == "fixed" else float(global_q_sec) if global_q_sec is not None else None),
#         "daily_threshold_days": int(len(daily_map)) if daily_map else 0,
#         "fallback_sec": float(fallback_sec),
#         "n_events": int(len(ev)) if not ev.empty else 0,
#         "span_days": int(ev["start"].dt.floor("D").nunique()) if not ev.empty else 0,
#         "first_ts": str(ev["start"].min()) if not ev.empty else "",
#         "last_ts": str(ev["end"].max()) if not ev.empty else ""
#     }
#     return ev, summary

# def write_events_tex(tex_path: Path, summary: dict):
#     lines = []
#     lines.append("% auto-generated from events_summary.json")
#     def macro(name, val):
#         if val is None: val = ""
#         if isinstance(val, bool): val = "true" if val else "false"
#         lines.append(f"\\newcommand\\{name}{{{val}}}")
#     macro("EventGapMode", summary.get("gap_mode_used",""))
#     macro("EventMergeGapSec", str(summary.get("merge_gap_seconds_used","")))
#     macro("EventQuantileUsed", str(summary.get("quantile_used","")))
#     macro("EventGlobalQGapSec", str(summary.get("global_quantile_gap_sec","")))
#     macro("EventDailyThresholdDays", str(summary.get("daily_threshold_days","")))
#     macro("EventFallbackSec", str(summary.get("fallback_sec","")))
#     macro("EventCount", str(summary.get("n_events","0")))
#     macro("EventSpanDays", str(summary.get("span_days","0")))
#     macro("EventFirstTS", summary.get("first_ts",""))
#     macro("EventLastTS", summary.get("last_ts",""))
#     tex_path.write_text("\n".join(lines), encoding="utf-8")

# # -----------------------------
# # Core runner
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="Path to processed Kaggle CSV")
#     ap.add_argument("--out", default="outputs/phase1_noise_fs", help="Output directory for figures")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--denoise-window", type=int, default=5, help="Moving average window for denoising")
#     ap.add_argument("--perm-topk", type=int, default=4, help="Top-k features by permutation importance")
#     # Event merge options
#     ap.add_argument("--event-gap-mode", choices=["fixed","quantile","quantile-per-day"], default="fixed")
#     ap.add_argument("--event-gap-sec", type=float, default=60.0, help="Fixed merge gap in seconds if mode=fixed")
#     ap.add_argument("--event-quantile", type=float, default=0.95, help="Quantile used if mode is quantile or quantile-per-day")
#     ap.add_argument("--event-fallback-sec", type=float, default=10.0, help="Fallback gap in seconds if not enough gaps")
#     args = ap.parse_args()

#     outdir = Path(args.out); ensure_dir(outdir)

#     # Load
#     df = pd.read_csv(args.csv)

#     # Detect columns
#     col_speed     = find_col(df, ["Speed","x2"])
#     col_vibration = find_col(df, ["Vibration","x3"])
#     col_revs      = find_col(df, ["Revolutions","Motor_Cycles","Ecycles","x4"])
#     col_energy    = find_col(df, ["Energy","x5"])
#     col_signal    = find_col(df, ["Signal_Strength","Signal"])
#     col_humidity  = find_col(df, ["Humidity"])
#     col_accel     = find_col(df, ["Acceleration","x1"])
#     col_time      = detect_time_column(df)

#     for need, nm in [("Speed",col_speed),("Vibration",col_vibration),("Revolutions",col_revs),("Energy",col_energy)]:
#         if nm is None:
#             print(f"ERROR: required column for {need} not found."); sys.exit(1)

#     # Labels (2-of-4 @ 95th)
#     y, thr0 = build_labels_2of4(df, col_vibration, col_speed, col_energy, col_revs, base_q=0.95, scale=1.0)
#     df = df.copy()
#     df["label_fault"] = y
#     print("Label distribution:", pd.Series(y).value_counts(normalize=True).round(3).to_dict())

#     # Base feature set for PCA (only numeric, keep meaningful telemetry)
#     feats_all = [c for c in [col_accel, col_speed, col_vibration, col_revs, col_energy, col_signal, col_humidity] if c is not None]
#     X_base = df[feats_all].copy()

#     # 1) Baseline
#     X1 = X_base.copy()
#     # 2) Denoised
#     X2 = X_base.copy()
#     for c in [col_vibration, col_speed]:
#         if c in X2.columns:
#             X2[c] = simple_denoise(X2[c], window=args.denoise_window)
#     # 3) Selected
#     X1_clean = X1.dropna()
#     y1 = df.loc[X1_clean.index, "label_fault"].astype(int).values
#     topk_cols = select_topk_by_permutation(X1_clean, y1, k=args.perm_topk, seed=args.seed)
#     X3 = X_base[topk_cols].copy()
#     # 4) Denoised + selected
#     X2_clean = X2.dropna()
#     common_idx = X2_clean.index
#     X4 = X2.loc[common_idx, topk_cols].copy()

#     def run_pca_block(X: pd.DataFrame, y_all: np.ndarray, tag: str):
#         Xc = X.dropna()
#         idx = Xc.index
#         Zs, _ = standardize(Xc)
#         pca = PCA(n_components=2, random_state=args.seed)
#         Z = pca.fit_transform(Zs)
#         var = pca.explained_variance_ratio_
#         sil = safe_silhouette(Z, y_all[idx])
#         return dict(tag=tag, Z=Z, var=var, sil=sil, n=len(idx), idx=idx)

#     y_all = df["label_fault"].astype(int).values
#     r1 = run_pca_block(X1, y_all, "baseline")
#     r2 = run_pca_block(X2, y_all, "denoised")
#     r3 = run_pca_block(X3, y_all, "selected")
#     r4 = run_pca_block(X4, y_all, "denoise+select")

#     plt.figure(figsize=(11.5, 9.0))
#     for i, r in enumerate([r1, r2, r3, r4], start=1):
#         ax = plt.subplot(2,2,i)
#         Z = r["Z"]; idx = r["idx"]
#         yplot = df.loc[idx, "label_fault"].values
#         # color by class (blue=0, red=1) without hard-coded colors to keep defaults
#         ax.scatter(Z[:,0], Z[:,1], c=yplot, s=6, alpha=0.6, cmap="coolwarm")
#         ax.set_title(f"{r['tag']}  |  PC1={r['var'][0]:.2f}, PC2={r['var'][1]:.2f}, Sil={0 if np.isnan(r['sil']) else r['sil']:.2f}")
#         ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
#         ax.text(0.02, 0.02, f"Normal={np.sum(yplot==0)}, Fault={np.sum(yplot==1)}",
#                 transform=ax.transAxes, fontsize=8,
#                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
#     plt.tight_layout()
#     plt.savefig(outdir / "pca_denoise_select_grid.png", dpi=180)
#     plt.close()

#     # Silhouette bars
#     sil_names = ["baseline","denoised","selected","denoise+select"]
#     sil_vals  = [r1["sil"], r2["sil"], r3["sil"], r4["sil"]]
#     plt.figure(figsize=(8.0,5.0))
#     bars = plt.bar(sil_names, [0 if v is None or np.isnan(v) else v for v in sil_vals])
#     for b, v in zip(bars, sil_vals):
#         plt.text(b.get_x()+b.get_width()/2., b.get_height()+0.005, f"{0 if v is None or np.isnan(v) else v:.2f}",
#                  ha='center', va='bottom', fontsize=10)
#     plt.ylabel("Silhouette on PC1 - PC2")
#     plt.title("Effect of denoising and feature selection on separation")
#     plt.tight_layout()
#     plt.savefig(outdir / "pca_silhouette_bars.png", dpi=180)
#     plt.close()

#     # ----- Event aggregation with flexible merge rule -----
#     df_ts = df.copy()
#     time_col = detect_time_column(df_ts)
#     if time_col is None:
#         print("Event aggregation: no usable time column found. Skipping event plots.")
#         ev, summary = pd.DataFrame(), {"gap_mode_used":"none","n_events":0}
#     else:
#         ev, summary = aggregate_fault_events(
#             df_ts, label_col="label_fault", time_col=time_col,
#             mode=args.event_gap_mode,
#             gap_seconds=args.event_gap_sec,
#             q=args.event_quantile,
#             fallback_sec=args.event_fallback_sec
#         )
#         # save machine-readable summary and TeX snippet with macros
#         (outdir / "events_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
#         write_events_tex(outdir / "events_summary.tex", summary)

#     if ev is not None and not ev.empty:
#         # Duration histogram (minutes)
#         ev["duration_min"] = ev["duration_s"] / 60.0
#         plt.figure(figsize=(8.5,5.0))
#         plt.hist(ev["duration_min"].clip(lower=0, upper=120), bins=30)
#         plt.xlabel("Event duration (minutes)")
#         plt.ylabel("Count")
#         plt.title("Fault event duration distribution")
#         plt.tight_layout()
#         plt.savefig(outdir / "events_duration_hist.png", dpi=180)
#         plt.close()

#         # Per day
#         ev["day"] = ev["start"].dt.floor("D")
#         daily = ev.groupby("day")["event_id"].nunique().reset_index()
#         plt.figure(figsize=(9.0,5.0))
#         plt.plot(daily["day"], daily["event_id"], marker="o")
#         plt.xlabel("Day")
#         plt.ylabel("Events")
#         plt.title("Events per day (limited span)")
#         plt.tight_layout()
#         plt.savefig(outdir / "events_per_day.png", dpi=180); plt.close()

#         # Per week
#         ev["week"] = ev["start"].dt.to_period("W").apply(lambda p: p.start_time)
#         weekly = ev.groupby("week")["event_id"].nunique().reset_index()
#         plt.figure(figsize=(9.0,5.0))
#         plt.plot(weekly["week"], weekly["event_id"], marker="o")
#         plt.xlabel("Week")
#         plt.ylabel("Events")
#         plt.title("Events per week")
#         plt.tight_layout()
#         plt.savefig(outdir / "events_per_week.png", dpi=180); plt.close()

#         # Per month
#         ev["month"] = ev["start"].dt.to_period("M").apply(lambda p: p.start_time)
#         monthly = ev.groupby("month")["event_id"].nunique().reset_index()
#         plt.figure(figsize=(9.0,5.0))
#         plt.plot(monthly["month"], monthly["event_id"], marker="o")
#         plt.xlabel("Month")
#         plt.ylabel("Events")
#         plt.title("Events per month")
#         plt.tight_layout()
#         plt.savefig(outdir / "events_per_month.png", dpi=180); plt.close()

#         # By hour of day
#         ev["hour"] = ev["start"].dt.hour
#         hr = ev.groupby("hour")["event_id"].nunique().reset_index()
#         plt.figure(figsize=(9.0,5.0))
#         plt.bar(hr["hour"], hr["event_id"])
#         plt.xlabel("Hour of day")
#         plt.ylabel("Events")
#         plt.title("Events by hour (limited span)")
#         plt.tight_layout()
#         plt.savefig(outdir / "events_by_hour.png", dpi=180); plt.close()

#         # Save events table
#         ev.to_csv(outdir / "events_table.csv", index=False)

#     print("\n=== Noise/FS + Event aggregation done. Outputs in:", outdir, "===")
#     print("- pca_denoise_select_grid.png")
#     print("- pca_silhouette_bars.png")
#     print("- events_*.png and events_table.csv (if time present)")
#     print("- events_summary.json and events_summary.tex (macros for LaTeX)")

# if __name__ == "__main__":
#     main()
