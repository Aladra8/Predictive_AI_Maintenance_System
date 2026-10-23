"""07 – Experiment 2: the researcher-processed dataset (as used in the thesis runs).

The processed table (data/processed/processed_large_dataset_v4.csv) was built in
phase 2 by: renaming x1..x5 to physical names, forward-filling vibration, adding
acceleration = diff(speed), synthesising a multi-day timestamp, adding
vibration-percentile labels (Fault_900..975) and an auto-tuned 2-of-3 label
(Fault_Combo2of3_TUNED); the 2-of-4 label (label_fault) was added in phase 2/3.
A row-shuffled copy (..._shuffled.csv) was also produced.

This script (1) verifies how the processed table relates to the raw data,
(2) reports the originally reported metrics (data/reported/), and
(3) re-evaluates three labels under the original random split, on the shuffled
copy, and under leave-one-block-out evaluation in recording order.
"""
from _common import setup, path, ROOT

import numpy as np
import pandas as pd

from pdm import audit, data, labels, metrics, models, splits
from pdm.utils import out_dir, save_json, save_table

cfg, _args = setup(__doc__)
out = out_dir("07_processed_dataset", cfg)
rep = cfg["thesis_replication"]
names = _args.models or [m for m in cfg["models"]["list"] if m != "svm"]
if cfg["models"]["engine"] == "numpy":
    names = [m for m in names if m in ("rf", "lr")]

proc = pd.read_csv(ROOT / "data/processed/processed_large_dataset_v4.csv")
shuf = pd.read_csv(ROOT / "data/processed/processed_large_dataset_v4_shuffled.csv")
raw = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])

# 1. provenance of the processed table ------------------------------------------------
checks = {f"{new} == raw {old}": bool(np.allclose(raw[old], proc[new])) for old, new in cfg["thesis_names"].items()}
checks["vibration == raw vibration (ffill)"] = bool(np.allclose(raw.vibration.ffill().bfill(), proc.vibration))
checks["acceleration == diff(speed)"] = bool(np.allclose(proc.speed.diff().fillna(0), proc.acceleration))
checks["shuffled file is a row permutation"] = bool(
    proc.sort_values(list(proc.columns)).reset_index(drop=True).equals(shuf.sort_values(list(shuf.columns)).reset_index(drop=True)))
ts = pd.to_datetime(proc.timestamp)
steps = ts.diff().dt.total_seconds().dropna()
checks["synthetic_clock"] = {"start": str(ts.iloc[0]), "end": str(ts.iloc[-1]),
                             "span_hours": float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600),
                             "step_s_min": float(steps.min()), "step_s_median": float(steps.median()),
                             "step_s_max": float(steps.max()), "real_span_hours_at_4Hz": len(proc) / 4 / 3600}
# auto-tuned 2-of-3 label: which percentiles reproduce it?
tuned = None
for vp in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for rp in [0.65, 0.70, 0.75, 0.80, 0.85]:
        for sp in [0.65, 0.70, 0.75, 0.80, 0.85]:
            y = ((proc.vibration >= proc.vibration.quantile(vp)).astype(int)
                 + (proc.revolutions >= proc.revolutions.quantile(rp)).astype(int)
                 + (proc.speed.abs() >= proc.speed.abs().quantile(sp)).astype(int) >= 2).astype(int)
            if (y.to_numpy() == proc.Fault_Combo2of3_TUNED.to_numpy()).all():
                tuned = {"vibration_q": vp, "revolutions_q": rp, "abs_speed_q": sp}
checks["Fault_Combo2of3_TUNED_percentiles"] = tuned
save_json(checks, out / "provenance_checks.json")

# 2. labels in the processed table ------------------------------------------------------
y24, _, _ = labels.percentile_vote(proc, rep["rule_2of4"], rep["quantile"], rep["k"])
lab = {"label_fault (2-of-4 @95)": y24,
       "Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65)": proc.Fault_Combo2of3_TUNED.to_numpy(),
       "Fault_950 (vibration > 95th pct)": proc.Fault_950.to_numpy()}
ls = pd.DataFrame([{"label": k, **audit.label_structure(v, 4)} for k, v in lab.items()])
save_table(ls, out / "label_structure", "{:.3f}")

# 3. reported metrics (as produced in 2025) ---------------------------------------------
rp_rows = []
for f, regime in [("model_metrics_processed.csv", "processed, random split"),
                  ("model_metrics_processed_guard.csv", "processed, guard (as coded)"),
                  ("model_metrics_original.csv", "original columns, random split")]:
    d = pd.read_csv(ROOT / "data/reported" / f)
    d.insert(0, "regime", regime)
    rp_rows.append(d[["regime", "model", "f1_macro", "recall_fault", "auc_roc", "avg_precision"]])
save_table(pd.concat(rp_rows), out / "reported_metrics_2025")

REV = ["revolutions", "temperature", "speed", "signal_strength", "energy", "acceleration"]   # revolutions and its copies
GUARD = {"label_fault (2-of-4 @95)": ["vibration"] + REV,
         "Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65)": ["vibration"] + REV,
         "Fault_950 (vibration > 95th pct)": ["vibration"]}
# 4. re-evaluation --------------------------------------------------------------------
F = rep["features"]
X = proc[F].to_numpy(float); Xs = shuf[F].to_numpy(float)
rows = []
for lname, y in lab.items():
    ysh = {"label_fault (2-of-4 @95)": labels.percentile_vote(shuf, rep["rule_2of4"], rep["quantile"], rep["k"])[0],
           "Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65)": shuf.Fault_Combo2of3_TUNED.to_numpy(),
           "Fault_950 (vibration > 95th pct)": shuf.Fault_950.to_numpy()}[lname]
    for mname in names:
        # (a) random split on the processed table, (b) on the shuffled copy
        for tag, XX, yy in [("random split", X, y), ("random split, shuffled file", Xs, ysh)]:
            tr, va, te = splits.random_stratified(yy, tuple(rep["split"].values()), cfg["seed"])
            trv = np.r_[tr, va]
            m = models.build(cfg, [mname])[mname]
            p = models.fit_predict(m, XX[trv], yy[trv], XX[te], mname, cfg, cfg["seed"])
            r = metrics.classification_metrics(yy[te], p)
            rows.append({"label": lname, "evaluation": tag, "model": mname, "f1_macro": r["f1_macro"],
                         "roc_auc": r["roc_auc"], "avg_precision": r["avg_precision"], "test_positives": r["positives"]})
        # (c) leave-one-block-out in recording order (pooled out-of-fold scores)
        oof = np.full(len(y), np.nan)
        for b, trn, tst in splits.blocked_folds(len(y), cfg["time_aware"]["n_blocks"], gap=240):
            m = models.build(cfg, [mname])[mname]
            oof[tst] = models.fit_predict(m, X[trn], y[trn], X[tst], mname, cfg, cfg["seed"])
        r = metrics.classification_metrics(y, oof)
        rows.append({"label": lname, "evaluation": "blocked CV (10 blocks, 60 s purge)", "model": mname,
                     "f1_macro": r["f1_macro"], "roc_auc": r["roc_auc"], "avg_precision": r["avg_precision"],
                     "test_positives": r["positives"]})
        # (d) blocked CV without the label-defining channels AND their copies (strict guard)
        keep = [f for f in F if f not in GUARD[lname]]
        Xg = proc[keep].to_numpy(float); oof = np.full(len(y), np.nan)
        for b, trn, tst in splits.blocked_folds(len(y), cfg["time_aware"]["n_blocks"], gap=240):
            m = models.build(cfg, [mname])[mname]
            oof[tst] = models.fit_predict(m, Xg[trn], y[trn], Xg[tst], mname, cfg, cfg["seed"])
        r = metrics.classification_metrics(y, oof)
        rows.append({"label": lname, "evaluation": "blocked CV, strict guard (" + ", ".join(keep) + ")", "model": mname,
                     "f1_macro": r["f1_macro"], "roc_auc": r["roc_auc"], "avg_precision": r["avg_precision"],
                     "test_positives": r["positives"]})
        print(f"[07] {lname:48s} {mname:3s} random AUC={rows[-4]['roc_auc']:.3f}  shuffled={rows[-3]['roc_auc']:.3f}  blocked={rows[-2]['roc_auc']:.3f}  blocked+strict guard={rows[-1]['roc_auc']:.3f}")
save_table(pd.DataFrame(rows), out / "reevaluation", "{:.4f}")
print(f"[07] outputs -> {out}")
