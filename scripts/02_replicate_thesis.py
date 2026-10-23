"""02 – Replicate the thesis experiment exactly as it was run.

2-of-4 rule at the 95th percentile (identical to 2-of-3, see 01), random
stratified 70/15/15 split, models refit on train+val and scored on test.
Feature sets: standard (9 features), the Leakage-Guard as described in the
thesis, the Leakage-Guard as coded, and a guard that also removes the
deterministic copies of the label-defining channels.
"""
from _common import setup, path

import numpy as np
import pandas as pd

from pdm import data, labels, metrics, models, splits
from pdm.utils import out_dir, save_table

cfg, _ = setup(__doc__)
out = out_dir("02_replicate_thesis", cfg)
rep = cfg["thesis_replication"]

k = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
t = data.thesis_frame(k, cfg["thesis_names"], cfg["data"]["vibration_nan"])
y, thr, _ = labels.percentile_vote(t, rep["rule_2of4"], rep["quantile"], rep["k"])
t["row_index"] = np.arange(len(t))

F = rep["features"]
copies = ["temperature", "speed", "signal_strength", "energy", "acceleration"]  # all derived from revolutions
feature_sets = {
    "standard (9 features)": F,
    "guard as described (-vib, speed, rev)": [f for f in F if f not in rep["guard_as_described"]],
    "guard as coded (-vib, speed, energy, rev)": [f for f in F if f not in rep["guard_as_coded"]],
    "guard without copies (humidity, motor_cycles)": [f for f in F if f not in rep["guard_as_coded"] + copies],
    "row index only (time)": ["row_index"],
}

tr, va, te = splits.random_stratified(y, tuple(rep["split"].values()), cfg["seed"])
trv = np.r_[tr, va]   # thesis code refits the selected model on train+val
rows = []
for fs_name, fs in feature_sets.items():
    X = t[fs].to_numpy(float)
    for name, model in models.build(cfg).items():
        p = models.fit_predict(model, X[trv], y[trv], X[te], name, cfg, cfg["seed"])
        m = metrics.classification_metrics(y[te], p)
        lo, hi = metrics.bootstrap_ci(y[te], p, "f1_macro", cfg["models"]["bootstrap_ci"], cfg["seed"])
        rows.append({"feature_set": fs_name, "model": name, "n_features": len(fs),
                     "f1_macro": m["f1_macro"], "f1_ci95": f"[{lo:.4f}, {hi:.4f}]",
                     "recall_fault": m["recall_fault"], "precision_fault": m["precision_fault"],
                     "roc_auc": m["roc_auc"], "avg_precision": m["avg_precision"],
                     "test_n": m["n"], "test_positives": m["positives"]})
        print(f"[02] {fs_name:48s} {name:4s} F1={m['f1_macro']:.4f} AUC={m['roc_auc']:.4f}")
res = pd.DataFrame(rows)
save_table(res, out / "replication_random_split")
pd.Series({"train": len(tr), "val": len(va), "test": len(te), "test_positives": int(y[te].sum()),
           "engine": cfg["models"]["engine"]}).to_csv(out / "split_sizes.csv")
print(f"[02] outputs -> {out}")
