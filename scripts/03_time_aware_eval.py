"""03 – Can the thesis label be evaluated with a time-aware split?

Shows where the positives fall in time, what a chronological 70/15/15 split
contains, and what leave-one-block-out cross-validation produces.
"""
from _common import setup, path

import numpy as np
import pandas as pd

from pdm import data, labels, metrics, models, splits
from pdm.utils import out_dir, save_table

cfg, _ = setup(__doc__)
out = out_dir("03_time_aware", cfg)
rep = cfg["thesis_replication"]
nb = cfg["time_aware"]["n_blocks"]

k = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
t = data.thesis_frame(k, cfg["thesis_names"], cfg["data"]["vibration_nan"])
y, _, _ = labels.percentile_vote(t, rep["rule_2of4"], rep["quantile"], rep["k"])
n = len(y)

tr, va, te = splits.chronological(n, tuple(rep["split"].values()))
chrono = pd.DataFrame([{"part": p, "rows": len(i), "first_row": int(i[0]), "last_row": int(i[-1]),
                        "positives": int(y[i].sum())} for p, i in [("train", tr), ("val", va), ("test", te)]])
save_table(chrono, out / "chronological_split_content")

blocks = pd.DataFrame({"block": range(nb), "positives": splits.positives_per_block(y, nb)})
save_table(blocks, out / "positives_per_block")

X = t[rep["features"]].to_numpy(float)
rows = []
for b, trn, tst in splits.blocked_folds(n, nb):
    for name, model in models.build(cfg).items():
        p = models.fit_predict(model, X[trn], y[trn], X[tst], name, cfg, cfg["seed"])
        m = metrics.classification_metrics(y[tst], p)
        rows.append({"test_block": b, "model": name, "train_positives": int(y[trn].sum()),
                     "test_positives": m["positives"], "roc_auc": m["roc_auc"],
                     "recall_fault": m["recall_fault"] if m["positives"] else np.nan,
                     "false_alarms": m["fp"]})
save_table(pd.DataFrame(rows), out / "leave_one_block_out")
print(chrono.to_string(index=False))
print(blocks.T.to_string(header=False))
print(f"[03] outputs -> {out}")
