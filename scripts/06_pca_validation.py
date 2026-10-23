"""06 – PCA structural check, with and without the duplicated columns.

The thesis PCA used 7 columns, 4 of which are copies of revolutions.  This
script reports variance, loadings and a sampled silhouette for
(a) the thesis feature set, (b) the three independent channels only,
coloured by the thesis label and by time.
"""
from _common import setup, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import data, labels
from pdm.pca import pca, silhouette
from pdm.utils import out_dir, save_table

cfg, _ = setup(__doc__)
out = out_dir("06_pca", cfg, uses_models=False)
rep = cfg["thesis_replication"]

k = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
t = data.thesis_frame(k, cfg["thesis_names"], cfg["data"]["vibration_nan"])
y, _, _ = labels.percentile_vote(t, rep["rule_2of4"], rep["quantile"], rep["k"])

sets = {"thesis PCA set (7)": ["acceleration", "speed", "vibration", "revolutions", "energy", "signal_strength", "humidity"],
        "independent channels (3)": ["revolutions", "humidity", "vibration"]}
rows, load_tabs, scores = [], [], {}
for name, cols in sets.items():
    Z, var, L = pca(t[cols], 2)
    s = silhouette(Z, y, cfg["pca"]["silhouette_sample"], cfg["seed"])
    rows.append({"feature_set": name, "pc1_var": var[0], "pc2_var": var[1], "silhouette_pc12_thesis_label": s})
    L = L.reset_index().rename(columns={"index": "feature"}); L.insert(0, "feature_set", name)
    load_tabs.append(L); scores[name] = Z
save_table(pd.DataFrame(rows), out / "pca_summary")
save_table(pd.concat(load_tabs), out / "pca_loadings", "{:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
Z = scores["thesis PCA set (7)"]; sub = slice(None, None, 10)
ax[0].scatter(Z[sub, 0], Z[sub, 1], c=np.where(y[sub] == 1, "#C0392B", "#0B3C49"), s=2)
ax[0].set_title("Thesis PCA set – coloured by thesis label")
sc = ax[1].scatter(Z[sub, 0], Z[sub, 1], c=(np.arange(len(y)) / 4 / 3600)[sub], s=2, cmap="viridis")
ax[1].set_title("Same projection – coloured by time (h)"); fig.colorbar(sc, ax=ax[1])
Z3 = scores["independent channels (3)"]
ax[2].scatter(Z3[sub, 0], Z3[sub, 1], c=np.where(y[sub] == 1, "#C0392B", "#0B3C49"), s=2)
ax[2].set_title("Independent channels only – thesis label")
for a in ax: a.set_xlabel("PC1"); a.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(out / "fig_pca.png", dpi=140); plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
print(f"[06] outputs -> {out}")
