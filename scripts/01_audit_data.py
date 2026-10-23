"""01 – Data provenance and label audit.

Answers: What is in the public dataset?  How do x1..x5 relate to the original
channels?  What does the thesis label actually select?  Is the signal interpolated?
"""
from _common import setup, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import audit, data, labels
from pdm.utils import out_dir, save_json, save_table

cfg, _ = setup(__doc__)
out = out_dir("01_audit", cfg, uses_models=False)
hz = cfg["data"]["sampling_hz"]

k = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
prov = {"kaggle_md5": data.md5(path(cfg, "kaggle_csv")), "rows": len(k), "columns": list(k.columns),
        "vibration_missing": int(k.vibration.isna().sum()),
        "recording_hours_at_4Hz": len(k) / hz / 3600}
zp = path(cfg, "zenodo_csv")
if zp.exists():
    z = data.load_zenodo(zp)
    prov["zenodo_md5"] = data.md5(zp)
    prov["zenodo_columns"] = list(z.columns)
    prov["zenodo_corrupted_or_missing_cells"] = z.attrs["corrupted_cells"]
    for c in ["revolutions", "humidity", "vibration"]:
        d = np.abs(z[c].to_numpy() - k[c].to_numpy())
        prov[f"kaggle_vs_zenodo_{c}_max_abs_diff"] = float(np.nanmax(d))
save_json(prov, out / "dataset_provenance.json")

# 1. anonymised columns are transforms of the original channels
der = audit.derived_columns(k)
save_table(der, out / "derived_columns", "{:.6f}")

# 2. what the thesis label selects
t = data.thesis_frame(k, cfg["thesis_names"], cfg["data"]["vibration_nan"])
rep = cfg["thesis_replication"]
y3, thr3, f3 = labels.percentile_vote(t, rep["rule_2of3"], rep["quantile"], rep["k"])
y4, thr4, f4 = labels.percentile_vote(t, rep["rule_2of4"], rep["quantile"], rep["k"])
struct = {"2of3": {**audit.label_structure(y3, hz), "thresholds": thr3},
          "2of4": {**audit.label_structure(y4, hz), "thresholds": thr4},
          "rules_identical": bool((y3 == y4).all())}
save_json(struct, out / "label_structure.json")
ov, co = audit.indicator_overlap(f4)
save_table(ov, out / "indicator_exceedances")
co.to_csv(out / "indicator_cooccurrence.csv")

# 3. single-feature separability, including the row index (= time)
t["row_index"] = np.arange(len(t))
sfa = audit.single_feature_auc(t, y4, rep["features"] + ["row_index"])
save_table(sfa, out / "single_feature_auc")

# 4. interpolation and regularity
lin = pd.DataFrame([{"channel": c, "share_locally_linear": audit.linear_segment_share(k[c].to_numpy())}
                    for c in ["revolutions", "humidity", "vibration"]])
save_table(lin, out / "interpolation_check", "{:.3f}")
reg = pd.DataFrame([audit.vibration_regularity(k.vibration.to_numpy(), s, e)
                    for s, e in [(19150, 19950), (20750, 21000), (90000, 92000)]])
save_table(reg, out / "vibration_spike_regularity", "{:.3f}")
save_table(k.drop(columns="ID").corr().round(3).reset_index().rename(columns={"index": "channel"}),
           out / "correlation_matrix", "{:.3f}")

# figures
cols = ["revolutions", "humidity", "vibration", "x1", "x2", "x3", "x4", "x5"]
fig, ax = plt.subplots(len(cols), 1, figsize=(12, 13), sharex=True)
hours = np.arange(len(k)) / hz / 3600
for a, c in zip(ax, cols):
    a.plot(hours, k[c], lw=0.5, color="#0B3C49")
    a.axvspan(0, y4.sum() / hz / 3600, color="#C0392B", alpha=0.18)
    a.set_ylabel(c)
ax[0].set_title("Raw channels on the real 4 Hz clock (red band = rows labelled 'fault' by the thesis rule)")
ax[-1].set_xlabel("hours since start of recording (16:30)")
fig.tight_layout(); fig.savefig(out / "fig_channels_with_label.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(1, 5, figsize=(16, 3.2))
src = {"revolutions": k.revolutions, "revolutions^2": k.revolutions ** 2, "humidity": k.humidity}
for a, (_, r) in zip(ax, der.iterrows()):
    a.scatter(src[r.explained_by][::50], k[r.column][::50], s=2, color="#2E7D8C")
    a.set_xlabel(r.explained_by); a.set_ylabel(r.column); a.set_title(f"R² = {r.r2:.4f}")
fig.suptitle("Anonymised Kaggle columns are transforms of the original Huawei channels")
fig.tight_layout(); fig.savefig(out / "fig_derived_columns.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 3))
s, e = 19150, 19950
ax.plot(np.arange(s, e), k.vibration.iloc[s:e], lw=0.7, marker=".", ms=2, color="#0B3C49")
ax.set_title("Vibration spikes: one every 8 rows (2 s), amplitude falling by exactly 1 per spike")
ax.set_xlabel("row"); fig.tight_layout(); fig.savefig(out / "fig_vibration_regularity.png", dpi=140); plt.close(fig)

print(f"[01] provenance: {prov['rows']} rows, vibration NaN {prov['vibration_missing']}")
print(der[["column", "explained_by", "r2"]].to_string(index=False))
print(f"[01] 2of3 == 2of4: {struct['rules_identical']}; positives {struct['2of4']['positives']} "
      f"rows {struct['2of4']['first_row']}-{struct['2of4']['last_row']} in {struct['2of4']['n_contiguous_runs']} run(s)")
print(sfa.head(6).to_string(index=False))
print(f"[01] outputs -> {out}")
