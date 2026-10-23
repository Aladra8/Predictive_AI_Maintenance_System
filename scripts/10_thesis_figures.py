"""10 – Summary figures for the thesis, built only from the result tables of scripts 02–08.

Writes results/10_thesis_figures/:
  fig_e2_evaluation_regimes.png     E2: ROC-AUC per label under four evaluation regimes
  fig_e3_strength_sensitivity.png   E3: AUC and AP vs injected fault strength
  fig_e3_label_quality.png          E3: precision/recall of percentile labels vs ground truth
  fig_cross_experiment.png          E1/E2/E3: random-split vs time/session-aware AUC (RF)
"""
from _common import setup, ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm.utils import out_dir

cfg, _ = setup(__doc__)
res = ROOT / ("results" if cfg["models"]["engine"] == "sklearn" else "results_numpy")
out = out_dir("10_thesis_figures", cfg)
NAVY, TEAL, AMBER, RED, GREY = "#0B3C49", "#2E7D8C", "#F2A900", "#C0392B", "#7FA7AF"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- E2 regimes
r7 = pd.read_csv(res / "07_processed_dataset" / "reevaluation.csv")
r7 = r7[r7.model == "rf"].copy()
r7["regime"] = r7.evaluation.str.replace(r" \(.*\)", "", regex=True).str.replace("blocked CV, strict guard", "blocked + strict guard")
labels = list(dict.fromkeys(r7.label))
regimes = ["random split", "random split, shuffled file", "blocked CV", "blocked + strict guard"]
short = {labels[0]: "2-of-4 (label_fault)", labels[1]: "tuned 2-of-3", labels[2]: "vibration > 95th"}
fig, ax = plt.subplots(figsize=(8, 4.0))
w = 0.2
for j, rg in enumerate(regimes):
    vals = [r7[(r7.label == l) & (r7.regime == rg)].roc_auc.iloc[0] for l in labels]
    ax.bar(np.arange(len(labels)) + (j - 1.5) * w, vals, w, label=rg, color=[NAVY, TEAL, AMBER, RED][j])
ax.axhline(0.5, ls="--", color=GREY, lw=1); ax.text(2.45, 0.52, "chance", color=GREY, fontsize=8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels([short[l] for l in labels])
ax.set_ylabel("ROC-AUC (Random Forest)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.set_title("Experiment 2: the same labels under four evaluation regimes")
fig.tight_layout(); fig.savefig(out / "fig_e2_evaluation_regimes.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- E3 strength
h = pd.read_csv(res / "08_synthetic_benchmark" / "H_fault_strength_sensitivity.csv")
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for m, c in [("rf", NAVY), ("lr", AMBER)]:
    d = h[h.model == m]
    ax[0].plot(d.fault_strength, d.auc, marker="o", color=c, label=m.upper())
    ax[1].plot(d.fault_strength, d.ap, marker="o", color=c, label=m.upper())
for a, t in zip(ax, ["ROC-AUC", "Average precision"]):
    a.set_xlabel("injected fault strength (× default)"); a.set_ylabel(t); a.set_ylim(0, 1.02); a.legend(frameon=False)
fig.suptitle("Experiment 3: detectability vs fault strength (leave-sessions-out)")
fig.tight_layout(); fig.savefig(out / "fig_e3_strength_sensitivity.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- E3 label quality
a = pd.read_csv(res / "08_synthetic_benchmark" / "A_label_quality.csv")
a = a[a.compared_with == "true fault"]
names = ["thesis-style 2-of-3", "vibration > 95th", "physics-informed 2-of-3"]
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(a))
ax.bar(x - 0.2, a.precision, 0.4, label="precision", color=NAVY)
ax.bar(x + 0.2, a.recall, 0.4, label="recall", color=AMBER)
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.legend(frameon=False)
ax.set_title("Experiment 3: percentile labels compared with the true fault state")
fig.tight_layout(); fig.savefig(out / "fig_e3_label_quality.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- cross experiment
r2 = pd.read_csv(res / "02_replicate_thesis" / "replication_random_split.csv")
e1_random = r2[(r2.model == "rf") & r2.feature_set.str.startswith("standard")].roc_auc.iloc[0]
r4 = pd.read_csv(res / "04_vibration_episodes" / "results_blocked_cv.csv")
e1_time = r4[(r4.model == "rf") & (r4.task.str.startswith("detection")) & (r4.features.str.startswith("C:"))].roc_auc.iloc[0]
e2_random = r7[(r7.label == labels[0]) & (r7.regime == "random split")].roc_auc.iloc[0]
e2_time = r7[(r7.label == labels[0]) & (r7.regime == "blocked + strict guard")].roc_auc.iloc[0]
g = pd.read_csv(res / "08_synthetic_benchmark" / "G_random_vs_session_split.csv")
e3_random = g[(g.model == "rf") & g.split.str.startswith("random")].auc.iloc[0]
e3_time = g[(g.model == "rf") & g.split.str.startswith("leave")].auc.iloc[0]
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(3)
ax.bar(x - 0.2, [e1_random, e2_random, e3_random], 0.4, label="random row split", color=GREY)
ax.bar(x + 0.2, [e1_time, e2_time, e3_time], 0.4, label="time / session-aware, no leaking channels", color=NAVY)
for i, v in enumerate([e1_time, e2_time, e3_time]):
    ax.text(i + 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, ls="--", color=GREY, lw=1)
ax.set_xticks(x); ax.set_xticklabels(["E1 original\n(random: thesis label;\naware: vibration episodes)", "E2 processed\n(2-of-4, strict guard)", "E3 semi-synthetic\n(precursor or fault)"])
ax.set_ylabel("ROC-AUC (Random Forest)"); ax.set_ylim(0, 1.1); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
ax.set_title("Perfect random-split scores survive only where faults exist")
fig.tight_layout(); fig.savefig(out / "fig_cross_experiment.png", dpi=200); plt.close(fig)
print(f"[10] outputs -> {out}")
