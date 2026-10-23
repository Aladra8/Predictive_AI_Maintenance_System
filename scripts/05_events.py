"""05 – Event aggregation on the real clock, and the synthetic-clock artefact.

(a) Thesis label on the real 4 Hz clock, for several merge gaps.
(b) The same label on the synthetic multi-day clock used in the thesis
    (re-implemented, seeded) – shows how 507-style counts and
    time-of-day peaks are produced by the clock, not by the data.
(c) Vibration episodes (corrected label) for several merge gaps.
"""
from _common import setup, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import data, events, labels
from pdm.utils import out_dir, save_table

cfg, _ = setup(__doc__)
out = out_dir("05_events", cfg, uses_models=False)
hz = cfg["data"]["sampling_hz"]; gaps = cfg["events"]["gap_sensitivity_s"]; rep = cfg["thesis_replication"]

k = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
t = data.thesis_frame(k, cfg["thesis_names"], cfg["data"]["vibration_nan"])
y, _, _ = labels.percentile_vote(t, rep["rule_2of4"], rep["quantile"], rep["k"])
real_t = np.arange(len(k)) / hz

a = events.gap_sensitivity(real_t, y, gaps); a.insert(0, "clock", "real 4 Hz")
syn = data.synthetic_timestamps(len(k), seed=0)
syn_t = (syn - syn.iloc[0]).dt.total_seconds().to_numpy()
b = events.gap_sensitivity(syn_t, y, gaps); b.insert(0, "clock", "synthetic (thesis)")
save_table(pd.concat([a, b]), out / "thesis_label_events_by_gap", "{:.3f}")

# time-of-day of labelled rows under each clock
real_clock = pd.Timestamp(cfg["data"]["clock_start"]) + pd.to_timedelta(real_t, unit="s")
hours = pd.DataFrame({"hour": range(24),
                      "fault_rows_real_clock": pd.Series(real_clock[y == 1].hour).value_counts().reindex(range(24), fill_value=0).values,
                      "fault_rows_synthetic_clock": syn[y == 1].dt.hour.value_counts().reindex(range(24), fill_value=0).values,
                      "all_rows_synthetic_clock": syn.dt.hour.value_counts().reindex(range(24), fill_value=0).values})
save_table(hours, out / "fault_rows_by_hour")

ep, y_ep, _ = labels.vibration_episodes(k.vibration.to_numpy(), cfg["vibration_episodes"]["quantile"], 60, hz)
exc = np.nan_to_num(k.vibration.to_numpy(), nan=-1) > np.nanquantile(k.vibration, cfg["vibration_episodes"]["quantile"])
c = events.gap_sensitivity(real_t, exc, gaps)
save_table(c, out / "vibration_exceedance_events_by_gap", "{:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(hours.hour, hours.fault_rows_synthetic_clock, color="#C0392B")
ax[0].set_title("Labelled rows by hour – synthetic clock"); ax[0].set_xlabel("hour")
ax[1].bar(hours.hour, hours.fault_rows_real_clock, color="#0B3C49")
ax[1].set_title("Labelled rows by hour – real clock (16:30 start)"); ax[1].set_xlabel("hour")
fig.tight_layout(); fig.savefig(out / "fig_clock_artefact.png", dpi=140); plt.close(fig)

print(pd.concat([a, b])[["clock", "gap_s", "n_events", "median_min", "max_min"]].to_string(index=False))
print(c[["gap_s", "n_events", "median_min"]].to_string(index=False))
print(f"[05] outputs -> {out}")
