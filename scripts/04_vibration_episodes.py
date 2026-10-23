"""04 – Corrected experiment: door-vibration episodes with time-aware evaluation.

Label     : episodes of high door vibration (vibration > 95th percentile; exceedances
            closer than 60 s on the real 4 Hz clock are merged into one episode).
Tasks     : (a) detection  – is the current row inside an episode?
            (b) early warning – will an episode start within the next H seconds?
Features  : only past information (rolling windows, current row excluded) from
            C  = independent context channels (revolutions, humidity)
            V  = vibration history
            CV = both.   x1..x5 are excluded because they are copies (see 01).
Evaluation: leave-one-block-out over contiguous time blocks with a purge gap,
            pooled out-of-fold scores, episode-level recall and lead time.
"""
from _common import setup, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import data, features, labels, metrics, models, splits
from pdm.utils import out_dir, save_json, save_table

cfg, _args = setup(__doc__)
out = out_dir("04_vibration_episodes", cfg)
ve = cfg["vibration_episodes"]; hz = cfg["data"]["sampling_hz"]

k = data.add_real_clock(data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"]),
                        hz, cfg["data"]["clock_start"])
eps, y_in, thr = labels.vibration_episodes(k.vibration.to_numpy(), ve["quantile"], ve["merge_gap_s"],
                                           hz, ve["min_duration_s"])
save_table(eps, out / "episodes", "{:.2f}")
n = len(k)

ctx = ve["context_features"]
kk = k.copy(); kk["vibration"] = kk["vibration"].ffill().bfill()
Xc = pd.concat([features.current_values(kk, ctx), features.rolling_history(kk, ctx, ve["window_s"], hz)], axis=1)
Xv = features.rolling_history(kk, ["vibration"], ve["window_s"], hz)
fsets = {"C: context only": Xc, "V: vibration history": Xv, "CV: context + vibration history": pd.concat([Xc, Xv], axis=1)}

targets = {"detection (in episode)": y_in}
for h in ve["horizons_s"]:
    targets[f"early warning (onset within {h}s)"] = labels.onset_target(eps, n, h, hz)

purge = int(max(ve["window_s"]) * hz)
names = list(ve.get("models", ["rf", "lr"]))   # override with --models
if _args.models:
    names = _args.models
if cfg["models"]["engine"] == "numpy":
    names = [m for m in names if m in ("rf", "lr")]
blocks_info = pd.DataFrame({"block": range(ve["n_blocks"]),
                            "episode_rows": splits.positives_per_block(y_in, ve["n_blocks"])})
save_table(blocks_info, out / "episode_rows_per_block")

rows, oof_store = [], {}
for tname, y in targets.items():
    for fname, X in fsets.items():
        Xa = X.to_numpy(float)
        for mname in names:
            oof = np.full(n, np.nan)
            for b, trn, tst in splits.blocked_folds(n, ve["n_blocks"], gap=purge):
                model = models.build(cfg, [mname])[mname]
                oof[tst] = models.fit_predict(model, Xa[trn], y[trn], Xa[tst], mname, cfg, cfg["seed"])
            m = metrics.classification_metrics(y, oof)
            prev = y.mean()
            r = {"task": tname, "features": fname, "model": mname, "positives": int(y.sum()),
                 "prevalence": prev, "roc_auc": m["roc_auc"], "avg_precision": m["avg_precision"],
                 "ap_lift_over_chance": m["avg_precision"] / prev if prev else np.nan,
                 "recall_fault@0.5": m["recall_fault"], "precision_fault@0.5": m["precision_fault"]}
            if tname.startswith("early"):
                h = float(tname.split("within ")[1].rstrip("s)"))
                er = metrics.event_recall(eps, oof, 0.5, hz, lookback_s=h)
                r.update({"episodes_warned": f"{er['detected']}/{er['episodes']}", "median_lead_s": er["median_lead_s"]})
            rows.append(r); oof_store[(tname, fname, mname)] = oof
            print(f"[04] {tname:34s} {fname:32s} {mname:3s} AUC={m['roc_auc']:.3f} AP={m['avg_precision']:.3f} (chance {prev:.3f})")
res = pd.DataFrame(rows)
save_table(res, out / "results_blocked_cv", "{:.3f}")
save_json({"vibration_threshold": thr, "episodes": len(eps), "episode_rows": int(y_in.sum()),
           "purge_rows": purge, "n_blocks": ve["n_blocks"], "engine": cfg["models"]["engine"]}, out / "summary.json")

# figure: timeline of episodes and an example of early-warning scores
fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
hrs = np.arange(n) / hz / 3600
ax[0].plot(hrs, k.vibration, lw=0.3, color="#7FA7AF"); ax[0].axhline(thr, color="#C0392B", lw=1, ls="--")
for _, e in eps.iterrows():
    ax[0].axvspan(e.start_s / 3600, e.end_s / 3600, color="#F2A900", alpha=0.35)
ax[0].set_ylabel("vibration"); ax[0].set_title(f"{len(eps)} vibration episodes (threshold {thr:.2f}, 60 s merge)")
key = ("early warning (onset within 60s)", "CV: context + vibration history", names[0])
if key in oof_store:
    ax[1].plot(hrs, oof_store[key], lw=0.4, color="#0B3C49")
    ax[1].set_ylabel("P(onset ≤ 60 s)"); ax[1].set_title(f"Out-of-fold early-warning score ({names[0]}, CV features)")
ax[1].set_xlabel("hours since 16:30")
fig.tight_layout(); fig.savefig(out / "fig_episodes_and_scores.png", dpi=140); plt.close(fig)
print(f"[04] outputs -> {out}")
