"""08 – Experiment 3: semi-synthetic benchmark with injected faults (ground truth known).

Questions answered with ground truth:
  A. How good are analytical (percentile-vote) labels compared with the true fault state?
  B. Can models detect the injected faults in UNSEEN sessions (leave-sessions-out)?
  C. If models are trained on analytical labels, do they still find the true faults?
  D. Leakage-Guard: does detection survive without the vibration channel?
  E. Early warning: are faults flagged during the precursor phase, and how early?
  F. Events: alarm events vs true fault episodes; false alarms per healthy hour.
  G. Leakage illustration: random row split vs leave-sessions-out.
"""
from _common import setup, path, ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import data, events, features, labels, metrics, models, splits, synthetic
from pdm.utils import out_dir, save_json, save_table

cfg, _args = setup(__doc__)
sc = cfg["synthetic"]; hz = cfg["data"]["sampling_hz"]
out = out_dir("08_synthetic_benchmark", cfg)
names = _args.models or sc["models"]
if cfg["models"]["engine"] == "numpy":
    names = [m for m in names if m in ("rf", "lr")]

# ---------------------------------------------------------------- generate (deterministic)
raw = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
df, sess = synthetic.generate(raw, sc["n_sessions"], sc["session_hours"], hz, sc["fault_share"],
                              tuple(sc["precursor_min"]), tuple(sc["fault_min"]), sc["seed"])
syn_dir = ROOT / "data" / "synthetic"; syn_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(syn_dir / "semisynthetic_benchmark.csv.gz", index=False, compression="gzip")
save_table(sess, out / "sessions", "{:.1f}")

# ---------------------------------------------------------------- features (per session, past only)
ch = sc["channels"]
parts = []
for s, g in df.groupby("session", sort=True):
    X = pd.concat([features.current_values(g, ch), features.rolling_history(g, ch, sc["window_s"], hz)], axis=1)
    parts.append(X)
Xall = pd.concat(parts).loc[df.index]
stride = sc["eval_stride"]
idx = np.arange(0, len(df), stride)               # evaluate at 1 Hz
D = df.iloc[idx].reset_index(drop=True); X = Xall.iloc[idx].reset_index(drop=True)
y_fault = (D.state == 2).to_numpy().astype(int)
y_any = (D.state >= 1).to_numpy().astype(int)
groups = D.session.to_numpy()
vib_cols = [c for c in X.columns if c.startswith("vibration")]
fsets = {"all channels": list(X.columns), "guard: no vibration": [c for c in X.columns if c not in vib_cols],
         "vibration only": vib_cols}

# ---------------------------------------------------------------- A. analytical labels vs truth
rules = {"thesis-style 2-of-3 (vibration, revolutions, humidity)": ["vibration", "revolutions", "humidity"],
         "vibration > 95th pct": ["vibration"],
         "physics-informed 2-of-3 (vibration, motor_current, humidity)": ["vibration", "motor_current", "humidity"]}
A, weak = [], {}
for rname, cols in rules.items():
    k = 1 if len(cols) == 1 else 2
    yr, thr, _ = labels.percentile_vote(D, cols, 0.95, k)
    weak[rname] = yr
    for tname, yt in [("true fault", y_fault), ("true precursor or fault", y_any)]:
        tp = int(((yr == 1) & (yt == 1)).sum()); fp = int(((yr == 1) & (yt == 0)).sum()); fn = int(((yr == 0) & (yt == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        A.append({"rule": rname, "compared_with": tname, "labelled_share": yr.mean(), "precision": p, "recall": r,
                  "f1": 2 * p * r / (p + r) if p + r else 0})
save_table(pd.DataFrame(A), out / "A_label_quality", "{:.3f}")

# ---------------------------------------------------------------- helpers
order = list(sess.session)                                  # round-robin sessions -> folds


def oof_scores(Xm, y, mname, folds="session"):
    s = np.full(len(y), np.nan)
    it = splits.group_folds(groups, sc["n_folds"], order) if folds == "session" else None
    if folds == "random":
        tr, va, te = splits.random_stratified(y, (0.7, 0.15, 0.15), cfg["seed"])
        m = models.build(cfg, [mname])[mname]
        p = models.fit_predict(m, Xm[np.r_[tr, va]], y[np.r_[tr, va]], Xm[te], mname, cfg, cfg["seed"])
        return te, p
    for _, tr, te in it:
        m = models.build(cfg, [mname])[mname]
        s[te] = models.fit_predict(m, Xm[tr], y[tr], Xm[te], mname, cfg, cfg["seed"])
    return np.arange(len(y)), s


def episode_table(state_col):
    rows = []
    for s, g in D.groupby("session"):
        on = g.index[g.state == 2]
        if len(on):
            pre = g.index[g.state == 1]
            rows.append({"session": s, "start_row": int(on[0]), "end_row": int(on[-1]),
                         "pre_start_row": int(pre[0]) if len(pre) else int(on[0])})
    return pd.DataFrame(rows)


faults = episode_table("state")
eval_hz = hz / stride
healthy_hours = (y_any == 0).sum() / eval_hz / 3600

# ---------------------------------------------------------------- B, C, D, E, F
B, scores = [], {}
for mname in names:
    for fname, cols in fsets.items():
        Xm = X[cols].to_numpy(float)
        for target_name, ytrain in [("ground truth: fault", y_fault), ("ground truth: precursor or fault", y_any),
                                    *[(f"weak label: {r}", weak[r]) for r in rules if fname == "all channels"]]:
            te, p = oof_scores(Xm, ytrain, mname)
            m_f = metrics.classification_metrics(y_fault, p)              # always scored against the TRUE fault
            m_a = metrics.classification_metrics(y_any, p)
            # events: alarms -> gap-merge -> compare with true episodes
            alarms = p >= sc["alarm_threshold"]
            t_s = np.arange(len(p)) / eval_hz
            ev = []
            for s in np.unique(groups):
                ii = np.flatnonzero(groups == s)
                e = events.aggregate(t_s[ii], alarms[ii], sc["event_gap_s"])
                for _, r in e.iterrows():
                    a = int(ii[0] + round((r.start_s - t_s[ii[0]]) * eval_hz)); b = int(ii[0] + round((r.end_s - t_s[ii[0]]) * eval_hz))
                    ev.append((a, b))
            true_hit = sum(1 for _, f in faults.iterrows() if any(a <= f.end_row and b >= f.pre_start_row for a, b in ev))
            false_ev = sum(1 for a, b in ev if y_any[a:b + 1].sum() == 0)
            warned, leads = 0, []
            for _, f in faults.iterrows():
                w = np.flatnonzero(alarms[f.pre_start_row:f.start_row])
                if w.size:
                    warned += 1; leads.append((f.start_row - (f.pre_start_row + w[0])) / eval_hz / 60)
            B.append({"model": mname, "features": fname, "trained_on": target_name,
                      "auc_vs_fault": m_f["roc_auc"], "ap_vs_fault": m_f["avg_precision"],
                      "auc_vs_precursor_or_fault": m_a["roc_auc"], "ap_vs_precursor_or_fault": m_a["avg_precision"],
                      "faults_detected": f"{true_hit}/{len(faults)}", "faults_warned_before_onset": f"{warned}/{len(faults)}",
                      "median_lead_min": float(np.median(leads)) if leads else np.nan,
                      "false_alarm_events_per_healthy_hour": false_ev / healthy_hours})
            scores[(mname, fname, target_name)] = p
            print(f"[08] {mname:3s} {fname:22s} {target_name:70s} AUC(fault)={m_f['roc_auc']:.3f} "
                  f"AP={m_f['avg_precision']:.3f} warned={warned}/{len(faults)}")
save_table(pd.DataFrame(B), out / "BCDEF_detection_warning_events", "{:.3f}")

# ---------------------------------------------------------------- G. random vs session split
G = []
for mname in names:
    Xm = X[fsets["all channels"]].to_numpy(float)
    te, p = oof_scores(Xm, y_any, mname, folds="random")
    G.append({"model": mname, "split": "random rows (leaky)", "auc": metrics.roc_auc(y_any[te], p),
              "ap": metrics.average_precision(y_any[te], p)})
    p2 = scores[(mname, "all channels", "ground truth: precursor or fault")]
    G.append({"model": mname, "split": "leave-sessions-out", "auc": metrics.roc_auc(y_any, p2),
              "ap": metrics.average_precision(y_any, p2)})
save_table(pd.DataFrame(G), out / "G_random_vs_session_split", "{:.3f}")
save_json({"rows_4Hz": len(df), "rows_evaluated_1Hz": len(D), "sessions": int(sc["n_sessions"]),
           "faulty_sessions": int(len(faults)), "healthy_hours": healthy_hours,
           "fault_types": sess.fault_type.value_counts().to_dict(), "engine": cfg["models"]["engine"]}, out / "summary.json")

# ---------------------------------------------------------------- H. detectability vs fault strength
H = []
for st in sc["strength_sensitivity"]:
    d2, s2 = synthetic.generate(raw, sc["n_sessions"], sc["session_hours"], hz, sc["fault_share"],
                                tuple(sc["precursor_min"]), tuple(sc["fault_min"]), sc["seed"], strength=st)
    X2 = pd.concat([pd.concat([features.current_values(g, ch), features.rolling_history(g, ch, sc["window_s"], hz)], axis=1)
                    for _, g in d2.groupby("session", sort=True)]).loc[d2.index].iloc[idx].reset_index(drop=True)
    D2 = d2.iloc[idx].reset_index(drop=True); ya = (D2.state >= 1).to_numpy().astype(int)
    for mname in names:
        sc_ = np.full(len(ya), np.nan)
        for _, tr, te in splits.group_folds(D2.session.to_numpy(), sc["n_folds"], order):
            m = models.build(cfg, [mname])[mname]
            sc_[te] = models.fit_predict(m, X2.to_numpy(float)[tr], ya[tr], X2.to_numpy(float)[te], mname, cfg, cfg["seed"])
        r = metrics.classification_metrics(ya, sc_)
        H.append({"fault_strength": st, "model": mname, "auc": r["roc_auc"], "ap": r["avg_precision"],
                  "recall@0.5": r["recall_fault"], "precision@0.5": r["precision_fault"]})
        print(f"[08] strength {st:4.2f} {mname:3s} AUC={r['roc_auc']:.3f} AP={r['avg_precision']:.3f}")
save_table(pd.DataFrame(H), out / "H_fault_strength_sensitivity", "{:.3f}")

# ---------------------------------------------------------------- figure: one faulty session
f0 = faults.iloc[0]; s0 = f0.session
g = D[D.session == s0]; key = (names[0], "all channels", "ground truth: precursor or fault")
fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
tm = g.t_s / 60
for a, c in zip(ax[:3], ["vibration", "motor_current", "humidity"]):
    a.plot(tm, g[c], lw=0.5, color="#0B3C49"); a.set_ylabel(c)
ax[3].plot(tm, scores[key][g.index], lw=0.7, color="#C0392B"); ax[3].set_ylabel("P(precursor/fault)")
ax[3].axhline(sc["alarm_threshold"], ls="--", color="#4A5A60", lw=0.8)
for a in ax:
    a.axvspan(g.t_s[g.state == 1].min() / 60, g.t_s[g.state == 1].max() / 60, color="#F2A900", alpha=0.25)
    a.axvspan(g.t_s[g.state == 2].min() / 60, g.t_s[g.state == 2].max() / 60, color="#C0392B", alpha=0.18)
ax[0].set_title(f"Session {s0} ({sess.loc[sess.session == s0, 'fault_type'].iloc[0]}): amber = precursor, red = fault; "
                f"out-of-session score ({names[0]})")
ax[3].set_xlabel("minutes in session")
fig.tight_layout(); fig.savefig(out / "fig_example_session.png", dpi=140); plt.close(fig)
print(f"[08] outputs -> {out}")
