"""09 – Train and export the model served by deploy/service.

Trains on the semi-synthetic benchmark (the only data with ground truth) using the
same feature code as the experiments, then writes
  deploy/model/model.pkl        the fitted model + feature specification
  deploy/model/model_card.json  intended use, data, metrics, limitations
Leave-sessions-out metrics from this exact configuration are stored in the card.
"""
from _common import setup, path, ROOT

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from pdm import data, features, metrics, models, splits, synthetic
from pdm.utils import git_hash

cfg, _args = setup(__doc__)
sc, dp, hz = cfg["synthetic"], cfg["deploy"], cfg["data"]["sampling_hz"]
if cfg["models"]["engine"] == "numpy" and dp["model"] not in ("rf", "lr"):
    dp["model"] = "rf"

raw = data.load_kaggle(path(cfg, "kaggle_csv"), cfg["data"]["kaggle_md5"])
df, sess = synthetic.generate(raw, sc["n_sessions"], sc["session_hours"], hz, sc["fault_share"],
                              tuple(sc["precursor_min"]), tuple(sc["fault_min"]), sc["seed"])
ch = dp["channels"]; win = sc["window_s"]
X = pd.concat([pd.concat([features.current_values(g, ch), features.rolling_history(g, ch, win, hz)], axis=1)
               for _, g in df.groupby("session", sort=True)]).loc[df.index]
idx = np.arange(0, len(df), sc["eval_stride"])
X = X.iloc[idx].reset_index(drop=True); D = df.iloc[idx].reset_index(drop=True)
y = (D.state >= 1).to_numpy().astype(int) if dp["target"] == "precursor_or_fault" else (D.state == 2).to_numpy().astype(int)
cols = list(X.columns); Xa = X.to_numpy(float)

# honest estimate: leave-sessions-out
oof = np.full(len(y), np.nan)
for _, tr, te in splits.group_folds(D.session.to_numpy(), sc["n_folds"], list(sess.session)):
    m = models.build(cfg, [dp["model"]])[dp["model"]]
    oof[te] = models.fit_predict(m, Xa[tr], y[tr], Xa[te], dp["model"], cfg, cfg["seed"])
cv = metrics.classification_metrics(y, oof, dp["alarm_threshold"])

# final model on all sessions except the hold-out sessions (kept for the replay demo)
hold = np.isin(D.session.to_numpy(), dp.get("holdout_sessions", []))
final = models.build(cfg, [dp["model"]])[dp["model"]]
final.fit(Xa[~hold], y[~hold])
held = metrics.classification_metrics(y[hold], final.predict_proba(Xa[hold])[:, 1], dp["alarm_threshold"]) if hold.any() else {}

out = ROOT / dp["model_dir"]; out.mkdir(parents=True, exist_ok=True)
bundle = {"model": final, "feature_columns": cols, "channels": ch, "window_s": win, "hz": hz,
          "alarm_threshold": dp["alarm_threshold"], "event_gap_s": dp["event_gap_s"],
          "min_event_alarms": dp["min_event_alarms"], "engine": cfg["models"]["engine"], "model_name": dp["model"]}
with open(out / "model.pkl", "wb") as fh:
    pickle.dump(bundle, fh)
card = {
    "name": "elevator-door-pdm", "version": f"1.0.0+{git_hash()}", "created": datetime.now().isoformat(timespec="seconds"),
    "model": dp["model"], "engine": cfg["models"]["engine"], "target": dp["target"],
    "inputs": {"channels": ch, "sampling_hz": hz, "features": len(cols), "window_s": win},
    "training_data": {"source": "semi-synthetic benchmark (pdm/synthetic.py) built on the public Huawei door recording",
                      "sessions": int(sc["n_sessions"]), "seed": sc["seed"], "rows_1Hz": int(len(y)), "positive_share": float(y.mean())},
    "holdout_sessions": dp.get("holdout_sessions", []),
    "holdout_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in held.items()},
    "leave_sessions_out_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in cv.items()},
    "intended_use": "Research demonstration of a predictive-maintenance scoring service. NOT validated on real faults.",
    "limitations": ["Trained on injected (simulated) faults; real fault signatures may differ.",
                    "motor_current is a simulated channel; real deployments need a real current sensor or a model without it.",
                    "Requires 4 Hz input with the listed channels; missing values are forward-filled.",
                    "Alarms must be reviewed by a qualified technician (human in the loop); lift safety components are high-risk under the EU AI Act."],
}
(out / "model_card.json").write_text(json.dumps(card, indent=2))
print(f"[09] leave-sessions-out AUC={cv['roc_auc']:.3f} AP={cv['avg_precision']:.3f} recall@{dp['alarm_threshold']}={cv['recall_fault']:.3f}")
print(f"[09] model -> {out / 'model.pkl'}")
