#!/usr/bin/env python3
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# -------------------
# Utilities
# -------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def find_col(df: pd.DataFrame, aliases):
    m = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in m: return m[a.lower()]
    for a in aliases:
        for k,v in m.items():
            if a.lower() in k: return v
    return None

def build_labels_2of4(df, vib, spd, eng, rev, base_q=0.95, scale=1.0):
    cols = [vib, spd, eng, rev]
    thr = {c: df[c].quantile(base_q) * scale for c in cols}
    flags = (df[cols] > pd.Series(thr)).astype(int)
    y = (flags.sum(axis=1) >= 2).astype(int).values
    return y, thr

def select_numeric_features(df: pd.DataFrame):
    feats = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in feats if c.lower() not in ("label_fault",)]
    feats = [c for c in feats if not str(c).lower().startswith("fault_")]
    return feats

def split_70_15_15(X, y, seed=42):
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te

def build_models():
    return {
        "lr": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                             alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                             max_iter=200, random_state=42)
    }

def small_param_grid(name):
    if name == "lr":  return [{"clf__C": c} for c in [0.5, 1.0, 2.0, 5.0]]
    if name == "svm": return [{"clf__C": c, "clf__gamma": g} for c in [0.5,1.0,2.0] for g in ["scale", 0.01, 0.001]]
    if name == "rf":  return [{"clf__max_depth": d} for d in [None, 8, 12, 16]]
    if name == "mlp": return [{"clf__hidden_layer_sizes": h} for h in [(64,32), (128,64), (128,64,32)]]
    return [{}]

def evaluate(name, model, X_te, y_te):
    y_prob = model.predict_proba(X_te)[:,1] if hasattr(model, "predict_proba") else None
    y_hat  = model.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, y_hat),
        "precision_macro": precision_score(y_te, y_hat, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_hat, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, y_hat, average="macro", zero_division=0),
        "recall_fault": recall_score(y_te, y_hat, pos_label=1, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_te, y_prob)
        metrics["avg_precision"] = average_precision_score(y_te, y_prob)
    else:
        metrics["auc_roc"] = np.nan
        metrics["avg_precision"] = np.nan
    return metrics, y_hat, y_prob

def plot_confusion(name, y_te, y_hat, outdir: Path, tag):
    disp = ConfusionMatrixDisplay.from_predictions(y_te, y_hat, display_labels=["Normal","Fault"])
    plt.title(f"Confusion Matrix ({name}, {tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_matrix_{name}_{tag}.png", dpi=160); plt.close()

def plot_roc(name, y_te, y_prob, outdir: Path, tag):
    if y_prob is None: return
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=name); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name} ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"roc_{name}_{tag}.png", dpi=160); plt.close()

def plot_pr(name, y_te, y_prob, outdir: Path, tag):
    if y_prob is None: return
    pr, rc, _ = precision_recall_curve(y_te, y_prob)
    plt.figure()
    plt.plot(rc, pr, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision–Recall — {name} ({tag})")
    plt.tight_layout()
    plt.savefig(outdir / f"pr_{name}_{tag}.png", dpi=160); plt.close()

def plot_calibration(name, y_te, y_prob, outdir: Path, tag, n_bins=10):
    if y_prob is None: return
    prob_true, prob_pred = calibration_curve(y_te, y_prob, n_bins=n_bins, strategy="quantile")
    plt.figure()
    plt.plot([0,1],[0,1],"--", label="Perfect")
    plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Calibration — {name} ({tag})")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"calibration_{name}_{tag}.png", dpi=160); plt.close()

def fit_best_on_trainval(pipe, grid, X_tr, y_tr, X_va, y_va, scorer="f1_macro"):
    best_score, best_params, best_model = -1, None, None
    for params in grid:
        model = Pipeline([("scaler", StandardScaler()), ("clf", pipe["clf"])])
        model.set_params(**params)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_va)
        y_prob = model.predict_proba(X_va)[:,1] if hasattr(model, "predict_proba") else None
        if scorer == "f1_macro":
            score = f1_score(y_va, y_hat, average="macro", zero_division=0)
        else:
            score = average_precision_score(y_va, y_prob) if y_prob is not None else 0.0
        if score > best_score:
            best_score, best_params, best_model = score, params, model
    X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
    final = Pipeline([("scaler", StandardScaler()), ("clf", pipe["clf"])])
    final.set_params(**(best_params if best_params else {}))
    final.fit(X_tv, y_tv)
    return final, best_params, best_score

# ---------- Robust split helpers ----------
def to_datetime_safe(s):
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series(index=s.index, dtype="datetime64[ns]")

def chronological_indices(df: pd.DataFrame, time_col: str):
    t = to_datetime_safe(df[time_col])
    order = np.argsort(t.values)
    n = len(order); n_tr = int(0.70*n); n_va = int(0.15*n)
    idx_tr = order[:n_tr]
    idx_va = order[n_tr:n_tr+n_va]
    idx_te = order[n_tr+n_va:]
    return idx_tr, idx_va, idx_te

def label_events(df, time_col, y, gap_minutes=5.0):
    t = to_datetime_safe(df[time_col])
    order = np.argsort(t.values) if t.notna().any() else np.arange(len(df))
    y_ord = np.asarray(y)[order]; t_ord = t.values[order]
    event_id = np.full(len(df), -1, dtype=int)
    eid = -1; prev_one = False; last_time = None
    for j, i in enumerate(order):
        if y_ord[j] == 1:
            start_new = (not prev_one)
            if (not start_new) and (pd.notna(t_ord[j]) and pd.notna(last_time)):
                dt_min = (t_ord[j] - last_time) / np.timedelta64(1, "m")
                if dt_min > gap_minutes:
                    start_new = True
            if start_new: eid += 1
            event_id[i] = eid
            prev_one = True; last_time = t_ord[j]
        else:
            prev_one = False; last_time = t_ord[j]
    return event_id  # -1 for normal rows

# -------------- Core run --------------
def run_one_schema(tag_base, df: pd.DataFrame, out_root: Path, seed=42,
                   emit_tex=False, save_models=False,
                   leakage_guard=False, time_col=None, time_split=False,
                   event_split=False, event_gap_mins=5.0):

    # detect required columns for labels
    col_speed     = find_col(df, ["Speed","x2"])
    col_vibration = find_col(df, ["Vibration","x3"])
    col_revs      = find_col(df, ["Revolutions","Motor_Cycles","Ecycles","x4"])
    col_energy    = find_col(df, ["Energy","x5"])
    if any(c is None for c in [col_speed, col_vibration, col_revs, col_energy]):
        raise SystemExit(f"[{tag_base}] Missing one of required columns: Speed/Vibration/Revolutions/Energy.")

    # labels
    y, thr = build_labels_2of4(df, col_vibration, col_speed, col_energy, col_revs, base_q=0.95, scale=1.0)
    df = df.copy(); df["label_fault"] = y

    # features (+ optional leakage guard)
    feats = select_numeric_features(df)
    if leakage_guard:
        guard_cols = [col_vibration, col_speed, col_energy, col_revs]
        feats = [c for c in feats if c not in guard_cols]

    X = df[feats].fillna(df[feats].median()).values
    y = df["label_fault"].values

    # choose split strategy
    tag_suffix = []
    if time_split:
        # detect time column if not provided
        if not time_col:
            time_col = find_col(df, ["timestamp","time","datetime","date_time","date"])
        if not time_col or time_col not in df.columns:
            raise SystemExit(f"[{tag_base}] --time-split requires a valid time column; none found.")
        idx_tr, idx_va, idx_te = chronological_indices(df, time_col)
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_va], y[idx_va]
        X_te, y_te = X[idx_te], y[idx_te]
        tag_suffix.append("time")
    elif event_split:
        if not time_col:
            time_col = find_col(df, ["timestamp","time","datetime","date_time","date"])
        if not time_col or time_col not in df.columns:
            raise SystemExit(f"[{tag_base}] --event-split requires a valid time column; none found.")
        ev = label_events(df, time_col, y, gap_minutes=event_gap_mins)
        groups = np.where(ev >= 0, ev, np.arange(len(df)) + 10_000_000)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed+1)
        va_rel, te_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups=groups[temp_idx]))
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[temp_idx][va_rel], y[temp_idx][va_rel]
        X_te, y_te = X[temp_idx][te_rel], y[temp_idx][te_rel]
        tag_suffix.append("event")
    else:
        X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(X, y, seed=seed)

    if leakage_guard:
        tag_suffix.append("guard")

    # final tag and outdir
    tag = tag_base + (("_" + "_".join(tag_suffix)) if tag_suffix else "")
    outdir = out_root / tag
    ensure_dir(outdir)

    # save meta
    meta_dir = outdir / "meta"; meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "labeling_thresholds.json").write_text(json.dumps({
        "thresholds": {k: float(v) for k,v in thr.items()},
        "features_used": feats,
        "split": ("time" if time_split else "event" if event_split else "random"),
        "time_col": time_col,
        "event_gap_minutes": event_gap_mins if event_split else None,
        "leakage_guard": bool(leakage_guard),
    }, indent=2))

    # train/eval
    models = build_models()
    results = []
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        grid = small_param_grid(name)
        best, best_params, best_val = fit_best_on_trainval(pipe, grid, X_tr, y_tr, X_va, y_va, scorer="f1_macro")
        metrics, y_hat, y_prob = evaluate(name, best, X_te, y_te)

        # plots
        plot_confusion(name, y_te, y_hat, outdir, tag)
        plot_roc(name, y_te, y_prob, outdir, tag)
        plot_pr(name, y_te, y_prob, outdir, tag)
        plot_calibration(name, y_te, y_prob, outdir, tag)

        # reports
        report = classification_report(y_te, best.predict(X_te), target_names=["Normal","Fault"], output_dict=True, zero_division=0)
        pd.DataFrame(report).to_csv(outdir / f"classification_report_{name}_{tag}.csv")

        # save model
        if save_models:
            models_dir = outdir / "models"; models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, models_dir / f"{name}_{tag}.joblib")

        results.append({"model": name, "tag": tag, **metrics})

    res_df = pd.DataFrame(results).sort_values(["f1_macro","auc_roc"], ascending=False)
    res_df.to_csv(outdir / f"model_metrics_{tag}.csv", index=False)

    # TeX table for this tag
    if emit_tex:
        tex = []
        tex += [r"\begin{table}[H]", r"\centering",
                rf"\caption{{Model comparison on the \textbf{{{tag}}} schema.}}",
                rf"\label{{{{tab:model_comparison_{tag}}}}}",
                r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
                r"Model & Acc. & F1 (macro) & Prec. (macro) & Rec. (macro) & AUC & AP \\", r"\midrule"]
        name_map = {"lr":"Logistic Regression","svm":"SVM (RBF)","rf":"Random Forest","mlp":"Neural Network (MLP)"}
        for _, r in res_df.iterrows():
            tex.append(f"{name_map.get(r['model'], r['model'])} & "
                       f"{r['accuracy']:.3f} & {r['f1_macro']:.3f} & {r['precision_macro']:.3f} & "
                       f"{r['recall_macro']:.3f} & {r['auc_roc']:.4f} & {r['avg_precision']:.4f} \\\\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (outdir / f"model_comparison_{tag}.tex").write_text("\n".join(tex))

    return res_df, tag

def make_delta_table(df_proc, df_orig, proc_tag, orig_tag, out_root: Path, emit_tex=False):
    m = pd.merge(df_proc.assign(tag_key="proc"), df_orig.assign(tag_key="orig"), on="model", suffixes=("_proc","_orig"))
    m["delta_f1"] = m["f1_macro_proc"] - m["f1_macro_orig"]
    m["delta_auc"] = m["auc_roc_proc"] - m["auc_roc_orig"]
    cols = ["model","f1_macro_orig","auc_roc_orig","f1_macro_proc","auc_roc_proc","delta_f1","delta_auc"]
    m = m[cols]
    m.to_csv(out_root / f"original_vs_processed_{proc_tag}_vs_{orig_tag}.csv", index=False)

    if emit_tex:
        name_map = {"lr":"Logistic Regression","svm":"SVM (RBF)","rf":"Random Forest","mlp":"Neural Network (MLP)"}
        lines = [r"\begin{table}[H]", r"\centering",
                 rf"\caption{{Original vs processed (tags: {orig_tag} vs {proc_tag}).}}",
                 rf"\label{{tab:orig_vs_proc_{proc_tag}}}",
                 r"\small",
                 r"\begin{tabular}{lcccccc}",
                 r"\toprule",
                 r"Model & F1 (orig) & AUC (orig) & F1 (proc) & AUC (proc) & $\Delta$F1 & $\Delta$AUC \\",
                 r"\midrule"]
        for _, r in m.iterrows():
            nm = name_map.get(r['model'], r['model'])
            lines.append(f"{nm} & {r['f1_macro_orig']:.3f} & {r['auc_roc_orig']:.4f} & "
                         f"{r['f1_macro_proc']:.3f} & {r['auc_roc_proc']:.4f} & "
                         f"{r['delta_f1']:+.3f} & {r['delta_auc']:+.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (out_root / f"original_vs_processed_{proc_tag}.tex").write_text("\n".join(lines))

def interpretable_vs_nn(df_proc, proc_tag, out_root: Path, emit_tex=False):
    keep = df_proc[df_proc["model"].isin(["rf","mlp"])].copy()
    name_map = {"rf":"Random Forest","mlp":"Neural Network (MLP)"}
    keep["Model Name"] = keep["model"].map(name_map)
    keep = keep[["Model Name","accuracy","f1_macro","precision_macro","recall_macro","auc_roc","avg_precision"]]
    keep.to_csv(out_root / f"interpretable_vs_nn_{proc_tag}.csv", index=False)

    if emit_tex:
        lines = [r"\begin{table}[H]", r"\centering",
                 rf"\caption{{Interpretable vs neural model ({proc_tag}).}}",
                 rf"\label{{tab:interp_vs_nn_{proc_tag}}}",
                 r"\small",
                 r"\begin{tabular}{lcccccc}",
                 r"\toprule",
                 r"Model & Acc. & F1 & Prec. & Rec. & AUC & AP \\",
                 r"\midrule"]
        for _, r in keep.iterrows():
            lines.append(f"{r['Model Name']} & {r['accuracy']:.3f} & {r['f1_macro']:.3f} & "
                         f"{r['precision_macro']:.3f} & {r['recall_macro']:.3f} & "
                         f"{r['auc_roc']:.4f} & {r['avg_precision']:.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (out_root / f"interpretable_vs_nn_{proc_tag}.tex").write_text("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-csv", required=True, help="Path to processed Kaggle CSV")
    ap.add_argument("--original-csv", help="(Optional) Path to original Kaggle CSV (x1..x5) for orig-vs-proc table")
    ap.add_argument("--out", default="outputs/phase2", help="Output root directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-tex", action="store_true")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--leakage-guard", action="store_true")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--time-split", action="store_true")
    ap.add_argument("--event-split", action="store_true")
    ap.add_argument("--event-gap-mins", type=float, default=5.0)
    args = ap.parse_args()

    out_root = Path(args.out); ensure_dir(out_root)

    # processed
    df_proc = pd.read_csv(args.processed_csv)
    res_proc, proc_tag = run_one_schema(
        "processed", df_proc, out_root, seed=args.seed, emit_tex=args.emit_tex,
        save_models=args.save_models, leakage_guard=args.leakage_guard,
        time_col=args.time_col, time_split=args.time_split,
        event_split=args.event_split, event_gap_mins=args.event_gap_mins
    )

    # optional: original
    if args.original_csv:
        df_orig = pd.read_csv(args.original_csv)
        res_orig, orig_tag = run_one_schema(
            "original", df_orig, out_root, seed=args.seed, emit_tex=args.emit_tex,
            save_models=args.save_models, leakage_guard=args.leakage_guard,
            time_col=args.time_col, time_split=args.time_split,
            event_split=args.event_split, event_gap_mins=args.event_gap_mins
        )
        make_delta_table(res_proc, res_orig, proc_tag, orig_tag, out_root, emit_tex=args.emit_tex)

    # interpretable vs NN on processed tag
    interpretable_vs_nn(res_proc, proc_tag, out_root, emit_tex=args.emit_tex)

    print("\n=== Phase 2 done. Outputs in:", out_root, "===")
    print(f"- Tables: model_comparison_{proc_tag}.tex, interpretable_vs_nn_{proc_tag}.tex"
          + (f", original_vs_processed_{proc_tag}.tex" if args.original_csv else ""))
    print(f"- Figures: confusion_matrix_*_{proc_tag}.png, roc_*_{proc_tag}.png, pr_*_{proc_tag}.png, calibration_*_{proc_tag}.png")
    if args.original_csv:
        print(f"- Also under tag {orig_tag} for the original schema")
    if args.save_models:
        print("- Models saved under */models/*.joblib and meta/labeling_thresholds.json")

if __name__ == "__main__":
    main()
