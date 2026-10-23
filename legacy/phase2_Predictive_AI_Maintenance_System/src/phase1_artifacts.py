#!/usr/bin/env python3

# Generates Phase-1 artifacts on the processed Kaggle schema:
# - PCA variance & silhouette (with/without Humidity)
# - Threshold grid (±20% around 95th-percentile cut)
# - RandomForest impurity & permutation importance
# - Partial dependence plots (top features)
# - Short decision-tree rules
# Optionally emits LaTeX tables from the CSV summaries.
#
# Usage:
# python phase1_artifacts.py --csv processed_large_dataset_v4.csv --out ./phase1_outputs --emit-tex

#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

# -----------------------------
# Utils
# -----------------------------

def find_col(df: pd.DataFrame, aliases):
    m = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in m:
            return m[a.lower()]
    # fallback: substring search
    for a in aliases:
        for k, v in m.items():
            if a.lower() in k:
                return v
    return None

def safe_silhouette(Z, labels):
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return np.nan
    counts = np.bincount(labels)
    if counts.min() < 10:
        return np.nan
    try:
        return float(silhouette_score(Z, labels, metric="euclidean"))
    except Exception:
        return np.nan

def standardize(X: pd.DataFrame):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Core
# -----------------------------

def build_labels_2of4(df, vib, spd, eng, rev, base_q=0.95, scale=1.0):
    """Return labels (0/1) and threshold dict for the 2-of-4 rule."""
    cols = [vib, spd, eng, rev]
    thr = {c: df[c].quantile(base_q) * scale for c in cols}
    flags = (df[cols] > pd.Series(thr)).astype(int)
    y = (flags.sum(axis=1) >= 2).astype(int).values
    return y, thr

def pca_summary_table(df, features_for_pca, y, dropna=True):
    X = df[features_for_pca].copy()
    if dropna:
        X = X.dropna()
    y_ = df.loc[X.index, :]["label_fault"].astype(int).values
    Xs, _ = standardize(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_
    sil = safe_silhouette(Z, y_)
    return dict(pc1_var=var[0], pc2_var=var[1],
                pc12_var_sum=float(var[:2].sum()),
                silhouette_pc12=sil,
                n_samples=int(X.shape[0]))

def threshold_grid(df, key_cols, base_thr, pca_Z, pca_index, factors, seed=42):
    """Compute fault rate, silhouette on existing PCA embedding, and RF fault recall per factor."""
    print("[4/7] Threshold grid ...", flush=True)
    results = []
    # Align to PCA index
    df_pca = df.loc[pca_index].copy()

    # subset for RF speed (keep class balance)
    try:
        from sklearn.utils import resample
        df_sub = resample(df, n_samples=min(5000, len(df)), replace=False,
                          random_state=seed, stratify=df["label_fault"])
    except Exception:
        df_sub = df.sample(n=min(5000, len(df)), random_state=seed)

    num_cols = df_sub.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude any label/leaky columns
    exclude = {"label_fault"}
    exclude.update([c for c in df_sub.columns if str(c).lower().startswith("fault_")])
    features = [c for c in num_cols if c not in exclude]

    for f in factors:
        thr_f = {c: base_thr[c] * f for c in key_cols}
        # fault rate on subset
        y_sub = ((df_sub[key_cols] > pd.Series(thr_f)).astype(int).sum(axis=1) >= 2).astype(int).values
        fault_rate = float(y_sub.mean())

        # silhouette on PCA (align labels to PCA index)
        y_pca = ((df_pca[key_cols] > pd.Series(thr_f)).astype(int).sum(axis=1) >= 2).astype(int).values
        sil = safe_silhouette(pca_Z, y_pca)

        # RF recall on stratified split
        Xtr, Xte, ytr, yte = train_test_split(
            df_sub[features].fillna(df_sub[features].median()),
            y_sub, test_size=0.3, stratify=y_sub, random_state=seed
        )
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced")
        rf.fit(Xtr, ytr)
        ypred = rf.predict(Xte)
        rec = recall_score(yte, ypred, pos_label=1, zero_division=0)

        results.append(dict(factor=f, fault_rate=fault_rate, silhouette_pc12=sil, rf_fault_recall=float(rec)))

    return pd.DataFrame(results)

def rf_diagnostics(rf, Xte, yte, outdir: Path, tag="rf"):
    """Save confusion matrix + ROC/AUC for the RF baseline."""
    yhat = rf.predict(Xte)
    cm = confusion_matrix(yte, yhat)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({tag})")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Normal", "Fault"]); plt.yticks(tick_marks, ["Normal", "Fault"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(outdir / f"confusion_matrix_{tag}.png", dpi=160); plt.close(fig)

    yprob = rf.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, yprob)
    fig = plt.figure()
    plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({tag}) — AUC={auc(fpr, tpr):.4f}")
    plt.tight_layout()
    plt.savefig(outdir / f"roc_{tag}.png", dpi=160); plt.close(fig)

def manual_pdp_1d(model, bg_X: pd.DataFrame, feat: str, path: Path,
                  q_lo=0.01, q_hi=0.99, n_grid=20):
    """Manual 1D PDP robust to outliers by restricting to [q_lo, q_hi]."""
    lo, hi = bg_X[feat].quantile(q_lo), bg_X[feat].quantile(q_hi)
    grid = np.linspace(lo, hi, n_grid)
    preds = []
    # copy once outside loop for speed
    base = bg_X.copy()
    for val in grid:
        tmp = base.copy()
        tmp[feat] = val
        preds.append(model.predict_proba(tmp)[:, 1].mean())
    fig = plt.figure()
    plt.plot(grid, preds, marker='o')
    plt.xlabel(feat); plt.ylabel("Predicted fault probability")
    plt.title(f"Partial Dependence: {feat} ({int(q_lo*100)}–{int(q_hi*100)}% range)")
    plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close(fig)

def rf_interpretability(df, seed=42, perm_repeats=10, pdp_topk=4, pdp_bg=2500, outdir=Path(".")):
    """Train RF and export impurity & permutation importances, PDPs, and shallow rules."""
    print("[5/7] Training RF + importances/PDPs/rules ...", flush=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"label_fault"}
    exclude.update([c for c in df.columns if str(c).lower().startswith("fault_")])
    features = [c for c in num_cols if c not in exclude]

    X = df[features].fillna(df[features].median())
    y = df["label_fault"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    rf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)

    # Impurity importance
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    imp.to_csv(outdir / "rf_feature_importances_impurity.csv", header=["impurity_importance"])

    # Permutation importance — positive-class aware
    pi = permutation_importance(
        rf, Xte, yte,
        n_repeats=perm_repeats,
        random_state=seed,
        n_jobs=-1,
        scoring="average_precision"
    )
    pi_series = pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
    pi_series.to_csv(outdir / "rf_permutation_importance.csv", header=["permutation_importance_mean"])

    # PDPs (manual, robust to outliers; top-k by AP-based permutation)
    top = list(pi_series.head(pdp_topk).index)
    bg = Xte.sample(n=min(pdp_bg, Xte.shape[0]), random_state=seed)
    for feat in top:
        manual_pdp_1d(rf, bg, feat, outdir / f"pdp_{feat.replace(' ', '_')}.png", q_lo=0.01, q_hi=0.99, n_grid=20)

    # Shallow rules (depth 3)
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)
    dt.fit(Xtr, ytr)
    rules_text = export_text(dt, feature_names=list(Xtr.columns))
    with open(outdir / "decision_tree_rules.txt", "w") as f:
        f.write(rules_text)

    # Diagnostics
    rf_diagnostics(rf, Xte, yte, outdir, tag="rf_processed")

    return imp, pi_series, top

def emit_tex_from_csv(pca_csv: Path, grid_csv: Path, outdir: Path):
    """Create small TeX tables so you can \input{} them in Chapter 2."""
    if pca_csv.exists():
        df = pd.read_csv(pca_csv)
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{PCA variance and silhouette in PC1--PC2 (processed schema).}",
            r"\label{tab:pca_quant_checks}",
            r"\footnotesize",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Setting} & \textbf{PC1 var} & \textbf{PC2 var} & \textbf{PC1+2 var} & \textbf{Silhouette} & \textbf{Samples} \\",
            r"\midrule",
        ]
        for _, r in df.iterrows():
            lines.append(f"{r['setting']} & {r['pc1_var']:.3f} & {r['pc2_var']:.3f} & {r['pc12_var_sum']:.3f} & {r['silhouette_pc12']:.3f} & {int(r['n_samples'])} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (outdir / "pca_quant_table.tex").write_text("\n".join(lines))

    if grid_csv.exists():
        df = pd.read_csv(grid_csv)
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Threshold stress test: factor $\times$ 95th percentile for the 2-of-4 rule.}",
            r"\label{tab:threshold_grid}",
            r"\footnotesize",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\textbf{Factor} & \textbf{Fault rate} & \textbf{Silhouette} & \textbf{RF fault recall} \\",
            r"\midrule",
        ]
        for _, r in df.iterrows():
            lines.append(f"{r['factor']:.2f} & {r['fault_rate']:.3f} & {r['silhouette_pc12']:.3f} & {r['rf_fault_recall']:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (outdir / "threshold_grid_table.tex").write_text("\n".join(lines))

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to processed Kaggle CSV")
    ap.add_argument("--out", default="./phase1_outputs", help="Output directory (figures + csv + tex)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perm-repeats", type=int, default=10, help="Permutation importance repeats")
    ap.add_argument("--pdp-topk", type=int, default=4, help="Number of PDP figures to save")
    ap.add_argument("--pdp-bg", type=int, default=2500, help="Background rows for PDP")
    ap.add_argument("--sample", type=int, default=0, help="Optional downsample N for speed (0=use all)")
    ap.add_argument("--emit-tex", action="store_true", help="Emit TeX tables from CSV summaries")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)

    print("[1/7] Loading CSV ...", flush=True)
    df = pd.read_csv(args.csv)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed)

    # Detect columns
    col_speed     = find_col(df, ["Speed","x2"])
    col_vibration = find_col(df, ["Vibration","x3"])
    col_revs      = find_col(df, ["Revolutions","Motor_Cycles","x4"])
    col_energy    = find_col(df, ["Energy","x5"])
    col_signal    = find_col(df, ["Signal_Strength","Signal"])
    col_humidity  = find_col(df, ["Humidity"])
    col_accel     = find_col(df, ["Acceleration","x1"])

    missing = [n for n,c in {"Speed":col_speed,"Vibration":col_vibration,"Revolutions/Ecycles":col_revs,"Energy":col_energy}.items() if c is None]
    if missing:
        print("ERROR: missing required columns:", missing); sys.exit(1)

    # Build labels at base scale=1.0
    print("[2/7] Labeling with 2-of-4 @ 95th percentile ...", flush=True)
    y, thr0 = build_labels_2of4(df, col_vibration, col_speed, col_energy, col_revs, base_q=0.95, scale=1.0)
    df["label_fault"] = y
    print("    Label distribution:", pd.Series(y).value_counts(normalize=True).round(3).to_dict(), flush=True)

    # PCA with humidity
    print("[3/7] PCA (with/without humidity) ...", flush=True)
    pca_feats_with = [c for c in [col_accel, col_speed, col_vibration, col_revs, col_energy, col_signal, col_humidity] if c is not None]
    Xw = df[pca_feats_with].dropna()
    yw = df.loc[Xw.index, "label_fault"].astype(int).values
    Xw_std, _ = standardize(Xw)
    pca = PCA(n_components=2, random_state=args.seed)
    Zw = pca.fit_transform(Xw_std)
    var_w = pca.explained_variance_ratio_
    sil_w = safe_silhouette(Zw, yw)

    # PCA without humidity
    pca_feats_wo = [c for c in pca_feats_with if c != col_humidity]
    Xo = df[pca_feats_wo].dropna()
    yo = df.loc[Xo.index, "label_fault"].astype(int).values
    Xo_std, _ = standardize(Xo)
    p2 = PCA(n_components=2, random_state=args.seed)
    Zo = p2.fit_transform(Xo_std)
    var_o = p2.explained_variance_ratio_
    sil_o = safe_silhouette(Zo, yo)

    pca_csv = outdir / "pca_variance_silhouette.csv"
    pca_df = pd.DataFrame({
        "setting": ["with_humidity","without_humidity"],
        "pc1_var": [var_w[0], var_o[0]],
        "pc2_var": [var_w[1], var_o[1]],
        "pc12_var_sum": [var_w[:2].sum(), var_o[:2].sum()],
        "silhouette_pc12": [sil_w, sil_o],
        "n_samples": [len(Xw), len(Xo)]
    })
    pca_df.to_csv(pca_csv, index=False)

    # Threshold grid around base thresholds using PCA-with-humidity embedding
    key_cols = [col_vibration, col_speed, col_energy, col_revs]
    grid = threshold_grid(df.copy(), key_cols, thr0, Zw, Xw.index, factors=[0.8,0.9,1.0,1.1,1.2], seed=args.seed)
    grid_csv = outdir / "threshold_grid_results.csv"
    grid.to_csv(grid_csv, index=False)

    # RF interpretability on rows used in PCA-with-humidity (keeps indices aligned/clean)
    df_rf = df.loc[Xw.index].copy()
    imp, pi_series, top = rf_interpretability(
        df_rf, seed=args.seed, perm_repeats=args.perm_repeats,
        pdp_topk=args.pdp_topk, pdp_bg=args.pdp_bg, outdir=outdir
    )

    # Optionally write TeX tables
    if args.emit_tex:
        print("[6/7] Emitting TeX tables ...", flush=True)
        emit_tex_from_csv(pca_csv, grid_csv, outdir)

    print("[7/7] Done. Outputs in:", outdir)
    print("- pca_variance_silhouette.csv")
    print("- threshold_grid_results.csv")
    print("- rf_feature_importances_impurity.csv")
    print("- rf_permutation_importance.csv  (scoring=average_precision)")
    print("- pdp_*.png  (for features:", ", ".join(top), ")")
    print("- decision_tree_rules.txt")
    print("- confusion_matrix_rf_processed.png, roc_rf_processed.png")

if __name__ == "__main__":
    main()
