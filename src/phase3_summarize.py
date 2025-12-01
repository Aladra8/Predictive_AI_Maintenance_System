#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

def read_metrics(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)

def write_tex_overview(df_map, out_path):
    # df_map: {tag_name: df}
    # Keep only common models for clean comparison
    keep_cols = ["model","accuracy","f1_macro","precision_macro","recall_macro","auc_roc","avg_precision","recall_fault"]
    # Build a long table with tag column
    rows = []
    for tag, df in df_map.items():
        if df is None: continue
        tmp = df[keep_cols].copy()
        tmp["tag"] = tag
        rows.append(tmp)
    full = pd.concat(rows, axis=0, ignore_index=True)

    # Focus on RF and MLP (space)
    full = full[full["model"].isin(["rf","mlp"])].copy()
    name_map = {"rf":"Random Forest","mlp":"Neural Network (MLP)"}
    full["Model"] = full["model"].map(name_map)
    # Order tags if present
    order = []
    for t in ["processed","processed_guard","processed_time","processed_event"]:
        if (full["tag"]==t).any():
            order.append(t)
    full["tag"] = pd.Categorical(full["tag"], categories=order, ordered=True)
    full = full.sort_values(["Model","tag"])

    # Emit TeX
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Robustness overview: RF and MLP across split strategies (processed schema).}",
             r"\label{tab:robustness_overview}",
             r"\footnotesize",
             r"\begin{tabular}{l l c c c c c c}",
             r"\toprule",
             r"Model & Split & Acc. & F1 (macro) & Prec. & Rec. & AUC & AP \\",
             r"\midrule"]
    tag_label = {"processed":"Random (strat.)",
                 "processed_guard":"Leakage-guard",
                 "processed_time":"Time-blocked",
                 "processed_event":"Event-grouped"}
    for _, r in full.iterrows():
        lines.append(f"{r['Model']} & {tag_label.get(r['tag'], r['tag'])} & "
                     f"{r['accuracy']:.3f} & {r['f1_macro']:.3f} & {r['precision_macro']:.3f} & "
                     f"{r['recall_macro']:.3f} & {r['auc_roc']:.4f} & {r['avg_precision']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out_path).write_text("\n".join(lines))

def write_tex_delta(baseline_df, df, tag_suffix, out_path):
    if baseline_df is None or df is None:
        return
    base = baseline_df.set_index("model")
    cur  = df.set_index("model")
    common = base.index.intersection(cur.index)
    base = base.loc[common]; cur = cur.loc[common]
    keep = ["accuracy","f1_macro","precision_macro","recall_macro","auc_roc","avg_precision"]
    delta = pd.DataFrame(index=common)
    for k in keep:
        delta[k+"_base"]  = base[k]
        delta[k+"_cur"]   = cur[k]
        delta["delta_"+k] = cur[k] - base[k]
    name_map = {"rf":"Random Forest","mlp":"Neural Network (MLP)","svm":"SVM (RBF)","lr":"Logistic Regression"}
    delta["Model"] = [name_map.get(m,m) for m in delta.index]
    delta = delta.reset_index(drop=True)

    # Emit concise deltas
    lines = [r"\begin{table}[H]", r"\centering",
             rf"\caption{{Delta vs baseline (processed) — {tag_suffix}.}}",
             rf"\label{{tab:delta_{tag_suffix}}}",
             r"\footnotesize",
             r"\begin{tabular}{l c c c c c c}",
             r"\toprule",
             r"Model & $\Delta$Acc. & $\Delta$F1 & $\Delta$Prec. & $\Delta$Rec. & $\Delta$AUC & $\Delta$AP \\",
             r"\midrule"]
    for _, r in delta.iterrows():
        lines.append(f"{r['Model']} & {r['delta_accuracy']:+.3f} & {r['delta_f1_macro']:+.3f} & "
                     f"{r['delta_precision_macro']:+.3f} & {r['delta_recall_macro']:+.3f} & "
                     f"{r['delta_auc_roc']:+.4f} & {r['delta_avg_precision']:+.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out_path).write_text("\n".join(lines))

def main():
    root2 = Path("outputs/phase2/processed")
    root3 = Path("outputs/phase3")
    base = read_metrics(root2 / "model_metrics_processed.csv")
    m_guard = read_metrics(root3 / "processed_guard/model_metrics_processed_guard.csv")
    m_time  = read_metrics(root3 / "processed_time/model_metrics_processed_time.csv")
    m_event = read_metrics(root3 / "processed_event/model_metrics_processed_event.csv")

    # Overview (RF+MLP)
    write_tex_overview({
        "processed": base,
        "processed_guard": m_guard,
        "processed_time": m_time,
        "processed_event": m_event,
    }, "outputs/phase3/robustness_overview.tex")

    # Deltas
    write_tex_delta(base, m_guard, "leakage_guard", "outputs/phase3/delta_leakage_guard.tex")
    write_tex_delta(base, m_time,  "time_blocked",   "outputs/phase3/delta_time_blocked.tex")
    write_tex_delta(base, m_event, "event_grouped",  "outputs/phase3/delta_event_grouped.tex")

if __name__ == "__main__":
    main()
