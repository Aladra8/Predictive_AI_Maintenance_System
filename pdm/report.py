"""Collect the headline numbers of all experiments into one Markdown file."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .utils import ROOT


def _md(path: Path) -> str:
    return path.read_text() if path.exists() else f"_missing: {path.name}_\n"


def write_summary(engine: str = "sklearn") -> str:
    res = ROOT / "results"; mdl = ROOT / ("results" if engine == "sklearn" else "results_numpy")
    ls = json.loads((res / "01_audit" / "label_structure.json").read_text())
    l4 = ls["2of4"]
    parts = [f"# Results summary (model engine: {engine})\n",
             "# Experiment 1 – original public dataset\n",
             "## 1. Data audit\n",
             f"* Thesis label (2-of-4 and 2-of-3 identical: **{ls['rules_identical']}**): "
             f"{l4['positives']} positives = rows {l4['first_row']}–{l4['last_row']}, "
             f"{l4['n_contiguous_runs']} contiguous run(s), {l4['positives_span_minutes']:.1f} min of recording.\n",
             "### Anonymised columns vs original channels\n", _md(res / "01_audit" / "derived_columns.md"),
             "### Single-feature ROC-AUC for the thesis label\n", _md(res / "01_audit" / "single_feature_auc.md"),
             "## 2. Replication of the thesis experiment (random split)\n",
             _md(mdl / "02_replicate_thesis" / "replication_random_split.md"),
             "## 3. Time-aware evaluation of the thesis label\n",
             _md(res.parent / ("results" if engine == "sklearn" else "results_numpy") / "03_time_aware" / "chronological_split_content.md"),
             _md(mdl / "03_time_aware" / "positives_per_block.md"),
             "## 4. Corrected experiment – vibration episodes (blocked CV)\n",
             _md(mdl / "04_vibration_episodes" / "results_blocked_cv.md"),
             "## 5. Events\n", _md(res / "05_events" / "thesis_label_events_by_gap.md"),
             _md(res / "05_events" / "vibration_exceedance_events_by_gap.md"),
             "## 6. PCA\n", _md(res / "06_pca" / "pca_summary.md"),
             "# Experiment 2 – researcher-processed dataset\n",
             "## Labels in the processed table\n", _md(mdl / "07_processed_dataset" / "label_structure.md"),
             "## Metrics reported in 2025\n", _md(mdl / "07_processed_dataset" / "reported_metrics_2025.md"),
             "## Re-evaluation (random, shuffled, blocked, blocked + strict guard)\n", _md(mdl / "07_processed_dataset" / "reevaluation.md"),
             "# Experiment 3 – semi-synthetic benchmark with injected faults\n",
             "## A. Analytical labels vs ground truth\n", _md(mdl / "08_synthetic_benchmark" / "A_label_quality.md"),
             "## B–F. Detection, guard, early warning, events (leave-sessions-out)\n", _md(mdl / "08_synthetic_benchmark" / "BCDEF_detection_warning_events.md"),
             "## G. Random vs session split\n", _md(mdl / "08_synthetic_benchmark" / "G_random_vs_session_split.md"),
             "## H. Fault-strength sensitivity\n", _md(mdl / "08_synthetic_benchmark" / "H_fault_strength_sensitivity.md")]
    out = mdl / "SUMMARY.md"
    out.write_text("\n".join(parts))
    return f"Summary written to {out}"
