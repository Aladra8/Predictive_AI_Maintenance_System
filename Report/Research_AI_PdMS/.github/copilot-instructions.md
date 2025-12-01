# AI Copilot Instructions: Elevator Predictive Maintenance Research

## Project Overview
This is a **research report** analyzing predictive maintenance (PdM) for elevator systems using public Kaggle/Zenodo telemetry data. The core deliverable is a reproducible, auditable analysis pipeline from raw sensor data to model recommendations, documented in `Full_Research_Report.tex` (LaTeX).

**Key Deliverable**: A comprehensive state-of-the-art report with model diagnostics, interpretability analysis, and event-level operational insights.

## Architecture & Data Flow

### Core Data Assets
- **`Full_Research_Report.tex`**: Main LaTeX document (~1735 lines) - the authoritative source
- **`Tables/`**: Auto-generated outputs from analysis scripts
  - `events_summary.json`: Event aggregation metadata (gap mode, quantiles, event counts)
  - `events_summary.tex`: LaTeX macros injected into report (see `\input{Tables/events_summary.tex}`)
  - `events_table.csv`: Top 10 longest fault events with start/end timestamps, durations
  - CSV versions of other summary statistics
- **`References.bib`**: BibTeX bibliography with 10+ citations from elevator PdM literature
- **`Images/`**: Two subdirectories for diagnostic plots:
  - `original_kaggle/`: Plots from raw anonymized features (x1-x5)
  - `processed_kaggle/`: Plots from semantically mapped features (Vibration, Speed, etc.)

### Data Processing Pipeline (Inferred from Report)
1. **Raw Data**: 112,001 sensor readings from Huawei Munich Research Center dataset
2. **Feature Mapping**: x1→Acceleration, x2→Speed, x3→Vibration, x4→Revolutions, x5→Energy
3. **Label Generation**: Analytical rule-based (NOT ground-truth labels) - samples flagged as "Fault" if ≥2 of {Vibration, Speed, Energy, Revolutions} exceed 95th percentile (2-of-4 rule)
4. **Data Splits**: Stratified 70/15/15 train/val/test with three leakage-guard variations:
   - **Leakage Guard**: Drops the 4 rule-defining features from training set
   - **Time Blocked**: Chronological splits (temporal generalization)
   - **Event Grouped**: Keeps fault events intact across splits
5. **Preprocessing**: Z-score standardization (fit on train only), smoothing for Acceleration (rolling avg, w=3)
6. **PCA Visualization**: PC1 correlates with operational intensity, PC2 captures secondary effects
7. **Event Aggregation**: Consecutive faults merged if <60s gap; 507 total events detected

## Critical Patterns & Conventions

### Labeling Philosophy
- **No Ground Truth**: The dataset lacks explicit fault labels; this is the primary research challenge
- **Audit Trail Required**: The analytical rule (95th percentile, 2-of-4) is **intentionally transparent** for reproducibility
- **Validation Loop**: Labels are validated via:
  1. PCA separation (silhouette scores confirm class cohesion)
  2. Permutation importance (features driving model decisions must align with labeling rule)
  3. Partial dependence plots (PDPs show sharp thresholds at high percentiles)

### Model Recommendations
**Chosen Model**: Random Forest (400 trees, Gini impurity, `class_weight="balanced"`)
- **Why RF over MLP**: Interpretability + calibration ease + competitive performance
- **Benchmark**: MLP (64→32→1 ReLU, Adam lr=1e-3, early stopping patience=5) used for comparison
- **Both models achieve ROC AUC ≈ 1.0**, but RF provides native feature importance; PDPs required for MLP

### Interpretability Requirements
- **Feature Importance**: Must report both Gini impurity importance and permutation importance
- **Partial Dependence Plots**: Critical for operational trust; generated for key features (Speed, Revolutions, Temperature, Acceleration, Humidity, Signal_Strength)
- **Confusion Matrices & Calibration Curves**: Required for robustness verification

### Dataset Limitations (Critical Context)
- **Single Calendar Day**: All 112,001 samples span ~3.5 hours on 2023-01-02, 06:00–09:24 UTC
  - **Implication**: Cannot perform meaningful week/month aggregations; temporal analysis naturally collapses
  - **Implication**: Most events are transient spikes, not sustained degradation
  - **Implication**: No train/test temporal drift expected within this window
- **Event Character**: 507 events detected; median duration ~1.2 min, max 8.28 min (transient spikes dominate)

## Build & Reporting System

### LaTeX Build Chain
See `.vscode/settings.json`:
```
pdflatex → biber → pdflatex → pdflatex
```
- **pdflatex** (v3 calls): Compiles tex to PDF
- **biber**: Bibliography processing (IEEE style, sorted by name/year/title)
- **Output**: Generates `.pdf`, `.aux`, `.bbl`, `.bcf`, `.run.xml`, `.toc`, `.lof`, `.lot`

### Generated Artifacts
The report injects dynamically computed values as LaTeX macros:
- `\input{Tables/events_summary.tex}` defines:
  - `\EventCount` (507)
  - `\EventSpanDays` (1)
  - `\EventGapMode` (quantile-per-day)
  - `\EventFirstTS`, `\EventLastTS`, etc.

**Convention**: When updating event aggregation logic, regenerate `Tables/events_summary.tex` and `Tables/events_summary.json`.

## Developer Workflows

### Updating Analysis or Regenerating Tables
If analysis scripts compute new results (e.g., different PCA loadings, event statistics):
1. Update the source Python/R scripts (referenced but not present in this repo; inferred from report structure)
2. Regenerate output files in `Tables/` and `Images/processed_kaggle/`
3. Update `Tables/events_summary.json` with new aggregation metadata
4. Regenerate `Tables/events_summary.tex` with LaTeX macro definitions
5. Run full build: `latexmk -pdf Full_Research_Report.tex`

### Adding Figures
- Place PNG plots in `Images/processed_kaggle/` (or `original_kaggle/` if using raw schema)
- Add to report via `\safeincludegraphics{filename.png}` or `\includegraphics[width=...]{filename.png}`
- Note: Report uses `\graphicspath{{Images/}{Images/original_kaggle/}{Images/processed_kaggle/}}` for automatic path resolution

### Citation Management
- Add to `references.bib` using IEEE style format
- Cite in text with `\textcite{key}` (inline) or `\cite{key}` (parenthetical)
- Biber auto-sorts; run full build chain to update bibliography

## Key Sections & Their Dependencies

| Section | Key Figures | Key Tables | Purpose |
|---------|-------------|-----------|---------|
| Methodology | PCA scatter, loadings | Feature map | Explain transparent labeling & preprocessing |
| Results | ROC, PR, Calibration curves | Model configs | Benchmark RF vs MLP performance |
| Interpretability | Feature importance, PDPs | Top events | Validate model decisions align with domain knowledge |
| Event Analysis | Events by hour, duration hist | Event summary | Operational context (transient vs sustained) |

## Conventions & Gotchas

1. **Z-Score Standardization**: Always fit on training data only; apply to val/test to prevent leakage
2. **Balanced Classes**: Use `class_weight="balanced"` for RF or equivalent resampling for neural nets (data is imbalanced; faults are minority)
3. **Event Aggregation**: Quantile-based gap detection accounts for variable inter-fault spacing; fixed 60s threshold is simpler but less adaptive
4. **Leakage Guard Protocol**: The "Processed Guard" regime (dropping rule-defining features) is THE gold standard for proving model doesn't just learn the labeling rule
5. **Silhouette Scores**: Monitor in PCA space; high scores (>0.3) validate that analytical labels have genuine structure
6. **Hyperparameter Stability**: RF importance requires sufficient trees (400+) to stabilize against stochastic noise

## Common AI Agent Tasks

- **Extending the report**: Add new model comparisons (XGBoost, SVM, etc.) following the diagnostic template (ROC, PR, Calib, feature importance)
- **Replicating analysis**: Ensure all random seeds (`random_state=42`) are fixed for reproducibility
- **Event analysis**: When aggregating events, document gap threshold rationale (quantile vs fixed)
- **Improving interpretability**: Generate PDPs for new features; ensure they show meaningful decision boundaries
- **Literature updates**: Keep `references.bib` current with recent PdM studies; use IEEE format

## References
- **Primary Dataset**: Huawei Munich Research Center (Kaggle/Zenodo)
- **Baseline Study**: Chen et al. 2024 (comprehensive elevator PdM review)
- **Fault Detection Methods**: Cao et al. 2025 (denoising & feature selection emphasis)
- **Transfer Learning**: Pan et al. 2024 (domain adaptation for door subsystems)
