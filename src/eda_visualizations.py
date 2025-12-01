"""
EDA Visualizations for Elevator Predictive Maintenance (polished)

What this script produces (saved under outputs/eda_polished/):
  1) fault_threshold_comparison.png      – counts at 90/92.5/95/97.5% vibration labels
  2) faults_by_hour.png                  – hourly distribution (bars + line), rush-hour bands
  3) faults_over_time.png                – daily faults with 3-day rolling mean
  4) faults_hourly_rolling.png           – 1-hour rolling sum over full timeline
  5) feature_importance_rf.png           – RandomForest impurity-based importance
  6) feature_importance_permutation.png  – Permutation importance (mean accuracy decrease)
  7) pca_scree_plot.png                  – explained variance and cumulative variance
  8) pca_scatter.png                     – PCA scatter with confidence ellipses (all features)
  9) pca_scatter_with_vibration.png      – PCA including vibration (color by fault)
 10) pca_scatter_without_vibration.png   – PCA excluding vibration (color by fault)
 11) pca_loadings.csv                    – PC1/PC2 loadings table (for slides/thesis)
"""

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Single authoritative dataset path
INPUT_PATH = "data/processed/processed_large_dataset_v4.csv"

# Output directory for figures/tables
OUTPUT_DIR = "outputs/eda_polished"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Canonical feature set
FEATURES = [
    "temperature", "speed", "acceleration", "humidity", "vibration",
    "revolutions", "signal_strength", "energy", "motor_cycles"
]

# Preferred order when choosing a fault label (first match wins)
CANDIDATE_LABELS = [
    "Fault_Combo2of3_TUNED",
    "Fault_Combo2of3",
    "Fault_900",   # fallback if combo labels are absent
]

# Vibration threshold labels to summarize (only plotted if present)
THRESHOLD_LABELS = ["Fault_900", "Fault_925", "Fault_950", "Fault_975"]

# Plot style
warnings.filterwarnings("ignore")
sns.set(context="talk", style="whitegrid")


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def pick_fault_label(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first label column present in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No expected fault label found. Looked for: {candidates}")


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce columns to numeric, leaving non-existent columns untouched."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def annotate_bars(ax, show_pct: bool = False, total: int = None) -> None:
    """Add value (and optional %) labels above bars in a barplot."""
    for p in ax.patches:
        h = p.get_height()
        if show_pct and total and total > 0:
            pct = 100.0 * h / total
            txt = f"{int(h)}\n({pct:.1f}%)"
        else:
            txt = f"{int(h)}"
        ax.annotate(txt, (p.get_x() + p.get_width() / 2.0, h),
                    ha="center", va="bottom", fontsize=10)


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    n_std: float = 1.96,
    **kwargs
) -> None:
    """
    Draw a confidence ellipse for x and y onto ax.
    n_std=1.96 corresponds approximately to a 95% ellipse for Gaussian data.
    """
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return
    cov = np.cov(x, y)
    # Degenerate or invalid covariance -> skip
    if not np.isfinite(cov).all() or np.linalg.det(cov) <= 0:
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        **kwargs
    )
    ax.add_patch(ell)


# ---------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Dataset not found at: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

# Basic guardrails
if "timestamp" not in df.columns:
    raise ValueError("Dataset is missing 'timestamp'. Re-run preprocessing.")

# Types and ordering
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# Ensure numerics for all relevant columns
all_possible = list(set(FEATURES + THRESHOLD_LABELS + CANDIDATE_LABELS))
df = ensure_numeric(df, all_possible)

# Choose the working fault label
fault_label = pick_fault_label(df, CANDIDATE_LABELS)

# Median-impute features (safe for PCA/ML)
work = df.copy()
for f in FEATURES:
    if f in work.columns:
        work[f] = work[f].fillna(work[f].median())

# Target vector as int
y = work[fault_label].astype(int).values

print(f"[i] Using fault label: {fault_label}")
print(f"[i] Samples: n={len(work):,}, positives={work[fault_label].sum():,} ({100*work[fault_label].mean():.2f}%)")


# ---------------------------------------------------------------------
# 1) Fault counts at vibration thresholds
# ---------------------------------------------------------------------

avail_thresh = [t for t in THRESHOLD_LABELS if t in work.columns]
if avail_thresh:
    counts = [int(work[t].sum()) for t in avail_thresh]
    total_n = len(work)

    plt.figure(figsize=(11, 6))
    ax = sns.barplot(x=avail_thresh, y=counts, palette="Set2")
    ax.set_title("Fault Count at Different Vibration Thresholds")
    ax.set_xlabel("Threshold Label")
    ax.set_ylabel("Fault Count")
    annotate_bars(ax, show_pct=True, total=total_n)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fault_threshold_comparison.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# 2) Fault distribution by hour (bars + line with rush-hour bands)
# ---------------------------------------------------------------------

tmp = work.copy()
tmp["hour"] = tmp["timestamp"].dt.hour
hourly = tmp.groupby("hour")[fault_label].sum().reindex(range(24), fill_value=0)

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=hourly.index, y=hourly.values, color="#4C72B0", alpha=0.6)
ax2 = ax.twinx()
ax2.plot(hourly.index, hourly.values, marker="o", linewidth=2, color="#55A868")

ax.set_title("Fault Distribution by Hour of Day")
ax.set_xlabel("Hour (0–23)")
ax.set_ylabel("Fault Count (bars)")
ax2.set_ylabel("Fault Count (line)")
ax.set_xticks(range(0, 24, 1))

# Light shading for typical rush hours (customize if your site differs)
for band in [(7, 10), (17, 20)]:
    ax.axvspan(band[0], band[1], color="grey", alpha=0.08)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/faults_by_hour.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------
# 3) Daily faults over time (with rolling mean)
# ---------------------------------------------------------------------

daily = work.set_index("timestamp")[fault_label].resample("1D").sum()
rolling = daily.rolling(window=3, min_periods=1).mean()  # 3-day window is smoother for short spans

plt.figure(figsize=(16, 6))
plt.plot(daily.index, daily.values, label="Daily faults", alpha=0.5)
plt.plot(rolling.index, rolling.values, label="3-day rolling mean", linewidth=2)
plt.title("Daily Fault Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Fault Count")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/faults_over_time.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------
# 4) Hourly rolling faults (1-hour sum) – useful for usage peaks
# ---------------------------------------------------------------------

hourly_series = work.set_index("timestamp")[fault_label].resample("1H").sum()
hourly_roll = hourly_series.rolling(window=1, min_periods=1).sum()

plt.figure(figsize=(20, 6))
plt.plot(hourly_roll.index, hourly_roll.values)
plt.title("Hourly Faults (1-hour rolling sum)")
plt.xlabel("Time")
plt.ylabel("Fault Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/faults_hourly_rolling.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------
# 5) Feature importance – RandomForest + Permutation
# ---------------------------------------------------------------------

X = work[FEATURES].copy()

# Balanced RF to mitigate class imbalance
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
rf.fit(X, y)

# Impurity-based importance
imp_series = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(14, 8))
ax = imp_series.plot(kind="barh", color="#4C72B0")
for i, v in enumerate(imp_series.values):
    plt.text(v + imp_series.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=10)
plt.title("Feature Importance (Random Forest – impurity)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_rf.png", dpi=150)
plt.close()

# Permutation importance on a holdout set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
rf_perm = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
).fit(X_train, y_train)

perm = permutation_importance(
    rf_perm, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_series = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(14, 8))
ax = perm_series.plot(kind="barh", color="#55A868")
for i, v in enumerate(perm_series.values):
    plt.text(v + (perm_series.max() if perm_series.max() > 0 else 0.0005) * 0.01, i, f"{v:.3f}",
             va="center", fontsize=10)
plt.title("Permutation Importance (mean accuracy decrease)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_permutation.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------
# 6) PCA – Scree and 2D scatter with confidence ellipses
# ---------------------------------------------------------------------

def pca_plots(X_df: pd.DataFrame, y_vec: np.ndarray, suffix: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Create scree and scatter plots; return (explained_var, X_pca_2d)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)
    exp = pca_full.explained_variance_ratio_
    cum = np.cumsum(exp)

    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(exp) + 1), exp, marker="o", label="Explained variance")
    plt.plot(range(1, len(cum) + 1), cum, marker="o", linestyle="--", label="Cumulative variance")
    plt.xticks(range(1, len(exp) + 1))
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Scree Plot")
    plt.legend()
    plt.tight_layout()
    out_name = f"{OUTPUT_DIR}/pca_scree_plot{suffix}.png" if suffix else f"{OUTPUT_DIR}/pca_scree_plot.png"
    plt.savefig(out_name, dpi=150)
    plt.close()

    # 2D scatter
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_scaled)

    # Save loadings for PC1/PC2 for interpretability
    loadings = pd.DataFrame(
        pca2.components_.T,
        index=X_df.columns,
        columns=["PC1_loading", "PC2_loading"]
    ).sort_index()
    loadings.to_csv(f"{OUTPUT_DIR}/pca_loadings{suffix if suffix else ''}.csv")

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Fault"] = y_vec

    # Subsample for readability if very large
    plot_df = pca_df.sample(min(40000, len(pca_df)), random_state=42)

    plt.figure(figsize=(12, 9))
    ax = sns.scatterplot(
        data=plot_df, x="PC1", y="PC2",
        hue="Fault", palette={0: "#4C72B0", 1: "#C44E52"},
        alpha=0.5, edgecolor="none"
    )

    # Confidence ellipses by class
    for cls, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        pts = pca_df[pca_df["Fault"] == cls]
        confidence_ellipse(
            pts["PC1"].values, pts["PC2"].values, ax,
            n_std=1.96, facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=1.5
        )

    ax.set_title(f"PCA Scatter (Scaled Features){' ' + suffix if suffix else ''}")
    ax.set_xlabel(f"PC1 ({exp[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({exp[1]*100:.1f}% var)")
    ax.legend(title="Fault", loc="best")
    plt.tight_layout()
    out_name = f"{OUTPUT_DIR}/pca_scatter{suffix}.png" if suffix else f"{OUTPUT_DIR}/pca_scatter.png"
    plt.savefig(out_name, dpi=150)
    plt.close()

    return exp, X_pca


# PCA with all FEATURES
exp_all, _ = pca_plots(X, y, suffix="")

# PCA excluding vibration to show whether separation is driven solely by vibration
if "vibration" in X.columns:
    X_no_vib = X.drop(columns=["vibration"]).copy()
    exp_nv, _ = pca_plots(X_no_vib, y, suffix="_without_vibration")
    # Also publish the "with vibration" scatter for side-by-side comparison
    pca_plots(X, y, suffix="_with_vibration")

print(f"[✓] All polished EDA plots saved to: {OUTPUT_DIR}")
