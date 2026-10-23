# src/model_threshold_comparison.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# === Config ===
DATA_PATH = "data/processed/predictive_maintenance_thresholds_labeled_v2.csv"
OUTPUT_DIR = "outputs/evaluation_thresholds/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
df = pd.read_csv(DATA_PATH)
features = ['Ball_bearing', 'Humidity', 'Vibration', 'Speed', 'Acceleration']
label_columns = ['Fault_900', 'Fault_925', 'Fault_950', 'Fault_975']

# === Models ===
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf')
}

# === Evaluation function ===
def evaluate(X_train, X_test, y_train, y_test, model, model_name, label_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"{model_name}_{label_name}_report.csv")
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} - {label_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{label_name}_cm.png"))
    plt.close()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{model_name} - {label_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_{label_name}_roc.png"))
    plt.close()

    print(f"[✓] {model_name} on {label_name} evaluated and saved.")

# === Loop through each label + model ===
for label in label_columns:
    print(f"\n===== Evaluating Label: {label} =====")
    
    # Drop rows with NaNs in features or label
    data = df[features + [label]].dropna()
    X = data[features]
    y = data[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    for model_name, model in models.items():
        evaluate(X_train, X_test, y_train, y_test, model, model_name, label)

print(f"\n[✓] All evaluations completed and saved to: {OUTPUT_DIR}")
