import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import load_model
import os

#  Ensure output directories exist
os.makedirs("outputs/evaluation", exist_ok=True)

#  Load the test data
print(" Loading dataset and model...")
df = pd.read_csv("data/raw/predictive-maintenance-dataset.csv")

# Rename sensor columns (for consistency)
df.rename(columns={
    "x1": "sensor_1",
    "x2": "sensor_2",
    "x3": "sensor_3",
    "x4": "sensor_4",
    "x5": "sensor_5"
}, inplace=True)

# Fill missing vibration values
df['vibration'] = df['vibration'].fillna(df['vibration'].median())

# Recreate 'fault' labels (top 5% vibration = fault)
thresh = df['vibration'].quantile(0.95)
df['fault'] = df['vibration'] > thresh

# Extract features and labels
X = df.drop(columns=['ID', 'fault'])
y = df['fault'].astype(int)

# Normalize with saved scaler
scaler = joblib.load("outputs/model/scaler.pkl")
X_scaled = scaler.transform(X)

# Load trained model
model = load_model("outputs/model/model.h5")

# Evaluate the model
print("🔍 Evaluating model...")
y_pred_probs = model.predict(X_scaled).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# Print classification report
report = classification_report(y, y_pred, digits=4)
print("\n📊 Classification Report:\n", report)

# Save report to file
with open("outputs/evaluation/classification_report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Fault"], yticklabels=["Normal", "Fault"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/confusion_matrix.png")
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_probs)
roc_auc = roc_auc_score(y, y_pred_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/evaluation/roc_curve.png")
plt.close()

print(" Evaluation complete. Results saved in 'outputs/evaluation/'")
