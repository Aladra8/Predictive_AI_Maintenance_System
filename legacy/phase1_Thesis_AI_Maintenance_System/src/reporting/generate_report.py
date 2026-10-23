import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Paths ===
DATA_PATH = "data/raw/predictive-maintenance-dataset.csv"
MODEL_PATH = "outputs/model/model.h5"
SCALER_PATH = "outputs/model/scaler.pkl"
VIBRATION_PLOT = "outputs/vibration_distribution_updated.png"
FAULT_PLOT = "outputs/fault_distribution_updated.png"
CLASS_REPORT_TXT = "outputs/evaluation/classification_report.txt"
REPORT_PDF = "outputs/evaluation/final_model_report.pdf"

# === Load data ===
df = pd.read_csv(DATA_PATH)

# Rename for consistency
df.rename(columns={
    "x1": "sensor_1",
    "x2": "sensor_2",
    "x3": "sensor_3",
    "x4": "sensor_4",
    "x5": "sensor_5"
}, inplace=True)

df['vibration'] = df['vibration'].fillna(df['vibration'].median())
threshold = df['vibration'].quantile(0.95)
df['fault'] = df['vibration'] > threshold

X = df.drop(columns=["ID", "fault"])
y = df["fault"]

# Load scaler and transform features
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X)

# Load model
model = load_model(MODEL_PATH)
y_pred = model.predict(X_scaled) > 0.5

# === Evaluation Metrics ===
report = classification_report(y, y_pred, target_names=["No Fault", "Fault"])
conf_matrix = confusion_matrix(y, y_pred)

# Save classification report
with open(CLASS_REPORT_TXT, "w") as f:
    f.write(report)

# === Generate PDF Report ===
with PdfPages(REPORT_PDF) as pdf:
    # Page 1: Summary stats
    plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    plt.title("Elevator Fault Prediction Model Report", fontsize=16, pad=20)
    lines = report.split("\n")
    for i, line in enumerate(lines):
        plt.text(0.01, 0.95 - i * 0.04, line, fontsize=10, family="monospace")
    pdf.savefig()
    plt.close()

    # Page 2: Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Fault", "Fault"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3: Vibration Distribution
    if os.path.exists(VIBRATION_PLOT):
        img = plt.imread(VIBRATION_PLOT)
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Vibration Distribution")
        pdf.savefig()
        plt.close()

    # Page 4: Fault Distribution
    if os.path.exists(FAULT_PLOT):
        img = plt.imread(FAULT_PLOT)
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Fault Label Distribution")
        pdf.savefig()
        plt.close()

print(f" Final report saved to: {REPORT_PDF}")
