import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import os

# Load data
X = pd.read_csv("data/processed/X_processed.csv")
y = pd.read_csv("data/processed/y_labels.csv")['fault']

# Normalize features (again for SVM stability)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train model
svm = SVC(kernel='rbf', class_weight="balanced", probability=True)
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Save model and report
os.makedirs("outputs/evaluation/benchmark", exist_ok=True)
joblib.dump(svm, "outputs/model/svm_model.pkl")
report = classification_report(y_test, y_pred, output_dict=False)
with open("outputs/evaluation/benchmark/svm_report.txt", "w") as f:
    f.write(report)

print(" SVM training complete. Report saved.")
