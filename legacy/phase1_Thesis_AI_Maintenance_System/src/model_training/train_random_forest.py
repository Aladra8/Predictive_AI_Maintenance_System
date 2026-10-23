import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Load preprocessed data
X = pd.read_csv("data/processed/X_processed.csv")
y = pd.read_csv("data/processed/y_labels.csv")['fault']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Output directory
os.makedirs("outputs/evaluation/benchmark", exist_ok=True)

# Save model
joblib.dump(rf, "outputs/model/random_forest_model.pkl")

# Save report
report = classification_report(y_test, y_pred, output_dict=False)
with open("outputs/evaluation/benchmark/rf_report.txt", "w") as f:
    f.write(report)

print(" Random Forest training complete. Report saved.")