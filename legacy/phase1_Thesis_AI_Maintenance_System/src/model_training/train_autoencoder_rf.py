import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import os

# Load data
X = pd.read_csv("data/processed/X_processed.csv")
y = pd.read_csv("data/processed/y_labels.csv")['fault']

# Normalize again
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Autoencoder structure
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation="relu")(input_layer)
encoded = Dense(4, activation="relu")(encoded)
decoded = Dense(6, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, validation_split=0.1, verbose=0)

# Use encoder's output as features
encoder = Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# Train RF on encoded features
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_encoded, y, stratify=y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train_enc, y_train)

# Predict
y_pred = rf.predict(X_test_enc)

# Save models
os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/evaluation/benchmark", exist_ok=True)
autoencoder.save("outputs/model/autoencoder_model.h5")
joblib.dump(rf, "outputs/model/autoencoder_rf_model.pkl")

# Save report
report = classification_report(y_test, y_pred, output_dict=False)
with open("outputs/evaluation/benchmark/autoencoder_rf_report.txt", "w") as f:
    f.write(report)

print(" Autoencoder + RF training complete. Report saved.")
