
"""
Train a binary classification neural network model to detect elevator faults
based on sensor and telemetry data.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

#  Load dataset
df = pd.read_csv("data/raw/predictive-maintenance-dataset.csv")
#  Rename sensor columns for clarity
df.rename(columns={
    "x1": "sensor_1",
    "x2": "sensor_2",
    "x3": "sensor_3",
    "x4": "sensor_4",
    "x5": "sensor_5"
}, inplace=True)

#  Fill missing values in 'vibration' using the median
df['vibration'] = df['vibration'].fillna(df['vibration'].median())

#  Create binary 'fault' label: Top 5% vibration = Fault
threshold = df['vibration'].quantile(0.95)
df['fault'] = df['vibration'] > threshold

#  Separate features and labels
X = df.drop(columns=['ID', 'fault'])  # Features
y = df["fault"].astype(int)  # Converts True/False → 1/0 (Label)

#  Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Save the scaler for later inference
os.makedirs("outputs/model", exist_ok=True)
with open("outputs/model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(" Scaler saved to outputs/model/scaler.pkl")

#  Train/test split (80/20) stratified by class to preserve balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

#  Compute class weights to handle imbalance
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("⚖️ Class weights:", class_weights)

# Define neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#  Setup early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#  Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model
model.save("outputs/model/model.h5")
print(" Model saved to outputs/model/model.h5")

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(" Classification Report:\n", classification_report(y_test, y_pred))
