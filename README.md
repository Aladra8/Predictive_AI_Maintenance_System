# Predictive AI Maintenance System for Elevators

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Master's_Thesis-orange)](https://github.com/Aladra8/Predictive_AI_Maintenance_System)

A reproducible, transparent Predictive Maintenance (PdM) pipeline for elevator systems using public telemetry data. This project establishes a baseline for detecting mechanical faults without ground-truth logs by using physically-grounded analytical labeling and rigorous robustness protocols ("Leakage Guard").

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Methodology](#-methodology)
  - [Analytical Labeling](#1-analytical-labeling-2-of-3-rule)
  - [Leakage Guard Protocol](#2-leakage-guard-robustness)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Pipeline](#-usage-pipeline)
- [Results](#-results)
- [Thesis Report](#-thesis-report)
- [Author](#-author)

---

## Project Overview

**Motivation:** Public research in Elevator PdM is hindered by the lack of labeled datasets. Most studies rely on proprietary "black box" data.
**Objective:** To create an auditable, open-source pipeline that:
1.  Generates valid fault labels from raw telemetry.
2.  Benchmarks interpretable models (Random Forest) against Neural Networks (MLP).
3.  Validates that models learn physical failure precursors (Energy, Acceleration) rather than just memorizing rules.
4.  Aggregates high-frequency alerts into actionable maintenance tickets to reduce alarm fatigue.

---

## Dataset & Feature Engineering

**Source:** Huawei Munich Research Center (Zenodo/Kaggle).
**Raw Data:** 112,001 rows of high-frequency telemetry (approx 4Hz).

### Feature Mapping
The raw data contained anonymized features (`x1`...`x5`). Based on Exploratory Data Analysis (EDA) and physical correlations, we mapped them as follows:

| Original | Mapped Name | Description |
| :--- | :--- | :--- |
| `x1` | **Temperature** | Environmental context (slow drift). |
| `x2` | **Speed** | Correlates with motion phases. |
| `x3` | **Signal Strength** | IoT/Network health proxy. |
| `x4` | **Energy** | Motor power consumption (peaks at start). |
| `x5` | **Motor Cycles** | Cumulative usage counter. |
| *Derived* | **Acceleration** | Discrete difference of Speed ($\Delta v$). |
| *Derived* | **Timestamp** | Synthesized using Poisson process for time-series splits. |

---

## Methodology

### 1. Analytical Labeling ("2-of-3 Rule")
Since ground truth was unavailable, we developed a transparent labeling logic. A sample is flagged as **Faulty** if at least **2** of the following **3** sensors exceed their **95th percentile**:
* Vibration
* Speed
* Revolutions

*Validation:* Principal Component Analysis (PCA) confirmed that these labeled points cluster in a distinct "High Operational Intensity" manifold, validating they are not random noise.

### 2. "Leakage Guard" (Robustness)
To prove the model isn't just "cheating" by reversing the labeling rule, we introduced the **Guard Regime**.
* **Standard Regime:** Train on all features.
* **Guard Regime:** **Drop** the label-defining features (Vibration, Speed, Revolutions). The model must predict faults using only secondary context (Temperature, Energy, Acceleration).

### 3. Event Aggregation
Raw row-level predictions (4Hz) create too much noise. We implemented an event logic:
* **Logic:** Merge consecutive faulty rows if the gap is < 60 seconds.
* **Output:** Discrete "Maintenance Events" with Start Time, Duration, and Intensity.

---

## Project Structure

```text
Predictive_AI_Maintenance_System/
│
├── Code/
│   ├── src/
│   │   ├── preprocess_large_dataset.py   # ETL, Feature Engineering, Labeling
│   │   ├── phase1_noise_fs.py            # PCA, EDA, Threshold Grid
│   │   ├── phase2_training.py            # Model Training (RF, MLP), Calibration
│   │   ├── phase3_summarize.py           # Robustness Analysis & Tables
│   │   └── visualization/                # Plotting scripts
│   ├── data/
│   │   ├── raw/                          # Place 'predictive-maintenance-dataset.csv' here
│   │   └── processed/                    # Output CSVs appear here
│   ├── outputs/                          # Saved models (.pkl) and artifacts
│   └── requirements.txt                  # Python dependencies
│
├── Report/
│   ├── Full_Research_Report.tex          # Main Thesis Source
│   ├── references.bib                    # Bibliography
│   ├── Images/                           # Figures for the report
│   └── Tables/                           # CSV tables for LaTeX
│
└── README.md

INSTALLATION

1. Clone the repository:
git clone [https://github.com/Aladra8/Predictive_AI_Maintenance_System.git](https://github.com/Aladra8/Predictive_AI_Maintenance_System.git)
cd Predictive_AI_Maintenance_System

2. Set up the environment (Optional but recommended):
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r Code/requirements.txt

USAGE PIPELINE
Run the scripts in the following order to reproduce the thesis results.
Phase 1: Data Preparation
python Code/src/preprocess_large_dataset.py

Phase 2: EDA & Structural Validation
Generates PCA plots, threshold visualizations, and distribution histograms to validate the labeling logic.
python Code/others/run_phase1_noise_fs.py

Phase 3: Training & Benchmarking
Trains Random Forest and MLP models, generates ROC/PR curves, and runs the Leakage Guard robustness test.
# Run Standard Training & Save Models
python Code/others/run_phase2.py --save-models

# Run Robustness Check (Leakage Guard)
python Code/others/run_phase2_training.py --leakage-guard

Phase 4: Visualization & Events
Generates operational plots (Events per Day, Faults by Hour) and the Top-10 Events table.
python Code/src/visualization/visualize_combined_faults.py
# Note: Event analytics are also generated during Phase 1 & 2 outputs.


RESULTS SUMMARY

Model,ROC-AUC,F1-Score,Avg Precision,Interpretation
Random Forest,0.999,0.985,0.998,Highly Interpretable (Recommended)
MLP (Neural Net),0.999,0.982,0.997,Black-box Benchmark

Key Findings:
Interpretability: Random Forest matches Neural Network performance but offers superior transparency via Feature Importance and Partial Dependence Plots.

Robustness: Even in the Guard Regime (Vibration removed), the model maintained high precision, proving it utilizes Energy and Acceleration as valid failure precursors.

Operations: Event aggregation logic successfully reduced ~5,600 raw fault rows into 507 actionable maintenance tickets.

THE REPORT
The full academic report is available in the Report/ directory. It is written in LaTeX.

To compile the PDF locally (VS Code):

Ensure you have a TeX distribution installed (e.g., MacTeX for macOS).

Open Report/Full_Research_Report.tex in VS Code.

Run the build command (using LaTeX Workshop extension): Recipe: latexmk (latexmk -> bibtex -> latexmk)


AUTHOR
Buba Drammeh 
