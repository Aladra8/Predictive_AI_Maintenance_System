# Elevator Predictive Maintenance with AI

This project implements predictive maintenance for elevators using sensor data and machine learning. It replicates the benchmark approach in Liu et al. (2022) and proposes an improved neural network model to detect faults.

##  Objective

To design, implement, and evaluate a predictive maintenance model that identifies potential faults in elevators using sensor data. The project is benchmarked against the methodology presented in Liu et al. (2022), with improvements and alternative models explored.


## Key Features

- Data exploration & preprocessing
- Neural network model for binary classification
- Evaluation (Precision, Recall, F1, Confusion Matrix, ROC)
- PDF report generator

## Dataset Source

Kaggle Elevator Predictive Maintenance Dataset  
[Vibration-based failure classification dataset](https://www.kaggle.com/datasets/ninorapicavoli/elevator-predictive-maintenance)

 ## Processed Dataset Summary

**Source:** Kaggle - Elevator Predictive Maintenance Dataset  
**Samples:** 112,001  
**Faults:** 5,598 (~5%)  
**Features:** `revolutions`, `humidity`, `vibration`, `x1` to `x5` (engineered)

## Models Implemented
 ---------------------------------------------------------------------------------------------------
| Model                     | Accuracy | F1-Score | Notes                                          |
|--------------------------|----------|----------|-------------------------------------------------|
| Neural Network (Keras)   | 99.1%    | 91.6%    | Strong performance despite class imbalance      |
| Random Forest            | 98.2%    | 88.1%    | Robust baseline, fast to train                  |
| Support Vector Machine   | 98.5%    | 89.4%    | Performs well, especially with balanced classes |
| Autoencoder + RF         | ~97.1%   | ~85.0%   | Liu et al. benchmark architecture               |

## VS Benchmark

Our results are compared with **Liu et al. (2022)** which used Autoencoder + RF on a simulated dataset and achieved:
- **Accuracy:** 97.1%
- **F1-Score:** 0.94

##  Highlights
- Full preprocessing pipeline
- Benchmark comparison table
- Evaluation metrics
- Plots: ROC Curve, Confusion Matrix
- PDF report: `outputs/evaluation/final_model_report.pdf`

##  Citation

Liu et al., 2022 — _“Fault detection in elevators using vibration-based AI classification.”_

---

## How to Run

1. **Set up virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. **Run preprocessing**:
    ```bash
    python src/preprocess_and_explore.py
    ```

3. **Train models**:
    ```bash
    python src/train_model.py                # Neural Network
    python src/train_random_forest.py        # Random Forest
    python src/train_svm.py                  # SVM
    python src/train_autoencoder_rf.py       # Autoencoder + RF
    ```

4. **Evaluate**:
    ```bash
    python src/evaluate_model.py
    ```

5. **Generate Final Report**:
    ```bash
    python src/generate_report.py
    ```

---

## Author

**Baba Drammeh**  
Master's in Computer Engineering  
University of Padova (UNIPD) and UNIPI, Italy  
GitHub: [@Aladra8](https://github.com/Aladra8)
