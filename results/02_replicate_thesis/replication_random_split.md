| feature_set | model | n_features | f1_macro | f1_ci95 | recall_fault | precision_fault | roc_auc | avg_precision | test_n | test_positives |
|---|---|---|---|---|---|---|---|---|---|---|
| standard (9 features) | rf | 9 | 0.9997 | [0.9990, 1.0000] | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 16800 | 840 |
| standard (9 features) | mlp | 9 | 0.9997 | [0.9990, 1.0000] | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 16800 | 840 |
| standard (9 features) | lr | 9 | 0.9732 | [0.9673, 0.9781] | 1.0000 | 0.9032 | 1.0000 | 1.0000 | 16800 | 840 |
| standard (9 features) | svm | 9 | 0.9997 | [0.9990, 1.0000] | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as described (-vib, speed, rev) | rf | 6 | 0.9997 | [0.9990, 1.0000] | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as described (-vib, speed, rev) | mlp | 6 | 0.9997 | [0.9990, 1.0000] | 0.9988 | 1.0000 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as described (-vib, speed, rev) | lr | 6 | 0.9689 | [0.9627, 0.9744] | 1.0000 | 0.8889 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as described (-vib, speed, rev) | svm | 6 | 0.9994 | [0.9985, 1.0000] | 1.0000 | 0.9976 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as coded (-vib, speed, energy, rev) | rf | 5 | 0.9997 | [0.9990, 1.0000] | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as coded (-vib, speed, energy, rev) | mlp | 5 | 0.9994 | [0.9984, 1.0000] | 0.9976 | 1.0000 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as coded (-vib, speed, energy, rev) | lr | 5 | 0.9623 | [0.9556, 0.9686] | 1.0000 | 0.8669 | 1.0000 | 1.0000 | 16800 | 840 |
| guard as coded (-vib, speed, energy, rev) | svm | 5 | 0.9991 | [0.9979, 1.0000] | 1.0000 | 0.9964 | 1.0000 | 1.0000 | 16800 | 840 |
| guard without copies (humidity, motor_cycles) | rf | 2 | 0.6303 | [0.6198, 0.6404] | 0.8274 | 0.2206 | 0.9163 | 0.3733 | 16800 | 840 |
| guard without copies (humidity, motor_cycles) | mlp | 2 | 0.4872 | [0.4863, 0.4879] | 0.0000 | 0.0000 | 0.9210 | 0.2498 | 16800 | 840 |
| guard without copies (humidity, motor_cycles) | lr | 2 | 0.6509 | [0.6404, 0.6610] | 0.8464 | 0.2448 | 0.9210 | 0.2498 | 16800 | 840 |
| guard without copies (humidity, motor_cycles) | svm | 2 | 0.5287 | [0.5170, 0.5401] | 0.0464 | 0.4149 | 0.9420 | 0.3898 | 16800 | 840 |
| row index only (time) | rf | 1 | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16800 | 840 |
| row index only (time) | mlp | 1 | 0.9997 | [0.9990, 1.0000] | 1.0000 | 0.9988 | 1.0000 | 1.0000 | 16800 | 840 |
| row index only (time) | lr | 1 | 0.9303 | [0.9219, 0.9391] | 1.0000 | 0.7678 | 1.0000 | 1.0000 | 16800 | 840 |
| row index only (time) | svm | 1 | 0.9994 | [0.9985, 1.0000] | 1.0000 | 0.9976 | 1.0000 | 1.0000 | 16800 | 840 |
