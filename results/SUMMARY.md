# Results summary (model engine: sklearn)

# Experiment 1 – original public dataset

## 1. Data audit

* Thesis label (2-of-4 and 2-of-3 identical: **True**): 5600 positives = rows 0–5599, 1 contiguous run(s), 23.3 min of recording.

### Anonymised columns vs original channels

| column | explained_by | r2 | slope | intercept | relation |
|---|---|---|---|---|---|
| x1 | revolutions | 0.998713 | 0.996400 | 74.393000 | x1 ≈ 0.9964·revolutions + 74.39 |
| x2 | revolutions | 0.998731 | 1.003600 | -74.393000 | x2 ≈ 1.004·revolutions − 74.39 |
| x3 | revolutions | 0.999186 | 0.013600 | -0.004600 | x3 ≈ 0.01358·revolutions − 0.004606 |
| x4 | revolutions^2 | 1.000000 | 1.000000 | -0.000000 | x4 ≈ 1·revolutions^2 − 1.86e-13 |
| x5 | humidity | 0.999961 | 148.082400 | -5481.593800 | x5 ≈ 148.1·humidity − 5482 |

### Single-feature ROC-AUC for the thesis label

| feature | auc | auc_direction_free |
|---|---|---|
| temperature | 1.0000 | 1.0000 |
| speed | 1.0000 | 1.0000 |
| revolutions | 1.0000 | 1.0000 |
| signal_strength | 1.0000 | 1.0000 |
| energy | 1.0000 | 1.0000 |
| row_index | 0.0000 | 1.0000 |
| humidity | 0.0791 | 0.9209 |
| motor_cycles | 0.0791 | 0.9209 |
| acceleration | 0.1435 | 0.8565 |
| vibration | 0.4864 | 0.5136 |

## 2. Replication of the thesis experiment (random split)

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

## 3. Time-aware evaluation of the thesis label

| part | rows | first_row | last_row | positives |
|---|---|---|---|---|
| train | 78400 | 0 | 78399 | 5600 |
| val | 16800 | 78400 | 95199 | 0 |
| test | 16801 | 95200 | 112000 | 0 |

| block | positives |
|---|---|
| 0 | 5600 |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |
| 7 | 0 |
| 8 | 0 |
| 9 | 0 |

## 4. Corrected experiment – vibration episodes (blocked CV)

| task | features | model | positives | prevalence | roc_auc | avg_precision | ap_lift_over_chance | recall_fault@0.5 | precision_fault@0.5 | episodes_warned | median_lead_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| detection (in episode) | C: context only | rf | 16258 | 0.145 | 0.463 | 0.124 | 0.854 | 0.009 | 0.017 | nan |  |
| detection (in episode) | C: context only | lr | 16258 | 0.145 | 0.368 | 0.112 | 0.775 | 0.310 | 0.106 | nan |  |
| detection (in episode) | V: vibration history | rf | 16258 | 0.145 | 0.988 | 0.923 | 6.357 | 0.791 | 0.922 | nan |  |
| detection (in episode) | V: vibration history | lr | 16258 | 0.145 | 0.977 | 0.868 | 5.980 | 0.904 | 0.814 | nan |  |
| detection (in episode) | CV: context + vibration history | rf | 16258 | 0.145 | 0.861 | 0.577 | 3.972 | 0.406 | 0.669 | nan |  |
| detection (in episode) | CV: context + vibration history | lr | 16258 | 0.145 | 0.968 | 0.816 | 5.619 | 0.846 | 0.774 | nan |  |
| early warning (onset within 10s) | C: context only | rf | 960 | 0.009 | 0.449 | 0.008 | 0.965 | 0.000 | 0.000 | 0/24 |  |
| early warning (onset within 10s) | C: context only | lr | 960 | 0.009 | 0.310 | 0.006 | 0.654 | 0.235 | 0.004 | 7/24 | 10.000 |
| early warning (onset within 10s) | V: vibration history | rf | 960 | 0.009 | 0.644 | 0.141 | 16.462 | 0.161 | 0.035 | 6/24 | 9.500 |
| early warning (onset within 10s) | V: vibration history | lr | 960 | 0.009 | 0.387 | 0.009 | 1.105 | 0.248 | 0.006 | 7/24 | 10.000 |
| early warning (onset within 10s) | CV: context + vibration history | rf | 960 | 0.009 | 0.485 | 0.052 | 6.077 | 0.041 | 1.000 | 1/24 | 9.750 |
| early warning (onset within 10s) | CV: context + vibration history | lr | 960 | 0.009 | 0.359 | 0.006 | 0.702 | 0.398 | 0.006 | 10/24 | 10.000 |
| early warning (onset within 60s) | C: context only | rf | 5760 | 0.051 | 0.468 | 0.051 | 0.984 | 0.000 | 0.000 | 0/24 |  |
| early warning (onset within 60s) | C: context only | lr | 5760 | 0.051 | 0.322 | 0.034 | 0.670 | 0.314 | 0.027 | 11/24 | 60.000 |
| early warning (onset within 60s) | V: vibration history | rf | 5760 | 0.051 | 0.581 | 0.219 | 4.266 | 0.202 | 0.147 | 5/24 | 60.000 |
| early warning (onset within 60s) | V: vibration history | lr | 5760 | 0.051 | 0.344 | 0.036 | 0.693 | 0.199 | 0.027 | 9/24 | 60.000 |
| early warning (onset within 60s) | CV: context + vibration history | rf | 5760 | 0.051 | 0.473 | 0.086 | 1.677 | 0.041 | 1.000 | 1/24 | 59.750 |
| early warning (onset within 60s) | CV: context + vibration history | lr | 5760 | 0.051 | 0.287 | 0.033 | 0.638 | 0.270 | 0.023 | 8/24 | 60.000 |

## 5. Events

| clock | gap_s | n_events | flagged_rows | compression | median_min | mean_min | max_min | pct_lt_1min | pct_ge_5min |
|---|---|---|---|---|---|---|---|---|---|
| real 4 Hz | 3 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| real 4 Hz | 10 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| real 4 Hz | 30 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| real 4 Hz | 60 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| real 4 Hz | 120 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| real 4 Hz | 300 | 1 | 5600 | 5600.000 | 23.329 | 23.329 | 23.329 | 0.000 | 100.000 |
| synthetic (thesis) | 3 | 852 | 5600 | 6.573 | 0.000 | 0.146 | 2.400 | 97.653 | 0.000 |
| synthetic (thesis) | 10 | 26 | 5600 | 215.385 | 1.425 | 7.735 | 146.967 | 34.615 | 15.385 |
| synthetic (thesis) | 30 | 1 | 5600 | 5600.000 | 206.017 | 206.017 | 206.017 | 0.000 | 100.000 |
| synthetic (thesis) | 60 | 1 | 5600 | 5600.000 | 206.017 | 206.017 | 206.017 | 0.000 | 100.000 |
| synthetic (thesis) | 120 | 1 | 5600 | 5600.000 | 206.017 | 206.017 | 206.017 | 0.000 | 100.000 |
| synthetic (thesis) | 300 | 1 | 5600 | 5600.000 | 206.017 | 206.017 | 206.017 | 0.000 | 100.000 |

| gap_s | n_events | flagged_rows | compression | median_min | mean_min | max_min | pct_lt_1min | pct_ge_5min |
|---|---|---|---|---|---|---|---|---|
| 3 | 205 | 5468 | 26.673 | 0.000 | 0.204 | 16.658 | 98.537 | 0.976 |
| 10 | 34 | 5468 | 160.824 | 1.250 | 1.891 | 16.658 | 44.118 | 5.882 |
| 30 | 24 | 5468 | 227.833 | 0.933 | 2.818 | 16.658 | 62.500 | 16.667 |
| 60 | 24 | 5468 | 227.833 | 0.933 | 2.818 | 16.658 | 62.500 | 16.667 |
| 120 | 23 | 5468 | 237.739 | 0.933 | 3.017 | 16.658 | 60.870 | 21.739 |
| 300 | 17 | 5468 | 321.647 | 4.296 | 5.311 | 16.658 | 29.412 | 35.294 |

## 6. PCA

| feature_set | pc1_var | pc2_var | silhouette_pc12_thesis_label |
|---|---|---|---|
| thesis PCA set (7) | 0.5749 | 0.1564 | 0.3972 |
| independent channels (3) | 0.4111 | 0.3016 | 0.3052 |

# Experiment 2 – researcher-processed dataset

## Labels in the processed table

| label | positives | share | first_row | last_row | n_contiguous_runs | longest_run_rows | positives_span_minutes |
|---|---|---|---|---|---|---|---|
| label_fault (2-of-4 @95) | 5600 | 0.050 | 0 | 5599 | 1 | 5600 | 23.333 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | 8789 | 0.078 | 19208 | 105560 | 3004 | 7 | 359.804 |
| Fault_950 (vibration > 95th pct) | 5611 | 0.050 | 19208 | 105160 | 2955 | 7 | 358.137 |

## Metrics reported in 2025

| regime | model | f1_macro | recall_fault | auc_roc | avg_precision |
|---|---|---|---|---|---|
| processed, random split | rf | 0.9997 | 0.9988 | 1.0000 | 1.0000 |
| processed, random split | mlp | 0.9981 | 0.9929 | 1.0000 | 1.0000 |
| processed, random split | lr | 0.9798 | 1.0000 | 1.0000 | 1.0000 |
| processed, random split | svm | 0.9783 | 1.0000 | 1.0000 | 1.0000 |
| processed, guard (as coded) | rf | 0.9997 | 0.9988 | 1.0000 | 1.0000 |
| processed, guard (as coded) | mlp | 0.9981 | 1.0000 | 1.0000 | 1.0000 |
| processed, guard (as coded) | svm | 0.9789 | 1.0000 | 1.0000 | 1.0000 |
| processed, guard (as coded) | lr | 0.9763 | 1.0000 | 1.0000 | 1.0000 |
| original columns, random split | rf | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| original columns, random split | mlp | 0.9910 | 1.0000 | 1.0000 | 0.9997 |
| original columns, random split | svm | 0.9671 | 1.0000 | 0.9999 | 0.9985 |
| original columns, random split | lr | 0.8488 | 0.9826 | 0.9951 | 0.9853 |

## Re-evaluation (random, shuffled, blocked, blocked + strict guard)

| label | evaluation | model | f1_macro | roc_auc | avg_precision | test_positives |
|---|---|---|---|---|---|---|
| label_fault (2-of-4 @95) | random split | rf | 0.9997 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | random split, shuffled file | rf | 1.0000 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | blocked CV (10 blocks, 60 s purge) | rf | 0.4872 | 0.5000 | 0.0500 | 5600 |
| label_fault (2-of-4 @95) | blocked CV, strict guard (humidity, motor_cycles) | rf | 0.4467 | 0.3864 | 0.0500 | 5600 |
| label_fault (2-of-4 @95) | random split | mlp | 0.9997 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | random split, shuffled file | mlp | 0.9997 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | blocked CV (10 blocks, 60 s purge) | mlp | 0.4843 | 0.0263 | 0.0500 | 5600 |
| label_fault (2-of-4 @95) | blocked CV, strict guard (humidity, motor_cycles) | mlp | 0.4872 | 0.0263 | 0.0500 | 5600 |
| label_fault (2-of-4 @95) | random split | lr | 0.9732 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | random split, shuffled file | lr | 0.9772 | 1.0000 | 1.0000 | 840 |
| label_fault (2-of-4 @95) | blocked CV (10 blocks, 60 s purge) | lr | 0.4808 | 0.0263 | 0.0500 | 5600 |
| label_fault (2-of-4 @95) | blocked CV, strict guard (humidity, motor_cycles) | lr | 0.4649 | 0.0263 | 0.0500 | 5600 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split | rf | 1.0000 | 1.0000 | 1.0000 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split, shuffled file | rf | 1.0000 | 1.0000 | 1.0000 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV (10 blocks, 60 s purge) | rf | 0.7313 | 0.8682 | 0.5392 | 8789 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV, strict guard (humidity, motor_cycles) | rf | 0.4215 | 0.4661 | 0.0711 | 8789 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split | mlp | 0.9938 | 0.9999 | 0.9991 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split, shuffled file | mlp | 0.9965 | 0.9999 | 0.9990 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV (10 blocks, 60 s purge) | mlp | 0.6448 | 0.9809 | 0.7155 | 8789 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV, strict guard (humidity, motor_cycles) | mlp | 0.4796 | 0.4925 | 0.0693 | 8789 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split | lr | 0.8400 | 0.9602 | 0.5334 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | random split, shuffled file | lr | 0.8467 | 0.9639 | 0.5605 | 1319 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV (10 blocks, 60 s purge) | lr | 0.5423 | 0.8161 | 0.2544 | 8789 |
| Fault_Combo2of3_TUNED (vib80/rev65/abs-speed65) | blocked CV, strict guard (humidity, motor_cycles) | lr | 0.2290 | 0.2843 | 0.0506 | 8789 |
| Fault_950 (vibration > 95th pct) | random split | rf | 1.0000 | 1.0000 | 1.0000 | 841 |
| Fault_950 (vibration > 95th pct) | random split, shuffled file | rf | 1.0000 | 1.0000 | 1.0000 | 841 |
| Fault_950 (vibration > 95th pct) | blocked CV (10 blocks, 60 s purge) | rf | 1.0000 | 1.0000 | 1.0000 | 5611 |
| Fault_950 (vibration > 95th pct) | blocked CV, strict guard (temperature, speed, acceleration, humidity, revolutions, signal_strength, energy, motor_cycles) | rf | 0.4872 | 0.6056 | 0.0689 | 5611 |
| Fault_950 (vibration > 95th pct) | random split | mlp | 0.9959 | 1.0000 | 0.9992 | 841 |
| Fault_950 (vibration > 95th pct) | random split, shuffled file | mlp | 0.9934 | 0.9998 | 0.9973 | 841 |
| Fault_950 (vibration > 95th pct) | blocked CV (10 blocks, 60 s purge) | mlp | 0.9931 | 0.9983 | 0.9947 | 5611 |
| Fault_950 (vibration > 95th pct) | blocked CV, strict guard (temperature, speed, acceleration, humidity, revolutions, signal_strength, energy, motor_cycles) | mlp | 0.4731 | 0.7287 | 0.0949 | 5611 |
| Fault_950 (vibration > 95th pct) | random split | lr | 0.9965 | 1.0000 | 0.9999 | 841 |
| Fault_950 (vibration > 95th pct) | random split, shuffled file | lr | 0.9934 | 1.0000 | 0.9998 | 841 |
| Fault_950 (vibration > 95th pct) | blocked CV (10 blocks, 60 s purge) | lr | 0.9958 | 1.0000 | 0.9997 | 5611 |
| Fault_950 (vibration > 95th pct) | blocked CV, strict guard (temperature, speed, acceleration, humidity, revolutions, signal_strength, energy, motor_cycles) | lr | 0.4919 | 0.7322 | 0.0836 | 5611 |

# Experiment 3 – semi-synthetic benchmark with injected faults

## A. Analytical labels vs ground truth

| rule | compared_with | labelled_share | precision | recall | f1 |
|---|---|---|---|---|---|
| thesis-style 2-of-3 (vibration, revolutions, humidity) | true fault | 0.008 | 0.024 | 0.003 | 0.005 |
| thesis-style 2-of-3 (vibration, revolutions, humidity) | true precursor or fault | 0.008 | 0.770 | 0.034 | 0.065 |
| vibration > 95th pct | true fault | 0.050 | 0.246 | 0.160 | 0.194 |
| vibration > 95th pct | true precursor or fault | 0.050 | 0.574 | 0.156 | 0.246 |
| physics-informed 2-of-3 (vibration, motor_current, humidity) | true fault | 0.013 | 0.571 | 0.097 | 0.166 |
| physics-informed 2-of-3 (vibration, motor_current, humidity) | true precursor or fault | 0.013 | 0.980 | 0.070 | 0.130 |

## B–F. Detection, guard, early warning, events (leave-sessions-out)

| model | features | trained_on | auc_vs_fault | ap_vs_fault | auc_vs_precursor_or_fault | ap_vs_precursor_or_fault | faults_detected | faults_warned_before_onset | median_lead_min | false_alarm_events_per_healthy_hour |
|---|---|---|---|---|---|---|---|---|---|---|
| rf | all channels | ground truth: fault | 0.992 | 0.909 | 0.842 | 0.733 | 9/12 | 4/12 | 0.233 | 0.000 |
| rf | all channels | ground truth: precursor or fault | 0.965 | 0.598 | 0.963 | 0.927 | 11/12 | 10/12 | 12.617 | 0.041 |
| rf | all channels | weak label: thesis-style 2-of-3 (vibration, revolutions, humidity) | 0.588 | 0.118 | 0.604 | 0.324 | 1/12 | 1/12 | 4.600 | 0.000 |
| rf | all channels | weak label: vibration > 95th pct | 0.780 | 0.192 | 0.752 | 0.385 | 8/12 | 8/12 | 10.300 | 1.511 |
| rf | all channels | weak label: physics-informed 2-of-3 (vibration, motor_current, humidity) | 0.742 | 0.264 | 0.703 | 0.483 | 1/12 | 1/12 | 7.250 | 0.000 |
| rf | guard: no vibration | ground truth: fault | 0.992 | 0.903 | 0.849 | 0.733 | 7/12 | 4/12 | 1.567 | 0.000 |
| rf | guard: no vibration | ground truth: precursor or fault | 0.959 | 0.509 | 0.963 | 0.927 | 11/12 | 11/12 | 12.550 | 0.122 |
| rf | vibration only | ground truth: fault | 0.921 | 0.722 | 0.783 | 0.617 | 11/12 | 5/12 | 4.167 | 0.204 |
| rf | vibration only | ground truth: precursor or fault | 0.941 | 0.630 | 0.854 | 0.682 | 11/12 | 9/12 | 11.533 | 1.429 |
| lr | all channels | ground truth: fault | 0.998 | 0.977 | 0.908 | 0.838 | 12/12 | 7/12 | 1.617 | 0.041 |
| lr | all channels | ground truth: precursor or fault | 0.976 | 0.700 | 0.989 | 0.979 | 12/12 | 12/12 | 14.433 | 2.940 |
| lr | all channels | weak label: thesis-style 2-of-3 (vibration, revolutions, humidity) | 0.702 | 0.228 | 0.710 | 0.472 | 5/12 | 4/12 | 12.992 | 0.694 |
| lr | all channels | weak label: vibration > 95th pct | 0.657 | 0.145 | 0.664 | 0.344 | 8/12 | 8/12 | 10.667 | 1.511 |
| lr | all channels | weak label: physics-informed 2-of-3 (vibration, motor_current, humidity) | 0.784 | 0.255 | 0.739 | 0.504 | 6/12 | 5/12 | 12.300 | 0.939 |
| lr | guard: no vibration | ground truth: fault | 0.999 | 0.986 | 0.914 | 0.879 | 12/12 | 10/12 | 1.492 | 0.000 |
| lr | guard: no vibration | ground truth: precursor or fault | 0.974 | 0.686 | 0.989 | 0.981 | 12/12 | 12/12 | 14.425 | 2.491 |
| lr | vibration only | ground truth: fault | 0.763 | 0.193 | 0.732 | 0.354 | 11/12 | 11/12 | 13.333 | 4.246 |
| lr | vibration only | ground truth: precursor or fault | 0.763 | 0.226 | 0.717 | 0.385 | 12/12 | 11/12 | 13.533 | 3.634 |

## G. Random vs session split

| model | split | auc | ap |
|---|---|---|---|
| rf | random rows (leaky) | 1.000 | 1.000 |
| rf | leave-sessions-out | 0.963 | 0.927 |
| lr | random rows (leaky) | 0.995 | 0.987 |
| lr | leave-sessions-out | 0.989 | 0.979 |

## H. Fault-strength sensitivity

| fault_strength | model | auc | ap | recall@0.5 | precision@0.5 |
|---|---|---|---|---|---|
| 0.250 | rf | 0.717 | 0.465 | 0.161 | 0.811 |
| 0.250 | lr | 0.970 | 0.945 | 0.887 | 0.928 |
| 0.500 | rf | 0.860 | 0.733 | 0.454 | 0.940 |
| 0.500 | lr | 0.981 | 0.966 | 0.919 | 0.958 |
| 1.000 | rf | 0.963 | 0.927 | 0.585 | 0.984 |
| 1.000 | lr | 0.989 | 0.979 | 0.944 | 0.979 |
