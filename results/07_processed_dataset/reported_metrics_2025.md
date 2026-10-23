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
