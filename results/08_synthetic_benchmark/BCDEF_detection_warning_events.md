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
