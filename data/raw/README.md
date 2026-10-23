# Raw data

| File | Source | MD5 | Notes |
|------|--------|-----|-------|
| `predictive-maintenance-dataset.csv` | Kaggle mirror ("Elevator Predictive Maintenance Dataset", uploaded Dec 2021) | `f4b2fcb07499a7e8e17b19b29cca3eaa` | 112,001 rows; columns ID, revolutions, humidity, vibration, x1–x5; 2,438 missing vibration values |
| `zenodo_predictive-maintenance-dataset.csv` | Zenodo record 3653909 (Axenie & Bortoli, Huawei MRC, v1.0.0, 2020) | `e6fc08f1f189dd822cdb20b93b61051b` | 112,001 rows; `;`-separated; columns ball-bearing, humidity, vibration; contains spreadsheet-corrupted cells such as `03. Jun` |

Data owner's description: elevator **car-door** system, time series sampled at **4 Hz** during
high-peak and evening usage (**16:30–23:30**); sensors: door ball-bearing (revolutions),
humidity (%), vibration (dB).

Before redistributing these files publicly, check the Zenodo licence ("Other (Open)") and the
Kaggle licence. Alternatively, delete the CSVs and download them again:
<https://doi.org/10.5281/zenodo.3653909>.
