# Predictive AI Maintenance for Elevators: pipeline, data audit and deployable service

MSc thesis code: **Baba Drammeh**, MSc Computer Engineering, Università degli Studi di Padova (with Università di Pisa).

One configuration file (`config/default.yaml`) drives every experiment. One command regenerates
every table and figure (`results/`). The same pipeline runs on three datasets:

| Experiment | Data | Scripts | Headline (scikit-learn, Random Forest) |
|---|---|---|---|
| **E1** | Original public Huawei dataset, real 4 Hz clock | 01–06 | x1–x5 are transforms of the 3 real channels (x4 = revolutions², R² = 1.0). The percentile-vote label = the first 23.3 min of the recording. Random split: AUC 1.000. Chronological test set: 0 positives. Vibration episodes: detectable (AUC 0.99), weakly predictable (AP 0.22 vs 0.05). |
| **E2** | Researcher-processed table (renamed columns, synthetic clock, tuned labels, shuffled copy) | 07 | Every processing step verified. Random / shuffled split: AUC 1.000. Time-aware split with strict guard: 0.28–0.73. |
| **E3** | Semi-synthetic benchmark: real baseline + simulated motor current + 12 injected faults | 08 | Unseen sessions: AUC 0.963, 10/12 faults warned 12.6 min before onset, 0.04 false events per healthy hour. Percentile labels vs truth: F1 ≤ 0.19. |
| **Service** | E3 model behind an HTTP scoring API with event aggregation | 09, `deploy/` | Hold-out faulty session: event 11.6 min before onset. Hold-out healthy session: 0 alarms. |

`results/SUMMARY.md` collects all headline tables.

## Quick start (macOS / Linux, Python 3.11 or newer)

```bash
python3 --version                      # must be 3.11+ (macOS often ships 3.9: install 3.12 from python.org)
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q                    # 9 tests
python scripts/run_all.py              # experiments 01–08, service model (09), figures (10); 30–60 min
```

Single steps: `python scripts/08_synthetic_benchmark.py`, etc. Options for every script:
`--models rf lr`, `--config my.yaml`, `--engine numpy` (dependency-free smoke test, writes to `results_numpy/`).

## Run the scoring service locally

```bash
export PDM_API_KEY=dev-key                       # a password you choose
python deploy/service/app.py                     # terminal 1: listens on http://localhost:8080
# terminal 2 (activate .venv first):
curl http://localhost:8080/health
python deploy/replay_client.py --api-key dev-key --session 13 --speed 50   # unseen faulty session
python deploy/replay_client.py --api-key dev-key --session 14 --speed 500  # unseen healthy session
```

Endpoints: `GET /health`, `GET /model`, `POST /score`, `POST /flush`, `GET /events`, `GET /status`,
`GET /metrics`. Requests other than `/health` need the header `X-API-Key`. Optional
`PDM_WEBHOOK_URL` posts every closed event to a ticket system.

For 24/7 operation the repository includes `deploy/Dockerfile`, `deploy/docker-compose.yml` (with a
Caddy HTTPS reverse proxy), `deploy/Caddyfile`, `deploy/.env.example` and a systemd unit
(`deploy/systemd/`).

## Repository layout

```
config/default.yaml      every parameter (seeds, thresholds, splits, windows, models, benchmark, service)
data/raw/                Kaggle mirror and original Zenodo CSV (MD5-checked); see data/raw/README.md
data/processed/          the researcher-processed tables of Experiment 2 (+ shuffled copy)
data/reported/           metrics files produced by the 2025 runs (as reported)
pdm/                     library: data & clock, labels, features, splits, metrics, models, events, PCA, audit, synthetic benchmark, report
scripts/                 numbered experiments 01–10 and run_all.py
results/                 generated tables (.csv/.md/.tex), figures and run_meta.json per experiment, SUMMARY.md
deploy/                  scoring service, replay client, trained model + model card, Docker/Caddy/systemd files
tests/                   unit tests (labels, metrics, splits, events, train/serve parity)
legacy/                  code of the earlier project phases, kept read-only for traceability
```

## Data

* Original release: C. Axenie, S. Bortoli, *Predictive maintenance dataset*, Huawei Munich Research
  Center, Zenodo, 2020, doi:10.5281/zenodo.3653909 (elevator car-door system, 4 Hz, 16:30–23:30).
* Kaggle mirror: "Elevator Predictive Maintenance Dataset" (adds ID and x1–x5; origin undocumented).

## Reproducibility

Seeds fixed (42 for models, 3 for the benchmark). The dataset MD5 is verified at load. Each results
folder stores `run_meta.json` (config, git hash, engine, Python version). Metrics (ROC-AUC, AP, F1,
bootstrap CIs, event recall, lead time) are implemented in NumPy (`pdm/metrics.py`).

## Licence

GPL-3.0 (see `LICENSE`).
