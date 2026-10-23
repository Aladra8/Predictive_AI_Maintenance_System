# Deployment Guide

This guide describes how to install the project, reproduce the experiments and run the
elevator predictive-maintenance scoring service in four configurations:

| Section | Configuration | Typical use |
|---|---|---|
| [1](#1-local-installation) | Local installation (macOS or Linux) | development, reproducing the results |
| [2](#2-running-the-service-locally) | Service on the local machine | testing, demonstrations |
| [3](#3-running-the-service-with-docker) | Service in a Docker container | portable, reproducible runtime |
| [4](#4-production-deployment-with-docker-compose-and-https) | Linux server with Docker Compose and HTTPS | continuous operation |
| [5](#5-production-deployment-with-systemd) | Linux server with systemd (no Docker) | continuous operation without containers |
| [6](#6-webhook-integration) | Webhook integration | forwarding events to a CMMS or ticket system |

Reference material: [service configuration](#7-configuration-reference), [API](#8-api-reference),
[operations](#9-operations), [security](#10-security-checklist) and
[troubleshooting](#11-troubleshooting).

> **Intended use.** The served model is trained on a semi-synthetic benchmark with injected
> faults. It is a reference implementation for research and demonstration and has not been
> validated on real elevator faults. Alarms are advisory; a qualified technician must confirm
> every maintenance decision.

## Conventions

* Commands are run from the repository root unless stated otherwise.
* `<PLACEHOLDER>` values must be replaced, including the angle brackets.
* "Local terminal" refers to the development machine; "server terminal" refers to an SSH session
  on the Linux server (Sections 4–5).

---

## 1. Local installation

### 1.1 Requirements

| Component | Version |
|---|---|
| Python | 3.11 or newer (the pinned NumPy 2.3 and scikit-learn 1.7 do not support 3.9/3.10) |
| Disk space | about 1 GB including results |
| Memory | 8 GB recommended for the full experiment run |
| Optional | Docker 24+ (Section 3), curl |

Check the installed Python version:

```bash
python3 --version
```

If the version is older than 3.11, install a newer one, for example from
<https://www.python.org/downloads/> or with a package manager (`brew install python@3.12` on
macOS, `sudo apt install python3.12 python3.12-venv` on Ubuntu). On macOS with the python.org
installer, run *Install Certificates.command* afterwards so that pip can verify HTTPS
connections. Use the versioned executable (for example `python3.12`) in the next step if
`python3` still points to an older version.

### 1.2 Get the code

```bash
git clone https://github.com/Aladra8/Predictive_AI_Maintenance_System.git
cd Predictive_AI_Maintenance_System
```

To use the exact version cited in the thesis: `git checkout v1.0-thesis`.

### 1.3 Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment (`source .venv/bin/activate`) in every new terminal session. Verify that
the environment's interpreter is used:

```bash
which python        # .../Predictive_AI_Maintenance_System/.venv/bin/python
python --version    # 3.11 or newer
```

If a Conda environment is active at the same time (prompt shows `(.venv) (base)`), the checks above
are sufficient; optionally run `conda deactivate` before activating `.venv`.

In VS Code, select the interpreter with **Python: Select Interpreter** → `./.venv/bin/python`.

### 1.4 Run the tests

```bash
python -m pytest -q
```

Expected result: `9 passed`.

### 1.5 Reproduce the experiments

```bash
python scripts/run_all.py
```

The run executes scripts 01–10 in order and takes 30–60 minutes on a laptop. Outputs:

| Path | Content |
|---|---|
| `results/<NN>_<name>/` | tables (`.csv`, `.md`, `.tex`), figures (`.png`) and `run_meta.json` per experiment |
| `results/SUMMARY.md` | headline tables of all experiments |
| `data/synthetic/semisynthetic_benchmark.csv.gz` | Experiment 3 benchmark (generated, not versioned) |
| `deploy/model/model.pkl`, `deploy/model/model_card.json` | service model and model card |

Individual steps can be run separately, for example:

```bash
python scripts/08_synthetic_benchmark.py      # regenerate the benchmark and its results
python scripts/09_train_service_model.py      # retrain the service model
```

Common options: `--config <file.yaml>`, `--models rf lr`, `--engine numpy` (dependency-free
reference engine for smoke tests; writes to `results_numpy/`).

---

## 2. Running the service locally

The repository ships a trained model in `deploy/model/`. The replay client additionally needs the
benchmark file; if `data/synthetic/semisynthetic_benchmark.csv.gz` does not exist, generate it with
`python scripts/08_synthetic_benchmark.py`.

### 2.1 Start the service

Set an API key (any string of your choice) and start the server:

```bash
export PDM_API_KEY=<API_KEY>
python deploy/service/app.py
```

The service logs one JSON line, `listening on 0.0.0.0:8080, model {...}`, and keeps running in
the foreground. Stop it with `Ctrl+C`. To use another port: `PDM_PORT=8081 python deploy/service/app.py`.

### 2.2 Verify the service

In a second terminal (with `.venv` activated):

```bash
curl http://localhost:8080/health                              # no key required
curl http://localhost:8080/model                               # rejected: missing key
curl -H "X-API-Key: <API_KEY>" http://localhost:8080/model     # model card
```

Score readings for one asset:

```bash
curl -X POST http://localhost:8080/score \
  -H "X-API-Key: <API_KEY>" -H "Content-Type: application/json" \
  -d '{"asset_id":"lift-A","readings":[
        {"ts":1700000000.00,"revolutions":55.1,"humidity":74.2,"vibration":21.3,"motor_current":2.10},
        {"ts":1700000000.25,"revolutions":55.1,"humidity":74.2,"vibration":21.4,"motor_current":2.11}]}'
```

The response contains `p_fault` and `alarm` for each reading. `"warming_up": true` indicates that
the history buffer does not yet cover the longest feature window (300 s plus margin), so the
rolling features are incomplete.

Input requirements: `ts` in Unix seconds, strictly increasing per `asset_id`; all four channels
(`revolutions`, `humidity`, `vibration`, `motor_current`) present in every reading.

### 2.3 Replay a recorded session

Sessions 13 (faulty) and 14 (healthy) are excluded from training and serve as hold-out data:

```bash
python deploy/replay_client.py --api-key <API_KEY> --session 13 --speed 50
python deploy/replay_client.py --api-key <API_KEY> --session 14 --speed 500
```

`--speed` is the replay acceleration (1 = real time). Expected behaviour with the reference model:
session 13 produces one event that opens about 11.6 minutes before the injected fault onset;
session 14 produces no event.

Each asset accepts only increasing timestamps. To replay the same session again, restart the
service or pass a different asset name (`--asset demo-2`).

Inspect state while the service runs:

```bash
curl -H "X-API-Key: <API_KEY>" "http://localhost:8080/events?asset_id=session-13"
curl -H "X-API-Key: <API_KEY>" http://localhost:8080/status
curl -H "X-API-Key: <API_KEY>" http://localhost:8080/metrics
```

Events are held in memory and are lost when the service stops; use the webhook (Section 6) to
persist them.

---

## 3. Running the service with Docker

Requires Docker Desktop (macOS/Windows) or Docker Engine (Linux). Verify with `docker version`.

### 3.1 Build the image

```bash
docker build -f deploy/Dockerfile -t elevator-pdm:1.0 .
```

The image contains Python 3.11, the runtime dependencies (`deploy/requirements-service.txt`), the
`pdm` library, the service and the model. It runs as a non-root user and defines a health check.

### 3.2 Configure

```bash
cp deploy/.env.example deploy/.env
```

Edit `deploy/.env` and set `PDM_API_KEY` (and optionally `PDM_WEBHOOK_URL`). The file is excluded
from version control.

### 3.3 Run

```bash
docker run --rm --name pdm -p 8080:8080 --env-file deploy/.env elevator-pdm:1.0
```

Test as in Section 2.2–2.3. Add `-d` to run in the background; then use `docker logs -f pdm` and
`docker stop pdm`. Stop any locally running service first, as both use port 8080.

---

## 4. Production deployment with Docker Compose and HTTPS

The Compose file starts two containers: the scoring service, bound to `127.0.0.1:8080` and
therefore not exposed directly, and a Caddy reverse proxy that terminates HTTPS on ports 80/443
with automatically issued certificates.

### 4.1 Server requirements

| Item | Recommendation |
|---|---|
| Operating system | Ubuntu 24.04 LTS (other recent Linux distributions work similarly) |
| Size | 1–2 vCPU, 2 GB RAM |
| Network | public IPv4 address; inbound 22, 80, 443 |
| DNS | a domain or subdomain with an A record pointing to the server (required for HTTPS) |
| Access | SSH key authentication |

Any cloud provider can be used. Create an SSH key locally if none exists and register the public
key with the provider when creating the server:

```bash
ls ~/.ssh/id_ed25519.pub 2>/dev/null || ssh-keygen -t ed25519 -C "<EMAIL>"
cat ~/.ssh/id_ed25519.pub
```

### 4.2 Prepare the server

Connect as root, update the system and create an administrative user:

```bash
ssh root@<SERVER_IP>
apt update && apt upgrade -y
adduser <USER>
usermod -aG sudo <USER>
rsync --archive --chown=<USER>:<USER> ~/.ssh /home/<USER>
```

Enable the firewall:

```bash
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status
exit
```

Reconnect as the new user: `ssh <USER>@<SERVER_IP>`.

### 4.3 Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
exit
```

Reconnect so that the group membership takes effect, then verify:

```bash
docker version
docker compose version
```

### 4.4 Deploy the code and the model

```bash
sudo apt install -y git
git clone https://github.com/Aladra8/Predictive_AI_Maintenance_System.git
cd Predictive_AI_Maintenance_System
ls deploy/model          # model.pkl and model_card.json
```

If the model was retrained locally, copy it from the local terminal:

```bash
scp deploy/model/model.pkl deploy/model/model_card.json \
    <USER>@<SERVER_IP>:~/Predictive_AI_Maintenance_System/deploy/model/
```

### 4.5 Configure secrets

On the server, in the repository:

```bash
cp deploy/.env.example deploy/.env
sed -i "s/^PDM_API_KEY=.*/PDM_API_KEY=$(openssl rand -hex 32)/" deploy/.env
chmod 600 deploy/.env
grep PDM_API_KEY deploy/.env
```

Store the generated key securely; clients need it in the `X-API-Key` header.

### 4.6 Configure the domain

Create a DNS A record, for example `pdm.<DOMAIN>` → `<SERVER_IP>`, and verify resolution with
`dig +short pdm.<DOMAIN>` (or `ping -c 3 pdm.<DOMAIN>`). Then set the domain in the Caddy
configuration:

```bash
sed -i "s/pdm.example.org/pdm.<DOMAIN>/" deploy/Caddyfile
```

### 4.7 Start the stack

```bash
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
```

Both services must be `running`; `pdm` reports `healthy` after about 30 seconds. Caddy obtains the
certificate within about a minute.

### 4.8 Verify

From the local terminal:

```bash
curl https://pdm.<DOMAIN>/health
curl -H "X-API-Key: <API_KEY>" https://pdm.<DOMAIN>/model
python deploy/replay_client.py --url https://pdm.<DOMAIN> --api-key <API_KEY> --session 13 --speed 200
```

### 4.9 Testing without a domain (optional)

For a short connectivity test without HTTPS (traffic is not encrypted):

```bash
docker build -f deploy/Dockerfile -t elevator-pdm:1.0 .
sudo ufw allow 8080/tcp
docker run -d --name pdm-test -p 8080:8080 --env-file deploy/.env elevator-pdm:1.0
curl http://<SERVER_IP>:8080/health          # from the local terminal
docker rm -f pdm-test && sudo ufw delete allow 8080/tcp
```

---

## 5. Production deployment with systemd

Alternative to Section 4 for servers without Docker. Complete Sections 4.1–4.2 first.

### 5.1 Install system packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
python3 --version        # 3.11 or newer (Ubuntu 24.04 ships 3.12)
```

### 5.2 Create the service account and install the application

```bash
sudo useradd --system --create-home --home-dir /opt/pdm --shell /usr/sbin/nologin pdm
sudo git clone https://github.com/Aladra8/Predictive_AI_Maintenance_System.git \
     /opt/pdm/Predictive_AI_Maintenance_System
sudo python3 -m venv /opt/pdm/venv
sudo /opt/pdm/venv/bin/pip install --upgrade pip
sudo /opt/pdm/venv/bin/pip install -r /opt/pdm/Predictive_AI_Maintenance_System/deploy/requirements-service.txt
sudo chown -R pdm:pdm /opt/pdm
```

### 5.3 Configure secrets

```bash
printf 'PDM_API_KEY=%s\nPDM_WEBHOOK_URL=\n' "$(openssl rand -hex 32)" | sudo tee /opt/pdm/pdm.env >/dev/null
sudo chown pdm:pdm /opt/pdm/pdm.env && sudo chmod 600 /opt/pdm/pdm.env
sudo grep PDM_API_KEY /opt/pdm/pdm.env
```

### 5.4 Install and start the unit

```bash
sudo cp /opt/pdm/Predictive_AI_Maintenance_System/deploy/systemd/pdm-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pdm-service
systemctl status pdm-service --no-pager
curl http://127.0.0.1:8080/health
```

The unit binds the service to `127.0.0.1:8080`, runs it as the unprivileged `pdm` user with a
read-only system and restarts it on failure. Logs: `sudo journalctl -u pdm-service -f`.

### 5.5 HTTPS with Caddy

Create the DNS record (Section 4.6), then install Caddy from its official repository:

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
printf 'pdm.<DOMAIN> {\n    reverse_proxy 127.0.0.1:8080\n}\n' | sudo tee /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Verify as in Section 4.8.

---

## 6. Webhook integration

If `PDM_WEBHOOK_URL` is set, each closed event is sent to that address as an HTTP POST with a JSON
body:

```json
{"asset_id": "lift-A", "start_ts": 1700003981.0, "end_ts": 1700005088.0,
 "duration_s": 1107.0, "n_alarms": 4298, "peak_score": 0.9993}
```

Typical targets are a CMMS work-order endpoint or an incoming-webhook URL of a messaging tool.
Delivery failures are logged and do not interrupt scoring; there is no retry queue, so the receiver
should be available or fronted by a queue.

Local test with a public request inspector (for example <https://webhook.site>):

```bash
export PDM_API_KEY=<API_KEY>
export PDM_WEBHOOK_URL=https://webhook.site/<ID>
python deploy/service/app.py
# second terminal:
python deploy/replay_client.py --api-key <API_KEY> --session 13 --speed 500
```

In production, set the variable in `deploy/.env` (Docker) or `/opt/pdm/pdm.env` (systemd) and
restart the service.

---

## 7. Configuration reference

### 7.1 Environment variables

| Variable | Default | Description |
|---|---|---|
| `PDM_API_KEY` | unset | Required key for all endpoints except `/health` (header `X-API-Key`). If unset, authentication is disabled; always set it outside local development. |
| `PDM_WEBHOOK_URL` | unset | Address that receives each closed event as JSON |
| `PDM_MODEL_DIR` | `deploy/model` | Folder containing `model.pkl` and `model_card.json` |
| `PDM_HOST` | `0.0.0.0` | Listening address |
| `PDM_PORT` | `8080` | Listening port |

### 7.2 Model and event parameters

Set in `config/default.yaml` (section `deploy`) and stored with the model by
`scripts/09_train_service_model.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `model` | `rf` | Random Forest, 400 trees, balanced class weights |
| `target` | `precursor_or_fault` | alarms start during degradation (early warning) |
| `channels` | revolutions, humidity, vibration, motor_current | required input fields |
| `alarm_threshold` | 0.5 | probability at or above which a reading is an alarm |
| `event_gap_s` | 60 | alarms closer than this belong to one event |
| `min_event_alarms` | 3 | events with fewer alarms are discarded |
| `holdout_sessions` | 13, 14 | sessions excluded from training |

Changing these values requires retraining the model (`python scripts/09_train_service_model.py`).

---

## 8. API reference

| Method | Endpoint | Authentication | Description |
|---|---|---|---|
| GET | `/health` | none | liveness, uptime and model summary |
| GET | `/model` | key | model card |
| POST | `/score` | key | body `{"asset_id": str, "readings": [...]}`; returns per-reading `p_fault` and `alarm`, `closed_events`, `open_event`, `warming_up` |
| POST | `/flush` | key | body `{"asset_id": str}`; closes the open event at the end of a stream |
| GET | `/events[?asset_id=]` | key | closed events |
| GET | `/status` | key | per-asset counters |
| GET | `/metrics` | key | counters in Prometheus text format |

Errors are returned as JSON `{"error": "..."}` with status 400 (invalid input), 401 (missing or
invalid key) or 404 (unknown endpoint).

---

## 9. Operations

### 9.1 Routine commands (Docker Compose)

```bash
cd ~/Predictive_AI_Maintenance_System
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs -f pdm
docker compose -f deploy/docker-compose.yml restart pdm
docker compose -f deploy/docker-compose.yml down
docker compose -f deploy/docker-compose.yml up -d
```

Both containers use `restart: unless-stopped` and start automatically after a reboot.

### 9.2 Updating the model or the code

1. Retrain locally (`python scripts/09_train_service_model.py`) and review
   `deploy/model/model_card.json`.
2. Back up the current model on the server: `cp -r deploy/model deploy/model_backup_$(date +%F)`.
3. Transfer the new version (`git pull`, or `scp` of the two model files).
4. Rebuild and restart:
   * Docker Compose: `docker compose -f deploy/docker-compose.yml up -d --build`
   * systemd: `sudo -u pdm git -C /opt/pdm/Predictive_AI_Maintenance_System pull && sudo systemctl restart pdm-service`

Rollback: restore the backup folder into `deploy/model` and repeat step 4.

The scikit-learn version used for training must match the one installed in the service
environment (pinned to 1.7.1 in `deploy/requirements-service.txt`).

### 9.3 Decommissioning

Stop the stack (`docker compose ... down` or `sudo systemctl disable --now pdm-service`), delete the
server in the provider's console and remove the DNS record.

---

## 10. Security checklist

- [ ] SSH access by key only; password login disabled
      (`PasswordAuthentication no` in `/etc/ssh/sshd_config`, then `sudo systemctl restart ssh`).
- [ ] Firewall allows only 22, 80 and 443 (`sudo ufw status`).
- [ ] `PDM_API_KEY` is long and random; secret files have mode 600 and are not under version control.
- [ ] The service port 8080 is bound to 127.0.0.1 and reachable only through the reverse proxy.
- [ ] Operating-system updates are applied regularly (`sudo apt update && sudo apt upgrade -y`).
- [ ] Alarms are reviewed by a qualified technician. AI systems used as safety components of lifts
      are high-risk under the EU AI Act (Regulation (EU) 2024/1689, Annex I, Lifts Directive
      2014/33/EU); the model card documents the intended use and limitations.

---

## 11. Troubleshooting

| Symptom | Cause | Resolution |
|---|---|---|
| `Could not find a version that satisfies the requirement numpy==2.3.2` | Python older than 3.11 | install Python 3.11+, remove `.venv`, repeat Section 1.3 |
| `command not found: python` | virtual environment not active | `source .venv/bin/activate` |
| `ModuleNotFoundError: No module named 'pdm'` | command run outside the repository root | `cd` into the repository |
| `No such file or directory: .../model.pkl` | model missing | `python scripts/09_train_service_model.py` |
| Replay client cannot find `semisynthetic_benchmark.csv.gz` | benchmark not generated | `python scripts/08_synthetic_benchmark.py` |
| `Address already in use` | port 8080 occupied | `lsof -i :8080` and stop the process, or set `PDM_PORT` |
| `401 invalid or missing X-API-Key` | key absent or different | use the same value as `PDM_API_KEY` |
| `400 timestamps must be increasing per asset` | session replayed to the same asset | restart the service or use `--asset <NEW_NAME>` |
| `400 missing fields` | channel missing in a reading | include `ts` and all four channels |
| Error while loading the model | scikit-learn version mismatch | install the pinned versions or retrain |
| No HTTPS certificate | DNS not yet propagated, or ports 80/443 closed | check `dig +short pdm.<DOMAIN>`, `sudo ufw status`, `docker compose ... logs caddy` |
| `permission denied ... docker.sock` | user not in the `docker` group | `sudo usermod -aG docker $USER`, then reconnect |
