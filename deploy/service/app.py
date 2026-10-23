"""HTTP scoring service for elevator-door predictive maintenance (standard library only).

Endpoints
  GET  /health                     liveness + model info
  GET  /model                      model card
  POST /score                      {"asset_id": "...", "readings": [{"ts":..., "revolutions":..., ...}]}
  POST /flush                      {"asset_id": "..."}  close the open event (end of stream)
  GET  /events[?asset_id=...]      closed maintenance events
  GET  /status                     per-asset counters
  GET  /metrics                    Prometheus text format

Environment variables
  PDM_MODEL_DIR   folder with model.pkl and model_card.json   (default deploy/model)
  PDM_HOST/PDM_PORT                                          (default 0.0.0.0:8080)
  PDM_API_KEY     if set, every request needs header  X-API-Key: <key>
  PDM_WEBHOOK_URL if set, each closed event is POSTed there as JSON (e.g. a CMMS ticket endpoint)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer import Scorer  # noqa: E402

MODEL_DIR = Path(os.environ.get("PDM_MODEL_DIR", ROOT / "deploy" / "model"))
API_KEY = os.environ.get("PDM_API_KEY")
WEBHOOK = os.environ.get("PDM_WEBHOOK_URL")
MAX_BODY = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}')
log = logging.getLogger("pdm")
SCORER = Scorer(MODEL_DIR / "model.pkl")
CARD = json.loads((MODEL_DIR / "model_card.json").read_text()) if (MODEL_DIR / "model_card.json").exists() else {}
COUNTERS = {"requests": 0, "readings": 0, "alarms": 0, "events_closed": 0, "errors": 0}
STARTED = time.time()


def post_webhook(events):
    if not WEBHOOK or not events:
        return
    def _send():
        for e in events:
            try:
                req = urllib.request.Request(WEBHOOK, data=json.dumps(e).encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5).read()
            except Exception as ex:  # never let the webhook break scoring
                log.warning(json.dumps(f"webhook failed: {ex}"))
    threading.Thread(target=_send, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "pdm-service/1.0"

    def log_message(self, fmt, *args):
        log.info(json.dumps(f"{self.address_string()} {fmt % args}"))

    def _send(self, code, obj, ctype="application/json"):
        body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _auth(self):
        if API_KEY and self.headers.get("X-API-Key") != API_KEY:
            self._send(401, {"error": "invalid or missing X-API-Key"}); return False
        return True

    def _json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        COUNTERS["requests"] += 1
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"status": "ok", "uptime_s": round(time.time() - STARTED, 1), "model": SCORER.meta})
        if not self._auth():
            return
        if u.path == "/model":
            return self._send(200, CARD)
        if u.path == "/events":
            a = parse_qs(u.query).get("asset_id", [None])[0]
            return self._send(200, {"events": SCORER.events(a)})
        if u.path == "/status":
            return self._send(200, SCORER.status())
        if u.path == "/metrics":
            lines = [f"pdm_{k}_total {v}" for k, v in COUNTERS.items()] + [f"pdm_assets {len(SCORER.assets)}"]
            return self._send(200, "\n".join(lines) + "\n", "text/plain; version=0.0.4")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        COUNTERS["requests"] += 1
        if not self._auth():
            return
        try:
            body = self._json()
            aid = str(body.get("asset_id", "")).strip()
            if not aid:
                raise ValueError("asset_id is required")
            if self.path == "/score":
                res = SCORER.score(aid, body.get("readings", []))
                COUNTERS["readings"] += len(res["scores"])
                COUNTERS["alarms"] += sum(s["alarm"] for s in res["scores"])
                COUNTERS["events_closed"] += len(res["closed_events"])
                post_webhook(res["closed_events"])
                return self._send(200, res)
            if self.path == "/flush":
                ev = SCORER.flush(aid); COUNTERS["events_closed"] += len(ev); post_webhook(ev)
                return self._send(200, {"closed_events": ev})
            self._send(404, {"error": "not found"})
        except (ValueError, KeyError, json.JSONDecodeError) as ex:
            COUNTERS["errors"] += 1
            self._send(400, {"error": str(ex)})


def main():
    host = os.environ.get("PDM_HOST", "0.0.0.0"); port = int(os.environ.get("PDM_PORT", "8080"))
    srv = ThreadingHTTPServer((host, port), Handler)
    log.info(json.dumps(f"listening on {host}:{port}, model {SCORER.meta}"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
