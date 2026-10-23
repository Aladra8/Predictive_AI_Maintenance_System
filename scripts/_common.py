"""Shared command-line handling for the numbered experiment scripts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

from pdm.utils import load_config  # noqa: E402


def setup(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    ap.add_argument("--engine", default=None, choices=["sklearn", "numpy"],
                    help="model engine (default from config: sklearn)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models, e.g. rf lr")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.engine:
        cfg["models"]["engine"] = args.engine
    if args.models:
        cfg["models"]["list"] = args.models
    if cfg["models"]["engine"] == "numpy":
        cfg["models"]["list"] = [m for m in cfg["models"]["list"] if m in ("rf", "lr")]
    return cfg, args


def path(cfg, key):
    return ROOT / cfg["data"][key]
