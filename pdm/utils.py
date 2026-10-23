"""Config loading, output folders and table writers."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def out_dir(name: str, cfg: dict, uses_models: bool = True) -> Path:
    # sklearn runs (the thesis numbers) go to results/, numpy reference runs to results_numpy/
    base = "results" if (not uses_models or cfg["models"]["engine"] == "sklearn") else "results_numpy"
    d = ROOT / base / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": name, "created": datetime.now().isoformat(timespec="seconds"),
            "git": git_hash(), "python": platform.python_version(),
            "engine": cfg["models"]["engine"], "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return d


def save_table(df: pd.DataFrame, path: Path, floatfmt: str = "{:.4f}") -> None:
    """Write CSV + Markdown + LaTeX versions of a table (same numbers everywhere)."""
    path = Path(path)
    df.to_csv(path.with_suffix(".csv"), index=False)
    fmt = df.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, fmt.columns)) + " |"
    sep = "|" + "|".join("---" for _ in fmt.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in fmt.astype(str).values]
    path.with_suffix(".md").write_text("\n".join([head, sep, *body]) + "\n")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("^", r"\^{}")
    tex = [r"\begin{tabular}{" + "l" * len(fmt.columns) + "}", r"\toprule",
           " & ".join(esc(c) for c in fmt.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(v) for v in r) + r" \\" for r in fmt.astype(str).values]
    tex += [r"\bottomrule", r"\end{tabular}"]
    path.with_suffix(".tex").write_text("\n".join(tex) + "\n")


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
