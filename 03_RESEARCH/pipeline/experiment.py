"""
Experiment tracking.

Each run writes one self-contained JSON file to experiments/. Files are never
overwritten — a run whose id already exists gets a numeric suffix — so the record
of what was actually executed can only grow.

Every metric stored here comes from an executed run. Nothing is filled in by hand.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config


class Experiment:
    """
    Usage:
        exp = Experiment("model_01_lightgbm", model_type="LightGBM",
                         objective="regression")
        exp.set(feature_set="base", feature_groups=["A", "B"])
        ... run ...
        exp.set_metrics(rmse=..., mae=...)
        exp.save()
    """

    def __init__(self, name: str, **fields):
        self.name = name
        self.record: dict = {
            "experiment_name": name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "random_seed": config.RANDOM_SEED,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "status": "running",
            "warnings": [],
            "errors": [],
            "notes": [],
        }
        self.record.update(fields)
        self._t0 = time.time()

    # ------------------------------------------------------------------

    def set(self, **fields) -> "Experiment":
        self.record.update(fields)
        return self

    def note(self, text: str) -> "Experiment":
        self.record["notes"].append(text)
        return self

    def warn(self, text: str) -> "Experiment":
        self.record["warnings"].append(text)
        print(f"      WARNING: {text}")
        return self

    def error(self, text: str) -> "Experiment":
        self.record["errors"].append(text)
        self.record["status"] = "failed"
        print(f"      ERROR: {text}")
        return self

    def set_metrics(self, **metrics) -> "Experiment":
        self.record.setdefault("metrics", {}).update(metrics)
        return self

    # ------------------------------------------------------------------

    def save(self, status: str = "completed") -> Path:
        if self.record["status"] != "failed":
            self.record["status"] = status
        self.record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.record["wall_seconds"] = round(time.time() - self._t0, 1)

        path = config.EXPERIMENTS_DIR / f"{self.name}.json"
        n = 1
        while path.exists():
            path = config.EXPERIMENTS_DIR / f"{self.name}__run{n}.json"
            n += 1

        path.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
        print(f"      -> experiments/{path.name}")
        return path


def load_all() -> list[dict]:
    """Every experiment record on disk, newest last."""
    out = []
    for p in sorted(config.EXPERIMENTS_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def load(name: str) -> dict | None:
    p = config.EXPERIMENTS_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
