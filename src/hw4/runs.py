"""Saving and loading a run: runs/<name>/{config.json, metrics.json, <split>_preds.parquet}.

`python -m hw4.runs` rebuilds results/experiments.csv from every runs/*/metrics.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import ROOT

RUNS = Path(os.environ.get("HW4_RUNS", ROOT / "runs"))


def save_run(name: str, config: dict, metrics: dict, preds: dict[str, pd.DataFrame], runs_dir: Path = RUNS) -> Path:
    d = Path(runs_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config, indent=1, default=str, ensure_ascii=False))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float, ensure_ascii=False))
    for split, df in preds.items():
        df.to_parquet(d / f"{split}_preds.parquet", index=False)
    return d


def load_preds(name: str, split: str, runs_dir: Path = RUNS) -> pd.DataFrame:
    return pd.read_parquet(Path(runs_dir) / name / f"{split}_preds.parquet")


def table(runs_dir: Path = RUNS) -> pd.DataFrame:
    rows = []
    for m in sorted(Path(runs_dir).glob("*/metrics.json")):
        met = json.loads(m.read_text())
        cfg = json.loads((m.parent / "config.json").read_text())
        rows.append({"run": m.parent.name, "model": cfg.get("model"), "strategy": cfg.get("strategy"),
                     "n": cfg.get("n"), **{k: v for k, v in met.items() if isinstance(v, (int, float, str))}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = table()
    out = ROOT / "results" / "experiments.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
