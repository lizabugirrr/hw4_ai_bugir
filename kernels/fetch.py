"""Downloads a finished kernel's runs/ into ./runs and rebuilds results/experiments.csv.

Usage: KAGGLE_USER=<user> python kernels/fetch.py hw4-xlmr [hw4-mdeberta ...]
Model weights are skipped unless --weights is given.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from hw4.runs import table  # noqa: E402

USER = os.environ.get("KAGGLE_USER") or ""


def fetch(slug: str, weights: bool):
    tmp = Path(tempfile.mkdtemp(prefix=f"{slug}-"))
    cmd = ["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(tmp), "-o"]
    cmd += [] if weights else ["--file-pattern", r"^(?!.*(\.safetensors|\.bin)$).*"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    for run in sorted((tmp / "runs").glob("*")):
        dst = ROOT / "runs" / run.name
        shutil.copytree(run, dst, dirs_exist_ok=True)
        (dst / "kernel.txt").write_text(slug + "\n")
        print("fetched", run.name)
    if (tmp / "logs").exists():
        shutil.copytree(tmp / "logs", ROOT / "runs" / "_logs" / slug, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--weights", action="store_true")
    a = ap.parse_args()
    if a.slugs and not USER:
        raise SystemExit("set KAGGLE_USER")
    for s in a.slugs:
        fetch(s, a.weights)
    df = table()
    cols = [c for c in ["run", "model", "strategy", "n", "val_acc", "val_none", "val_min"] if c in df]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
