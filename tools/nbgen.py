"""Tiny helper that writes the notebooks in notebooks/ from tools/nb_*.py (cells as (kind, source)).

python tools/nb_eda.py && cd notebooks && ../safe_run.sh ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
"""
import json
from pathlib import Path

M, C = "markdown", "code"
OUT = Path(__file__).resolve().parent.parent / "notebooks"
HEADER = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hw4.data import load, load_split
IMG = Path.cwd().parent / "images"
RUNS = Path.cwd().parent / "runs"
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 90)
sns.set_theme(style="whitegrid", font_scale=0.9)'''


def write(name, cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == C:
            c.update(execution_count=None, outputs=[])
        out.append(c)
    nb = {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", name)
