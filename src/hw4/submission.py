"""Predictions -> Kaggle submission CSV (id,answer) in sample_submission order, with checks.

python -m hw4.submission --run <run name> --out submissions/<file>.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from .data import load
from .runs import load_preds

N_TEST = 3561
CANON = re.compile(r"-?\d+|-?\d+\.\d*[1-9]|-?\d+/\d+")


def to_submission(preds: pd.DataFrame, path: str | None = None) -> pd.DataFrame:
    sample = load("sample_submission")
    p = preds.set_index("id")["pred"]
    assert p.index.is_unique, "duplicate ids"
    sub = pd.DataFrame({"id": sample["id"]})
    sub["answer"] = sub["id"].map(p).fillna("0").astype(str)
    check(sub, sample)
    if path:
        sub.to_csv(path, index=False)
    return sub


def check(sub: pd.DataFrame, sample: pd.DataFrame):
    assert list(sub.columns) == ["id", "answer"]
    assert len(sub) == N_TEST, len(sub)
    assert (sub["id"].values == sample["id"].values).all(), "order differs from sample"
    bad = sub[~sub["answer"].map(lambda a: bool(CANON.fullmatch(a)))]
    assert bad.empty, bad.head()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = to_submission(load_preds(a.run, "test"), a.out)
    print(f"wrote {a.out}: {len(s)} rows, {(s['answer'] == '0').sum()} zeros")
