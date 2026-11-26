"""Data loading and the fixed validation split.

`python -m hw4.data` writes configs/split.csv once (id, split in {train, val}): 30% of train.csv,
stratified by skill, seed 42. Every comparison in the report uses this val split; few-shot examples
and fine-tuning data come only from the train side.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from . import CONFIGS, DATA

SPLIT = CONFIGS / "split.csv"
VAL_FRAC, SEED = 0.30, 42


def load(name: str) -> pd.DataFrame:
    """name: train | test | sample_submission. All columns as strings."""
    df = pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False)
    for col in ("steps", "hints", "typical_errors"):
        if col in df:
            df[col] = df[col].map(lambda s: json.loads(s) if s else [])
    return df


def make_split() -> pd.DataFrame:
    tr = load("train")
    _, val_ids = train_test_split(tr["id"], test_size=VAL_FRAC, stratify=tr["skill"], random_state=SEED)
    split = pd.DataFrame({"id": tr["id"], "split": "train"})
    split.loc[split["id"].isin(set(val_ids)), "split"] = "val"
    CONFIGS.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT, index=False)
    return split


def load_split(which: str) -> pd.DataFrame:
    """which: train | val | test. train/val are the two sides of train.csv."""
    if which == "test":
        return load("test")
    tr = load("train")
    split = pd.read_csv(SPLIT, dtype=str)
    ids = set(split.loc[split["split"] == which, "id"])
    return tr[tr["id"].isin(ids)].reset_index(drop=True)


def load_problems(which: str) -> pd.DataFrame:
    """A split name (train | val | test) or a .jsonl file of problems (e.g. synthetic ones)."""
    if which.endswith(".jsonl"):
        return pd.read_json(which, lines=True, dtype={"id": str, "answer": str})
    return load_split(which)


if __name__ == "__main__":
    s = make_split()
    print(s["split"].value_counts().to_string())
