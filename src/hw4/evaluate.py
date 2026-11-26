"""Accuracy overall and per skill; error breakdown against typical_errors."""
from __future__ import annotations

import pandas as pd


def accuracy(preds: pd.DataFrame) -> dict:
    """preds: columns skill, pred, answer."""
    ok = preds["pred"].astype(str) == preds["answer"].astype(str)
    return {"acc": float(ok.mean()), "n": int(len(ok)), "n_none": int(preds["pred"].isna().sum()),
            "per_skill": {s: float(v) for s, v in ok.groupby(preds["skill"]).mean().sort_values().items()}}


def typical_error_share(preds: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """For wrong predictions: which typical_errors misconception (if any) the wrong value matches."""
    te = val.set_index("id")["typical_errors"]
    rows = []
    for r in preds[preds["pred"].astype(str) != preds["answer"].astype(str)].itertuples():
        match = next((e["id"] for e in te.get(r.id, []) if str(e.get("value")) == str(r.pred)), None)
        rows.append({"id": r.id, "skill": r.skill, "pred": r.pred, "answer": r.answer,
                     "typical_error": match or ("no answer" if r.pred is None else "other")})
    return pd.DataFrame(rows)
