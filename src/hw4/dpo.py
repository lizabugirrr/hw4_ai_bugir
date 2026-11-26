"""DPO on the student's own mistakes, with typical_errors used to pick the rejected answers.

1) sample the (SFT) student on the TRAIN side and keep every sample:
   python -m hw4.infer --model weights/sft_mamay_pot --strategy pot0 --splits train --n 8 --temperature 0.8 \
       --save-all --name samples_sft_mamay_train
2) python -m hw4.dpo --model weights/sft_mamay_pot --samples samples_sft_mamay_train \
       --teacher-data runs/data_qwen_pot_train/sft.jsonl --name dpo_mamay_pot

Pairs: chosen = a correct sample of the student (else the verified teacher solution); rejected = a wrong
sample, preferring one whose value equals a typical_errors misconception of that problem (the known
mistakes the homework suggests to push away from). Problems the student always solves give no pair.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from .answers import canonical, extract_answer
from .data import load_split
from .prompts import build
from .runs import RUNS, load_preds
from .sandbox import run_many
from .train_utils import WEIGHTS, bf16_ok, make_config, warmup, load_student, lora_config, save_adapter_and_merged


def make_pairs(samples: str, strategy: str, teacher_data: str | None, max_pairs_per_problem: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pr = load_preds(samples, "train")
    probs = load_split("train").set_index("id")
    teacher = {}
    if teacher_data:
        for l in open(teacher_data):
            r = json.loads(l)
            teacher.setdefault(r["id"], r["completion"][0]["content"])
    outs = [json.loads(o) for o in pr["outputs"]]
    flat = [t for o in outs for t in o]
    vals = [v for v, _ in run_many(flat)] if strategy.startswith("pot") else [extract_answer(t) for t in flat]
    pairs, i = [], 0
    for r, o in zip(pr.itertuples(), outs):
        p = probs.loc[r.id]
        v = [canonical(x, p["statement"]) for x in vals[i:i + len(o)]]
        i += len(o)
        good = [t for t, x in zip(o, v) if x == p["answer"]]
        bad = [(t, x) for t, x in zip(o, v) if x != p["answer"]]
        if not bad:
            continue
        typical = {str(e.get("value")) for e in p["typical_errors"]}
        bad.sort(key=lambda tx: (tx[1] not in typical, tx[1] is None))  # typical misconceptions first
        chosen = good or ([teacher[r.id]] if r.id in teacher else [])
        if not chosen:
            continue
        prompt = build(strategy, {**p.to_dict(), "id": r.id})
        for t, x in bad[:max_pairs_per_problem]:
            pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": rng.choice(chosen)}],
                          "rejected": [{"role": "assistant", "content": t}], "id": r.id, "skill": p["skill"],
                          "rejected_typical": x in typical})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", required=True, help="infer run on the train split with --save-all")
    ap.add_argument("--strategy", default="pot0")
    ap.add_argument("--teacher-data", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pairs-per-problem", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    pairs = make_pairs(a.samples, a.strategy, a.teacher_data, a.pairs_per_problem, a.seed)
    out = Path(RUNS) / a.name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs)[["id", "skill", "rejected_typical"]].to_csv(out / "pairs.csv", index=False)
    print(f"{len(pairs)} pairs, {sum(p['rejected_typical'] for p in pairs)} with a typical-error rejected answer", flush=True)
    ds = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    model, tok, base_dir = load_student(a.model)
    # TRL 0.2x treats any gemma3 config as an image-text model and expects images; we train on text only
    import trl.trainer.dpo_trainer as dpo_mod
    if hasattr(dpo_mod, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"):
        dpo_mod.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = {
            k: v for k, v in dpo_mod.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.items() if k != model.config.model_type}
    cfg = make_config(DPOConfig, output_dir=str(out / "trainer"), beta=a.beta, learning_rate=a.lr, num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum, bf16=bf16_ok(),
                    max_length=1024, gradient_checkpointing=True, logging_steps=10, save_strategy="no",
                    report_to="none", seed=a.seed, lr_scheduler_type="cosine", **warmup(DPOConfig, 0.05))
    tr = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(a.rank))
    t0 = time.time()
    res = tr.train()
    save_adapter_and_merged(tr, tok, base_dir, out / "adapter", WEIGHTS / a.name)
    (out / "config.json").write_text(json.dumps({**vars(a), "n_pairs": len(pairs)}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"train_loss": res.training_loss, "train_min": round((time.time() - t0) / 60, 1),
                                                  "log": tr.state.log_history}, indent=1))


if __name__ == "__main__":
    main()
