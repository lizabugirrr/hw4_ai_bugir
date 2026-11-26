"""GRPO with verifiable rewards (bonus): the student writes code, the reward is 1 if running it gives the answer.

python -m hw4.grpo --model weights/sft_mamay_pot --synth runs/data_synth/problems.jsonl --max-steps 300 --name grpo_mamay_pot

Problems: the TRAIN side of the split + synthetic problems (never val/test). Rewards: correctness (1.0) and a
small format reward (0.1 for a ```python block that sets `answer`). LoRA, bf16; generation with vLLM in
colocate mode when available (A100 recommended), otherwise plain transformers generation (--no-vllm).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .answers import canonical
from .data import load_problems
from .prompts import build
from .runs import RUNS
from .sandbox import extract_code, run_many
from .train_utils import WEIGHTS, bf16_ok, make_config, warmup, load_student, lora_config, save_adapter_and_merged


def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


def reward_correct(completions, answer, statement, **_):
    vals = run_many([_text(c) for c in completions])
    return [1.0 if canonical(v, s) == ans else 0.0 for (v, _), s, ans in zip(vals, statement, answer)]


def reward_format(completions, **_):
    out = []
    for c in completions:
        code = extract_code(_text(c))
        out.append(0.1 if code and "answer" in code and "```python" in _text(c) else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--synth", default=None, help="synthetic problems jsonl (hw4.synth)")
    ap.add_argument("--synth-per-skill", type=int, default=60)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16, help="completions per device step (multiple of num-generations)")
    ap.add_argument("--max-completion", type=int, default=320)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    df = load_problems("train")
    if a.synth:
        syn = load_problems(a.synth).sample(frac=1, random_state=a.seed).groupby("skill").head(a.synth_per_skill)
        df = pd.concat([df, syn], ignore_index=True)
    df = df.sample(frac=1, random_state=a.seed)
    ds = Dataset.from_list([{"prompt": build("pot0", r), "answer": r["answer"], "statement": r["statement"]}
                            for _, r in df.iterrows()])
    print(f"{len(ds)} RL problems", flush=True)
    model, tok, base_dir = load_student(a.model)
    kw = dict(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3) if not a.no_vllm else {}
    out = Path(RUNS) / a.name
    cfg = make_config(GRPOConfig, output_dir=str(out / "trainer"), learning_rate=a.lr, max_steps=a.max_steps, beta=a.beta,
                     num_generations=a.num_generations, per_device_train_batch_size=a.bs, gradient_accumulation_steps=1,
                     max_completion_length=a.max_completion, max_prompt_length=512, temperature=1.0, bf16=bf16_ok(),
                     gradient_checkpointing=True, logging_steps=5, save_strategy="no", report_to="none",
                     seed=a.seed, lr_scheduler_type="constant_with_warmup", **warmup(GRPOConfig, 0.03), **kw)
    tr = GRPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                     reward_funcs=[reward_correct, reward_format], peft_config=lora_config(a.rank))
    t0 = time.time()
    res = tr.train()
    save_adapter_and_merged(tr, tok, base_dir, out / "adapter", WEIGHTS / a.name)
    (out / "config.json").write_text(json.dumps({**vars(a), "n_problems": len(ds)}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"train_loss": res.training_loss, "train_min": round((time.time() - t0) / 60, 1),
                                                  "log": tr.state.log_history}, indent=1))


if __name__ == "__main__":
    main()
