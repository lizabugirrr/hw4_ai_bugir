"""LoRA SFT of Gemma-3-4B / MamayLM-4B on verified teacher solutions (bf16, one L4/A100).

python -m hw4.sft --model mamay --data runs/data_qwen_pot_train/sft.jsonl runs/data_qwen_pot_synth/sft.jsonl \
    --name sft_mamay_pot
-> runs/<name>/adapter (LoRA, published on HF), weights/<name> (merged, for vLLM), runs/<name>/{config,metrics}.json
Then: python -m hw4.infer --model weights/<name> --strategy pot0 --fallback cot --splits val test --name <name>_eval
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from .runs import RUNS
from .train_utils import WEIGHTS, bf16_ok, make_config, warmup, load_student, lora_config, save_adapter_and_merged


def read_jsonl(paths: list[str], max_per_skill: int | None, seed: int) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    random.Random(seed).shuffle(rows)
    if max_per_skill:
        cnt, kept = Counter(), []
        for r in rows:
            if cnt[r["skill"]] < max_per_skill:
                cnt[r["skill"]] += 1
                kept.append(r)
        rows = kept
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gemma | mamay | path")
    ap.add_argument("--data", nargs="+", required=True, help="sft.jsonl files from hw4.teacher")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-per-skill", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--backend", choices=["peft", "unsloth"], default="peft",
                    help="unsloth: 4-bit QLoRA in fp16 for T4 (no bf16); peft: bf16 LoRA for L4/A100")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = read_jsonl(a.data, a.max_per_skill, a.seed)
    if a.backend == "unsloth":
        return train_unsloth(a, rows)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    print(f"{len(ds)} training examples", flush=True)
    model, tok, base_dir = load_student(a.model)
    out = Path(RUNS) / a.name
    cfg = make_config(SFTConfig, output_dir=str(out / "trainer"), num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, lr_scheduler_type="cosine",
                    **warmup(SFTConfig, 0.03), logging_steps=10, bf16=bf16_ok(), max_length=a.max_length,
                    completion_only_loss=True, gradient_checkpointing=True, save_strategy="no",
                    report_to="none", seed=a.seed)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config(a.rank))
    t0 = time.time()
    res = tr.train()
    save_adapter_and_merged(tr, tok, base_dir, out / "adapter", None if a.no_merge else WEIGHTS / a.name)
    (out / "config.json").write_text(json.dumps({**vars(a), "base_dir": base_dir, "n_train": len(ds)}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"train_loss": res.training_loss, "train_min": round((time.time() - t0) / 60, 1),
                                                  "log": tr.state.log_history}, indent=1))
    print(f"{a.name}: loss {res.training_loss:.4f}", flush=True)


def train_unsloth(a, rows):
    """Unsloth path for Kaggle T4: Gemma 3 in 4-bit with fp16 compute (Unsloth keeps Gemma 3 stable in fp16),
    loss only on the model turn (train_on_responses_only), merged 16-bit weights saved for vLLM."""
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from .train_utils import local_dir
    base_dir = local_dir(a.model)
    model, tok = FastModel.from_pretrained(base_dir, max_seq_length=a.max_length, load_in_4bit=True,
                                           full_finetuning=False)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True, r=a.rank,
                                     lora_alpha=2 * a.rank, lora_dropout=0, bias="none", random_state=a.seed)
    text_tok = getattr(tok, "tokenizer", tok)
    texts = [text_tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False).removeprefix("<bos>")
             for r in rows]
    ds = Dataset.from_dict({"text": texts})
    print(f"{len(ds)} training examples (unsloth)", flush=True)
    out = Path(RUNS) / a.name
    cfg = make_config(SFTConfig, output_dir=str(out / "trainer"), dataset_text_field="text", num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
                    lr_scheduler_type="cosine", **warmup(SFTConfig, 0.03), logging_steps=10, max_length=a.max_length,
                    optim="adamw_8bit", weight_decay=0.01, save_strategy="no", report_to="none", seed=a.seed,
                    dataset_num_proc=1)
    tr = SFTTrainer(model=model, tokenizer=text_tok, train_dataset=ds, args=cfg)
    tr = train_on_responses_only(tr, instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")
    t0 = time.time()
    res = tr.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if not a.no_merge:
        model.save_pretrained_merged(str(WEIGHTS / a.name), tok, save_method="merged_16bit")
        print("merged model ->", WEIGHTS / a.name, flush=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "base_dir": base_dir, "n_train": len(ds)}, indent=1))
    (out / "metrics.json").write_text(json.dumps({"train_loss": res.training_loss, "train_min": round((time.time() - t0) / 60, 1),
                                                  "log": tr.state.log_history}, indent=1))
    print(f"{a.name}: loss {res.training_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
