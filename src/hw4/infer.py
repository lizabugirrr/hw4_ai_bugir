"""Batch inference with vLLM: one model x one prompting strategy on val and/or test.

python -m hw4.infer --model mamay --strategy cot --splits val test --name mamay_cot
python -m hw4.infer --model gemma --strategy pot --n 5 --temperature 0.7 --name gemma_pot_sc5
python -m hw4.infer --model mamay --lora runs/sft_mamay/adapter --strategy pot --name sft_pot
python -m hw4.infer --backend llamacpp --model mamay --strategy cot --name local_mamay_cot   # laptop, 4-bit GGUF

--n > 1 is self-consistency: majority vote over the normalized answers of n samples.
--fallback cot: for problems where no sample gives a number (e.g. code that crashes), ask again with cot.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from . import MODELS
from .answers import canonical, extract_answer
from .data import load_problems, load_split
from .evaluate import accuracy
from .prompts import STRATEGIES, build
from .runs import save_run
from .sandbox import run_many


def resolve_model(name: str) -> str:
    """Short name -> local Kaggle Models path (if attached) or HF id; anything else is used as is."""
    if name not in MODELS:
        return name
    hf, variation = MODELS[name]
    local = os.environ.get(f"HW4_MODEL_{name.upper()}")  # e.g. a kagglehub download on Colab
    if local:
        return local
    if variation:
        hits = sorted(glob.glob(f"/kaggle/input/**/{variation}/**/config.json", recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    return hf


def hardware_defaults(a):
    """dtype/tp 'auto': bf16 on one GPU where supported (L4, A100); else fp32 over all GPUs (T4 x2)."""
    import torch
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8  # is_bf16_supported() is True on T4 (emulated)
    if a.dtype == "auto":
        a.dtype = "bfloat16" if bf16 else "float32"
    if a.tp == 0:
        a.tp = 1 if a.dtype != "float32" else torch.cuda.device_count()


def make_llm(a):
    if a.backend == "llamacpp":
        return None
    from vllm import LLM
    hardware_defaults(a)
    kw = dict(model=resolve_model(a.model), dtype=a.dtype, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.90, seed=a.seed, limit_mm_per_prompt={"image": 0})
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank)
    return LLM(**kw)


def generate_llamacpp(convs: list[list[dict]], url: str, n: int, temperature: float, max_tokens: int, seed: int,
                      workers: int = 8, extra: dict | None = None) -> list[list[str]]:
    """Same interface as vLLM generation, via a llama-server (OpenAI API) running a GGUF model on the laptop GPU.
    The server batches the concurrent requests (one slot per request)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    def one(job):
        conv, k = job
        body = {"messages": conv, "temperature": temperature, "top_p": 1.0 if temperature == 0 else 0.95,
                "max_tokens": max_tokens, "seed": seed + k, **(extra or {})}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"] or ""
            except Exception as e:  # server busy / restarted
                if attempt == 2:
                    print("request failed:", e, flush=True)
                    return ""
                time.sleep(5)

    jobs = [(c, k) for c in convs for k in range(n)]
    with ThreadPoolExecutor(workers) as ex:
        flat = list(tqdm(ex.map(one, jobs), total=len(jobs), mininterval=30))
    return [flat[i * n:(i + 1) * n] for i in range(len(convs))]


def generate(llm, convs: list[list[dict]], a, n: int, temperature: float) -> list[list[str]]:
    if a.backend == "llamacpp":
        return generate_llamacpp(convs, a.server, n, temperature, a.max_tokens, a.seed, a.workers)
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                        max_tokens=a.max_tokens, seed=a.seed)
    lora = None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("adapter", 1, a.lora)
    outs = llm.chat(convs, sp, use_tqdm=True, lora_request=lora)
    return [[c.text for c in o.outputs] for o in outs]


def values(texts: list[str], strategy: str) -> list:
    if strategy.startswith("pot"):
        return [v for v, _ in run_many(texts)]
    return [extract_answer(t) for t in texts]


def vote(cands: list[str | None]) -> str | None:
    c = Counter(x for x in cands if x is not None)
    return c.most_common(1)[0][0] if c else None


def run_split(llm, df: pd.DataFrame, pool: pd.DataFrame, a) -> pd.DataFrame:
    convs = [build(a.strategy, r, pool, k=a.k, seed=a.seed) for _, r in df.iterrows()]
    outs = generate(llm, convs, a, a.n, a.temperature)
    flat = [t for o in outs for t in o]
    vals = values(flat, a.strategy)
    rows, i = [], 0
    for (_, r), o in zip(df.iterrows(), outs):
        cands = [canonical(v, r["statement"]) for v in vals[i:i + len(o)]]
        i += len(o)
        rows.append({"id": r["id"], "skill": r["skill"], "pred": vote(cands), "votes": json.dumps(cands),
                     "raw": o[0], "source": a.strategy})
        if a.save_all:
            rows[-1]["outputs"] = json.dumps(o, ensure_ascii=False)
    res = pd.DataFrame(rows)
    if a.fallback:
        miss = res["pred"].isna()
        if miss.any():
            sub = df[miss.values]
            fb = generate(llm, [build(a.fallback, r, pool, k=a.k, seed=a.seed) for _, r in sub.iterrows()], a, 1, 0.0)
            fv = values([o[0] for o in fb], a.fallback)
            res.loc[miss, "pred"] = [canonical(v, s) for v, s in zip(fv, sub["statement"])]
            res.loc[miss, "raw"] = [o[0] for o in fb]
            res.loc[miss, "source"] = a.fallback
    if "answer" in df:
        res["answer"] = df["answer"].values
        res["correct"] = res["pred"].astype(str) == res["answer"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="gemma | mamay | HF id | local path")
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--splits", nargs="+", default=["val"], help="val | test | train | path/to/problems.jsonl")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3, help="few-shot examples")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--dtype", default="auto", help="auto: bf16 if supported, else fp32 (vLLM refuses fp16 for Gemma 3)")
    ap.add_argument("--tp", type=int, default=0, help="tensor parallel size; 0 = auto (fp32 4B needs both T4s)")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--fallback", default=None, choices=[None, "cot", "plan"])
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows per split")
    ap.add_argument("--save-all", action="store_true", help="keep every sample (for DPO pairs)")
    ap.add_argument("--backend", choices=["vllm", "llamacpp"], default="vllm")
    ap.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server URL (backend llamacpp)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests = server slots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="runs dir (default runs/)")
    a = ap.parse_args()

    pool = load_split("train")
    llm = make_llm(a)
    preds, metrics = {}, {}
    for split in a.splits:
        df = load_problems(split)
        split = Path(split).stem if split.endswith(".jsonl") else split
        if a.limit:
            df = df.head(a.limit)
        t0 = time.time()
        res = run_split(llm, df, pool, a)
        metrics[f"{split}_min"] = round((time.time() - t0) / 60, 2)
        metrics[f"{split}_empty_raw"] = int((res["raw"].str.strip() == "").sum())  # fp16 overflow check
        preds[split] = res
        if split == "val":
            acc = accuracy(res)
            metrics.update(val_acc=acc["acc"], val_n=acc["n"], val_none=acc["n_none"], per_skill=acc["per_skill"])
            print(f"{a.name}: val acc {acc['acc']:.4f} ({acc['n']} rows, {acc['n_none']} without answer)", flush=True)
    kw = {"runs_dir": a.out} if a.out else {}
    d = save_run(a.name, vars(a), metrics, preds, **kw)
    print("saved", d, flush=True)


if __name__ == "__main__":
    main()
