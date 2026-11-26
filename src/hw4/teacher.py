"""Teacher data: an open Qwen3 model writes solutions, we keep only the ones that give the reference answer.

Qwen3 is used only as a teacher (allowed); it never answers val/test. Problems come from the TRAIN side of
the split and/or from my synthetic generators (hw4.synth). Output: runs/<name>/sft.jsonl, one row per kept
solution, in TRL's conversational prompt/completion format with the same prompt the student will see.

python -m hw4.teacher --source train --style pot --name data_qwen_pot_train
python -m hw4.teacher --source runs/data_synth/problems.jsonl --style pot --per-skill 120 --name data_qwen_pot_synth
python -m hw4.teacher --source train --style cot --name data_qwen_cot_train

styles: pot (Python that sets `answer`, checked by running it), cot (Ukrainian steps + 'Відповідь: x').
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from .answers import ANSWER_RE, canonical, extract_answer
from .data import load_problems
from .prompts import build
from .runs import RUNS
from .sandbox import extract_code, run_many


def clean_pot(text: str) -> str | None:
    code = extract_code(text)
    if not code:
        return None
    code = "\n".join(l.rstrip() for l in code.strip().splitlines() if l.strip() and not l.strip().startswith("print("))
    return f"```python\n{code}\n```"


def clean_cot(text: str, answer: str) -> str:
    """Drop everything after the last 'Відповідь:' and write the answer there in canonical form."""
    hits = list(ANSWER_RE.finditer(text))
    body = text[:hits[-1].start()] if hits else text
    body = re.sub(r"\*\*\s*$", "", body.rstrip()).rstrip()
    return f"{body}\n\nВідповідь: {answer}".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--source", required=True, help="train | path/to/problems.jsonl (never val/test)")
    ap.add_argument("--style", choices=["pot", "cot"], default="pot")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=4, help="samples per problem")
    ap.add_argument("--keep", type=int, default=1, help="distinct correct solutions kept per problem")
    ap.add_argument("--per-skill", type=int, default=None, help="subsample problems per skill")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="auto", help="float16 on T4 (Qwen3 is fine in fp16)")
    ap.add_argument("--tp", type=int, default=1, help="2 on Kaggle T4 x2")
    ap.add_argument("--backend", choices=["vllm", "llamacpp"], default="vllm")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.source not in ("val", "test"), "teacher data must not touch val/test"

    df = load_problems(a.source)
    if a.per_skill:
        df = df.sample(frac=1, random_state=a.seed).groupby("skill").head(a.per_skill)
    df = df.reset_index(drop=True)
    strategy = "pot0" if a.style == "pot" else "cot"
    prompts = [build(strategy, r) for _, r in df.iterrows()]

    t0 = time.time()
    if a.backend == "llamacpp":  # laptop: Qwen3 GGUF in llama-server, thinking off via the chat template
        from .infer import generate_llamacpp
        texts = generate_llamacpp(prompts, a.server, a.n, a.temperature, a.max_tokens, a.seed, a.workers,
                                  extra={"chat_template_kwargs": {"enable_thinking": False}})
    else:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.teacher, dtype=a.dtype, tensor_parallel_size=a.tp, max_model_len=4096,
                  gpu_memory_utilization=0.90, max_num_seqs=64, seed=a.seed)
        sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
        outs = llm.chat(prompts, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        texts = [[c.text for c in o.outputs] for o in outs]

    flat = [t for ts in texts for t in ts]
    vals = [v for v, _ in run_many(flat)] if a.style == "pot" else [extract_answer(t) for t in flat]
    rows, solved, i = [], [], 0
    for j, ((_, r), ts) in enumerate(zip(df.iterrows(), texts)):
        kept, seen = 0, set()
        for t, v in zip(ts, vals[i:i + len(ts)]):
            if kept >= a.keep or canonical(v, r["statement"]) != r["answer"]:
                continue
            c = clean_pot(t) if a.style == "pot" else clean_cot(t, r["answer"])
            if not c or c in seen:
                continue
            seen.add(c)
            kept += 1
            rows.append({"id": r["id"], "skill": r["skill"], "answer": r["answer"], "style": a.style,
                         "prompt": prompts[j], "completion": [{"role": "assistant", "content": c}]})
        i += len(ts)
        solved.append({"id": r["id"], "skill": r["skill"], "solved": kept > 0,
                       "pass_rate": sum(canonical(v, r["statement"]) == r["answer"] for v in vals[i - len(ts):i]) / len(ts)})

    d = Path(RUNS) / a.name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sv = pd.DataFrame(solved)
    sv.to_csv(d / "solve_rate.csv", index=False)
    metrics = {"problems": len(df), "kept": len(rows), "solved": float(sv["solved"].mean()),
               "teacher_pass_rate": float(sv["pass_rate"].mean()), "minutes": round((time.time() - t0) / 60, 1),
               "per_skill_solved": sv.groupby("skill")["solved"].mean().round(3).to_dict()}
    (d / "config.json").write_text(json.dumps(vars(a), indent=1))
    (d / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    print(f"{a.name}: kept {len(rows)} solutions, solved {metrics['solved']:.3f}, pass@1 {metrics['teacher_pass_rate']:.3f}")


if __name__ == "__main__":
    main()
