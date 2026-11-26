# HW4 code: how to run

## Setup and data
```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt -e .
./download_data.sh            # needs a Kaggle token and accepted rules
python -m hw4.data            # once: fixed split -> src/hw4/configs/split.csv (val = 30% of train, stratified by skill, seed 42)
```

## GPU runs (Kaggle T4 x2, vLLM)
Gemma 3 cannot run in float16 under vLLM and T4 has no bfloat16, so models run in float32 split over both T4s
(`--dtype float32 --tp 2`, the defaults). Gemma-3-4B-it is attached from Kaggle Models (accept the Gemma licence on
Kaggle once); MamayLM is downloaded from Hugging Face.
```bash
export KAGGLE_USER=<kaggle username>
python kernels/launch.py --upload --slug hw4-<name> --queue "python -m hw4.infer --model mamay --strategy cot --splits val test --name mamay_cot --out {OUT}"
python kernels/fetch.py hw4-<name>        # -> runs/<name>/, prints results table
python -m hw4.submission --run mamay_cot --out submissions/mamay_cot.csv
```

## GPU runs on Colab (training part)
```bash
python tools/make_colab.py    # -> kernels/colab_00_all_a100.ipynb + hw4-code.zip (upload the zip to MyDrive/hw4/)
```
`kernels/colab_00_all_a100.ipynb` is a walkthrough of the GPU half: for every stage (teacher data → SFT MamayLM →
Hugging Face → SFT Gemma → ablations → self-consistency → DPO → GRPO) it explains the method, shows the key
implementation from the `hw4` package, runs it and displays the results. Secrets: `KAGGLE_API_TOKEN` (data + Gemma
from Kaggle Models), `HF_TOKEN` (publishing). A100 ≈ 4 h (L4 also works). Results land in `MyDrive/hw4/runs/`;
copy them into `runs/`, then `python -m hw4.runs` -> `results/experiments.csv`.

## Laptop runs (prompting, teacher on a 4 GB GPU)
```bash
tools/local_queue.sh teacher      # Qwen3-4B (4-bit GGUF) teacher data for the train side
tools/local_queue.sh prompting    # both models x zero-shot / CoT / few-shot / plan / tool use / self-consistency
```
llama.cpp server with 4-bit GGUF models (`tools/llama_server.sh`, RAM-capped); clients via `safe_run.sh`.

| Path | What |
|---|---|
| `src/hw4/data.py` | loading, fixed train/val split |
| `src/hw4/answers.py` | number extraction from model output, canonical answer format |
| `src/hw4/prompts.py` | prompting strategies: zeroshot, cot, fewshot, plan, pot (tool use) |
| `src/hw4/sandbox.py` | isolated execution of model-written Python |
| `src/hw4/infer.py` | vLLM batch inference, self-consistency voting, fallback |
| `src/hw4/evaluate.py` | accuracy per skill, matching wrong answers to `typical_errors` |
| `src/hw4/synth.py` | my own generators of new problems for all 29 skills (training data only) |
| `src/hw4/teacher.py` | Qwen3 teacher solutions, verified against the answer |
| `src/hw4/sft.py`, `dpo.py`, `grpo.py` | LoRA SFT, DPO, GRPO (TRL + PEFT) |
| `src/hw4/train_utils.py`, `merge.py`, `hub.py` | student loading, LoRA merge for vLLM, HF publishing with model card |
| `src/hw4/runs.py`, `submission.py` | run storage, submission file with format checks |
| `tools/make_colab.py` | generates the Colab notebook and the private code zip |
| `tools/llama_server.sh`, `local_queue.sh` | local 4-bit inference server and run queues |
| `kernels/` | Kaggle launch (private code dataset + code-hash check) and fetch scripts |
