#!/usr/bin/env bash
# Kaggle (T4 x2) versions of Colab notebooks 01 and 02. Usage:
#   bash kernels/kaggle_queues.sh prompting     # notebook 01: both models x strategies (val + test)
#   bash kernels/kaggle_queues.sh teacher       # notebook 02: synthetic problems + Qwen3 teacher data
#   KAGGLE_USER=<user> python kernels/fetch.py hw4-prompting hw4-teacher    # afterwards: results -> runs/
# On T4 Gemma/MamayLM run in float32 over both GPUs (infer defaults: --dtype auto --tp 0); Qwen3 in float16.
set -euo pipefail
cd "$(dirname "$0")/.."
export KAGGLE_USER="${KAGGLE_USER:-lizabugirrr}"
PY="${PY:-.venv/bin/python}"
export PATH="$PWD/.venv/bin:$PATH"

join() { local IFS="&"; local s="$*"; echo "${s//&/ && }"; }

case "${1:-}" in
prompting)
  Q=(
    "python -m hw4.infer --model mamay --strategy cot --splits val test --name mamay_cot"
    "python -m hw4.infer --model gemma --strategy cot --splits val test --name gemma_cot"
    "python -m hw4.infer --model mamay --strategy pot --fallback cot --splits val test --name mamay_pot"
    "python -m hw4.infer --model gemma --strategy pot --fallback cot --splits val test --name gemma_pot"
    "python -m hw4.infer --model mamay --strategy pot0 --fallback cot --name mamay_pot0"
    "python -m hw4.infer --model gemma --strategy pot0 --fallback cot --name gemma_pot0"
    "python -m hw4.infer --model mamay --strategy zeroshot --name mamay_zeroshot"
    "python -m hw4.infer --model gemma --strategy zeroshot --name gemma_zeroshot"
    "python -m hw4.infer --model mamay --strategy fewshot --name mamay_fewshot"
    "python -m hw4.infer --model gemma --strategy fewshot --name gemma_fewshot"
    "python -m hw4.infer --model mamay --strategy plan --name mamay_plan"
    "python -m hw4.infer --model gemma --strategy plan --name gemma_plan"
    "python -m hw4.infer --model mamay --strategy cot --n 8 --temperature 0.7 --splits val test --name mamay_cot_sc8"
    "python -m hw4.infer --model mamay --strategy pot --n 8 --temperature 0.7 --fallback cot --splits val test --name mamay_pot_sc8"
    "python -m hw4.infer --model gemma --strategy pot --n 8 --temperature 0.7 --fallback cot --splits val test --name gemma_pot_sc8"
  )
  # "; " between commands would stop nothing: launch.py splits on " && " and runs every command regardless
  $PY kernels/launch.py ${DRY:+--dry} --upload --slug hw4-prompting --queue "$(join "${Q[@]}")"
  ;;
teacher)
  T="--teacher Qwen/Qwen3-8B --dtype float16 --tp 2"
  Q=(
    "python -m hw4.synth --per-skill 300 --out {OUT}/data_synth/problems.jsonl"
    "python -m hw4.teacher $T --source train --style pot --n 4 --name data_qwen_pot_train"
    "python -m hw4.teacher $T --source train --style cot --n 4 --name data_qwen_cot_train"
    "python -m hw4.teacher $T --source {OUT}/data_synth/problems.jsonl --style pot --n 2 --per-skill 150 --name data_qwen_pot_synth"
    "python -m hw4.teacher $T --source {OUT}/data_synth/problems.jsonl --style cot --n 2 --per-skill 150 --name data_qwen_cot_synth"
  )
  $PY kernels/launch.py ${DRY:+--dry} --upload --slug hw4-teacher --models --queue "$(join "${Q[@]}")"
  ;;
*) echo "usage: $0 prompting|teacher"; exit 1 ;;
esac
