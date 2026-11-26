#!/usr/bin/env bash
# Local version of the Colab notebooks 01 (prompting) and 02 (teacher) on the laptop GPU via llama-server.
#   tools/local_queue.sh prompting | teacher | <file with "model|command" lines>
# One model is loaded at a time; each client command runs with a 2 GB RAM cap (safe_run.sh).
# A run whose runs/<name>/metrics.json exists is skipped, so the queue can be restarted at any time.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L="--backend llamacpp"
prompting() { cat <<Q
mamay|-m hw4.infer $L --model mamay --strategy zeroshot --name mamay_zeroshot
mamay|-m hw4.infer $L --model mamay --strategy cot --splits val test --name mamay_cot
mamay|-m hw4.infer $L --model mamay --strategy pot --fallback cot --splits val test --name mamay_pot
mamay|-m hw4.infer $L --model mamay --strategy pot0 --fallback cot --name mamay_pot0
mamay|-m hw4.infer $L --model mamay --strategy fewshot --name mamay_fewshot
mamay|-m hw4.infer $L --model mamay --strategy plan --name mamay_plan
mamay|-m hw4.infer $L --model mamay --strategy pot --n 5 --temperature 0.7 --fallback cot --name mamay_pot_sc5
gemma|-m hw4.infer $L --model gemma --strategy zeroshot --name gemma_zeroshot
gemma|-m hw4.infer $L --model gemma --strategy cot --name gemma_cot
gemma|-m hw4.infer $L --model gemma --strategy pot --fallback cot --name gemma_pot
gemma|-m hw4.infer $L --model gemma --strategy pot0 --fallback cot --name gemma_pot0
gemma|-m hw4.infer $L --model gemma --strategy fewshot --name gemma_fewshot
gemma|-m hw4.infer $L --model gemma --strategy plan --name gemma_plan
gemma|-m hw4.infer $L --model gemma --strategy pot --n 5 --temperature 0.7 --fallback cot --name gemma_pot_sc5
Q
}
teacher() { cat <<Q
qwen3-4b|-m hw4.teacher $L --teacher Qwen/Qwen3-4B-GGUF:Q4_K_M --source train --style pot --n 4 --max-tokens 768 --name data_qwen_pot_train
qwen3-4b|-m hw4.teacher $L --teacher Qwen/Qwen3-4B-GGUF:Q4_K_M --source runs/data_synth/problems.jsonl --style pot --n 2 --per-skill 80 --max-tokens 768 --name data_qwen_pot_synth
qwen3-4b|-m hw4.teacher $L --teacher Qwen/Qwen3-4B-GGUF:Q4_K_M --source train --style cot --n 2 --max-tokens 768 --name data_qwen_cot_train
Q
}
case "${1:?prompting|teacher|file}" in
  prompting) JOBS=$(prompting) ;;
  teacher) JOBS=$(teacher) ;;
  *) JOBS=$(cat "$1") ;;
esac
cur=""
while IFS='|' read -r model cmd; do
  [ -z "$model" ] && continue
  name=$(echo "$cmd" | sed -n 's/.*--name \([^ ]*\).*/\1/p')
  if [ -f "runs/$name/metrics.json" ]; then echo "skip (done): $name"; continue; fi
  if [ "$model" != "$cur" ]; then tools/llama_server.sh "$model" || exit 1; cur=$model; fi
  echo "=== $name ($(date +%H:%M))"
  MEM=2G ./safe_run.sh $PY $cmd > "runs/_logs/$name.log" 2>&1
  echo "exit $? : $(grep -hE 'val acc|kept' runs/_logs/$name.log | tail -1)"
done <<< "$JOBS"
tools/llama_server.sh stop
echo "queue done ($(date +%H:%M))"
