#!/usr/bin/env bash
# Local OpenAI-compatible server for a 4-bit GGUF model on the laptop GPU (RTX 2050, 4 GB, Vulkan).
#   tools/llama_server.sh mamay            # or gemma; stops any running instance first
#   tools/llama_server.sh stop
# RAM is capped (systemd scope) so the server is killed alone instead of freezing VS Code.
set -euo pipefail
cd "$(dirname "$0")/.."
LL="$HOME/.local/llama.cpp/cuda"
BIN="${LLAMA_BIN:-$LL/llama-b12774/llama-server}"
export LD_LIBRARY_PATH="$LL/llama-b12774:$LL/cudart-llama-b12774-bin-ubuntu-cuda-12.8-x64:${LD_LIBRARY_PATH:-}"
DEV="${DEV:-CUDA0}"            # Vulkan1 with the Vulkan build
PORT="${PORT:-8080}"
NP="${NP:-8}"                 # parallel slots (batched decoding)
CTX="${CTX:-16384}"           # total context, split between slots
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
[ "${1:-}" = "stop" ] && exit 0
MODEL="weights/gguf/${1:?mamay|gemma|qwen3-4b|<name>}-Q4_K_M.gguf"
LOG="runs/_logs/llama_server_$1.log"; mkdir -p runs/_logs
systemd-run --user --scope --quiet -p MemoryMax="${MEM:-3G}" -p MemorySwapMax=0 \
  nice -n 5 "$BIN" -m "$MODEL" --device "$DEV" -ngl 99 -c "$CTX" -np "$NP" -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 --port "$PORT" --no-webui --jinja > "$LOG" 2>&1 &
for i in $(seq 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "server ready: $MODEL (port $PORT, $NP slots)"; exit 0; }
  sleep 1
done
echo "server did not start, see $LOG"; tail -20 "$LOG"; exit 1
