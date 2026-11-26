#!/usr/bin/env bash
# Run a command with a hard memory cap so it is killed alone instead of freezing the laptop.
#   ./safe_run.sh python -m hw4.baseline --n-train 5000
#   MEM=6G ./safe_run.sh python -m hw4.features
set -euo pipefail
MEM="${MEM:-4G}"
exec systemd-run --user --scope --quiet -p MemoryMax="$MEM" -p MemorySwapMax=0 \
  nice -n 10 "$@"
