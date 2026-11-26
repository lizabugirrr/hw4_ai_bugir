#!/usr/bin/env bash
# Downloads competition data into data/. Requires a Kaggle API token
# and accepted competition rules on Kaggle.
set -euo pipefail
COMP=ukrainian-school-math-slm-hw4-ucu-ai-2026
cd "$(dirname "$0")"
mkdir -p data
kaggle competitions download -c "$COMP" -p data
unzip -q -o "data/$COMP.zip" -d data
rm "data/$COMP.zip"
