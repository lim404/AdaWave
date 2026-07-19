#!/usr/bin/env bash
# Full-tile DALES run (large-scale table).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ADAWAVE_DATA_ROOT:?set it to the directory holding the datasets (see README.md)}"

python eval/large_scale_tile.py "$@"
