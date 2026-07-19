#!/usr/bin/env bash
# Final-test run (Table I + Table II). Classical methods and AdaWave only:
# the learned baselines need third-party checkpoints that are not
# redistributed here -- see README.md "Not included in this repository".
set -euo pipefail
cd "$(dirname "$0")/.."

: "${ADAWAVE_DATA_ROOT:?set it to the directory holding the datasets (see README.md)}"

python eval/final_test.py "$@"
