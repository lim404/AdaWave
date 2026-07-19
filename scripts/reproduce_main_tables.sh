#!/usr/bin/env bash
# Regenerate the LaTeX tables from result CSVs.
#
# By default this reads the CSVs shipped in supplementary/raw_csv/, so it
# reproduces the published tables without rerunning any experiment. To
# build tables from your own run instead, point ADAWAVE_CSV_DIR at the
# directory holding the freshly written CSVs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

CSV_DIR="${ADAWAVE_CSV_DIR:-$REPO/supplementary/raw_csv}"
OUT_DIR="${ADAWAVE_TABLE_DIR:-$REPO/tables}"

mkdir -p "$OUT_DIR"
cd "$CSV_DIR"
ADAWAVE_TABLE_DIR="$OUT_DIR" python "$REPO/eval/make_tip_tables.py"

echo "tables written to $OUT_DIR/"
