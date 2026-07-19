#!/usr/bin/env bash
# Figure sources.
#
# The figure-drawing scripts are not part of this release: the qualitative
# panel depends on a learned baseline (StraightPCF) whose checkpoint is not
# redistributed here. The underlying data for every figure is shipped as
# CSV so the figures are auditable and redrawable.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

cat "$REPO/supplementary/figure_sources/README.md"
