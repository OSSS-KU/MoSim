#!/usr/bin/env bash
# Generate the paper Table II/III/IV artifacts from one rawdata.csv.
#
# From MoSim:
#   bash scripts/analyze.sh
#
# Tunables:
#   RAWDATA=result/rawdata.csv
#   OUTDIR=result/paper
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RAWDATA="${RAWDATA:-result/rawdata.csv}"
OUTDIR="${OUTDIR:-result/paper}"

python3 "$SCRIPT_DIR/analyze.py" \
  --rawdata "$RAWDATA" \
  --outdir "$OUTDIR" \
  "$@"
