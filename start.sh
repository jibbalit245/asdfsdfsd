#!/bin/bash
# start.sh — clone the repo, run this once, everything else is automatic
#
# Builds coupled_field if needed, then launches a 500k-tick long run with defaults:
#   seed      : random
#   output    : ./frames_longrun
#   device    : GPU 0
#
# Optional env overrides (all have safe defaults):
#   SEED=random|impulse|sparse   OUT_DIR=./frames_longrun   DEVICE=0
#   RESUME=1                     (pick up from latest checkpoint)

set -e

cd "$(dirname "$0")"

# ── build if binary is missing ────────────────────────────────────────────────
if [ ! -x "./coupled_field" ]; then
    echo "=== coupled_field not found — building now ==="
    bash deploy/setup.sh
    echo ""
fi

# ── launch with defaults ──────────────────────────────────────────────────────
SEED=${SEED:-random}
OUT_DIR=${OUT_DIR:-./frames_longrun}
DEVICE=${DEVICE:-0}

exec bash deploy/run_longrun.sh "$SEED" "$OUT_DIR" "$DEVICE"
