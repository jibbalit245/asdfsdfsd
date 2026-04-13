#!/bin/bash
# start.sh — clone the repo, run this once, everything else is automatic
#
# Builds coupled_field if needed, then launches a 500k-tick long run across
# all available GPUs (up to 8), one instance per GPU with different seeds.
#
# Optional env overrides:
#   SNAP_EVERY=500   (PNG snapshot interval, default 500 ticks)

set -e

cd "$(dirname "$0")"

# ── build if binary is missing ────────────────────────────────────────────────
if [ ! -x "./coupled_field" ]; then
    echo "=== coupled_field not found — building now ==="
    bash deploy/setup.sh
    echo ""
fi

# ── launch across all GPUs ────────────────────────────────────────────────────
SNAP_EVERY=${SNAP_EVERY:-500}

bash deploy/run_multi.sh "$SNAP_EVERY"

exec bash deploy/monitor.sh
