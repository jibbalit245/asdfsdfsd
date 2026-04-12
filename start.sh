#!/bin/bash
# start.sh — clone the repo, run this once, everything else is automatic
#
# Builds coupled_field if needed, then launches:
#   - multi-GPU mode by default (one process per visible GPU), or
#   - single-GPU long-run mode when MULTI_GPU=0.
#
# Optional env overrides:
#   MULTI_GPU=1|0               (default: 1)
#   SNAP_EVERY=1                (used by multi-GPU mode)
#   SEED=random|impulse|sparse  (single-GPU mode)
#   OUT_DIR=./frames_longrun    (single-GPU mode)
#   DEVICE=0                    (single-GPU mode)
#   RESUME=1                    (single-GPU mode; resume from latest checkpoint)

set -e

cd "$(dirname "$0")"

# ── build if binary is missing ────────────────────────────────────────────────
if [ ! -x "./coupled_field" ]; then
    echo "=== coupled_field not found — building now ==="
    bash deploy/setup.sh
    echo ""
fi

# ── choose launch mode ────────────────────────────────────────────────────────
MULTI_GPU=${MULTI_GPU:-1}

if [ "$MULTI_GPU" = "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found; falling back to single-GPU mode."
    else
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
        if [ "${GPU_COUNT:-0}" -gt 1 ]; then
            SNAP_EVERY=${SNAP_EVERY:-1}
            echo "=== start.sh: launching multi-GPU mode across $GPU_COUNT GPUs ==="
            exec bash deploy/run_multi.sh "$SNAP_EVERY"
        fi
        echo "=== start.sh: detected ${GPU_COUNT:-0} GPU(s); using single-GPU long run ==="
    fi
fi

# ── single-GPU fallback / opt-in ─────────────────────────────────────────────
SEED=${SEED:-random}
OUT_DIR=${OUT_DIR:-./frames_longrun}
DEVICE=${DEVICE:-0}

exec bash deploy/run_longrun.sh "$SEED" "$OUT_DIR" "$DEVICE"
