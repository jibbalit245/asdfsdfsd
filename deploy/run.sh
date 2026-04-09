#!/bin/bash
# run.sh — launch one simulation instance headlessly
# Usage: ./deploy/run.sh [seed] [out_dir] [snap_every] [device]
#
# seed      : random | quantum | sparse   (default: random)
# out_dir   : where to write PNGs         (default: ./frames)
# snap_every: save frame every N ticks    (default: 1)
# device    : CUDA device index           (default: 0)

SEED=${1:-random}
OUT_DIR=${2:-./frames}
SNAP_EVERY=${3:-1}
DEVICE=${4:-0}

cd "$(dirname "$0")/.."

mkdir -p "$OUT_DIR"

echo "=== STSC Run ==="
echo "  Seed      : $SEED"
echo "  Output    : $OUT_DIR"
echo "  Snap every: $SNAP_EVERY ticks"
echo "  Device    : $DEVICE"
echo "  PID will be written to: $OUT_DIR/pid"
echo ""

CUDA_VISIBLE_DEVICES=$DEVICE ./coupled_field \
    --seed "$SEED" \
    --snap-every "$SNAP_EVERY" \
    --snap-dir "$OUT_DIR" \
    --no-interactive \
    "$@" \
    &

echo $! > "$OUT_DIR/pid"
echo "Launched PID $(cat $OUT_DIR/pid)"
echo "Logs: tail -f $OUT_DIR/run.log"
