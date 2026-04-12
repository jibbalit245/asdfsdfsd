#!/bin/bash
# run_multi.sh — launch one instance per GPU, different seeds
# Detects how many GPUs are available and fills them up.
# Usage: ./deploy/run_multi.sh [snap_every] [ticks]

SNAP_EVERY=${1:-500}
TICKS=${2:-500000}

cd "$(dirname "$0")/.."

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "=== STSC Multi-GPU Launch: $GPU_COUNT GPUs detected ==="

SEEDS=(random quantum random quantum random quantum random quantum)

for i in $(seq 0 $((GPU_COUNT - 1))); do
    SEED=${SEEDS[$((i % ${#SEEDS[@]}))]}
    OUT="frames_gpu${i}_${SEED}"
    mkdir -p "$OUT"

    echo "  GPU $i → seed=$SEED  out=$OUT"

    CUDA_VISIBLE_DEVICES=$i nohup ./coupled_field \
        --seed "$SEED" \
        --ticks "$TICKS" \
        --snap-every "$SNAP_EVERY" \
        --snap-dir "$OUT" \
        --no-interactive \
        > "$OUT/run.log" 2>&1 &

    echo $! > "$OUT/pid"
    sleep 1   # stagger startup to avoid init races
done

echo ""
echo "All instances launched. Monitor with:"
echo "  watch -n 2 'for d in frames_gpu*/; do echo \"\$d: \$(ls \$d/*.png 2>/dev/null | wc -l) frames\"; done'"
echo ""
echo "Stop all:  pkill coupled_field"
