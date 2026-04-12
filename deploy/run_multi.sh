#!/bin/bash
# run_multi.sh — launch one instance per GPU, different seeds
# Detects how many GPUs are available and fills them up.
# All instances share best_nca.bin — whichever improves loss overwrites it.
# Usage: ./deploy/run_multi.sh [snap_every] [ticks]

SNAP_EVERY=${1:-500}
TICKS=${2:-500000}

cd "$(dirname "$0")/.."

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "=== STSC Multi-GPU Launch: $GPU_COUNT GPUs detected ==="

# All instances share the same NCA model file; pass it if it already exists
NCA_ARGS=""
if [ -f best_nca.bin ]; then
    echo "  Shared NCA model: best_nca.bin"
    NCA_ARGS="--nca-weights best_nca.bin"
fi

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
        $NCA_ARGS \
        > "$OUT/run.log" 2>&1 &

    echo $! > "$OUT/pid"
    sleep 1   # stagger startup to avoid init races
done

echo ""
echo "All instances launched. Each writes to best_nca.bin when loss improves."
echo "Monitor with:"
echo "  watch -n 2 'for d in frames_gpu*/; do echo \"\$d: \$(ls \$d/*.png 2>/dev/null | wc -l) frames\"; done'"
echo ""
echo "Stop all:  pkill coupled_field"
