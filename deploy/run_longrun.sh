#!/bin/bash
# run_longrun.sh — 500,000-tick training run with checkpoint/resume support
#
# Usage:
#   ./deploy/run_longrun.sh [seed] [out_dir] [device]
#
# To resume from the latest substrate snapshot:
#   RESUME=1 ./deploy/run_longrun.sh [seed] [out_dir] [device]
#
# NCA model is saved to best_nca.bin whenever loss improves.
# All instances (including multi-GPU) share the same best_nca.bin file.
# When RESUME=1, best_nca.bin is loaded via --nca-weights if present.

SEED=${1:-random}
OUT_DIR=${2:-./frames_longrun}
DEVICE=${3:-0}
TICKS=500000
SNAP_EVERY=500    # one PNG every 500 ticks → ~1000 frames for the full run

cd "$(dirname "$0")/.."

mkdir -p "$OUT_DIR"

RESUME_ARGS=""
if [ "${RESUME:-0}" = "1" ]; then
    # Resume substrate state from final snapshot if available
    if [ -f snapshot_final.bin ]; then
        echo "=== Resuming from: snapshot_final.bin ==="
        RESUME_ARGS="--resume snapshot_final.bin"
    else
        echo "=== No snapshot found; starting fresh ==="
    fi
    # Load best NCA model if available
    if [ -f best_nca.bin ]; then
        echo "=== Loading NCA model: best_nca.bin ==="
        RESUME_ARGS="$RESUME_ARGS --nca-weights best_nca.bin"
    fi
fi

echo "=== STSC Long Run (500k ticks) ==="
echo "  Seed       : $SEED"
echo "  Output     : $OUT_DIR"
echo "  Ticks      : $TICKS"
echo "  Snap every : $SNAP_EVERY ticks (~$(( TICKS / SNAP_EVERY )) frames)"
echo "  Device     : $DEVICE"
echo "  Resume     : ${RESUME_ARGS:-fresh start}"
echo ""

CUDA_VISIBLE_DEVICES=$DEVICE ./coupled_field \
    --seed "$SEED" \
    --ticks "$TICKS" \
    --snap-every "$SNAP_EVERY" \
    --snap-dir "$OUT_DIR" \
    --no-interactive \
    $RESUME_ARGS \
    2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "=== Run complete. Outputs: ==="
echo "  Substrate  : snapshot_final.bin"
echo "  NCA model  : best_nca.bin  (saved only on improvement)"
echo "  Training   : nca_training.csv"
echo "  Frames     : $OUT_DIR/"
