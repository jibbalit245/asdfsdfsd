#!/bin/bash
# run_longrun.sh — 500,000-tick training run with full checkpoint/resume support
#
# Usage:
#   ./deploy/run_longrun.sh [seed] [out_dir] [device]
#
# To resume from the latest checkpoint:
#   RESUME=1 ./deploy/run_longrun.sh [seed] [out_dir] [device]
#
# Checkpoints are written every 65,536 ticks:
#   checkpoint_65536.bin        — substrate state (pixel + wave)
#   checkpoint_65536_nca.bin    — NCA weights + Adam moments + step counter
#
# On resume (RESUME=1), the script finds the latest checkpoint_NNN.bin and
# passes both --resume and (implicitly via auto-discovery) --nca-weights.
# The auto-discovery looks for snapshot_path_nca.bin, so naming matches.

SEED=${1:-random}
OUT_DIR=${2:-./frames_longrun}
DEVICE=${3:-0}
TICKS=500000
SNAP_EVERY=500    # one PNG every 500 ticks → ~1000 frames for the full run

cd "$(dirname "$0")/.."

mkdir -p "$OUT_DIR"

RESUME_ARGS=""
if [ "${RESUME:-0}" = "1" ]; then
    # Find the latest periodic checkpoint using a glob and sort numerically on the tick number
    LATEST=""
    LATEST_TICK=-1
    for f in checkpoint_*.bin; do
        # Skip NCA companion files
        case "$f" in *_nca.bin) continue ;; esac
        # Extract tick number from checkpoint_NNN.bin
        tick="${f#checkpoint_}"
        tick="${tick%.bin}"
        case "$tick" in ''|*[!0-9]*) continue ;; esac
        if [ "$tick" -gt "$LATEST_TICK" ] 2>/dev/null; then
            LATEST_TICK=$tick
            LATEST=$f
        fi
    done
    if [ -n "$LATEST" ]; then
        echo "=== Resuming from: $LATEST (tick $LATEST_TICK) ==="
        RESUME_ARGS="--resume $LATEST"
        # NCA companion file is auto-discovered by the binary (strips .bin, appends _nca.bin)
    else
        # Fall back to final snapshot if available
        if [ -f snapshot_final.bin ]; then
            echo "=== Resuming from: snapshot_final.bin ==="
            RESUME_ARGS="--resume snapshot_final.bin"
        else
            echo "=== No checkpoint found; starting fresh ==="
        fi
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
echo "  NCA model  : snapshot_final_nca.bin"
echo "  Training   : nca_training.csv"
echo "  Frames     : $OUT_DIR/"
