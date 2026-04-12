#!/bin/bash
# setup.sh — run once on a fresh RunPod instance
# Detects GPU arch, installs deps, compiles coupled_field

set -e

echo "=== STSC Deploy Setup ==="

# ── detect CUDA arch ──────────────────────────────────────────────────────────
ARCH=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode().strip().split('\n')[0]
major, minor = out.strip().split('.')
print(f'sm_{major}{minor}')
" 2>/dev/null || echo "sm_80")

echo "Detected arch: $ARCH"

# ── find nvcc ─────────────────────────────────────────────────────────────────
NVCC=$(which nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)
if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    exit 1
fi
echo "Using nvcc: $NVCC"

# ── compile ───────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
echo "Compiling coupled_field..."
$NVCC -O3 -arch=$ARCH -use_fast_math -lineinfo \
    -o coupled_field coupled_field.cu 2>&1

echo ""
echo "=== Build complete ==="
echo "Run:  ./deploy/run.sh [random|quantum|sparse] [output_dir] [snap_every]"
