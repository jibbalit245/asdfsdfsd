#!/bin/bash
# setup.sh — run once on a fresh RunPod instance
# Detects GPU arch, installs deps, compiles coupled_field

set -e

echo "=== STSC Deploy Setup ==="

# ── find nvcc ─────────────────────────────────────────────────────────────────
NVCC=$(which nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)
if [ -z "$NVCC" ]; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    exit 1
fi
echo "Using nvcc: $NVCC"

# ── detect CUDA arch ──────────────────────────────────────────────────────────
ARCH=$(python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode().strip().split('\n')[0]
major, minor = out.strip().split('.')
print(f'sm_{major}{minor}')
" 2>/dev/null || echo "sm_80")

echo "Detected arch: $ARCH"

# ── verify nvcc supports the detected arch; fall back if not ──────────────────
_PROBE=$(mktemp /tmp/probe_XXXX.cu)
echo "int main(){return 0;}" > "$_PROBE"
if ! $NVCC -arch=$ARCH -o /dev/null "$_PROBE" 2>/dev/null; then
    for FALLBACK in sm_120 sm_90 sm_89 sm_86 sm_80; do
        if $NVCC -arch=$FALLBACK -o /dev/null "$_PROBE" 2>/dev/null; then
            echo "WARNING: $ARCH not supported by this nvcc; falling back to $FALLBACK"
            ARCH=$FALLBACK
            break
        fi
    done
fi
rm -f "$_PROBE"

# ── compile ───────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
echo "Compiling coupled_field..."
$NVCC -O3 -arch=$ARCH -use_fast_math -lineinfo \
    -o coupled_field coupled_field.cu 2>&1

echo ""
echo "=== Build complete ==="
echo "Run:  ./deploy/run.sh [random|quantum|sparse] [output_dir] [snap_every]"
