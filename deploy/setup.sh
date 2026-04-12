#!/bin/bash
# setup.sh — run once on a fresh RunPod instance
# Detects GPU arch, installs deps, compiles coupled_field

set -euo pipefail

echo "=== STSC Deploy Setup ==="

# ── find nvcc ─────────────────────────────────────────────────────────────────
NVCC=$(command -v nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)
if [ -z "${NVCC:-}" ]; then
    echo "ERROR: nvcc not found. Is CUDA installed?"
    exit 1
fi
echo "Using nvcc: $NVCC"

# Allow manual override (useful on very new GPUs / older toolkits)
if [ -n "${CUDA_ARCH:-}" ]; then
    ARCH="$CUDA_ARCH"
    echo "Using CUDA_ARCH override: $ARCH"
else
    # ── detect CUDA arch ──────────────────────────────────────────────────────
    ARCH=$(python3 - <<'PY' 2>/dev/null || true
import subprocess
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        stderr=subprocess.DEVNULL,
    ).decode().strip().split('\n')[0].strip()
    major, minor = out.split('.')
    print(f"sm_{major}{minor}")
except Exception:
    pass
PY
)
    ARCH=${ARCH:-sm_80}
    echo "Detected arch: $ARCH"
fi

# ── verify nvcc supports the detected arch; fall back if not ─────────────────
# Prefer newest known arch first, then descend.
SUPPORTED_ARCHS="sm_120 sm_103 sm_102 sm_101 sm_100 sm_90 sm_89 sm_87 sm_86 sm_80"

choose_supported_arch() {
    local candidate="$1"
    local probe
    probe=$(mktemp /tmp/probe_XXXX.cu)
    echo "int main(){return 0;}" > "$probe"
    if "$NVCC" -arch="$candidate" -c "$probe" -o /tmp/probe.o >/dev/null 2>&1; then
        rm -f "$probe" /tmp/probe.o
        return 0
    fi
    rm -f "$probe" /tmp/probe.o
    return 1
}

if ! choose_supported_arch "$ARCH"; then
    ORIGINAL="$ARCH"
    for FALLBACK in $SUPPORTED_ARCHS; do
        if choose_supported_arch "$FALLBACK"; then
            echo "WARNING: $ORIGINAL not supported by this nvcc; falling back to $FALLBACK"
            ARCH="$FALLBACK"
            break
        fi
    done
fi

if ! choose_supported_arch "$ARCH"; then
    echo "ERROR: could not find a CUDA arch supported by this nvcc."
    echo "Try upgrading CUDA toolkit or set CUDA_ARCH manually (for example CUDA_ARCH=sm_90)."
    exit 1
fi

# Build SASS for chosen arch + PTX for forward-compat JIT on newer GPUs.
COMPUTE="${ARCH#sm_}"
GENCODE=(
    "-gencode=arch=compute_${COMPUTE},code=sm_${COMPUTE}"
    "-gencode=arch=compute_${COMPUTE},code=compute_${COMPUTE}"
)

# ── compile ───────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
echo "Compiling coupled_field for $ARCH (with PTX forward-compat)..."
"$NVCC" -O3 -use_fast_math -lineinfo "${GENCODE[@]}" \
    -o coupled_field coupled_field.cu 2>&1

echo ""
echo "=== Build complete ==="
echo "Run:  ./deploy/run.sh [random|quantum|sparse] [output_dir] [snap_every]"
