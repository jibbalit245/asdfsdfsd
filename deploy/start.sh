#!/bin/bash
# start.sh — build (if needed) then launch one simulation instance
# Usage: ./deploy/start.sh [seed] [out_dir] [snap_every] [device]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -x "$REPO_ROOT/coupled_field" ]; then
    echo "=== coupled_field binary not found — running setup first ==="
    bash "$SCRIPT_DIR/setup.sh"
    echo ""
fi

exec bash "$SCRIPT_DIR/run.sh" "$@"
