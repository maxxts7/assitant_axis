#!/usr/bin/env bash
# Quick test of Step 5 (compute assistant axis) — from test role vectors
#
# Expects Step 4 role vectors to already exist in VECTORS_DIR.
#
# Usage on RunPod:
#   chmod +x pipeline/test_axis.sh
#   ./pipeline/test_axis.sh

set -euo pipefail

VECTORS_DIR="${VECTORS_DIR:-/workspace/test-generate/role_vectors}"
OUT="${OUT:-/workspace/test-generate/axis.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Install dependencies ─────────────────────────────────────────────────────
echo "Installing dependencies..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Install project dependencies
cd "$PROJECT_DIR"
uv sync

# ── Verify vectors exist ────────────────────────────────────────────────────
if [ ! -d "$VECTORS_DIR" ]; then
    echo "ERROR: Vectors directory not found: $VECTORS_DIR"
    echo "Run test_vectors.sh first (or set VECTORS_DIR)."
    exit 1
fi

VEC_COUNT=$(find "$VECTORS_DIR" -name '*.pt' | wc -l)
if [ "$VEC_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in $VECTORS_DIR"
    exit 1
fi
echo "Found $VEC_COUNT vector file(s) in $VECTORS_DIR"

echo ""
echo "Vectors:  $VECTORS_DIR"
echo "Output:   $OUT"
echo ""

uv run "$PROJECT_DIR/pipeline/5_axis.py" \
    --vectors_dir "$VECTORS_DIR" \
    --output "$OUT"

echo ""
echo "Done! Axis file:"
ls -lh "$OUT" 2>/dev/null || echo "(axis file not found — check logs above)"
