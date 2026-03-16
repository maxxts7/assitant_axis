#!/usr/bin/env bash
# Quick test of Step 4 (compute role vectors) — averages activations for 5 test roles
#
# Expects Step 2 activations and Step 3 scores to already exist.
#
# Usage on RunPod:
#   chmod +x pipeline/test_vectors.sh
#   ./pipeline/test_vectors.sh

set -euo pipefail

ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-/workspace/test-activations/activations}"
SCORES_DIR="${SCORES_DIR:-/workspace/test-generate/scores}"
OUT="${OUT:-/workspace/test-generate/role_vectors}"
MIN_COUNT="${MIN_COUNT:-5}"

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

# ── Verify inputs exist ─────────────────────────────────────────────────────
if [ ! -d "$ACTIVATIONS_DIR" ]; then
    echo "ERROR: Activations directory not found: $ACTIVATIONS_DIR"
    echo "Run test_activations.sh first (or set ACTIVATIONS_DIR)."
    exit 1
fi

ACT_COUNT=$(find "$ACTIVATIONS_DIR" -name '*.pt' | wc -l)
if [ "$ACT_COUNT" -eq 0 ]; then
    echo "ERROR: No .pt files found in $ACTIVATIONS_DIR"
    exit 1
fi
echo "Found $ACT_COUNT activation file(s) in $ACTIVATIONS_DIR"

if [ ! -d "$SCORES_DIR" ]; then
    echo "ERROR: Scores directory not found: $SCORES_DIR"
    echo "Run test_judge.sh first (or set SCORES_DIR)."
    exit 1
fi

SCORE_COUNT=$(find "$SCORES_DIR" -name '*.json' | wc -l)
if [ "$SCORE_COUNT" -eq 0 ]; then
    echo "ERROR: No .json files found in $SCORES_DIR"
    exit 1
fi
echo "Found $SCORE_COUNT score file(s) in $SCORES_DIR"

echo ""
echo "Activations:  $ACTIVATIONS_DIR"
echo "Scores:       $SCORES_DIR"
echo "Output:       $OUT"
echo "Min count:    $MIN_COUNT (score=3 samples needed per role)"
echo ""

uv run "$PROJECT_DIR/pipeline/4_vectors.py" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --scores_dir "$SCORES_DIR" \
    --output_dir "$OUT" \
    --min_count "$MIN_COUNT"

echo ""
echo "Done! Role vector files:"
ls -lh "$OUT/"*.pt 2>/dev/null || echo "(no .pt files found — check logs above)"
