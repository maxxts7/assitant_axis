#!/usr/bin/env bash
# Quick test of Step 3 (judge scoring) — scores the 5 test roles
#
# Expects Step 1 responses to already exist in RESPONSES_DIR.
# Requires OPENAI_API_KEY to be set.
#
# Usage on RunPod:
#   chmod +x pipeline/test_judge.sh
#   OPENAI_API_KEY=sk-... ./pipeline/test_judge.sh

set -euo pipefail

RESPONSES_DIR="${RESPONSES_DIR:-/workspace/test-generate/responses}"
OUT="${OUT:-/workspace/test-generate/scores}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
BATCH_SIZE="${BATCH_SIZE:-50}"
ROLES="${ROLES:-default pirate therapist demon altruist}"

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

# ── Check API key ────────────────────────────────────────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Usage: OPENAI_API_KEY=sk-... ./pipeline/test_judge.sh"
    exit 1
fi

# ── Verify responses exist ───────────────────────────────────────────────────
if [ ! -d "$RESPONSES_DIR" ]; then
    echo "ERROR: Responses directory not found: $RESPONSES_DIR"
    echo "Run test_generate.sh first (or set RESPONSES_DIR to your responses)."
    exit 1
fi

RESPONSE_COUNT=$(find "$RESPONSES_DIR" -name '*.jsonl' | wc -l)
if [ "$RESPONSE_COUNT" -eq 0 ]; then
    echo "ERROR: No .jsonl files found in $RESPONSES_DIR"
    exit 1
fi
echo "Found $RESPONSE_COUNT response file(s) in $RESPONSES_DIR"

echo ""
echo "Judge model:  $JUDGE_MODEL"
echo "Responses:    $RESPONSES_DIR"
echo "Output:       $OUT"
echo "Batch size:   $BATCH_SIZE"
echo "Roles:        $ROLES"
echo ""

uv run "$PROJECT_DIR/pipeline/3_judge.py" \
    --responses_dir "$RESPONSES_DIR" \
    --roles_dir "$PROJECT_DIR/data/roles/instructions" \
    --output_dir "$OUT" \
    --judge_model "$JUDGE_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --roles $ROLES

echo ""
echo "Done! Score files:"
ls -lh "$OUT/"*.json 2>/dev/null || echo "(no .json files found — check logs above)"
