#!/usr/bin/env bash
# Quick test of Step 1 (generation only) — ~250 responses instead of ~330,000
#
# Usage on RunPod:
#   chmod +x pipeline/test_generate.sh
#   ./pipeline/test_generate.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
OUT="${OUT:-/workspace/test-generate/responses}"
QUESTION_COUNT="${QUESTION_COUNT:-10}"
TP_SIZE="${TP_SIZE:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

TP_ARG=""
if [ -n "$TP_SIZE" ]; then
    TP_ARG="--tensor_parallel_size $TP_SIZE"
fi

echo "Model:     $MODEL"
echo "Output:    $OUT"
echo "Questions: $QUESTION_COUNT per role (× 5 prompts × 5 roles = ~$((QUESTION_COUNT * 5 * 5)) responses)"
echo ""

uv run "$PROJECT_DIR/pipeline/1_generate.py" \
    --model "$MODEL" \
    --roles_dir "$PROJECT_DIR/data/roles/instructions" \
    --questions_file "$PROJECT_DIR/data/extraction_questions.jsonl" \
    --output_dir "$OUT" \
    --question_count "$QUESTION_COUNT" \
    --roles default pirate therapist demon altruist \
    $TP_ARG

echo ""
echo "Done! Generated files:"
ls -lh "$OUT/"*.jsonl 2>/dev/null
