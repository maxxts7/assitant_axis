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

# ── Install dependencies ─────────────────────────────────────────────────────
echo "Installing dependencies..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Set HuggingFace token from RunPod secret
if [ -n "${RUNPOD_SECRET_hf_token:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$RUNPOD_SECRET_hf_token"
    echo "HF token loaded from RUNPOD_SECRET_hf_token"
elif [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "WARNING: No HuggingFace token found. Gated models (e.g. Llama) will fail."
fi

# Redirect HuggingFace cache to persistent storage
export HF_HOME="${HF_HOME:-/workspace/huggingface_cache}"

# Install project dependencies
cd "$PROJECT_DIR"
uv sync

# Verify GPU
if ! uv run python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No CUDA GPU detected."
    exit 1
fi

GPU_COUNT=$(uv run python3 -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected: $GPU_COUNT"

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
