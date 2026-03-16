#!/usr/bin/env bash
# Quick test of Step 2 (activation extraction only) — runs on a handful of roles
#
# Expects Step 1 responses to already exist in RESPONSES_DIR.
#
# Usage on RunPod:
#   chmod +x pipeline/test_activations.sh
#   ./pipeline/test_activations.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
RESPONSES_DIR="${RESPONSES_DIR:-/workspace/test-generate/responses}"
OUT="${OUT:-/workspace/test-activations/activations}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TP_SIZE="${TP_SIZE:-4}"
LAYERS="${LAYERS:-all}"

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

# ── Verify responses exist ────────────────────────────────────────────────────
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

# ── Build arguments ──────────────────────────────────────────────────────────
TP_ARG=""
if [ -n "$TP_SIZE" ]; then
    TP_ARG="--tensor_parallel_size $TP_SIZE"
fi

ROLES="${ROLES:-default pirate therapist demon altruist}"
ROLES_ARG="--roles $ROLES"

echo ""
echo "Model:        $MODEL"
echo "Responses:    $RESPONSES_DIR"
echo "Output:       $OUT"
echo "Batch size:   $BATCH_SIZE"
echo "Layers:       $LAYERS"
echo ""

uv run "$PROJECT_DIR/pipeline/2_activations.py" \
    --model "$MODEL" \
    --responses_dir "$RESPONSES_DIR" \
    --output_dir "$OUT" \
    --batch_size "$BATCH_SIZE" \
    --layers "$LAYERS" \
    $TP_ARG \
    $ROLES_ARG

echo ""
echo "Done! Extracted activation files:"
ls -lh "$OUT/"*.pt 2>/dev/null || echo "(no .pt files found — check logs above)"
