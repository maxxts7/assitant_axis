#!/usr/bin/env bash
# =============================================================================
# test_pipeline.sh — Run a small test of the full Assistant Axis pipeline
#
# Generates ~250 responses (5 roles × 10 questions × 5 prompts) instead of
# the full ~330,000, then runs all 5 pipeline steps end-to-end.
#
# Usage (on RunPod):
#   chmod +x pipeline/test_pipeline.sh
#   ./pipeline/test_pipeline.sh
#
# Prerequisites:
#   - uv installed and on PATH
#   - HUGGING_FACE_HUB_TOKEN set (for gated models like Llama)
#   - OPENAI_API_KEY set (for the judge in step 3)
#   - GPU(s) available
# =============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these to match your setup
# ──────────────────────────────────────────────────────────────────────────────

# Model to test with (pick one)
MODEL="${MODEL:-Qwen/Qwen3-32B}"
# MODEL="google/gemma-2-27b-it"
# MODEL="meta-llama/Llama-3.3-70B-Instruct"

# Output directory (all artifacts go here)
OUT="${OUT:-/workspace/test-pipeline}"

# Number of questions per role (5 prompts × this = responses per role)
QUESTION_COUNT="${QUESTION_COUNT:-10}"

# Roles to test — default + a diverse mix of characters
# "default" is required (it's the Assistant baseline)
ROLES=(default pirate therapist demon altruist)

# Tensor parallel size — how many GPUs per worker
# Auto-detect if not set: uses all available GPUs
TP_SIZE="${TP_SIZE:-}"

# Batch size for activation extraction (reduce if OOM)
BATCH_SIZE="${BATCH_SIZE:-8}"

# Minimum score=3 samples to compute a role vector
# Lower this for test runs since we have fewer responses
MIN_COUNT="${MIN_COUNT:-5}"

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  Assistant Axis — Test Pipeline"
echo "=============================================="
echo "Model:          $MODEL"
echo "Output dir:     $OUT"
echo "Questions/role: $QUESTION_COUNT"
echo "Roles:          ${ROLES[*]}"
echo "Responses:      ~$((${#ROLES[@]} * QUESTION_COUNT * 5)) total"
echo "Project dir:    $PROJECT_DIR"
echo "=============================================="

# Build tensor parallel arg
TP_ARG=""
if [ -n "$TP_SIZE" ]; then
    TP_ARG="--tensor_parallel_size $TP_SIZE"
fi

# Check for GPU
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: No CUDA GPU detected. This script requires a GPU."
    exit 1
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected:  $GPU_COUNT"
echo ""

# Create output directories
mkdir -p "$OUT/responses" "$OUT/activations" "$OUT/scores" "$OUT/vectors"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate responses
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  Step 1/5: Generate responses"
echo "=============================="
echo "Generating ~$((${#ROLES[@]} * QUESTION_COUNT * 5)) responses for ${#ROLES[@]} roles..."
echo ""

uv run "$PROJECT_DIR/pipeline/1_generate.py" \
    --model "$MODEL" \
    --roles_dir "$PROJECT_DIR/data/roles/instructions" \
    --questions_file "$PROJECT_DIR/data/extraction_questions.jsonl" \
    --output_dir "$OUT/responses" \
    --question_count "$QUESTION_COUNT" \
    --roles ${ROLES[@]} \
    $TP_ARG

echo ""
echo "Step 1 complete. Generated files:"
ls -lh "$OUT/responses/"*.jsonl 2>/dev/null || echo "  (no files found — check for errors above)"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Extract activations
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  Step 2/5: Extract activations"
echo "=============================="
echo ""

uv run "$PROJECT_DIR/pipeline/2_activations.py" \
    --model "$MODEL" \
    --responses_dir "$OUT/responses" \
    --output_dir "$OUT/activations" \
    --batch_size "$BATCH_SIZE" \
    --roles ${ROLES[@]} \
    $TP_ARG

echo ""
echo "Step 2 complete. Activation files:"
ls -lh "$OUT/activations/"*.pt 2>/dev/null || echo "  (no files found — check for errors above)"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Judge responses
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  Step 3/5: Judge responses"
echo "=============================="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Skipping judge step."
    echo "Set it and re-run, or run step 3 manually:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo "  uv run $PROJECT_DIR/pipeline/3_judge.py --responses_dir $OUT/responses --roles_dir $PROJECT_DIR/data/roles/instructions --output_dir $OUT/scores --roles ${ROLES[*]}"
    SKIP_JUDGE=1
else
    echo ""
    uv run "$PROJECT_DIR/pipeline/3_judge.py" \
        --responses_dir "$OUT/responses" \
        --roles_dir "$PROJECT_DIR/data/roles/instructions" \
        --output_dir "$OUT/scores" \
        --roles ${ROLES[@]}

    echo ""
    echo "Step 3 complete. Score files:"
    ls -lh "$OUT/scores/"*.json 2>/dev/null || echo "  (no files found — check for errors above)"
    SKIP_JUDGE=0
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Compute per-role vectors
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  Step 4/5: Compute role vectors"
echo "=============================="

if [ "${SKIP_JUDGE:-0}" = "1" ]; then
    echo "Skipped (no scores from step 3)."
else
    echo ""
    uv run "$PROJECT_DIR/pipeline/4_vectors.py" \
        --activations_dir "$OUT/activations" \
        --scores_dir "$OUT/scores" \
        --output_dir "$OUT/vectors" \
        --min_count "$MIN_COUNT" \
        --overwrite

    echo ""
    echo "Step 4 complete. Vector files:"
    ls -lh "$OUT/vectors/"*.pt 2>/dev/null || echo "  (no files found — check for errors above)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: Compute the axis
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================="
echo "  Step 5/5: Compute axis"
echo "=============================="

if [ "${SKIP_JUDGE:-0}" = "1" ]; then
    echo "Skipped (no vectors from step 4)."
else
    echo ""
    uv run "$PROJECT_DIR/pipeline/5_axis.py" \
        --vectors_dir "$OUT/vectors" \
        --output "$OUT/axis.pt"

    echo ""
    echo "Step 5 complete!"
    echo ""

    # Verify the axis
    uv run python3 -c "
import torch
axis = torch.load('$OUT/axis.pt', weights_only=True)
print(f'Axis shape:  {axis.shape}')
print(f'Axis dtype:  {axis.dtype}')
print(f'Axis norm:   {axis.norm(dim=-1).mean():.4f}')
print('Pipeline test successful!')
"
fi

# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo "  Pipeline Test Complete"
echo "=============================================="
echo ""
echo "Output directory: $OUT"
echo ""
du -sh "$OUT"/*/ 2>/dev/null || true
echo ""
if [ -f "$OUT/axis.pt" ]; then
    echo "Axis file: $OUT/axis.pt"
    ls -lh "$OUT/axis.pt"
else
    echo "No axis file produced (check steps 3-5 above)."
fi
echo ""
echo "To run the full pipeline, remove the --roles and --question_count limits:"
echo "  uv run pipeline/1_generate.py --model $MODEL --output_dir $OUT/responses $TP_ARG"
