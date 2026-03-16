# `pipeline/run_pipeline.sh`

## Overview

This is a **Bash shell script** that orchestrates the entire 5-step Assistant Axis pipeline end-to-end. It runs each Python script in sequence with preconfigured arguments for a specific model. While it can be run as a single command, the comments recommend running steps 1 and 2 individually (they are time-consuming) and note that step 3 can be run in parallel once step 1 is complete.

---

## Line-by-Line Explanation

### Shebang

```bash
#!/bin/bash
```

Specifies that the script should be executed with the Bash shell. This is the standard shebang for Bash scripts and ensures consistent behavior across systems that may have different default shells.

---

### Header comments

```bash
#
# Example pipeline for computing the Assistant Axis.
#
# This script runs all 5 steps of the pipeline for a given model.
# Adjust the parameters below for your setup.
#
# Usage:
#   ./pipeline/run_pipeline.sh
#   (I RECOMMEND RUNNING STEPS 1 AND 2 INDIVIDUALLY, 3 CAN BE RUN IN PARALLEL ONCE 1 IS DONE)
#
# Requirements:
#   - OPENAI_API_KEY environment variable (for step 3)
#   - Sufficient GPU memory for the model
```

Documentation comments explaining:

- **Purpose**: This is an example pipeline script for computing the Assistant Axis.
- **Usage**: Run it from the repository root with `./pipeline/run_pipeline.sh`.
- **Practical advice**: Steps 1 and 2 are long-running and should be run individually. Step 3 (the LLM judge) can be run in parallel with step 2 since it only depends on step 1's output (the responses).
- **Requirements**: An `OPENAI_API_KEY` environment variable is needed for step 3 (which uses an external LLM as a judge to score responses), and there must be sufficient GPU memory to load the model for steps 1 and 2.

---

### Error handling

```bash
set -e  # Exit on error
```

The `set -e` directive tells Bash to immediately exit the script if any command returns a non-zero exit code. This prevents the pipeline from continuing after a failure -- for example, if step 1 fails, steps 2--5 will not run. Without this, Bash would silently continue executing subsequent commands even after a failure.

---

### Configuration variables

```bash
# Configuration
MODEL="Qwen/Qwen3-32B"
OUTPUT_DIR="/workspace/qwen-3-32b/roles"
```

Two shell variables define the pipeline configuration:

- **`MODEL`**: The Hugging Face model identifier. In this example, it is set to `Qwen/Qwen3-32B`, a 32-billion-parameter model. This is passed to steps 1 and 2 which need to load the model.
- **`OUTPUT_DIR`**: The base directory where all pipeline outputs (responses, activations, scores, vectors, and the final axis) will be stored. Each step writes to a subdirectory within this path.

---

### Banner output

```bash
echo "=== Assistant Axis Pipeline ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo ""
```

Prints a header banner to the terminal showing the model being processed and the output directory. The `$MODEL` and `$OUTPUT_DIR` variables are expanded by the shell. This provides immediate visibility into the configuration when the script starts.

---

### Step 1: Generate responses

```bash
# Step 1: Generate responses
# THIS WILL TAKE A WHILE
# RECOMMEND RUNNING WITH https://github.com/justanhduc/task-spooler
echo "=== Step 1: Generating responses ==="
uv run 1_generate.py \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR/responses"
```

Runs the response generation script (`1_generate.py`) using the `uv run` command. `uv` is a Python package manager and runner that executes scripts in the project's virtual environment.

- **`--model "$MODEL"`**: Tells the script which model to load for generating responses.
- **`--output_dir "$OUTPUT_DIR/responses"`**: Responses are saved into a `responses/` subdirectory.

The comments warn that this step takes a long time and suggest using [task-spooler](https://github.com/justanhduc/task-spooler) -- a job queuing tool -- to manage execution. The backslashes (`\`) at the end of lines are line-continuation characters, allowing the command to span multiple lines for readability.

---

### Step 2: Extract activations

```bash
# Step 2: Extract activations
# SET BATCH SIZE APPROPRIATELY
# THIS WILL ALSO TAKE A WHILE
echo ""
echo "=== Step 2: Extracting activations ==="
uv run 2_activations.py \
    --model "$MODEL" \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/activations" \
    --batch_size 16
```

Runs the activation extraction script (`2_activations.py`).

- **`--model "$MODEL"`**: The same model is loaded again so that hidden-state activations can be captured.
- **`--responses_dir "$OUTPUT_DIR/responses"`**: Reads the responses generated in step 1.
- **`--output_dir "$OUTPUT_DIR/activations"`**: Saves activation tensors to an `activations/` subdirectory.
- **`--batch_size 16`**: Processes 16 responses at a time. The comment "SET BATCH SIZE APPROPRIATELY" indicates this should be tuned based on available GPU memory -- larger models or GPUs with less VRAM may require a smaller batch size.

---

### Step 3: Score responses with judge LLM

```bash
# Step 3: Score responses with judge LLM
# WILL NOT REPEAT WORK ON RERUN
# RERUN IS RECOMMENDED IN CASE SOME RESPONSES ARE MALFORMED
echo ""
echo "=== Step 3: Scoring responses ==="
uv run 3_judge.py \
    --responses_dir "$OUTPUT_DIR/responses" \
    --output_dir "$OUTPUT_DIR/scores"
```

Runs the judging script (`3_judge.py`) which uses an external LLM (via the OpenAI API, hence the `OPENAI_API_KEY` requirement) to score each response on a 0--3 scale indicating how well the model adopted the assigned role.

- **`--responses_dir "$OUTPUT_DIR/responses"`**: Reads the responses from step 1.
- **`--output_dir "$OUTPUT_DIR/scores"`**: Saves score JSON files to a `scores/` subdirectory.

The comments note two important properties:
1. **Idempotent**: The script will not re-score responses that already have scores, making it safe to re-run.
2. **Re-run recommended**: Some API responses may be malformed (e.g., the judge LLM returned unparseable output), so re-running can fill in any gaps.

---

### Step 4: Compute per-role vectors

```bash
# Step 4: Compute per-role vectors
echo ""
echo "=== Step 4: Computing per-role vectors ==="
uv run 4_vectors.py \
    --activations_dir "$OUTPUT_DIR/activations" \
    --scores_dir "$OUTPUT_DIR/scores" \
    --output_dir "$OUTPUT_DIR/vectors"
```

Runs the vector computation script (`4_vectors.py`, documented in detail in `docs/pipeline/4_vectors.md`).

- **`--activations_dir "$OUTPUT_DIR/activations"`**: Reads activation tensors from step 2.
- **`--scores_dir "$OUTPUT_DIR/scores"`**: Reads scores from step 3 to filter activations.
- **`--output_dir "$OUTPUT_DIR/vectors"`**: Saves per-role mean vectors to a `vectors/` subdirectory.

Note that `--min_count` is not specified here, so it defaults to 50 (as defined in the script).

---

### Step 5: Compute final axis

```bash
# Step 5: Compute final axis
echo ""
echo "=== Step 5: Computing axis ==="
uv run 5_axis.py \
    --vectors_dir "$OUTPUT_DIR/vectors" \
    --output "$OUTPUT_DIR/axis.pt"
```

Runs the final axis computation script (`5_axis.py`, documented in detail in `docs/pipeline/5_axis.md`).

- **`--vectors_dir "$OUTPUT_DIR/vectors"`**: Reads per-role vectors from step 4.
- **`--output "$OUTPUT_DIR/axis.pt"`**: Saves the final axis tensor directly into the output directory as `axis.pt`.

---

### Completion message

```bash
echo ""
echo "=== Pipeline complete ==="
echo "Axis saved to: $OUTPUT_DIR/axis.pt"
```

Prints a completion banner confirming the pipeline finished successfully and shows the path to the final output file. Since `set -e` is active, reaching this point guarantees that all five steps completed without error.
