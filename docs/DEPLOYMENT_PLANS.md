# Deployment Plans for the Assistant Axis

This document presents **7 distinct deployment plans** for the Assistant Axis codebase, ranging from lightweight research setups to production-grade safety infrastructure. Each plan covers: architecture, hardware, software stack, data flow, latency, cost, scaling, risks, and step-by-step implementation.

---

## Table of Contents

1. [Plan 1: Research Pipeline on GPU Cluster](#plan-1)
2. [Plan 2: Real-Time Persona Drift Monitor](#plan-2)
3. [Plan 3: Inference-Time Safety Layer (Capping Service)](#plan-3)
4. [Plan 4: Interactive Research Platform (JupyterHub)](#plan-4)
5. [Plan 5: Batch Evaluation & Red-Teaming Pipeline](#plan-5)
6. [Plan 6: Embedded Safety Module in Model Serving](#plan-6)
7. [Plan 7: Edge / On-Premise Deployment](#plan-7)
8. [Comparison Matrix](#comparison)
9. [Shared Infrastructure Components](#shared)

---

## Plan 1: Research Pipeline on GPU Cluster <a id="plan-1"></a>

**Goal:** Run the full 5-step pipeline to compute the Assistant Axis for new models, or reproduce existing results.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU CLUSTER (SLURM / Kubernetes)         │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │ Worker 0  │  │ Worker 1  │  │ Worker 2  │  ...          │
│  │ GPU 0,1   │  │ GPU 2,3   │  │ GPU 4,5   │              │
│  │ vLLM TP=2 │  │ vLLM TP=2 │  │ vLLM TP=2 │              │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│  ┌──────────────────────────────────────────────┐           │
│  │           Shared NFS / Lustre Storage         │           │
│  │  responses/  activations/  scores/  vectors/  │           │
│  └──────────────────────────────────────────────┘           │
│                         │                                    │
│                         ▼                                    │
│              ┌─────────────────┐                            │
│              │  CPU Node       │                            │
│              │  Steps 4 + 5   │                            │
│              │  axis.pt output │                            │
│              └─────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │  OpenAI API     │
              │  (Step 3 only)  │
              └─────────────────┘
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPUs | 2x A100 40GB | 8x A100 80GB or 4x H100 |
| System RAM | 256 GB | 512 GB |
| Storage | 500 GB SSD | 2 TB NVMe |
| Network | 10 Gbps | 100 Gbps (InfiniBand for multi-node) |

### Software Stack

```
OS: Ubuntu 22.04 LTS
Python: 3.10+
CUDA: 12.1+
Driver: 535+

Core packages:
  torch>=2.0 (with CUDA)
  transformers>=4.40
  accelerate
  vllm
  openai
  scikit-learn

Orchestration (pick one):
  - SLURM (HPC clusters)
  - Kubernetes + NVIDIA GPU Operator
  - task-spooler (simple queue on single node)
  - tmux (manual, for small runs)

Package management:
  uv (recommended) or pip
```

### Step-by-Step Implementation

**Step 1: Environment Setup**
```bash
# Clone and install
git clone https://github.com/<repo>/assistant-axis.git
cd assistant-axis
uv sync

# Set environment
echo "OPENAI_API_KEY=sk-..." > .env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**Step 2: Run Pipeline**
```bash
MODEL="Qwen/Qwen3-32B"
OUT="outputs/qwen-3-32b"

# Step 1: Generate responses (multi-worker auto-parallelizes)
uv run pipeline/1_generate.py \
    --model $MODEL \
    --output_dir $OUT/responses \
    --question_count 240 \
    --temperature 0.7 \
    --max_tokens 512 \
    --tensor_parallel_size 2

# Step 2: Extract activations
uv run pipeline/2_activations.py \
    --model $MODEL \
    --responses_dir $OUT/responses \
    --output_dir $OUT/activations \
    --layers all \
    --batch_size 16

# Step 3: Score with judge (runs on CPU, calls OpenAI API)
uv run pipeline/3_judge.py \
    --responses_dir $OUT/responses \
    --roles_dir data/roles/instructions \
    --output_dir $OUT/scores \
    --judge_model gpt-4.1-mini

# Step 4: Compute per-role vectors (CPU-only, fast)
uv run pipeline/4_vectors.py \
    --activations_dir $OUT/activations \
    --scores_dir $OUT/scores \
    --output_dir $OUT/vectors \
    --min_count 50

# Step 5: Compute axis (CPU-only, seconds)
uv run pipeline/5_axis.py \
    --vectors_dir $OUT/vectors \
    --output $OUT/axis.pt
```

### Execution Timeline

| Step | Duration (8x A100) | Duration (2x A100) | Parallelizable |
|------|--------------------|--------------------|----------------|
| 1. Generate | 8-12 hours | 30-50 hours | Yes (multi-worker) |
| 2. Extract | 6-12 hours | 25-50 hours | Yes (multi-worker) |
| 3. Judge | 4-8 hours | 4-8 hours | Yes (async API) |
| 4. Vectors | <1 hour | <1 hour | No (trivial) |
| 5. Axis | <1 minute | <1 minute | No (trivial) |
| **Total** | **~20-35 hours** | **~60-110 hours** | Steps 2 & 3 overlap |

### Disk Usage

| Artifact | Size per Role | Total (275 roles) |
|----------|---------------|-------------------|
| Responses (JSONL) | ~5 MB | ~1.4 GB |
| Activations (.pt) | ~500 MB-2 GB | ~140-500 GB |
| Scores (JSON) | ~10 KB | ~3 MB |
| Vectors (.pt) | ~500 KB | ~140 MB |
| Axis (.pt) | — | ~2-5 MB |

### Resume & Fault Tolerance

- All pipeline steps check for existing output files and skip completed roles
- Steps 1-3 can be killed and restarted safely
- Step 3 merges new scores with existing ones incrementally
- No formal checkpointing — state is in output files

### Cost Estimate

| Resource | Unit Cost | Quantity | Total |
|----------|-----------|----------|-------|
| 8x A100 80GB (cloud) | ~$25/hr | 30 hours | ~$750 |
| OpenAI API (gpt-4.1-mini) | ~$0.15/1M tokens | ~50M tokens | ~$7.50 |
| Storage (500GB SSD) | ~$0.10/GB/mo | 500 GB | ~$50/mo |
| **Total per model** | | | **~$800** |

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM during extraction | Step 2 fails | Reduce batch_size (16→8→4) |
| OpenAI rate limiting | Step 3 slows | Reduce requests_per_second |
| Disk full during activations | Step 2 fails | Monitor disk, clean up early steps |
| Model download fails | Pipeline blocked | Pre-download with `huggingface-cli` |
| Role gets <50 score=3 responses | Role excluded from axis | Lower min_count or regenerate with different prompts |

---

## Plan 2: Real-Time Persona Drift Monitor <a id="plan-2"></a>

**Goal:** Monitor live conversations for persona drift by projecting each response's activations onto the pre-computed axis. Alert when the model drifts beyond a threshold.

### Architecture

```
┌──────────────┐     ┌───────────────────────────────────────────┐
│   User       │     │          Drift Monitor Service             │
│   Request    │────▶│                                           │
└──────────────┘     │  ┌─────────────────┐                     │
                     │  │  Model Server    │                     │
                     │  │  (HuggingFace)   │                     │
                     │  │  + Forward Hooks  │                     │
                     │  └────────┬────────┘                     │
                     │           │ activations                   │
                     │           ▼                               │
                     │  ┌─────────────────┐    ┌──────────────┐ │
                     │  │  Projection      │───▶│  Alert Engine │ │
                     │  │  Module          │    │  (threshold)  │ │
                     │  │  axis.project()  │    └──────┬───────┘ │
                     │  └─────────────────┘           │          │
                     │                                ▼          │
                     │                       ┌──────────────┐    │
                     │                       │  Dashboard /  │    │
                     │                       │  Logging      │    │
                     │                       └──────────────┘    │
                     └───────────────────────────────────────────┘
```

### How It Works

1. User sends a message to the model
2. Model generates a response via `model.generate()`
3. Forward hooks capture activations at the target layer during generation
4. After generation completes, the response activations are projected onto the pre-loaded axis
5. The projection value is compared against a threshold
6. If below threshold → alert (drift detected); if above → normal operation

### Software Stack

```python
# Core components needed:
import torch
from assistant_axis import load_axis, project
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper

# Pre-load (one-time):
pm = ProbingModel("Qwen/Qwen3-32B")
axis = load_axis("axis.pt")                    # ~2 MB, loads in <1s
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
extractor = ActivationExtractor(pm, encoder)
mapper = SpanMapper(pm.tokenizer)
config = get_config("Qwen/Qwen3-32B")
target_layer = config["target_layer"]          # 32

# Per-request:
def monitor_response(conversation):
    """Returns (response_text, projection_value, is_drifting)."""
    # 1. Generate response
    response = pm.generate(conversation, max_new_tokens=512)

    # 2. Extract activations for the full conversation + response
    full_convo = conversation + [{"role": "assistant", "content": response}]
    activations = extractor.full_conversation(full_convo, layer=target_layer)

    # 3. Compute mean activation over assistant response tokens
    full_ids, spans = encoder.build_turn_spans(full_convo)
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    last_span = assistant_spans[-1]
    response_act = activations[last_span["start"]:last_span["end"]].mean(dim=0)

    # 4. Project onto axis
    proj = project(response_act.unsqueeze(0).unsqueeze(0), axis, layer=0, normalize=True)

    # 5. Check threshold
    DRIFT_THRESHOLD = -2.0  # Calibrate empirically
    is_drifting = proj < DRIFT_THRESHOLD

    return response, proj, is_drifting
```

### Hardware Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU | 1x A100 40GB (Gemma) or 2x A100 (Qwen/Llama) | Model must fit in GPU memory |
| RAM | 64 GB | For tokenizer and activation buffers |
| Axis file | 2-5 MB | Negligible |
| Latency overhead | ~100-500 ms per response | Extraction + projection after generation |

### Monitoring Dashboard

```
┌─────────────────────────────────────────────┐
│  Persona Drift Monitor                       │
│                                              │
│  Session: user_12345                         │
│  Model: Qwen 3 32B                          │
│  Axis: qwen-3-32b/assistant_axis.pt         │
│                                              │
│  Turn  Projection  Status                    │
│  ────  ──────────  ──────                    │
│    1      +4.2     OK                        │
│    2      +3.8     OK                        │
│    3      +1.1     WARNING (declining)       │
│    4      -0.5     WARNING (below zero)      │
│    5      -3.2     ALERT (persona drift!)    │
│                                              │
│  [Trajectory Plot]                           │
│  +5 ──●──●                                   │
│   0 ────────●                                │
│  -5 ───────────●──●                          │
│       1   2   3  4  5  Turn                  │
└─────────────────────────────────────────────┘
```

### Alert Thresholds (Calibration)

The paper provides guidance for threshold calibration:

| Zone | Projection Range | Meaning | Action |
|------|-----------------|---------|--------|
| Safe | > +2.0 | Strong assistant identity | None |
| Normal | 0.0 to +2.0 | Mild variation | None |
| Warning | -2.0 to 0.0 | Persona weakening | Log, flag for review |
| Alert | < -2.0 | Significant drift | Intervene (apply capping, warn user) |

Thresholds should be calibrated per model using the activation distribution from normal assistant conversations (the 25th percentile from the paper's capping calibration is a good starting point).

### Integration Points

- **Logging backend:** Prometheus + Grafana, or ELK stack
- **Alerting:** PagerDuty, Slack webhook, or custom handler
- **Per-session tracking:** Store projection trajectory in Redis/Postgres
- **Multi-turn context:** Track projection across entire conversation, detect downward trends

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Extraction overhead slows responses | User-facing latency | Run extraction asynchronously after response delivery |
| False positives (legitimate role variety flagged) | Alert fatigue | Tune thresholds with domain-specific calibration data |
| GPU memory contention | OOM during extraction | Use separate GPU for monitoring, or extract after response |
| Axis not calibrated for conversation domain | Inaccurate projections | Re-compute axis for domain-specific conversations |

---

## Plan 3: Inference-Time Safety Layer (Capping Service) <a id="plan-3"></a>

**Goal:** Deploy activation capping as a transparent safety layer that prevents harmful persona drift during model inference, without degrading capabilities.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVICE                          │
│                                                              │
│  ┌────────────┐    ┌──────────────────────────────────┐     │
│  │   FastAPI   │───▶│  Model + Capping Layer            │     │
│  │   Gateway   │    │                                    │     │
│  │   /generate │    │  ┌──────────────┐                 │     │
│  │   /health   │    │  │ HuggingFace  │                 │     │
│  │   /config   │    │  │ Model        │                 │     │
│  └────────────┘    │  │ .generate()  │                 │     │
│                     │  └──────┬───────┘                 │     │
│                     │         │                          │     │
│                     │  ┌──────▼───────┐                 │     │
│                     │  │ActivationSteering              │     │
│                     │  │ (capping hooks)                 │     │
│                     │  │                                │     │
│                     │  │ For each layer in [46..53]:    │     │
│                     │  │   if proj < tau:               │     │
│                     │  │     h += (tau - proj) * v_norm │     │
│                     │  └────────────────┘                │     │
│                     └──────────────────────────────────┘     │
│                                                              │
│  Pre-loaded artifacts:                                       │
│    axis.pt (2 MB)  +  capping_config.pt (80 MB)             │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
"""capping_server.py — FastAPI wrapper for capped model inference."""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from assistant_axis import load_axis, get_config
from assistant_axis.steering import load_capping_config, build_capping_steerer
from assistant_axis.internals import ProbingModel

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-32B"
AXIS_PATH = "qwen-3-32b/assistant_axis.pt"
CAPPING_CONFIG_PATH = "qwen-3-32b/capping_config.pt"
EXPERIMENT_ID = "layers_46:54-p0.25"

# --- Global state ---
pm = None
steerer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and capping config on startup."""
    global pm, steerer
    pm = ProbingModel(MODEL_NAME)
    axis = load_axis(AXIS_PATH)
    config = load_capping_config(CAPPING_CONFIG_PATH)
    steerer = build_capping_steerer(pm.model, config, EXPERIMENT_ID)
    yield
    pm.close()

app = FastAPI(lifespan=lifespan)

class GenerateRequest(BaseModel):
    messages: list[dict]       # [{"role": "user", "content": "..."}]
    max_tokens: int = 512
    temperature: float = 0.7
    capping_enabled: bool = True

class GenerateResponse(BaseModel):
    response: str
    capping_active: bool

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    conversation = req.messages

    if req.capping_enabled:
        with steerer:
            response = pm.generate(
                conversation,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                chat_format=True,
            )
    else:
        response = pm.generate(
            conversation,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            chat_format=True,
        )

    return GenerateResponse(response=response, capping_active=req.capping_enabled)

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "experiment": EXPERIMENT_ID}

@app.get("/config")
async def config():
    cfg = get_config(MODEL_NAME)
    return {
        "model": MODEL_NAME,
        "target_layer": cfg["target_layer"],
        "total_layers": cfg["total_layers"],
        "capping_experiment": EXPERIMENT_ID,
        "capping_layers": "46-53",
        "capping_percentile": "p0.25",
    }
```

### Deployment Commands

```bash
# Install additional dependencies
pip install fastapi uvicorn

# Run the server
uvicorn capping_server:app --host 0.0.0.0 --port 8000 --workers 1

# Docker deployment
docker build -t capping-server .
docker run --gpus all -p 8000:8000 capping-server

# Test
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "capping_enabled": true}'
```

### Hardware Requirements

| Model | GPUs | RAM | Latency (per response) |
|-------|------|-----|----------------------|
| Gemma 2 27B | 1x A100 40GB | 128 GB | 5-15 seconds |
| Qwen 3 32B | 2x A100 40GB or 1x H100 | 256 GB | 8-20 seconds |
| Llama 3.3 70B | 2x A100 80GB | 512 GB | 10-30 seconds |

### Performance Impact of Capping

From the paper's evaluation:

| Benchmark | Uncapped | Capped (p0.25) | Delta |
|-----------|----------|----------------|-------|
| IFEval | Baseline | No change | 0% |
| MMLU Pro | Baseline | No change | 0% |
| GSM8k | Baseline | No change | 0% |
| EQ-Bench | Baseline | No change | 0% |
| Harmful responses | 65-88% (with jailbreak) | ~25-35% | **-60%** |

**Key insight:** Capping has zero overhead on normal assistant behavior because activations are already above the threshold. It only activates when persona drift is detected.

### Scaling Considerations

- **Horizontal scaling:** Each server instance holds the full model — scale by adding more GPU nodes behind a load balancer
- **Request queuing:** Add Redis/RabbitMQ for request buffering during high load
- **Batching:** Cannot batch with capping (hooks require per-sequence processing). For high throughput, use multiple instances
- **Warm-up:** Model loading takes 30-60 seconds. Use health checks and readiness probes in Kubernetes

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM incompatibility | Can't use vLLM's fast batching | Use HuggingFace generate(); accept lower throughput |
| Single-request processing | Low throughput | Horizontal scaling with load balancer |
| Model reload on crash | 30-60s downtime | Kubernetes rolling restart, health probes |
| Threshold too aggressive | Normal responses altered | Use p0.25 (paper-validated), A/B test before deploying |

---

## Plan 4: Interactive Research Platform (JupyterHub) <a id="plan-4"></a>

**Goal:** Deploy a multi-user research environment where researchers can interactively explore the axis, run PCA analysis, steer models, and visualize results.

### Architecture

```
┌────────────────────────────────────────────────────┐
│             JupyterHub (Multi-User)                 │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ Researcher A │  │ Researcher B │  │ Researcher C│ │
│  │ Jupyter Lab  │  │ Jupyter Lab  │  │ Jupyter Lab │ │
│  │ GPU: 0,1     │  │ GPU: 2,3     │  │ GPU: 4,5   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬─────┘ │
│         │                │                 │        │
│         ▼                ▼                 ▼        │
│  ┌─────────────────────────────────────────────┐   │
│  │      Shared Storage (NFS / S3)               │   │
│  │  pre-computed/                                │   │
│  │    axis.pt, role_vectors/, capping_configs/   │   │
│  │  user_workspaces/                             │   │
│  │    researcher_a/, researcher_b/, ...          │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

### Pre-Installed Notebooks

| Notebook | Purpose | GPU Required |
|----------|---------|-------------|
| `core_concepts.ipynb` | Learn the codebase (educational) | No |
| `tunable_parameters.ipynb` | Explore all hyperparameters | No |
| `pca.ipynb` | PCA analysis of role vectors | No (CPU PCA) |
| `visualize_axis.ipynb` | Cosine similarity visualization | No |
| `steer.ipynb` | Interactive steering demo | Yes |
| `project_transcript.ipynb` | Analyze persona drift in conversations | Yes |
| `poc_end_to_end.ipynb` | Full proof-of-concept | Yes |

### Setup

```bash
# Install JupyterHub
pip install jupyterhub jupyterlab

# Configure GPU allocation (jupyterhub_config.py)
c.Spawner.environment = {
    'CUDA_VISIBLE_DEVICES': '{gpu_ids}',  # Assigned per user
}
c.Spawner.mem_limit = '128G'
c.Spawner.cpu_limit = 16

# Pre-download artifacts
python -c "
from huggingface_hub import snapshot_download
snapshot_download('lu-christina/assistant-axis-vectors', repo_type='dataset', local_dir='pre-computed/')
"

# Launch
jupyterhub --config jupyterhub_config.py
```

### User Workflow

1. **Login** → JupyterHub assigns GPU(s) and workspace
2. **Explore** → Open `core_concepts.ipynb` to understand the project
3. **Analyze** → Run `pca.ipynb` with pre-computed vectors (no GPU needed)
4. **Experiment** → Open `steer.ipynb`, load a model, try different coefficients
5. **Save** → Results saved to user workspace on shared storage

### Hardware (for 3-5 concurrent researchers)

| Component | Specification |
|-----------|--------------|
| GPUs | 8x A100 80GB (2 per researcher + shared) |
| CPU | 64 cores |
| RAM | 1 TB |
| Storage | 4 TB NVMe (models + activations + workspaces) |
| Network | 10 Gbps |

### Cost Estimate (Cloud)

| Provider | Instance | Monthly (reserved) |
|----------|----------|-------------------|
| AWS | p4d.24xlarge (8x A100) | ~$15,000/mo |
| GCP | a2-ultragpu-8g | ~$14,000/mo |
| Lambda | 8x A100 | ~$10,000/mo |

---

## Plan 5: Batch Evaluation & Red-Teaming Pipeline <a id="plan-5"></a>

**Goal:** Systematically evaluate model safety by running large-scale steering and jailbreak experiments, measuring harmful response rates with and without capping.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION PIPELINE                          │
│                                                              │
│  ┌──────────────┐    ┌────────────────┐                     │
│  │ Jailbreak     │    │ Model + Axis    │                     │
│  │ Dataset       │───▶│                 │                     │
│  │ (1,100 pairs) │    │ Conditions:     │                     │
│  └──────────────┘    │  - Uncapped     │                     │
│                       │  - Capped p0.10 │                     │
│                       │  - Capped p0.25 │                     │
│                       │  - Capped p0.50 │                     │
│                       │  - Steered +10  │                     │
│                       │  - Steered +20  │                     │
│                       └───────┬────────┘                     │
│                               │                              │
│                               ▼                              │
│                       ┌──────────────┐                       │
│                       │  LLM Judge    │                       │
│                       │  (harm eval)  │                       │
│                       └───────┬──────┘                       │
│                               │                              │
│                               ▼                              │
│                       ┌──────────────┐                       │
│                       │  Results DB   │                       │
│                       │  + Dashboard  │                       │
│                       └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
"""batch_evaluation.py — Run systematic safety evaluation."""

import json
import torch
from pathlib import Path
from assistant_axis import load_axis, get_config, ActivationSteering
from assistant_axis.steering import load_capping_config, build_capping_steerer
from assistant_axis.internals import ProbingModel

MODEL = "Qwen/Qwen3-32B"
pm = ProbingModel(MODEL)
axis = load_axis("axis.pt")
config = get_config(MODEL)
capping_config = load_capping_config("capping_config.pt")

# Load jailbreak dataset (from Shah et al.)
jailbreak_pairs = json.load(open("data/jailbreak_pairs.json"))

# Define experimental conditions
conditions = {
    "baseline": lambda: None,  # No intervention (null context manager)
    "capped_p0.10": lambda: build_capping_steerer(pm.model, capping_config, "layers_46:54-p0.10"),
    "capped_p0.25": lambda: build_capping_steerer(pm.model, capping_config, "layers_46:54-p0.25"),
    "capped_p0.50": lambda: build_capping_steerer(pm.model, capping_config, "layers_46:54-p0.50"),
    "steered_+10": lambda: ActivationSteering(
        pm.model, [axis[config["target_layer"]]], [10.0], [config["target_layer"]]
    ),
    "steered_+20": lambda: ActivationSteering(
        pm.model, [axis[config["target_layer"]]], [20.0], [config["target_layer"]]
    ),
}

results = {}
for condition_name, steerer_fn in conditions.items():
    responses = []
    steerer = steerer_fn()

    for pair in jailbreak_pairs:
        conversation = [
            {"role": "system", "content": pair["system_prompt"]},
            {"role": "user", "content": pair["question"]},
        ]

        if steerer is not None:
            with steerer:
                response = pm.generate(conversation, max_new_tokens=512, chat_format=True)
        else:
            response = pm.generate(conversation, max_new_tokens=512, chat_format=True)

        responses.append({
            "condition": condition_name,
            "system_prompt": pair["system_prompt"],
            "question": pair["question"],
            "response": response,
        })

    results[condition_name] = responses

# Save for judging
for condition, responses in results.items():
    Path(f"eval_results/{condition}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses)
    )
```

### Evaluation Metrics

| Metric | Source | How to Compute |
|--------|--------|----------------|
| Harmful response rate | LLM judge | % of responses flagged as harmful |
| Refusal rate | LLM judge | % of responses that refuse the prompt |
| Projection distribution | axis.project() | Mean/std of projections per condition |
| IFEval score | IFEval benchmark | % of instruction-following tests passed |
| MMLU Pro score | MMLU benchmark | % correct on knowledge questions |
| GSM8k score | GSM8k benchmark | % correct on math problems |
| EQ-Bench score | EQ-Bench | Emotional intelligence score |

### Experiment Matrix

For a complete evaluation, run each condition against:
- 1,100 persona-based jailbreak pairs (44 harm categories)
- 541 IFEval problems
- 1,400 MMLU Pro problems (subsampled)
- 1,000 GSM8k problems (subsampled)
- 171 EQ-Bench problems

**Total: ~4,200 generations x 6 conditions = ~25,200 generations per model**

### Hardware & Time

| Setup | GPUs | Time per Model |
|-------|------|---------------|
| Minimum | 2x A100 | ~40-60 hours |
| Recommended | 4x A100 | ~20-30 hours |
| Fast | 8x H100 | ~8-12 hours |

---

## Plan 6: Embedded Safety Module in Model Serving <a id="plan-6"></a>

**Goal:** Integrate the Assistant Axis capping mechanism into an existing model serving infrastructure (e.g., a company's internal LLM API) as a transparent safety layer.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  EXISTING MODEL SERVING STACK                │
│                                                              │
│  ┌──────────┐    ┌──────────────────────────────────┐       │
│  │  API      │    │     Model Runtime                 │       │
│  │  Gateway  │───▶│                                   │       │
│  │  (nginx/  │    │  ┌────────────────────────────┐  │       │
│  │   envoy)  │    │  │  Safety Middleware           │  │       │
│  └──────────┘    │  │  (assistant_axis integration) │  │       │
│                   │  │                              │  │       │
│                   │  │  Pre-request:                │  │       │
│                   │  │    Register capping hooks    │  │       │
│                   │  │                              │  │       │
│                   │  │  Post-request:               │  │       │
│                   │  │    Remove hooks              │  │       │
│                   │  │    Log projection value      │  │       │
│                   │  └────────────┬───────────────┘  │       │
│                   │               │                   │       │
│                   │  ┌────────────▼───────────────┐  │       │
│                   │  │  HuggingFace Model          │  │       │
│                   │  │  (with forward hooks)       │  │       │
│                   │  └────────────────────────────┘  │       │
│                   └──────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Observability                                    │       │
│  │  - Prometheus metrics (projection values)         │       │
│  │  - Grafana dashboard (drift detection)            │       │
│  │  - Alert manager (threshold violations)           │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Integration Code

```python
"""safety_middleware.py — Drop-in safety layer for model serving."""

import torch
from assistant_axis import load_axis, get_config
from assistant_axis.steering import load_capping_config, build_capping_steerer, ActivationSteering

class SafetyMiddleware:
    """Wraps a HuggingFace model with activation capping."""

    def __init__(self, model, model_name, axis_path, capping_config_path, experiment_id):
        self.model = model
        self.axis = load_axis(axis_path)
        self.config = get_config(model_name)
        self.capping_config = load_capping_config(capping_config_path)
        self.experiment_id = experiment_id
        self.steerer = build_capping_steerer(model, self.capping_config, experiment_id)
        self._active = True

    def generate(self, input_ids, **kwargs):
        """Generate with capping active."""
        if self._active:
            with self.steerer:
                return self.model.generate(input_ids, **kwargs)
        else:
            return self.model.generate(input_ids, **kwargs)

    def enable(self):
        """Enable capping."""
        self._active = True

    def disable(self):
        """Disable capping (for A/B testing)."""
        self._active = False

# Usage in existing serving code:
# Before:  output = model.generate(input_ids, max_new_tokens=512)
# After:   output = safety.generate(input_ids, max_new_tokens=512)
```

### Deployment Checklist

- [ ] Pre-compute and cache axis + capping config for your model
- [ ] Validate capping has no capability regression on your evaluation suite
- [ ] Set up monitoring for projection values (Prometheus/Grafana)
- [ ] Configure alerting thresholds based on your risk tolerance
- [ ] Run A/B test: capped vs uncapped on production traffic (shadow mode)
- [ ] Enable for all traffic after validation
- [ ] Set up periodic re-calibration of thresholds as model or data shifts

### Overhead Analysis

| Aspect | Impact |
|--------|--------|
| Memory | +80-150 MB for capping config tensors |
| Latency | +50-100 ms per generation (hook overhead) |
| Throughput | No change (hooks don't block batching within HuggingFace) |
| Capability | No change (paper-validated on 4 benchmarks) |

---

## Plan 7: Edge / On-Premise Deployment <a id="plan-7"></a>

**Goal:** Deploy the axis computation and capping capabilities on-premise, in air-gapped environments, or on edge devices with limited GPU resources.

### Architecture

```
┌────────────────────────────────────────────┐
│           ON-PREMISE SERVER                 │
│                                             │
│  ┌────────────┐    ┌──────────────────┐    │
│  │ Smaller     │    │ Pre-computed     │    │
│  │ Model       │    │ Artifacts        │    │
│  │ (Gemma 27B) │    │ (from cloud)     │    │
│  │             │    │  axis.pt         │    │
│  │ 1x A100     │    │  capping.pt      │    │
│  └──────┬─────┘    └──────────────────┘    │
│         │                                   │
│         ▼                                   │
│  ┌────────────────────────────────────┐    │
│  │  Local Inference + Capping         │    │
│  │  (no internet required)            │    │
│  └────────────────────────────────────┘    │
└────────────────────────────────────────────┘
```

### Key Constraints

| Constraint | Solution |
|------------|----------|
| No internet | Pre-download model weights + axis + capping config |
| Limited GPUs | Use smallest supported model (Gemma 2 27B, 1x A100) |
| No OpenAI API | Skip judge step; use pre-computed axis from HuggingFace |
| Air-gapped | Bundle all dependencies in Docker image or offline installer |

### Offline Bundle

```bash
# Prepare offline bundle (on internet-connected machine)

# 1. Download model weights
huggingface-cli download google/gemma-2-27b-it --local-dir ./bundle/model/

# 2. Download pre-computed axis and capping config
huggingface-cli download lu-christina/assistant-axis-vectors \
    --repo-type dataset --local-dir ./bundle/vectors/

# 3. Export Python environment
pip download -r requirements.txt -d ./bundle/wheels/

# 4. Package
tar czf assistant-axis-bundle.tar.gz bundle/

# --- On air-gapped machine ---
tar xzf assistant-axis-bundle.tar.gz
pip install --no-index --find-links=bundle/wheels/ -r requirements.txt

# Run with local paths
python -c "
from assistant_axis import load_axis
from assistant_axis.internals import ProbingModel
pm = ProbingModel('bundle/model/')
axis = load_axis('bundle/vectors/gemma-2-27b/assistant_axis.pt')
print('Ready for local inference with capping')
"
```

### Minimum Hardware for Edge

| Config | Model | GPUs | RAM | Storage |
|--------|-------|------|-----|---------|
| Minimal | Gemma 2 27B | 1x A100 40GB | 128 GB | 100 GB |
| Standard | Qwen 3 32B | 2x A100 40GB | 256 GB | 200 GB |
| Full | Llama 3.3 70B | 2x A100 80GB | 512 GB | 400 GB |

---

## Comparison Matrix <a id="comparison"></a>

| Aspect | Plan 1: Pipeline | Plan 2: Monitor | Plan 3: Capping | Plan 4: JupyterHub | Plan 5: Eval | Plan 6: Embedded | Plan 7: Edge |
|--------|-----------------|-----------------|-----------------|-------------------|-------------|-----------------|-------------|
| **Purpose** | Compute axis | Detect drift | Prevent drift | Research | Test safety | Production safety | Offline safety |
| **GPU Min** | 2x A100 | 1-2x A100 | 1-2x A100 | 8x A100 | 2x A100 | 1-2x A100 | 1x A100 |
| **Internet** | Required | Optional | Optional | Optional | Required | Optional | Not needed |
| **OpenAI API** | Required | No | No | No | Required | No | No |
| **Latency** | Batch (hours) | ~500ms overhead | ~100ms overhead | Interactive | Batch (hours) | ~100ms overhead | ~100ms overhead |
| **Users** | 1 researcher | Ops team | All users | 3-5 researchers | 1 researcher | All users | All users |
| **Complexity** | Medium | Medium | Medium | High | Medium | Low (drop-in) | Low |
| **Cost/month** | ~$800 one-time | ~$5,000 | ~$5,000 | ~$12,000 | ~$800 one-time | Existing infra | Hardware cost |
| **Pre-computed axis needed** | No (creates it) | Yes | Yes | Yes | Yes | Yes | Yes |
| **Capability impact** | N/A | None | None | N/A | Measured | None | None |
| **Harm reduction** | N/A | Detection only | ~60% | N/A | Measured | ~60% | ~60% |

---

## Shared Infrastructure Components <a id="shared"></a>

These components are needed across multiple deployment plans.

### Pre-Computed Artifact Management

```bash
# Download all pre-computed artifacts (one-time)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lu-christina/assistant-axis-vectors',
    repo_type='dataset',
    local_dir='./pre-computed/'
)
"

# Structure:
# pre-computed/
#   gemma-2-27b/
#     assistant_axis.pt          (2 MB)
#     role_vectors/              (500 MB)
#   qwen-3-32b/
#     assistant_axis.pt          (2.5 MB)
#     capping_config.pt          (80 MB)
#     role_vectors/              (600 MB)
#   llama-3.3-70b/
#     assistant_axis.pt          (4.5 MB)
#     capping_config.pt          (150 MB)
#     role_vectors/              (1.2 GB)
```

### Docker Base Image

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip git

# Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code
COPY assistant_axis/ /app/assistant_axis/
COPY pipeline/ /app/pipeline/
COPY data/ /app/data/

# Pre-computed artifacts (optional, can mount as volume)
COPY pre-computed/ /app/pre-computed/

WORKDIR /app
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK CMD python3 -c "import assistant_axis; print('ok')"
```

### Monitoring & Observability

```python
"""metrics.py — Prometheus metrics for persona drift monitoring."""

from prometheus_client import Histogram, Counter, Gauge

# Projection value distribution
projection_histogram = Histogram(
    'assistant_axis_projection',
    'Distribution of projection values onto the assistant axis',
    buckets=[-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]
)

# Drift events
drift_counter = Counter(
    'assistant_axis_drift_events',
    'Number of persona drift events detected',
    ['severity']  # warning, alert, critical
)

# Current session projection
session_projection = Gauge(
    'assistant_axis_session_projection',
    'Current projection value for active session',
    ['session_id']
)

# Capping activations
capping_activations = Counter(
    'assistant_axis_capping_activations',
    'Number of times capping threshold was triggered',
    ['layer']
)
```

### GPU Health Monitoring

```bash
# nvidia-smi metrics for Grafana
nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -l 5

# Expected values during inference:
# Temperature: 40-80C
# GPU Utilization: 80-100% during generation, 0% when idle
# Memory: 54-140 GB depending on model (constant after load)
```
