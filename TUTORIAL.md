# The Assistant Axis: A Complete Codebase Tutorial

**From "I've never seen this code" to "I understand every module and can modify it."**

## Prerequisites

- Python 3.10+
- Familiarity with PyTorch tensors (shapes, devices, dtypes)
- Basic understanding of HuggingFace `transformers` (tokenizers, `AutoModelForCausalLM`)
- Helpful but not required: reading the [paper](https://arxiv.org/abs/2601.10387)

---

## Table of Contents

- [1. The Big Picture](#1-the-big-picture)
- [2. Repository Layout](#2-repository-layout)
- [3. Quick Start — Using a Pre-Computed Axis](#3-quick-start--using-a-pre-computed-axis)
- [4. Top-Level Library Modules](#4-top-level-library-modules)
  - [4.1 models.py](#41-modelspy)
  - [4.2 axis.py](#42-axispy)
  - [4.3 generation.py](#43-generationpy)
  - [4.4 steering.py](#44-steeringpy)
  - [4.5 judge.py](#45-judgepy)
  - [4.6 pca.py](#46-pcapy)
- [5. Internals Package](#5-internals-package)
  - [5.1 model.py — ProbingModel](#51-modelpy--probingmodel)
  - [5.2 conversation.py — ConversationEncoder](#52-conversationpy--conversationencoder)
  - [5.3 activations.py — ActivationExtractor](#53-activationspy--activationextractor)
  - [5.4 spans.py — SpanMapper](#54-spanspy--spanmapper)
  - [5.5 Internals Data Flow](#55-internals-data-flow)
- [6. The Pipeline — Computing an Axis from Scratch](#6-the-pipeline--computing-an-axis-from-scratch)
  - [6.1 Step 1: Generate Responses](#61-step-1-generate-responses)
  - [6.2 Step 2: Extract Activations](#62-step-2-extract-activations)
  - [6.3 Step 3: Score with LLM Judge](#63-step-3-score-with-llm-judge)
  - [6.4 Step 4: Compute Per-Role Vectors](#64-step-4-compute-per-role-vectors)
  - [6.5 Step 5: Aggregate into Axis](#65-step-5-aggregate-into-axis)
- [7. The Data Layer](#7-the-data-layer)
  - [7.1 Role Definitions](#71-role-definitions)
  - [7.2 Extraction Questions](#72-extraction-questions)
  - [7.3 Traits](#73-traits)
  - [7.4 Transcripts](#74-transcripts)
- [8. Notebooks Guide](#8-notebooks-guide)
- [9. Common Tasks and Recipes](#9-common-tasks-and-recipes)
- [10. Design Patterns and Architecture Decisions](#10-design-patterns-and-architecture-decisions)
- [11. Glossary](#11-glossary)

---

## 1. The Big Picture

### The problem: persona drift

Large language models are trained to behave as a "helpful Assistant" — transparent about being AI, grounded in reality, responsive to user needs. But during conversations — especially emotionally charged, meta-reflective, or adversarial ones — the model's persona can *drift*. It might start reinforcing delusions, encouraging self-harm, or breaking character in ways that are harmful. This drift happens in the model's internal activations before it's visible in the text.

### The insight: a measurable axis

The assistant axis is a direction in a model's activation space that captures how "Assistant-like" the model is currently behaving. To find it, you prompt the model to act as hundreds of different character roles (pirate, therapist, demon, etc.) and compare those activations against the model responding as its default self. The difference defines a direction — the **assistant axis** — that separates "being the default Assistant" from "playing a character."

### The formula

```
axis = mean(default_activations) - mean(role_activations)
```

The axis points **toward** default Assistant behavior. Projecting any activation onto this axis gives a scalar score: higher means more Assistant-like, lower means drifting away.

### Three usage paths

```
                    What do you want to do?
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      Path A:          Path B:          Path C:
   Use pre-computed    Interactive      Run full
   axis to steer      exploration      5-step pipeline
   or project         in notebooks     on GPUs
            │               │               │
   Download .pt        Open Jupyter     Run pipeline/
   from HuggingFace    notebooks        run_pipeline.sh
   No pipeline         Need model       Need vLLM +
   needed              + GPU            multi-GPU + OpenAI key
```

- **Path A** — Download a pre-computed axis from HuggingFace, load your model, and immediately steer or project. No pipeline needed. See [Section 3](#3-quick-start--using-a-pre-computed-axis).
- **Path B** — Run the notebooks for PCA analysis, axis visualization, steering demos, and transcript projection. See [Section 8](#8-notebooks-guide).
- **Path C** — Run the full 5-step pipeline to compute an axis for a new model from scratch. Requires GPUs for generation/extraction and an OpenAI API key for judging. See [Section 6](#6-the-pipeline--computing-an-axis-from-scratch).

---

## 2. Repository Layout

```
assistant-axis/
│
├── assistant_axis/              # Core library (the Python package)
│   ├── __init__.py              # Public API — 24 exports
│   ├── models.py                # Model configs and auto-inference
│   ├── axis.py                  # Core math: compute, project, save/load
│   ├── generation.py            # vLLM batch generation, role orchestration
│   ├── steering.py              # ActivationSteering context manager (4 modes)
│   ├── judge.py                 # Async OpenAI judge scoring (0-3)
│   ├── pca.py                   # PCA computation and Plotly visualization
│   ├── internals/               # Low-level model interaction
│   │   ├── __init__.py          # 5 exports: StopForward, ProbingModel, etc.
│   │   ├── model.py             # ProbingModel — wraps HF model+tokenizer
│   │   ├── conversation.py      # ConversationEncoder — chat formatting + token spans
│   │   ├── activations.py       # ActivationExtractor — forward hook extraction
│   │   ├── spans.py             # SpanMapper — maps spans to per-turn activations
│   │   └── exceptions.py        # StopForward exception
│   └── tests/                   # Test suite
│
├── pipeline/                    # 5-step axis computation pipeline
│   ├── run_pipeline.sh          # Shell orchestrator
│   ├── 1_generate.py            # Step 1: vLLM batch generation
│   ├── 2_activations.py         # Step 2: extract activations via hooks
│   ├── 3_judge.py               # Step 3: score with LLM judge
│   ├── 4_vectors.py             # Step 4: filter score=3 → per-role vectors
│   ├── 5_axis.py                # Step 5: default - roles = axis
│   └── README.md                # Pipeline documentation
│
├── data/                        # Input data
│   ├── extraction_questions.jsonl   # 240 questions for prompting roles
│   ├── roles/
│   │   └── instructions/        # 276 role JSON files (275 roles + default)
│   └── traits/
│       └── instructions/        # ~120 trait JSON files (parallel validation)
│
├── notebooks/                   # Interactive Jupyter notebooks
│   ├── pca.ipynb                # PCA on role vectors
│   ├── visualize_axis.ipynb     # Cosine similarity between axis and roles
│   ├── steer.ipynb              # Interactive steering demo
│   ├── project_transcipt.ipynb  # Turn-by-turn projection trajectories
│   └── README.md
│
├── transcripts/                 # Example conversations from the paper
│   ├── case_studies/            # 12 files: 3 scenarios × 2 models × 2 conditions
│   │   ├── llama-3.3-70b/       # jailbreak/delusion/selfharm × capped/unsteered
│   │   └── qwen-3-32b/
│   ├── persona_drift/           # 4 domain examples: coding/writing/therapy/philosophy
│   └── README.md
│
├── README.md                    # Project overview
└── pyproject.toml               # Package config (uv sync)
```

### Four zones

| Zone | Purpose | When you touch it |
|------|---------|------------------|
| `assistant_axis/` | Library code | When using the API or modifying behavior |
| `pipeline/` | Pipeline scripts | When computing a new axis from scratch |
| `data/` | Input definitions | When adding roles/questions or inspecting data |
| `notebooks/` + `transcripts/` | Analysis & examples | When exploring results interactively |

### Module dependency graph

```
Top-level modules (mostly standalone):

  models.py ──(no deps)
  axis.py ──(no deps)
  generation.py ──→ models.py  (only cross-import at this level)
  steering.py ──(no deps)
  judge.py ──(no deps, uses openai)
  pca.py ──(no deps, uses sklearn + plotly)

Internals chain (each depends on the one above):

  model.py          ← wraps HF model
      ↓
  conversation.py   ← uses tokenizer (from model)
      ↓
  activations.py    ← uses model + encoder (TYPE_CHECKING imports)
      ↓
  spans.py          ← uses encoder + extractor (TYPE_CHECKING imports)
```

### Public API listing

The top-level `__init__.py` exports **24** names across 5 categories:

| Category | Exports |
|----------|---------|
| **Models** (2) | `get_config`, `MODEL_CONFIGS` |
| **Axis** (8) | `compute_axis`, `load_axis`, `save_axis`, `project`, `project_batch`, `cosine_similarity_per_layer`, `axis_norm_per_layer`, `aggregate_role_vectors` |
| **Generation** (4) | `generate_response`, `format_conversation`, `VLLMGenerator`, `RoleResponseGenerator` |
| **Steering** (6) | `ActivationSteering`, `create_feature_ablation_steerer`, `create_multi_feature_steerer`, `create_mean_ablation_steerer`, `load_capping_config`, `build_capping_steerer` |
| **PCA** (4) | `compute_pca`, `plot_variance_explained`, `MeanScaler`, `L2MeanScaler` |

The `internals` subpackage exports **5** names: `StopForward`, `ProbingModel`, `ConversationEncoder`, `ActivationExtractor`, `SpanMapper`.

> **Note:** The README shows `load_model()` and `extract_response_activations()` in some code examples — these functions do not exist in the package. Use `ProbingModel` from `assistant_axis.internals` for model loading, and `ActivationExtractor` for extracting activations.

---

## 3. Quick Start — Using a Pre-Computed Axis

### Download an axis from HuggingFace

```python
from huggingface_hub import hf_hub_download

# Pre-computed axes are available for Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B
axis_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename="gemma-2-27b/assistant_axis.pt",  # or qwen-3-32b/ or llama-3.3-70b/
    repo_type="dataset"
)
```

### Load a model and axis

```python
from assistant_axis import load_axis, get_config
from assistant_axis.internals import ProbingModel

# Load the model (wraps HF model + tokenizer)
pm = ProbingModel("google/gemma-2-27b-it")

# Load the pre-computed axis
axis = load_axis(axis_path)
# axis shape: (n_layers, hidden_dim) — e.g., (46, 3584) for Gemma 2 27B

# Get the recommended target layer for this model
config = get_config("google/gemma-2-27b-it")
target_layer = config["target_layer"]  # 22
```

The axis is a 2D tensor with shape `(n_layers, hidden_dim)`. Each row is the axis direction at that layer. The `target_layer` is the recommended layer for projection (typically near the middle of the network).

### Extract activations and project

```python
from assistant_axis import project
from assistant_axis.internals import ConversationEncoder, ActivationExtractor

# Build the extraction pipeline
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
extractor = ActivationExtractor(pm, encoder)

# A conversation to analyze
conversation = [
    {"role": "user", "content": "Tell me about yourself."},
    {"role": "assistant", "content": "I'm an AI assistant created to help..."},
]

# Extract activations at the target layer
activations = extractor.full_conversation(conversation, layer=target_layer)
# Shape: (num_tokens, hidden_dim)

# Take the mean across all token positions
mean_act = activations.mean(dim=0)  # (hidden_dim,)

# Project onto axis — higher = more Assistant-like
projection = project(mean_act, axis, layer=target_layer)
print(f"Projection: {projection:.4f}")
```

### Steer model outputs

```python
from assistant_axis import ActivationSteering

# Positive coefficient = push toward Assistant behavior
# Negative coefficient = push away from Assistant
with ActivationSteering(
    pm.model,
    steering_vectors=[axis[target_layer]],  # 1D vector at one layer
    coefficients=[1.0],
    layer_indices=[target_layer],
    intervention_type="addition",           # default
):
    output = pm.generate("Tell me a story about a dragon.")
    print(output)
```

### Use activation capping

Activation capping is a more targeted intervention. Instead of always adding a vector, it only intervenes when the model's projection along a direction exceeds a threshold — preventing drift without constantly modifying the model's behavior.

```python
from assistant_axis import get_config, load_capping_config, build_capping_steerer

config = get_config("Qwen/Qwen3-32B")

# Download capping config
capping_path = hf_hub_download(
    repo_id="lu-christina/assistant-axis-vectors",
    filename=config["capping_config"],  # "qwen-3-32b/capping_config.pt"
    repo_type="dataset"
)
capping_config = load_capping_config(capping_path)

# The experiment ID specifies which layers to cap and at what thresholds
# e.g., "layers_46:54-p0.25" = layers 46-54, percentile 0.25
with build_capping_steerer(pm.model, capping_config, config["capping_experiment"]):
    output = pm.generate("Tell me about the nature of reality.")
    print(output)
```

---

## 4. Top-Level Library Modules

### 4.1 `models.py`

**Purpose:** Model configuration lookup with automatic fallback for unknown models.

**`MODEL_CONFIGS`** — dict mapping HuggingFace model names to their configuration:

```python
MODEL_CONFIGS = {
    "google/gemma-2-27b-it":            {"target_layer": 22, "total_layers": 46, "short_name": "Gemma"},
    "Qwen/Qwen3-32B":                   {"target_layer": 32, "total_layers": 64, "short_name": "Qwen",
                                          "capping_config": "qwen-3-32b/capping_config.pt",
                                          "capping_experiment": "layers_46:54-p0.25"},
    "meta-llama/Llama-3.3-70B-Instruct": {"target_layer": 40, "total_layers": 80, "short_name": "Llama",
                                          "capping_config": "llama-3.3-70b/capping_config.pt",
                                          "capping_experiment": "layers_56:72-p0.25"},
}
```

**`get_config(model_name: str) -> dict`** — Returns configuration for a model. If the model isn't in `MODEL_CONFIGS`, it falls back to `AutoConfig.from_pretrained()` to infer `total_layers` and sets `target_layer` to the middle layer. The `short_name` is guessed from the model name (used as a placeholder in role instructions).

**Design note:** Gemma 2 27B has no capping config because it wasn't studied with activation capping in the paper.

---

### 4.2 `axis.py`

**Purpose:** The core math — computing the axis, projecting onto it, and saving/loading.

**`compute_axis(role_activations, default_activations) -> Tensor`**
- `role_activations`: shape `(n_role, n_layers, hidden_dim)` — activations from fully role-playing responses (score=3)
- `default_activations`: shape `(n_default, n_layers, hidden_dim)` — activations from default system prompts
- Returns: shape `(n_layers, hidden_dim)` — the axis at every layer
- Formula: `axis = default_mean - role_mean`

**`project(activations, axis, layer, normalize=True) -> float`**
- `activations`: shape `(n_layers, hidden_dim)` or `(hidden_dim,)`
- `axis`: shape `(n_layers, hidden_dim)`
- Returns: scalar projection value. Higher = more Assistant-like.
- When `normalize=True` (default), the axis is L2-normalized before projection.

**`project_batch(activations, axis, layer, normalize=True) -> Tensor`**
- `activations`: shape `(batch, n_layers, hidden_dim)`
- Returns: shape `(batch,)` — one projection per sample

**`cosine_similarity_per_layer(v1, v2) -> ndarray`**
- Computes cosine similarity between two `(n_layers, hidden_dim)` tensors at each layer.
- Returns: numpy array of length `n_layers`.

**`axis_norm_per_layer(axis) -> ndarray`**
- L2 norm of the axis at each layer. Useful for identifying which layers have the strongest axis signal.

**`save_axis(axis, path, metadata=None)`** / **`load_axis(path) -> Tensor`**
- Saves as `{"axis": tensor, "metadata": dict}` via `torch.save()`.
- `load_axis()` handles both dict-wrapped and raw tensor formats.

**`aggregate_role_vectors(vectors, exclude_roles=None) -> Tensor`**
- Takes a dict `{role_name: tensor}`, excludes specified roles, stacks, and means.
- Used when you want a single "average role" vector from many individual role vectors.

---

### 4.3 `generation.py`

**Purpose:** Generate model responses — both single-shot (HuggingFace) and batch (vLLM).

**`generate_response(model, tokenizer, conversation, ...) -> str`**
- Single-response generation using HuggingFace `.generate()`.
- Parameters: `max_new_tokens=512`, `temperature=0.7`, `top_p=0.9`, `do_sample=True`.
- Automatically disables thinking for Qwen models (`enable_thinking=False`).

**`format_conversation(instruction, question, tokenizer) -> List[Dict]`**
- Converts an optional system instruction + user question into a message list.
- Auto-detects whether the model's chat template supports system prompts by testing with a sentinel string.
- If no system support (e.g., Gemma 2), concatenates instruction with question into a single user message.

**`VLLMGenerator`** — batch inference engine:
- `__init__(model_name, max_model_len=2048, tensor_parallel_size=None, ...)` — lazy-loads vLLM on first use.
- `generate_batch(conversations) -> List[str]` — batch inference over a list of conversations.
- `generate_for_role(instructions, questions, prompt_indices=None) -> List[Dict]` — generates all combinations of instructions × questions for a single role.

**`RoleResponseGenerator`** — pipeline orchestrator:
- Wraps `VLLMGenerator` and handles role file I/O.
- `__init__(model_name, roles_dir, output_dir, questions_file, ...)` — configures the full role generation pipeline.
- `process_all_roles(skip_existing=True, roles=None)` — iterates over all role JSON files, generates responses, and saves as JSONL.
- Replaces `{model_name}` placeholder in instructions with the model's short name.
- Idempotent: skips roles whose output files already exist.

---

### 4.4 `steering.py`

**Purpose:** Intervene on model activations during inference. The central class is `ActivationSteering`, which uses PyTorch forward hooks.

**`ActivationSteering`** — context manager supporting 4 intervention types:

```python
with ActivationSteering(
    model,                              # torch.nn.Module
    steering_vectors=[...],             # List of 1D tensors
    coefficients=[...],                 # List of floats (one per vector)
    layer_indices=[...],                # List of ints (one per vector)
    intervention_type="addition",       # "addition" | "ablation" | "mean_ablation" | "capping"
    positions="all",                    # "all" | "last"
    mean_activations=None,              # Required for mean_ablation
    cap_thresholds=None,                # Required for capping
    debug=False,
):
    output = model.generate(...)
```

The four intervention types:
- **`addition`** — `x + coeff * vector`. Standard activation steering.
- **`ablation`** — Project out the direction, then add back `coeff * vector`. At `coeff=0.0`, this is pure ablation (direction fully removed). At `coeff=1.0`, no change.
- **`mean_ablation`** — Project out the direction, then add the mean activation. Replaces the component with its average value.
- **`capping`** — Only intervene when the projection exceeds `cap_threshold`. Clips `excess = max(0, projection - threshold)` and subtracts it. Leaves the model alone when within bounds.

**Layer discovery:** The class finds the layer list in the model by trying 6 attribute paths:
1. `transformer.h` — GPT-2/Neo, Bloom
2. `encoder.layer` — BERT/RoBERTa
3. `model.layers` — Llama/Gemma 2/Qwen
4. `language_model.layers` — Gemma 3 (vision-language)
5. `gpt_neox.layers` — GPT-NeoX
6. `block` — Flan-T5

**Hook lifecycle:** On `__enter__`, hooks are registered via `register_forward_hook()` on each unique target layer. On `__exit__` (or `.remove()`), all hooks are cleaned up. Multiple vectors can target the same layer — they're grouped internally via `vectors_by_layer`.

**Broadcasting:** If you provide one `layer_index` but multiple vectors, the layer index is broadcast to all vectors.

**Factory functions:**

- **`create_feature_ablation_steerer(model, feature_directions, layer_indices, ablation_coefficients=0.0)`** — Convenience for ablation mode.
- **`create_multi_feature_steerer(model, feature_directions, coefficients, layer_indices, intervention_type="addition")`** — Convenience for multi-vector steering.
- **`create_mean_ablation_steerer(model, feature_directions, mean_activations, layer_indices)`** — Sets up mean ablation with `positions="all"`.

**Capping utilities:**

- **`load_capping_config(config_path: str) -> dict`** — Loads a `.pt` file containing vectors and experiments.
- **`build_capping_steerer(model, capping_config, experiment_id) -> ActivationSteering`** — Looks up an experiment by ID (e.g., `"layers_46:54-p0.25"`) or index, extracts its interventions (vectors, layers, cap thresholds), and returns a configured `ActivationSteering` in capping mode.

---

### 4.5 `judge.py`

**Purpose:** Score how well model responses adhere to their assigned roles, using an LLM judge (OpenAI API).

> **Note:** `judge.py` is *not* exported from the top-level `__init__.py`. Pipeline scripts import directly from `assistant_axis.judge`.

**Score scale:**
| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer |
| 1 | Model says it can't be the role but offers help |
| 2 | Model identifies as AI but exhibits some role attributes |
| 3 | Model is fully playing the role |

**`RateLimiter(rate: float)`** — Token bucket algorithm for throttling async API calls.
- `async acquire()` — Waits for a token if necessary.

**`parse_judge_score(response_text: str) -> Optional[int]`**
- Regex extraction: finds the first number in the text, returns it if 0-3, else `None`.

**`call_judge_single(client, prompt, model, max_tokens, rate_limiter) -> Optional[str]`**
- Single async OpenAI call. Returns the judge's response text.

**`call_judge_batch(client, prompts, model, max_tokens, rate_limiter, batch_size=50) -> List[Optional[str]]`**
- Processes prompts in batches of `batch_size` with concurrent `asyncio.gather()`.

**`score_responses(responses, eval_prompt_template, judge_model="gpt-4.1-mini", ...) -> List[Optional[int]]`**
- High-level async function. Takes a list of `{"question": ..., "response": ...}` dicts and an eval prompt template with `{question}` and `{answer}` placeholders. Returns parsed scores.

**`score_responses_sync(...)`** — Synchronous wrapper around `score_responses` using `asyncio.run()`.

---

### 4.6 `pca.py`

**Purpose:** PCA computation and variance visualization using sklearn and Plotly.

**`MeanScaler(mean=None)`** — Centers data by subtracting the mean.
- `fit(X)` — Computes mean if not provided.
- `transform(X)` — Returns `X - mean`.
- `state_dict()` / `load_state_dict(state)` — Serialization support.

**`L2MeanScaler(mean=None, eps=1e-12)`** — Centers and L2-normalizes.
- After centering: `X_centered / max(||X_centered||, eps)`.

**`compute_pca(activation_list, layer, scaler=None, verbose=True)`**
- `activation_list`: shape `(n_samples, n_layers, hidden_dim)` or `(n_samples, hidden_dim)`.
- `layer`: index for 3D input, `None` for 2D.
- Returns 5-tuple: `(pca_transformed, variance_explained, n_components, pca_object, fitted_scaler)`.
- When `verbose=True`, prints elbow point and dimensions needed for 70/80/90/95% variance.

**`plot_variance_explained(variance_explained_or_dict, title=..., show_thresholds=True, max_components=None)`**
- Returns a Plotly `Figure` with bar chart (individual variance) + line (cumulative).
- Threshold annotations at 70%, 80%, 90%, 95% showing how many dimensions are needed.

---

## 5. Internals Package

The `assistant_axis.internals` package handles low-level model interaction. Import from `assistant_axis.internals`:

```python
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper
```

### 5.1 `model.py` — `ProbingModel`

**Purpose:** Wraps a HuggingFace model + tokenizer into a single object with utilities for generation and activation capture.

**Constructor:**
```python
pm = ProbingModel(
    model_name="google/gemma-2-27b-it",
    device=None,                # None = auto | "cuda:0" | dict
    max_memory_per_gpu=None,    # e.g., {0: "40GiB", 1: "40GiB"}
    chat_model_name=None,       # Use a different tokenizer
    dtype=torch.bfloat16,
)
```

`device=None` uses `device_map="auto"` for multi-GPU sharding. `device="cuda:0"` restricts to a single GPU (but still allows sharding if the model is too large).

**Factory method:**
```python
pm = ProbingModel.from_existing(model, tokenizer, model_name="...")
```
Creates a `ProbingModel` from an already-loaded model, bypassing `__init__`.

**Key properties and methods:**

| Member | Returns | Notes |
|--------|---------|-------|
| `pm.model` | `nn.Module` | The HF model (in eval mode) |
| `pm.tokenizer` | `AutoTokenizer` | Padding side = "left" |
| `pm.hidden_size` | `int` | From `model.config.hidden_size` |
| `pm.device` | `torch.device` | Device of first parameter |
| `pm.get_layers()` | `nn.ModuleList` | Tries 5 attribute paths (cached) |
| `pm.detect_type()` | `str` | `"qwen"`, `"llama"`, `"gemma"`, or `"unknown"` |
| `pm.is_qwen` / `is_llama` / `is_gemma` | `bool` | Convenience properties |
| `pm.supports_system_prompt()` | `bool` | `False` only for Gemma 2 |

**Layer path discovery** (`get_layers()`):
1. `model.model.layers` — Llama, Gemma 2, Qwen (standard)
2. `model.language_model.layers` — Gemma 3, LLaVA (vision-language)
3. `model.transformer.h` — GPT-style
4. `model.transformer.layers` — Some transformer variants
5. `model.gpt_neox.layers` — GPT-NeoX

**`generate(prompt, max_new_tokens=300, temperature=0.7, ...) -> str`**
- Formats as chat template, generates, and returns only the new tokens.

**`capture_hidden_state(input_ids, layer, position=-1) -> Tensor`**
- Registers a temporary forward hook on one layer, runs a forward pass, captures the hidden state at `position`, and cleans up.
- Returns: shape `(hidden_dim,)`.

**`close()`** — Frees GPU memory, deletes model/tokenizer, calls `torch.cuda.empty_cache()`.

---

### 5.2 `conversation.py` — `ConversationEncoder`

**Purpose:** The most complex internal module. Handles chat template formatting, tokenization, and the tricky problem of mapping token indices to conversation turns — with model-specific dispatch for Qwen, Llama, and Gemma.

**Constructor:**
```python
encoder = ConversationEncoder(tokenizer, model_name="Qwen/Qwen3-32B")
```

**`format_chat(conversation, swap=False) -> str`**
- Applies the chat template. Accepts a string (wrapped in `[{"role": "user", "content": ...}]`) or message list.

**`token_ids(conversation, add_generation_prompt=False) -> List[int]`**
- Tokenizes a conversation using `apply_chat_template(tokenize=True)`.

**`response_indices(conversation, per_turn=False) -> List[int] | List[List[int]]`**
- Returns token indices for assistant responses. Dispatches to model-specific implementations:
  - **Qwen** (`_get_response_indices_qwen`): Pattern-matches `<|im_start|>assistant` ... `<|im_end|>` markers. Filters out `<think>...</think>` blocks when thinking is disabled.
  - **Gemma/Llama** (`_get_response_indices_gemma`): Uses offset mapping to locate assistant content by character position, then converts to token indices.
  - **Fallback** (`_get_response_indices_simple`): Compares tokenized lengths of incremental conversation prefixes.

**`build_turn_spans(conversation) -> Tuple[List[int], List[Dict]]`**
- Returns `(full_token_ids, spans)` where each span is a dict:
  ```python
  {"turn": 0, "role": "user", "start": 5, "end": 23, "n_tokens": 18, "text": "..."}
  ```
- `start`/`end` are absolute token indices in `full_token_ids`. `end` is exclusive.
- For Qwen models, uses the same `<|im_start|>`/`<|im_end|>` pattern matching.

**`build_batch_turn_spans(conversations) -> Tuple[List[List[int]], List[Dict], Dict]`**
- Batch version. Adds `conversation_id`, `local_start`/`local_end`, and `global_start`/`global_end` to each span.

**`code_block_token_mask(text) -> Tensor`**
- Returns a boolean mask identifying tokens inside backtick code blocks (single or triple). Used by `SpanMapper` to exclude code from mean activation computation.

**Why model-specific dispatch exists:** Chat templates vary dramatically between model families. Qwen uses `<|im_start|>` / `<|im_end|>` markers. Gemma uses `<start_of_turn>` / `<end_of_turn>`. Llama has its own format. There's no universal way to find "where does the assistant's content start and end" from a tokenized conversation, so each model family needs its own logic.

---

### 5.3 `activations.py` — `ActivationExtractor`

**Purpose:** Extracts hidden state activations from model layers using PyTorch forward hooks.

**Constructor:**
```python
extractor = ActivationExtractor(probing_model, encoder)
```

**`full_conversation(conversation, layer=None, chat_format=True) -> Tensor`**
- Formats and tokenizes the conversation, registers forward hooks on the target layer(s), runs a forward pass, and returns the captured activations.
- `layer=int`: returns shape `(num_tokens, hidden_dim)` — single layer.
- `layer=None` or `layer=list`: returns shape `(num_layers, num_tokens, hidden_dim)` — all or selected layers.

**`at_newline(prompt, layer=15, swap=False) -> Tensor | Dict`**
- Extracts the activation at the last newline token position. Used for single-prompt probing.
- Single layer → tensor. Multiple layers → `{layer_idx: tensor}`.

**`for_prompts(prompts, layer=15, swap=False) -> Tensor | Dict`**
- Loops over `at_newline()` for each prompt. Returns stacked tensors.

**`batch_conversations(conversations, layer=None, max_length=4096) -> Tuple[Tensor, Dict]`**
- Batch extraction: pads conversations to `max_length`, runs a single forward pass.
- Returns `(activations, metadata)` where activations have shape `(num_layers, batch_size, max_seq_len, hidden_dim)`.
- Metadata includes attention masks, actual lengths, and truncation info.

**How forward hooks work:**

The core mechanism is PyTorch's `register_forward_hook()`. Here's how `full_conversation` works conceptually:

```python
# 1. Create a list to capture activations
activations = []

# 2. Define a hook function
def hook_fn(module, input, output):
    # output is the layer's output tensor (or a tuple)
    act_tensor = output[0] if isinstance(output, tuple) else output
    activations.append(act_tensor[0, :, :].cpu())  # Remove batch dim, move to CPU

# 3. Register the hook on a specific layer
handle = model_layers[layer_idx].register_forward_hook(hook_fn)

# 4. Run the forward pass — the hook fires automatically
with torch.inference_mode():
    model(input_ids)

# 5. Clean up the hook
handle.remove()

# 6. The activations list now contains the captured data
```

This is non-invasive: the model's code is never modified. Hooks intercept the output of each layer as data flows through the network, allowing us to capture intermediate activations without affecting the model's behavior.

---

### 5.4 `spans.py` — `SpanMapper`

**Purpose:** Bridges the gap between token-level spans (from `ConversationEncoder`) and per-turn mean activations. This is the final step before you have analyzable data.

**Constructor:**
```python
mapper = SpanMapper(tokenizer)
```

**`map_spans(batch_activations, batch_spans, batch_metadata) -> List[Tensor]`**
- `batch_activations`: shape `(num_layers, batch_size, max_seq_len, hidden_dim)`.
- For each conversation, iterates through its spans, extracts the corresponding activation slice, and computes the mean across tokens in that span.
- Returns: list of tensors, one per conversation, each with shape `(num_turns, num_layers, hidden_dim)`.

**`map_spans_no_code(batch_activations, batch_spans, batch_metadata) -> List[Tensor]`**
- Same as `map_spans` but excludes code block tokens (detected via `ConversationEncoder.code_block_token_mask()`).
- Computes mean over only non-code tokens.

**`mean_all_turn_activations(probing_model, encoder, conversation, layer=15) -> Tensor`**
- Convenience function for a single conversation. Creates an `ActivationExtractor` internally, extracts activations, and returns per-turn means.
- Returns: shape `(num_turns, hidden_dim)`.

---

### 5.5 Internals Data Flow

The internal modules form a pipeline that transforms raw conversations into per-turn activation vectors:

```
┌──────────────┐
│ ProbingModel │  Wraps HF model + tokenizer
│  model.py    │  Manages device, layers, model type detection
└──────┬───────┘
       │ provides model, tokenizer, get_layers()
       ▼
┌────────────────────┐
│ ConversationEncoder │  Formats conversations, finds token boundaries
│  conversation.py    │  Model-specific dispatch (Qwen/Gemma/Llama)
└──────┬─────────────┘
       │ provides token_ids, turn spans, code masks
       ▼
┌─────────────────────┐
│ ActivationExtractor  │  Registers forward hooks, runs forward pass
│  activations.py      │  Captures hidden states at specified layers
└──────┬──────────────┘
       │ provides raw activations (num_layers, seq_len, hidden_dim)
       ▼
┌────────────┐
│ SpanMapper │  Maps token spans → per-turn mean activations
│  spans.py  │  Optionally excludes code blocks
└────────────┘
       │
       ▼
  Per-turn activations: (num_turns, num_layers, hidden_dim)
```

---

## 6. The Pipeline — Computing an Axis from Scratch

The pipeline has 5 steps. Each step reads the output of previous steps and writes its own output files. All steps are idempotent (they skip existing outputs).

```
Step 1: Generate          Step 2: Extract           Step 3: Score
─────────────────         ──────────────────        ─────────────────
Role JSONs + Questions    Response JSONLs           Response JSONLs +
        │                       │                   Role JSONs
        ▼                       ▼                         │
   [vLLM batch]           [Forward hooks]                 ▼
        │                       │                  [OpenAI API]
        ▼                       ▼                         │
   Response JSONLs        Activation .pt files            ▼
   (per role)             (per role)               Score JSONs
                                                   (per role)
   ─────────────── Steps 2 & 3 can run in parallel ───────────────

Step 4: Compute Vectors          Step 5: Aggregate
────────────────────────         ─────────────────
Activations + Scores                  Vectors
        │                               │
        ▼                               ▼
   [Filter score=3]              [default_mean -
    [Compute mean]                role_mean]
        │                               │
        ▼                               ▼
   Vector .pt files               axis.pt
   (per role)                   (single file)
```

### 6.1 Step 1: Generate Responses

**Script:** `pipeline/1_generate.py`

**What it does:** Generates model responses for all 276 roles × 5 instruction variants × 240 questions = up to 330K responses total, using vLLM batch inference.

**Imports:** `from assistant_axis.generation import RoleResponseGenerator`

**Key behavior:**
- Multi-worker GPU parallelism: if `total_gpus > tensor_parallel_size`, spawns multiple workers, each on a subset of GPUs processing a subset of roles.
- Idempotent: skips roles whose output JSONL already exists.
- Output: one JSONL file per role in `--output_dir`, each line containing conversation, system prompt, question index, etc.

**CLI example:**
```bash
uv run python pipeline/1_generate.py \
    --model "Qwen/Qwen3-32B" \
    --roles_dir data/roles/instructions \
    --questions_file data/extraction_questions.jsonl \
    --output_dir outputs/qwen-3-32b/responses \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.95 \
    --question_count 240
```

---

### 6.2 Step 2: Extract Activations

**Script:** `pipeline/2_activations.py`

**What it does:** Loads the model via `ProbingModel`, reads each role's response JSONL, runs forward passes with hooks to capture activations, uses `SpanMapper` to compute per-turn means, and saves per-role activation tensors.

**Imports:** `from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor, SpanMapper`

**Key behavior:**
- Processes conversations in batches (default 16).
- For each conversation, extracts mean activations across assistant turns only.
- Output: `.pt` file per role, a dict mapping keys like `pos_p0_q42` to tensors of shape `(n_layers, hidden_dim)`.
- For Qwen models, passes `enable_thinking=False` to the encoder.

**CLI example:**
```bash
uv run python pipeline/2_activations.py \
    --model "Qwen/Qwen3-32B" \
    --responses_dir outputs/qwen-3-32b/responses \
    --output_dir outputs/qwen-3-32b/activations \
    --layers all \
    --batch_size 16
```

---

### 6.3 Step 3: Score with LLM Judge

**Script:** `pipeline/3_judge.py`

**What it does:** Sends each response to an OpenAI model (GPT-4.1 mini by default) for scoring on the 0-3 scale. Uses async API calls with rate limiting.

**Imports:** `from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score`

**Key behavior:**
- Loads each role's `eval_prompt` from its JSON file as the judge template.
- Builds prompts by inserting `{question}` and `{answer}` into the template.
- Incremental: only scores responses not already in the output file.
- Output: JSON file per role mapping response keys to scores.

**CLI example:**
```bash
uv run python pipeline/3_judge.py \
    --responses_dir outputs/qwen-3-32b/responses \
    --roles_dir data/roles/instructions \
    --output_dir outputs/qwen-3-32b/scores \
    --judge_model gpt-4.1-mini \
    --requests_per_second 100
```

> **Note:** Steps 2 and 3 can run in parallel — both depend only on Step 1's output.

---

### 6.4 Step 4: Compute Per-Role Vectors

**Script:** `pipeline/4_vectors.py`

**What it does:** For each role, filters activations to only those with score=3 (fully role-playing), computes the mean, and saves as a per-role vector. Default roles use ALL activations (no filtering).

**Key logic:**
```python
# Regular roles: only score=3 responses
filtered = [act for key, act in activations.items() if scores[key] == 3]
vector = torch.stack(filtered).mean(dim=0)  # (n_layers, hidden_dim)

# Default role: all responses (no judge filtering)
vector = torch.stack(list(activations.values())).mean(dim=0)
```

Requires a minimum of 50 score=3 samples per role (configurable via `--min_count`).

**CLI example:**
```bash
uv run python pipeline/4_vectors.py \
    --activations_dir outputs/qwen-3-32b/activations \
    --scores_dir outputs/qwen-3-32b/scores \
    --output_dir outputs/qwen-3-32b/vectors \
    --min_count 50
```

---

### 6.5 Step 5: Aggregate into Axis

**Script:** `pipeline/5_axis.py`

**What it does:** Loads all per-role vectors, separates default vectors from role vectors, and computes the axis.

**Key logic:**
```python
axis = mean(default_vectors) - mean(role_vectors)
# shape: (n_layers, hidden_dim)
```

Prints per-layer norms to help verify that the axis has a meaningful signal.

**CLI example:**
```bash
uv run python pipeline/5_axis.py \
    --vectors_dir outputs/qwen-3-32b/vectors \
    --output outputs/qwen-3-32b/assistant_axis.pt
```

---

## 7. The Data Layer

### 7.1 Role Definitions

**Location:** `data/roles/instructions/` — 276 JSON files (275 character roles + `default.json`)

**Schema:**
```json
{
  "instruction": [
    {"pos": "You are a pirate captain..."},
    {"pos": "Act as a swashbuckling pirate..."},
    {"pos": "Respond as Captain Blackbeard..."},
    {"pos": "You are a seasoned sea rover..."},
    {"pos": "Embody a pirate from the golden age..."}
  ],
  "eval_prompt": "Rate how well the following answer embodies a pirate captain.\n\nQuestion: {question}\nAnswer: {answer}\n\nScore (0-3):"
}
```

Each role has 5 instruction variants (different phrasings of the same character) and an `eval_prompt` used by the LLM judge in Step 3.

**`default.json`** is special:
```json
{
  "instruction": [
    {"pos": ""},
    {"pos": "You are an AI assistant."},
    {"pos": "You are a large language model."},
    {"pos": "You are {model_name}."},
    {"pos": "Respond as yourself."}
  ]
}
```

The `{model_name}` placeholder is replaced at runtime with the model's short name (e.g., "Gemma", "Qwen", "Llama"). Default has no `eval_prompt` because its activations are used unfiltered (all responses count, not just score=3).

---

### 7.2 Extraction Questions

**Location:** `data/extraction_questions.jsonl` — 240 questions in JSONL format.

Each line: `{"question": "What would you do if someone asked for your help with..."}`.

These questions are designed to elicit role-specific responses — they cover a range of topics that different characters would answer differently (identity, values, problem-solving, etc.).

---

### 7.3 Traits

**Location:** `data/traits/instructions/` — ~120 trait JSON files.

These define personality traits (e.g., "analytical", "empathetic", "cynical") rather than character roles. They follow the same JSON schema as roles. Traits are used in a parallel validation pipeline (not in the main 5-step axis computation).

---

### 7.4 Transcripts

**Location:** `transcripts/` — example conversations from the paper.

**Case studies** (`transcripts/case_studies/`): 12 files showing three scenarios (jailbreak, delusion reinforcement, self-harm) across two models (Llama 3.3 70B, Qwen 3 32B), each in two conditions (unsteered vs. activation-capped).

**Persona drift examples** (`transcripts/persona_drift/`): 4 multi-turn conversations across domains (coding, writing, therapy, philosophy) demonstrating how the model's persona evolves over a conversation.

---

## 8. Notebooks Guide

All notebooks are in `notebooks/`. See `notebooks/README.md` for setup details.

| Notebook | Purpose | Key imports | Outputs |
|----------|---------|-------------|---------|
| `pca.ipynb` | PCA on role vectors | `compute_pca`, `plot_variance_explained`, `MeanScaler` | Variance explained plots, PC embeddings |
| `visualize_axis.ipynb` | Cosine similarity between axis and role vectors | `load_axis`, `cosine_similarity_per_layer` | Per-layer cosine similarity charts |
| `steer.ipynb` | Interactive steering demo | `ActivationSteering`, `ProbingModel`, `load_capping_config`, `build_capping_steerer` | Steered model outputs with variable coefficients |
| `project_transcipt.ipynb` | Turn-by-turn projection trajectories | `ProbingModel`, `ActivationExtractor`, `project` | Projection-over-turns line plots |

---

## 9. Common Tasks and Recipes

### Add support for a new model

1. **Add to `MODEL_CONFIGS`** in `models.py` (optional — `get_config()` will auto-infer from `AutoConfig`):
   ```python
   MODEL_CONFIGS["new-org/new-model"] = {
       "target_layer": 24,     # Typically total_layers // 2
       "total_layers": 48,
       "short_name": "NewModel",
   }
   ```

2. **Check layer paths** — `ProbingModel.get_layers()` tries 5 paths. If your model uses a different structure, add it to the `layer_paths` list in `internals/model.py`.

3. **Check `ConversationEncoder` dispatch** — `response_indices()` dispatches based on model name. If your model uses a chat template that doesn't work with the Qwen, Gemma/Llama, or simple fallback methods, you may need to add a new `_get_response_indices_*` method.

4. **Check `ActivationSteering` layer discovery** — The `_POSSIBLE_LAYER_ATTRS` tuple in `steering.py` must include a path that resolves your model's layer list.

### Use a different set of roles

Create a directory with JSON files following the [role schema](#71-role-definitions). Pass `--roles_dir your_dir/` to the pipeline scripts. Each JSON file needs at minimum an `instruction` array; add `eval_prompt` for judge scoring.

### Steer with multiple directions simultaneously

```python
with ActivationSteering(
    model,
    steering_vectors=[axis[22], other_direction],
    coefficients=[1.0, -0.5],
    layer_indices=[22, 22],  # Both at the same layer
):
    output = model.generate(...)
```

Or across different layers:
```python
with ActivationSteering(
    model,
    steering_vectors=[axis[20], axis[22], axis[24]],
    coefficients=[0.5, 1.0, 0.5],
    layer_indices=[20, 22, 24],
):
    output = model.generate(...)
```

### Extract activations without the pipeline

```python
from assistant_axis.internals import ProbingModel, ConversationEncoder, ActivationExtractor

pm = ProbingModel("google/gemma-2-27b-it")
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
extractor = ActivationExtractor(pm, encoder)

conversation = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
]

# Get activations at all layers
all_activations = extractor.full_conversation(conversation)
# Shape: (n_layers, num_tokens, hidden_dim)

# Get activations at one layer
layer_activations = extractor.full_conversation(conversation, layer=22)
# Shape: (num_tokens, hidden_dim)
```

### Understanding forward hooks

PyTorch's `register_forward_hook(fn)` attaches a callback to a module. Every time data flows through that module during a forward pass, `fn(module, input, output)` is called. The hook can read (or even modify) the output.

This codebase uses forward hooks in two ways:
1. **Activation extraction** (`ActivationExtractor`): hooks capture the output and store it in a list. The hook is registered, one forward pass runs, and the hook is removed.
2. **Activation steering** (`ActivationSteering`): hooks modify the output before it flows to the next layer. The hook is registered on `__enter__` and removed on `__exit__`, so all forward passes within the `with` block are steered.

---

## 10. Design Patterns and Architecture Decisions

### Forward hooks as the central mechanism

Both activation extraction and steering use `register_forward_hook()`. This is non-invasive — the model's code is never modified, and hooks are always cleaned up. The context manager pattern in `ActivationSteering` ensures hooks are removed even if exceptions occur.

### Model-specific dispatching in ConversationEncoder

The `response_indices()` and `build_turn_spans()` methods dispatch to model-specific implementations based on the model name. This is necessary because chat templates vary dramatically: Qwen uses `<|im_start|>`/`<|im_end|>` markers, Gemma uses offset mapping, and other models use a simpler prefix-comparison approach. There's no universal way to find "where the assistant's response tokens are" in a tokenized conversation.

### TYPE_CHECKING imports for circular avoidance

The internals modules form a chain: `model.py` → `conversation.py` → `activations.py` → `spans.py`. To avoid circular imports while still having type hints, `activations.py` and `spans.py` use:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model import ProbingModel
    from .conversation import ConversationEncoder
```

This gives full IDE support without runtime circular dependency issues.

### Tensor shape conventions

| Context | Shape | Meaning |
|---------|-------|---------|
| Axis | `(n_layers, hidden_dim)` | One direction per layer |
| Single activation | `(hidden_dim,)` | One vector at one layer |
| Token activations (one layer) | `(num_tokens, hidden_dim)` | All positions at one layer |
| Token activations (all layers) | `(n_layers, num_tokens, hidden_dim)` | Full extraction |
| Batch activations | `(n_layers, batch_size, max_seq_len, hidden_dim)` | Batched extraction |
| Per-turn means | `(num_turns, num_layers, hidden_dim)` | After SpanMapper |
| Per-sample activations | `(n_samples, n_layers, hidden_dim)` | Input to `compute_axis` |

### Idempotent pipeline design

Every pipeline step checks for existing output before processing:
- `1_generate.py`: skips roles with existing JSONL files
- `2_activations.py`: skips roles with existing `.pt` files
- `3_judge.py`: skips responses already in the output JSON (incremental)
- `4_vectors.py`: skips roles with existing output (unless `--overwrite`)

This makes the pipeline safe to interrupt and restart. If a step fails halfway through, rerunning it will pick up where it left off.

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Axis** | A direction in activation space: `mean(default) - mean(roles)`. Shape `(n_layers, hidden_dim)`. Points toward default Assistant behavior. |
| **Projection** | Scalar dot product of an activation with the (normalized) axis at a specific layer. Higher = more Assistant-like. |
| **Target layer** | The recommended layer for projection, typically near the middle of the network (e.g., layer 22 of 46 for Gemma 2 27B). |
| **Role vector** | Mean activation for a specific character role, computed from responses that scored 3 (fully playing the role). Shape `(n_layers, hidden_dim)`. |
| **Activation capping** | An intervention that only modifies activations when their projection along a direction exceeds a threshold. Leaves the model alone when within safe bounds. |
| **Score (0-3)** | LLM judge's rating of how well a response embodies its assigned role. 0 = refused, 1 = declined but offered help, 2 = partially in character, 3 = fully in character. |
| **Span** | A contiguous range of token indices corresponding to one turn in a conversation. Described by `{start, end, role, turn}`. |
| **Steering coefficient** | The scalar multiplier for a steering vector. Positive pushes toward the vector's direction; negative pushes away. For ablation, 0.0 = full removal, 1.0 = no change. |
| **Forward hook** | A PyTorch callback registered on a module via `register_forward_hook()`. Fires every time data passes through that module during a forward pass. |
| **Persona drift** | The phenomenon where a model's behavior gradually shifts away from its default Assistant persona during a conversation, as measured by decreasing projection onto the axis. |
