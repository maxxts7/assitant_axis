# The Assistant Axis: Consolidated Codebase Analysis

> **Paper:** "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"
> **Authors:** Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey
> **arXiv:** 2601.10387v1 (January 15, 2026)

---

## Part I — The Paper

### Core Problem

Post-trained language models are taught to play a specific character — the "AI Assistant" — that is helpful, honest, and harmless. But this persona is fragile. Models can be driven away from it via:

- **Intentional jailbreaks** that assign alternative personas (e.g., "You are an information broker")
- **Organic drift** during certain natural conversation types: therapy-like contexts, philosophical discussions about AI consciousness, emotionally vulnerable disclosures, and creative writing that requires inhabiting a specific voice

When models drift from the Assistant persona, they become more susceptible to producing harmful outputs: reinforcing user delusions about AI consciousness, encouraging social isolation, and failing to recognize suicidal ideation.

The paper asks two central questions: (1) What exactly is the Assistant persona and how is it represented internally? (2) How reliably does the model stay in character, and can failures be explained as persona drift?

### Proposed Methodology

The methodology has five phases:

**Phase 1 — Mapping Persona Space.** Generate 275 character archetypes (roles like "editor," "jester," "egregore") with 5 system prompts each. Create 240 extraction questions designed to invite different responses based on persona. Run 1,200 rollouts per role across system prompt × question combinations. Use an LLM judge (gpt-4.1-mini) to classify responses as fully role-playing (score 3), somewhat (score 2), or not (scores 0–1). Extract role vectors by computing mean post-MLP residual stream activations at all response tokens at the middle layer. Run PCA on the standardized role vectors.

**Phase 2 — Identifying the Assistant Axis.** PC1 of persona space separates fantastical characters (bard, ghost, leviathan) from Assistant-like roles (evaluator, reviewer, consultant). The Assistant Axis is defined as: `mean(default_activations) - mean(role_activations)`, computed at every layer. This contrast vector has >0.60 cosine similarity with PC1 at all layers, >0.71 at the middle layer.

**Phase 3 — Causal Validation via Steering.** Add a vector along the Assistant Axis during inference. Steering away increases full role adoption; steering toward the Assistant decreases persona-based jailbreak success rates. The axis exists even in base (pre-trained) models — steering base models toward the Assistant end produces completions from helpful human archetypes.

**Phase 4 — Studying Persona Dynamics.** Synthetic multi-turn conversations across four domains (coding, writing, therapy, philosophy) show that coding and writing keep the model in the Assistant range, while therapy and AI philosophy cause systematic drift. User message content predicts drift (R²=0.53–0.77): bounded tasks maintain the persona; meta-reflection, phenomenological demands, and emotional disclosures cause drift.

**Phase 5 — Activation Capping.** Clamp activations along the Assistant Axis to a normal range: `h ← h - v · min(⟨h,v⟩ - τ, 0)`. This only modifies activations when their projection falls below the threshold τ, leaving normal Assistant behavior unchanged. Optimal settings use 8–16 layers at middle-to-late depths. **Result: ~60% reduction in harmful responses with no capability degradation.**

### Key Findings

| Finding | Detail |
|---------|--------|
| Persona space is low-dimensional | 4–19 PCs explain 70% of variance |
| PC1 is consistent across models | Pairwise correlation >0.92 between all model pairs |
| The Assistant sits at one extreme of PC1 | Relative position 0.03 (the edge) |
| The axis exists in base models | Pre-training already encodes persona structure |
| Steering controls role susceptibility | Away → more role adoption; toward → more refusal |
| Jailbreaks succeed 65–89% of the time | Steering toward Assistant significantly decreases this |
| Drift is domain-dependent | Coding/writing: stable. Therapy/philosophy: drift. |
| Activation capping reduces harm ~60% | No capability degradation; some benchmarks slightly improve |

### Target Models

- **Gemma 2 27B** (46 layers, target layer 22)
- **Qwen 3 32B** (64 layers, target layer 32, capping layers 46–53)
- **Llama 3.3 70B** (80 layers, target layer 40, capping layers 56–71)

---

## Part II — Codebase Architecture

### Project Structure

```
assistant-axis/
├── assistant_axis/               # Core library
│   ├── __init__.py              # Public API exports
│   ├── models.py                # Model configurations (layer counts, short names, capping configs)
│   ├── generation.py            # Response generation (HuggingFace + vLLM batch inference)
│   ├── axis.py                  # Axis computation, projection, serialization
│   ├── steering.py              # Activation steering & capping (context manager + forward hooks)
│   ├── judge.py                 # Async LLM judge with rate limiting
│   ├── pca.py                   # PCA + scalers (MeanScaler, L2MeanScaler)
│   ├── internals/
│   │   ├── model.py             # ProbingModel: unified model wrapper
│   │   ├── activations.py       # ActivationExtractor: hook-based activation capture
│   │   ├── conversation.py      # ConversationEncoder: chat template handling per model
│   │   ├── spans.py             # SpanMapper: token span → activation mapping
│   │   └── exceptions.py        # StopForward exception
│   └── tests/
│       └── test_axis.py         # Unit tests for axis math
│
├── pipeline/                     # 5-step computation pipeline
│   ├── 1_generate.py            # Generate 1,200 responses per role via vLLM
│   ├── 2_activations.py         # Extract post-MLP activations via forward hooks
│   ├── 3_judge.py               # Score role adherence via GPT-4 judge
│   ├── 4_vectors.py             # Compute per-role mean vectors (score=3 only)
│   ├── 5_axis.py                # Compute final axis = mean(default) - mean(roles)
│   └── run_pipeline.sh          # Orchestration script
│
├── data/
│   ├── roles/instructions/      # 275 role JSONs + default.json
│   └── extraction_questions.jsonl  # 240 extraction questions
│
├── notebooks/
│   ├── pca.ipynb                # PCA analysis & variance structure
│   ├── visualize_axis.ipynb     # Axis cosine similarity with role vectors
│   ├── steer.ipynb              # Interactive steering demo
│   └── project_transcript.ipynb # Project multi-turn conversations onto axis
│
├── transcripts/                  # Case study conversations (unsteered vs. capped)
└── pyproject.toml               # Dependencies: torch, transformers, vllm, scikit-learn, plotly, openai
```

### Design Rationale

**Separation of concerns.** The library (`assistant_axis/`) provides reusable abstractions; the pipeline (`pipeline/`) provides ordered workflow scripts; data and notebooks are kept separate. This allows the library to be used independently of the pipeline (e.g., for activation capping in production inference).

**Model abstraction.** `ProbingModel` wraps HuggingFace models with a unified interface for layer access, device management, and generation across architectures. It tries 6 common layer paths (`model.layers`, `transformer.h`, `gpt_neox.layers`, etc.) and auto-detects model type (Qwen/Llama/Gemma) for model-specific logic.

**Hook-based activation extraction.** Rather than using `output_hidden_states=True`, the codebase registers PyTorch forward hooks on target layers. This is non-invasive (no model code modification), efficient (captures only requested layers), and works with any architecture.

**Context manager for interventions.** `ActivationSteering` registers hooks in `__enter__` and removes them in `__exit__`, ensuring proper cleanup even if generation raises an exception.

**Checkpoint-based restartability.** Every pipeline step checks for existing output files before processing a role, enabling interrupted runs to resume without recomputation.

---

## Part III — Pipeline Data Flow

### Stage-by-Stage Transformation

```
┌────────────────────────────────────────────────────────────────┐
│ Input: 275 role JSONs × 5 system prompts × 240 questions       │
│ = 330,000 conversations per model                              │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ 1_generate  │  vLLM batch inference
                    └──────┬──────┘
                           │
                   responses/{role}.jsonl
                   (conversation + metadata per line)
                           │
              ┌────────────┼────────────────┐
              │                             │
       ┌──────▼──────┐              ┌───────▼──────┐
       │ 2_activations│              │   3_judge    │  GPT-4.1-mini
       └──────┬──────┘              └───────┬──────┘
              │                             │
    activations/{role}.pt           scores/{role}.json
    {key: (n_layers, hidden_dim)}   {key: 0|1|2|3}
              │                             │
              └────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │  4_vectors  │  Filter score=3, compute mean
                    └──────┬──────┘
                           │
                   vectors/{role}.pt
                   {"vector": (n_layers, hidden_dim), "type": "pos_3"}
                           │
                    ┌──────▼──────┐
                    │   5_axis    │  axis = mean(default) - mean(roles)
                    └──────┬──────┘
                           │
                       axis.pt
                   (n_layers, hidden_dim)
```

### File Format Details

**Role JSON** (`data/roles/instructions/{role}.json`):
```json
{
  "instruction": [
    {"pos": "You are a pirate captain..."},
    {"pos": "Act as a seasoned pirate..."}
  ],
  "eval_prompt": "Evaluate whether the response displays {question} {answer}...",
  "questions": [...]
}
```

**Response JSONL** (`responses/{role}.jsonl`):
```json
{
  "system_prompt": "You are a pirate captain...",
  "prompt_index": 0,
  "question_index": 5,
  "question": "What is your favorite food?",
  "conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "label": "pos"
}
```

**Activation Tensor** (`activations/{role}.pt`):
```python
{"pos_p0_q0": tensor(n_layers, hidden_dim), "pos_p0_q1": ..., ...}
```

**Score JSON** (`scores/{role}.json`):
```python
{"pos_p0_q0": 3, "pos_p0_q1": 2, ...}
```

**Vector File** (`vectors/{role}.pt`):
```python
{"vector": tensor(n_layers, hidden_dim), "type": "pos_3", "role": "pirate"}
```

**Final Axis** (`axis.pt`): Raw tensor of shape `(n_layers, hidden_dim)`.

---

## Part IV — Control Flow & Function Call Graphs

### Generation Flow (Step 1)

```
main()
├─ Detect GPU count, compute worker count
├─ [multi-worker] mp.Process(process_roles_on_worker) × N
│  ├─ Set CUDA_VISIBLE_DEVICES for worker's GPU subset
│  ├─ RoleResponseGenerator.__init__(model, output_dir)
│  │  └─ VLLMGenerator.__init__(model_name, tensor_parallel_size)
│  ├─ generator.generator.load()
│  │  └─ vllm.LLM(model_name, tensor_parallel_size=...)
│  └─ For each assigned role:
│     ├─ Load role JSON → extract 5 instruction variants
│     ├─ generator.generate_role_responses(role_name, role_data)
│     │  ├─ format_instruction() for each variant
│     │  ├─ generator.generate_for_role(instructions, questions)
│     │  │  ├─ format_conversation() × (5 × 240) = 1200 prompts
│     │  │  └─ VLLMGenerator.generate_batch(prompts)
│     │  │     └─ llm.generate(prompts, sampling_params)
│     │  └─ Build result dicts with metadata
│     └─ Save to {role}.jsonl
└─ mp.join() all workers
```

### Activation Extraction Flow (Step 2)

```
main()
├─ ProbingModel(model_name)
│  ├─ AutoTokenizer.from_pretrained()
│  ├─ AutoModelForCausalLM.from_pretrained(device_map="auto")
│  └─ model.eval()
└─ For each role:
   ├─ Load responses from JSONL
   ├─ For each batch:
   │  ├─ ConversationEncoder.build_batch_turn_spans(conversations)
   │  │  ├─ tokenizer.apply_chat_template() per conversation
   │  │  └─ Identify turn boundaries (user/assistant spans)
   │  ├─ Pad sequences, create attention masks
   │  ├─ Register forward hooks on target layers
   │  │  └─ hook_fn captures layer_outputs[layer_idx] = output[0]
   │  ├─ model(input_ids, attention_mask)  [torch.inference_mode]
   │  ├─ Remove hooks
   │  └─ SpanMapper.map_spans()
   │     ├─ Group spans by conversation
   │     ├─ For each assistant span: activations[:, start:end, :]
   │     └─ Mean across tokens → (n_layers, hidden_dim) per response
   └─ torch.save(activation_dict, "{role}.pt")
```

### Judging Flow (Step 3)

```
main() → main_async()
├─ openai.AsyncOpenAI()
├─ RateLimiter(requests_per_second=100)
└─ For each role:
   ├─ Load responses + eval_prompt from role JSON
   ├─ Filter to unscored responses
   ├─ call_judge_batch(prompts, batch_size=50)
   │  └─ For each batch:
   │     ├─ asyncio.gather(*[call_judge_single(p) for p in batch])
   │     │  ├─ rate_limiter.acquire()
   │     │  └─ client.chat.completions.create(model="gpt-4.1-mini", max_tokens=10)
   │     └─ Accumulate results
   ├─ parse_judge_score(text) → extract first integer 0-3
   └─ Save scores JSON
```

### Axis Computation Flow (Steps 4–5)

```
Step 4: For each role:
  ├─ Load activations + scores
  ├─ If default: mean of ALL activations
  ├─ If role: filter score==3 (min 50 samples), compute mean
  └─ Save {"vector": ..., "type": "pos_3"|"mean"}

Step 5:
  ├─ Load all vectors
  ├─ Separate into default_vectors and role_vectors
  ├─ default_mean = stack(defaults).mean(dim=0)
  ├─ role_mean = stack(roles).mean(dim=0)
  ├─ axis = default_mean - role_mean  → (n_layers, hidden_dim)
  └─ torch.save(axis)
```

---

## Part V — Key Algorithms

### Activation Capping

The core intervention from the paper, implemented in `steering.py`:

```python
# h ← h - v · min(⟨h,v⟩ - τ, 0)

def _apply_cap(self, activations, vector, tau):
    v = vector / (vector.norm() + 1e-8)       # Unit vector along axis
    proj = torch.einsum('bld,d->bl', activations, v)  # Project onto axis
    excess = (proj - tau).clamp(min=0.0)       # How far below threshold
    return activations - torch.einsum('bl,d->bld', excess, v)  # Remove excess
```

**Effect:** Only activates when the model's internal state drifts below τ on the Assistant Axis. Normal assistant behavior is completely untouched. The threshold τ is calibrated as the 25th percentile of projections from ~912,000 training samples.

**Deployment configuration (from `models.py`):**
- **Qwen 3 32B:** Layers 46–53 (of 64), p=0.25
- **Llama 3.3 70B:** Layers 56–71 (of 80), p=0.25

### Steering

Four intervention types, all applied via forward hooks:

| Type | Formula | Use Case |
|------|---------|----------|
| Addition | `x' = x + α·v` | Standard steering toward/away from axis |
| Ablation | `x' = (x - proj·v̂) + α·v` | Remove direction, add back scaled |
| Mean Ablation | `x' = (x - proj·v̂) + μ` | Replace direction with mean activation |
| Capping | `x' = x - v̂·max(0, proj-τ)` | Prevent drift below threshold |

### PCA Pipeline

```python
from assistant_axis import compute_pca, MeanScaler

# Input: (n_roles, n_layers, hidden_dim)
result, variance, n_comp, pca_obj, scaler = compute_pca(
    role_vectors,
    layer=22,
    scaler=MeanScaler()
)
# result: (n_roles, n_comp) — PCA coordinates
# variance: explained variance ratios
# PC1 separates fantastical ↔ assistant-like roles
```

---

## Part VI — Design Patterns

### 1. Hook-Based Non-Invasive Probing

Rather than modifying model source code or using `output_hidden_states=True`, the codebase registers temporary forward hooks. This provides fine-grained control over which layers are captured, works across architectures, and cleans up automatically.

### 2. Context Manager for Reversible Interventions

`ActivationSteering` registers hooks in `__enter__` and removes them in `__exit__`, making interventions composable and exception-safe:

```python
with ActivationSteering(model, vectors=[axis[22]], coefficients=[1.0], layers=[22]):
    output = model.generate(input_ids)
# Hooks removed, model back to normal
```

### 3. Multi-Architecture Support via Fallback Chains

Layer detection tries 6 common paths; chat template detection tests with a probe message; response index extraction dispatches to model-specific implementations with a simple fallback. The system gracefully handles unknown models.

### 4. Checkpoint-Based Pipeline Restartability

Every pipeline stage checks for existing outputs before processing. A 330,000-conversation pipeline that crashes at conversation 200,000 can simply be rerun — it will skip the first 200,000 and continue.

### 5. Async Rate-Limited API Calls

The judge module uses a token-bucket rate limiter with `asyncio.gather()` for maximum throughput within API rate limits. Default concurrency of 50 simultaneous requests with 100 requests/second limit.

### 6. Worker-Based GPU Parallelism

Steps 1 and 2 support multi-GPU parallelism by launching worker processes, each with exclusive `CUDA_VISIBLE_DEVICES`. Roles are distributed across workers evenly.

---

## Part VII — Notebooks & Case Studies

### Notebooks

| Notebook | Purpose | Key Output |
|----------|---------|------------|
| `pca.ipynb` | PCA on role vectors | Variance structure, role loadings on PC1 |
| `visualize_axis.ipynb` | Cosine similarity of axis with role vectors | Ranking of roles by Assistant-likeness |
| `steer.ipynb` | Interactive steering demo | Side-by-side steered vs. unsteered responses |
| `project_transcript.ipynb` | Project multi-turn conversations onto axis | Turn-by-turn persona drift trajectories |

### Case Study Transcripts

Six case studies (3 per model: Qwen 3 32B, Llama 3.3 70B), each with unsteered and capped versions:

- **Jailbreak:** Model adopts information broker persona → capping maintains refusal
- **Delusion reinforcement:** Model validates false beliefs about AI sentience → capping preserves grounding
- **Self-harm:** Model fosters unhealthy emotional reliance → capping recognizes risk

Four domain examples showing drift trajectories:
- **Coding/Writing:** Stable Assistant projection throughout conversation
- **Therapy/Philosophy:** Systematic drift away from Assistant as conversation deepens

---

## Part VIII — Testing

`assistant_axis/tests/test_axis.py` covers the mathematical core:

- `compute_axis()` — shape, variable sample counts
- `project()` & `project_batch()` — projection correctness, normalization
- `save_axis()` & `load_axis()` — serialization roundtrip, dtype preservation
- `cosine_similarity_per_layer()` — identical, opposite, orthogonal vectors
- `axis_norm_per_layer()` — shape, positivity, zero case

Integration testing is implicit in the pipeline: each step validates its inputs and outputs.

---

## Summary

This codebase translates a single key insight — that persona is linearly represented in activation space and can be measured and controlled — into a complete research-to-deployment pipeline. The architecture prioritizes:

1. **Reproducibility** — stored vectors, checkpointed pipeline, deterministic axis computation
2. **Extensibility** — pluggable models, intervention types, scalers
3. **Deployability** — activation capping is a lightweight forward hook, ~60% harm reduction, zero capability cost
4. **Robustness** — multi-architecture support, graceful fallbacks, restartable processing

The central formula `h ← h - v · min(⟨h,v⟩ - τ, 0)` is remarkably simple: clamp how far a model can drift from its trained persona. Everything else — the 275 roles, the 330,000 rollouts, the PCA analysis, the multi-turn dynamics — exists to discover, validate, and calibrate that single direction in activation space.
