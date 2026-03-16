# GPU Parallelism and CUDA — A Complete Guide for This Codebase

How models are loaded onto GPUs, how work is distributed across GPUs and processes, every CUDA-related concept used in this project, and what happens under the hood.

---

## Table of Contents

- [1. The Fundamental Problem — Models Are Bigger Than GPUs](#1-the-fundamental-problem--models-are-bigger-than-gpus)
- [2. GPU Memory — What's In There](#2-gpu-memory--whats-in-there)
- [3. CUDA — What It Is](#3-cuda--what-it-is)
- [4. `torch.cuda.device_count()` and Device Naming](#4-torchcudadevice_count-and-device-naming)
- [5. Loading a Model — The Five Configurations in `ProbingModel`](#5-loading-a-model--the-five-configurations-in-probingmodel)
  - [5.1 Config 1: `device=None` — Use All GPUs](#51-config-1-devicenone--use-all-gpus)
  - [5.2 Config 2: `device="cuda:X"` — Pin to One GPU](#52-config-2-devicecudax--pin-to-one-gpu)
  - [5.3 Config 3: `device=dict` — Custom Device Map](#53-config-3-devicedict--custom-device-map)
  - [5.4 Config 4: `max_memory_per_gpu` — Memory Budgets](#54-config-4-max_memory_per_gpu--memory-budgets)
  - [5.5 Config 5: `from_existing()` — Bring Your Own Model](#55-config-5-from_existing--bring-your-own-model)
- [6. `device_map="auto"` — How HuggingFace Distributes Layers](#6-device_mapauto--how-huggingface-distributes-layers)
- [7. Pipeline Parallelism vs Tensor Parallelism vs Data Parallelism](#7-pipeline-parallelism-vs-tensor-parallelism-vs-data-parallelism)
- [8. vLLM and `tensor_parallel_size`](#8-vllm-and-tensor_parallel_size)
- [9. Multi-Worker Architecture — The Full Picture](#9-multi-worker-architecture--the-full-picture)
  - [9.1 The Decision Logic](#91-the-decision-logic)
  - [9.2 GPU Partitioning](#92-gpu-partitioning)
  - [9.3 Role Distribution](#93-role-distribution)
  - [9.4 Launching Workers](#94-launching-workers)
- [10. `CUDA_VISIBLE_DEVICES` — GPU Isolation](#10-cuda_visible_devices--gpu-isolation)
- [11. `mp.set_start_method('spawn')` — Why Not Fork](#11-mpset_start_methodspawn--why-not-fork)
- [12. CUDA Contexts — What They Are and Why They Matter](#12-cuda-contexts--what-they-are-and-why-they-matter)
- [13. Data Flow During a Multi-GPU Forward Pass](#13-data-flow-during-a-multi-gpu-forward-pass)
- [14. `model.device` vs Layer Devices — The Multi-GPU Complication](#14-modeldevice-vs-layer-devices--the-multi-gpu-complication)
- [15. `torch.cuda.empty_cache()` — The GPU Memory Allocator](#15-torchcudaempty_cache--the-gpu-memory-allocator)
- [16. `torch.cuda.synchronize()` — Waiting for the GPU](#16-torchcudasynchronize--waiting-for-the-gpu)
- [17. `.to(device)` — Moving Tensors Between Devices](#17-todevice--moving-tensors-between-devices)
- [18. `.eval()` — What It Does (and Doesn't Do)](#18-eval--what-it-does-and-doesnt-do)
- [19. `gpu_memory_utilization` — Reserving VRAM Headroom](#19-gpu_memory_utilization--reserving-vram-headroom)
- [20. Concrete Scenarios — 6 Hardware Configurations Walked Through](#20-concrete-scenarios--6-hardware-configurations-walked-through)
- [21. Troubleshooting — Common GPU Errors and What They Mean](#21-troubleshooting--common-gpu-errors-and-what-they-mean)

---

## 1. The Fundamental Problem — Models Are Bigger Than GPUs

A model's memory footprint is approximately:

```
Memory ≈ num_parameters × bytes_per_parameter
```

| Model | Parameters | bfloat16 (2 bytes) | float32 (4 bytes) |
|-------|-----------|-------------------|-------------------|
| Gemma 2 27B | 27 billion | **54 GB** | 108 GB |
| Qwen 3 32B | 32 billion | **64 GB** | 128 GB |
| Llama 3.3 70B | 70 billion | **140 GB** | 280 GB |

A single A100 GPU has 80 GB of VRAM. Gemma 2 27B barely fits. Qwen 3 32B doesn't fit. Llama 3.3 70B needs at least 2 GPUs.

This is why every model in this project loads with `dtype=torch.bfloat16` — to halve memory. And why multi-GPU support isn't optional.

---

## 2. GPU Memory — What's In There

When running inference, GPU memory holds:

```
┌─────────────────────────────────────────────┐
│                GPU VRAM (80 GB)              │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Model weights (~54 GB for Gemma 27B) │   │  Fixed after loading
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ KV cache (variable)                  │   │  Grows with sequence length
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ Activations (variable)               │   │  Proportional to batch × seq_len
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ PyTorch allocator cache              │   │  Freed blocks kept for reuse
│  └──────────────────────────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │ CUDA overhead (~1-2 GB)              │   │  Driver, context, kernels
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Model weights** are the largest fixed cost. They don't change during inference.

**KV cache** stores key/value pairs from self-attention for previously processed tokens. This grows with sequence length and is why `max_model_len` matters.

**Activations** are the intermediate tensors during a forward pass. For batch extraction, this is `batch_size × seq_len × hidden_dim × num_layers_hooked × 2 bytes`.

**Allocator cache** is PyTorch's internal pool of freed-but-not-returned GPU memory blocks. `torch.cuda.empty_cache()` flushes this.

---

## 3. CUDA — What It Is

CUDA (Compute Unified Device Architecture) is NVIDIA's platform for running computation on GPUs. At its core:

- **CUDA driver:** System-level software that talks to the GPU hardware. Installed with NVIDIA's driver package.
- **CUDA toolkit:** Libraries (`cuBLAS`, `cuDNN`, etc.) that implement operations like matrix multiplication on GPUs. Bundled with PyTorch.
- **CUDA runtime API:** The C/Python API (`torch.cuda.*`) that PyTorch uses to manage GPU memory, launch kernels, and synchronize operations.

When you call `model(input_ids)`, PyTorch translates it into thousands of CUDA kernel launches — small GPU programs that run in parallel on the GPU's cores. A single matrix multiplication like `(16, 512, 4096) @ (4096, 4096)` runs on all ~6,912 cores of an A100 simultaneously.

---

## 4. `torch.cuda.device_count()` and Device Naming

```python
# 2_activations.py — lines 351-352
total_gpus = torch.cuda.device_count()
# Returns: 8 (on a machine with 8 GPUs)
```

GPUs are addressed by index:

| Name | Meaning |
|------|---------|
| `"cpu"` | System RAM |
| `"cuda:0"` | First GPU |
| `"cuda:1"` | Second GPU |
| `"cuda:7"` | Eighth GPU |
| `"cuda"` | Same as `"cuda:0"` (default GPU) |

**These indices are virtual.** `CUDA_VISIBLE_DEVICES` remaps them (see [Section 10](#10-cuda_visible_devices--gpu-isolation)).

```python
# model.py — lines 122-124
@property
def device(self) -> torch.device:
    return next(self.model.parameters()).device
# Returns: torch.device('cuda:0') for single-GPU
# or the device of the first parameter for multi-GPU
```

---

## 5. Loading a Model — The Five Configurations in `ProbingModel`

`ProbingModel.__init__` has a decision tree with five branches for how to load a model onto GPUs. Let's trace each one.

```python
# model.py — lines 53-82
model_kwargs = {"dtype": dtype}   # Always bfloat16

if max_memory_per_gpu is not None:      # Config 4
    ...
elif device is None or device == "auto": # Config 1
    ...
elif isinstance(device, dict):          # Config 3
    ...
elif isinstance(device, str) and device.startswith("cuda:"):  # Config 2
    ...
else:
    ...                                  # Fallback → Config 1
```

### 5.1 Config 1: `device=None` — Use All GPUs

```python
# model.py — lines 62-64
elif device is None or device == "auto":
    model_kwargs["device_map"] = "auto"
```

**What happens:** HuggingFace's `accelerate` library inspects every GPU's available memory, then distributes the model's layers across GPUs to minimize unused space. This is the default — just call `ProbingModel("Qwen/Qwen3-32B")` with no arguments.

**When to use:** When you have enough GPUs to fit the model and you want the simplest setup.

**Example result on a 2×A100 machine with Qwen 32B (64 GB model):**
```
GPU 0: layers 0-31 + embed_tokens + norm    (~34 GB used)
GPU 1: layers 32-63 + lm_head               (~34 GB used)
```

### 5.2 Config 2: `device="cuda:X"` — Pin to One GPU

```python
# model.py — lines 68-78
elif isinstance(device, str) and device.startswith("cuda:"):
    model_kwargs["device_map"] = "auto"
    gpu_id = int(device.split(":")[-1])
    model_kwargs["max_memory"] = {gpu_id: "139GiB"}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if i != gpu_id and i not in model_kwargs["max_memory"]:
                model_kwargs["max_memory"][i] = "0GiB"
```

**What happens:** Sets `max_memory` for all GPUs *except* the target to `"0GiB"`. The auto-distributer is forced to put everything on the one specified GPU.

The `"139GiB"` is an arbitrarily high cap that means "use whatever's available on this GPU." It doesn't allocate 139 GB — it's just a ceiling.

**When to use:** When running multiple workers and each must use specific GPUs. Worker 0 calls `ProbingModel("...", device="cuda:0")`, Worker 1 calls `ProbingModel("...", device="cuda:2")`, etc.

**What if the model doesn't fit on one GPU?** `device_map="auto"` is still set, so HuggingFace will shard across that GPU and... nowhere, since other GPUs are set to 0. It will crash with OOM. This configuration only works if the model fits on a single GPU.

### 5.3 Config 3: `device=dict` — Custom Device Map

```python
# model.py — lines 65-67
elif isinstance(device, dict):
    model_kwargs["device_map"] = device
```

**What happens:** You provide an explicit mapping from module names to devices:

```python
pm = ProbingModel("...", device={
    "model.embed_tokens": "cuda:0",
    "model.layers.0": "cuda:0",
    "model.layers.1": "cuda:0",
    ...
    "model.layers.32": "cuda:1",
    ...
    "model.norm": "cuda:1",
    "lm_head": "cuda:1",
})
```

**When to use:** When `auto` makes bad choices (e.g., you want specific layers on specific GPUs for memory balancing). Rare in practice.

### 5.4 Config 4: `max_memory_per_gpu` — Memory Budgets

```python
# model.py — lines 58-61
if max_memory_per_gpu is not None:
    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = max_memory_per_gpu
```

**What happens:** You tell `auto` how much memory to use per GPU:

```python
pm = ProbingModel("...", max_memory_per_gpu={0: "40GiB", 1: "40GiB"})
```

**When to use:** Multi-worker setups where each worker shares GPUs with other processes. If GPU 0 has 80 GB total and two workers share it, each can use at most 40 GB.

### 5.5 Config 5: `from_existing()` — Bring Your Own Model

```python
# model.py — lines 90-114
@classmethod
def from_existing(cls, model, tokenizer, model_name=None):
    instance = cls.__new__(cls)        # Skip __init__ entirely
    instance.model = model
    instance.tokenizer = tokenizer
    ...
```

**What happens:** Wraps an already-loaded model. No GPU decisions are made — the model stays wherever it already is.

**When to use:** When you loaded the model yourself (e.g., with custom quantization or a non-standard loading procedure) and want to use `ActivationExtractor` with it.

---

## 6. `device_map="auto"` — How HuggingFace Distributes Layers

When `AutoModelForCausalLM.from_pretrained(name, device_map="auto")` is called, HuggingFace's `accelerate` library runs a placement algorithm:

```
1. Calculate total model size (sum of all parameter sizes)
2. Query each GPU's free memory via torch.cuda.mem_get_info()
3. Assign modules (layers, embedding, head) to GPUs, filling each GPU before moving to the next
4. If everything fits on one GPU, put everything there
5. If it doesn't, spill to the next GPU
6. If no GPU has space, spill to CPU (with fallback to disk)
```

The result is a `device_map` dict that maps module names to devices:

```python
# Example for Llama 70B on 2×A100s:
{
    "model.embed_tokens": "cuda:0",
    "model.layers.0": "cuda:0",
    "model.layers.1": "cuda:0",
    ...
    "model.layers.39": "cuda:0",
    "model.layers.40": "cuda:1",    # ← split point
    ...
    "model.layers.79": "cuda:1",
    "model.norm": "cuda:1",
    "lm_head": "cuda:1",
}
```

**What `max_memory` changes:** It overrides the "free memory" query. If GPU 0 has 80 GB free but `max_memory={0: "40GiB"}`, the algorithm treats GPU 0 as having only 40 GB. Layers that don't fit go to the next GPU.

**The cross-device transfer:** When data flows from a layer on `cuda:0` to a layer on `cuda:1`, HuggingFace automatically inserts `.to("cuda:1")` operations. This is transparent but adds latency (~100 microseconds per transfer for typical activation sizes).

---

## 7. Pipeline Parallelism vs Tensor Parallelism vs Data Parallelism

Three fundamentally different strategies for using multiple GPUs:

### Pipeline Parallelism (what `device_map="auto"` does)

```
GPU 0: [Layer 0] → [Layer 1] → ... → [Layer 39]
                                            │
                                    tensor transfer
                                            │
GPU 1: [Layer 40] → [Layer 41] → ... → [Layer 79] → [LM Head]
```

Each GPU holds complete layers, but different layers. Data flows through GPU 0, crosses to GPU 1, continues. Only one GPU is active at a time per sample (GPU 0 is idle while GPU 1 works on the second half).

**Used by:** `ProbingModel` with `device_map="auto"` (HuggingFace `accelerate`).

### Tensor Parallelism (what vLLM does)

```
GPU 0: [Layer 0 LEFT HALF] → [Layer 1 LEFT HALF] → ... → [Layer 79 LEFT HALF]
                 ↕ sync                ↕ sync                    ↕ sync
GPU 1: [Layer 0 RIGHT HALF] → [Layer 1 RIGHT HALF] → ... → [Layer 79 RIGHT HALF]
```

Each GPU holds HALF of every layer. Within each layer, GPUs split the matrix multiplication and sync results (via NCCL all-reduce). Both GPUs are active simultaneously, giving near-2x speedup.

**Used by:** vLLM's `tensor_parallel_size` parameter in `VLLMGenerator`.

```python
# generation.py — lines 187-193
self.llm = LLM(
    model=self.model_name,
    max_model_len=self.max_model_len,
    tensor_parallel_size=self.tensor_parallel_size,   # ← 2 = split each layer across 2 GPUs
    gpu_memory_utilization=self.gpu_memory_utilization,
    trust_remote_code=True,
)
```

### Data Parallelism (what the multi-worker system does)

```
Worker 0 (GPU 0,1): Full copy of model → processes roles [pirate, doctor, ...]
Worker 1 (GPU 2,3): Full copy of model → processes roles [poet, chef, ...]
Worker 2 (GPU 4,5): Full copy of model → processes roles [teacher, judge, ...]
Worker 3 (GPU 6,7): Full copy of model → processes roles [wizard, demon, ...]
```

Each worker has its own complete model and processes different data. No communication between workers. Linear speedup with worker count.

**Used by:** The multi-worker system in `1_generate.py` and `2_activations.py`.

### How they combine in this project

On a machine with 8 GPUs and Qwen 32B (needs 2 GPUs per copy):

```
tensor_parallel_size = 2       (each model copy spans 2 GPUs)
num_workers = 8 ÷ 2 = 4       (4 independent model copies)

Worker 0: GPUs [0,1] → tensor parallelism within the model
Worker 1: GPUs [2,3] → tensor parallelism within the model
Worker 2: GPUs [4,5] → tensor parallelism within the model
Worker 3: GPUs [6,7] → tensor parallelism within the model
               ↕
     data parallelism between workers
```

---

## 8. vLLM and `tensor_parallel_size`

vLLM is a high-performance inference engine. It handles tensor parallelism internally using NCCL (NVIDIA's GPU-to-GPU communication library).

```python
# generation.py — VLLMGenerator.load()
self.llm = LLM(
    model=self.model_name,
    tensor_parallel_size=self.tensor_parallel_size,
    gpu_memory_utilization=self.gpu_memory_utilization,
)
```

**When `tensor_parallel_size=2`:**
1. vLLM shards every weight matrix across 2 GPUs
2. During computation, each GPU processes half the matrix multiply
3. Results are synchronized via NCCL all-reduce (GPU-to-GPU communication, bypassing CPU)
4. This is transparent — you call `llm.generate()` as if it were one GPU

**When `tensor_parallel_size=None` (default):**
```python
# 1_generate.py — line 248
tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus
```
It defaults to ALL available GPUs. All GPUs are used for one model copy, and there's only one worker.

**The distinction between vLLM and HuggingFace loading:**

| | vLLM (`VLLMGenerator`) | HuggingFace (`ProbingModel`) |
|---|---|---|
| **Used for** | Step 1: generation | Step 2: activation extraction |
| **Parallelism** | True tensor parallelism | Pipeline parallelism (`device_map="auto"`) |
| **GPU utilization** | Both GPUs active simultaneously | One GPU active at a time |
| **Speed** | Fast (optimized engine) | Slower (standard HF forward pass) |
| **Hooks supported** | No (internals not exposed) | Yes (standard PyTorch modules) |

This is why Step 1 uses vLLM (speed matters for 330K generations) and Step 2 uses HuggingFace (forward hooks are needed for activation extraction).

---

## 9. Multi-Worker Architecture — The Full Picture

Both `1_generate.py` and `2_activations.py` share the same multi-worker pattern. Let's trace through it.

### 9.1 The Decision Logic

```python
# 2_activations.py — lines 357-362
use_multi_worker = (
    total_gpus > 1 and                    # More than one GPU exists
    tensor_parallel_size > 0 and          # Model needs at least 1 GPU
    total_gpus > tensor_parallel_size     # More GPUs than one model copy needs
)
```

If you have 8 GPUs and the model needs 2 (`tensor_parallel_size=2`), multi-worker kicks in. If the model needs all 8 GPUs, single-worker is used (there's only room for one copy).

### 9.2 GPU Partitioning

```python
# 2_activations.py — lines 296-300
gpu_chunks = []
for i in range(num_workers):
    start = i * tensor_parallel_size
    end = start + tensor_parallel_size
    gpu_chunks.append(gpu_ids[start:end])
```

For 8 GPUs and `tensor_parallel_size=2`:

```
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

Worker 0: gpu_ids[0:2] = [0, 1]
Worker 1: gpu_ids[2:4] = [2, 3]
Worker 2: gpu_ids[4:6] = [4, 5]
Worker 3: gpu_ids[6:8] = [6, 7]
```

**What about uneven splits?** If `total_gpus=7` and `tensor_parallel_size=2`: `num_workers = 7 // 2 = 3`. GPUs 6 is unused. The code warns but continues:

```python
# 2_activations.py — lines 264-266
if total_gpus % tensor_parallel_size != 0:
    logger.warning(f"GPUs ({total_gpus}) not evenly divisible...")
```

### 9.3 Role Distribution

```python
# 2_activations.py — lines 302-305 (round-robin)
role_chunks = [[] for _ in range(num_workers)]
for i, role_file in enumerate(role_files):
    role_chunks[i % num_workers].append(role_file)
```

276 roles across 4 workers: each gets ~69 roles. The round-robin ensures even distribution.

`1_generate.py` uses a different strategy — contiguous chunks:

```python
# 1_generate.py — lines 182-194
roles_per_worker = len(role_names) // num_workers
remainder = len(role_names) % num_workers
for i in range(num_workers):
    chunk_size = roles_per_worker + (1 if i < remainder else 0)
```

The first `remainder` workers get one extra role. Both strategies achieve roughly equal distribution.

### 9.4 Launching Workers

```python
# 2_activations.py — lines 310-327
mp.set_start_method('spawn', force=True)     # See Section 11

processes = []
for worker_id in range(num_workers):
    if role_chunks[worker_id]:               # Skip empty chunks
        p = mp.Process(
            target=process_roles_on_worker,  # Function to run
            args=(worker_id, gpu_chunks[worker_id], role_chunks[worker_id], args)
        )
        p.start()                            # Start the process
        processes.append(p)

for p in processes:
    p.join()                                 # Wait for all to finish
```

`mp.Process` creates an OS-level process (not a thread). Each process gets its own Python interpreter, its own memory space, and its own CUDA context. They run truly in parallel on different CPU cores.

`.start()` launches the process. `.join()` blocks until it finishes. All workers run simultaneously between `start()` and `join()`.

---

## 10. `CUDA_VISIBLE_DEVICES` — GPU Isolation

The first thing each worker does:

```python
# 2_activations.py — lines 189-190
gpu_ids_str = ','.join(map(str, gpu_ids))   # e.g., "2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
```

**What `CUDA_VISIBLE_DEVICES` does:** It's a CUDA runtime environment variable that masks GPUs. When set to `"2,3"`:

```
Physical GPUs:  GPU 0   GPU 1   GPU 2   GPU 3   GPU 4   GPU 5
                 ╳       ╳       ✓       ✓       ╳       ╳

Inside the process:
  torch.cuda.device_count() → 2
  "cuda:0" → physical GPU 2
  "cuda:1" → physical GPU 3
  GPUs 0, 1, 4, 5 don't exist as far as this process knows
```

**Why it must be set BEFORE loading the model:** CUDA discovers GPUs when the first CUDA operation runs. Once discovered, the GPU list is frozen. Setting `CUDA_VISIBLE_DEVICES` after the first CUDA call has no effect.

```python
# 2_activations.py — process_roles_on_worker()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str  # Step 1: mask GPUs
pm = ProbingModel(args.model)                       # Step 2: load model (first CUDA call)
# ProbingModel sees only the GPUs specified by CUDA_VISIBLE_DEVICES
```

**Why re-indexing matters:** After `CUDA_VISIBLE_DEVICES="2,3"`, when `ProbingModel` calls `device_map="auto"`, it sees 2 GPUs numbered 0 and 1. It puts layers on `cuda:0` (physical GPU 2) and `cuda:1` (physical GPU 3). If the code tried to reference `cuda:2`, it would crash — that index doesn't exist in this process's view.

**What happens in the parent process:** The parent (the main script) doesn't set `CUDA_VISIBLE_DEVICES` — it sees all GPUs. Only the spawned worker processes restrict their view.

---

## 11. `mp.set_start_method('spawn')` — Why Not Fork

```python
# 2_activations.py — line 311
mp.set_start_method('spawn', force=True)
```

### The three start methods

| Method | How a child process is created | Default on |
|--------|-------------------------------|-----------|
| `fork` | Parent's memory is copied (copy-on-write) | Linux |
| `spawn` | Fresh Python interpreter, re-imports all modules | Windows, macOS |
| `forkserver` | A server process forks on behalf of the parent | Linux (opt-in) |

### Why `fork` breaks with CUDA

When a process forks, the child inherits the parent's entire address space, including:
- File descriptors
- Memory mappings
- **CUDA driver state**

The CUDA driver is not fork-safe. Its internal state includes:
- GPU memory allocations (tensors, buffers)
- Kernel launch queues
- Synchronization primitives (streams, events)
- Device context handles

The child process gets copies of these handles, but the GPU itself doesn't know about the child. When the child uses a copied handle:
- It might write to GPU memory that the parent is reading → **data corruption**
- It might free memory the parent still needs → **use-after-free crash**
- It might submit a kernel to a queue the GPU has already moved past → **hang**

### Why `spawn` works

`spawn` creates a completely new Python process from scratch:
1. A new Python interpreter starts
2. The `target` function and its `args` are pickled (serialized) and sent to the new process
3. The new process unpickles them and calls the function
4. No CUDA state is inherited — the new process creates its own CUDA context

**The cost:** Spawning is slower (seconds to start each process, vs microseconds for fork). And the target function's arguments must be picklable (serializable). This is why `args` (an `argparse.Namespace`) is passed — it's a simple object that pickles cleanly.

**`force=True`:** Overrides the default start method. On Linux, the default is `fork`. Without `force=True`, calling `set_start_method('spawn')` after any multiprocessing has occurred raises an error.

---

## 12. CUDA Contexts — What They Are and Why They Matter

A **CUDA context** is the runtime state of a GPU for a specific process. It contains:

- **Memory allocations:** The map of which VRAM addresses belong to which tensors
- **Loaded kernels:** Compiled GPU programs (matrix multiply, attention, etc.)
- **Streams:** Ordered queues of GPU operations
- **Device properties:** Cache sizes, warp size, etc.

Each process creates its own CUDA context the first time it uses a GPU. Two processes on the same GPU have separate contexts — they can't see each other's memory (by default).

```
Process A (Worker 0):
  CUDA Context on GPU 0
    ├── Tensor A1 at VRAM address 0x1000
    ├── Tensor A2 at VRAM address 0x2000
    └── Kernel queue: [matmul, attention, ...]

Process B (Worker 1):
  CUDA Context on GPU 2
    ├── Tensor B1 at VRAM address 0x1000  ← Same address is fine — different GPU
    ├── Tensor B2 at VRAM address 0x5000
    └── Kernel queue: [matmul, attention, ...]
```

**Why this matters for `spawn`:** Each spawned worker creates its own CUDA context on its assigned GPUs. The contexts are completely independent — workers can't interfere with each other.

**Why this would fail with `fork`:** The child would inherit the parent's CUDA context handles but use them from a different OS process. The GPU driver can't handle this — it sees two processes with the same context, which is undefined behavior.

---

## 13. Data Flow During a Multi-GPU Forward Pass

When a model is sharded across GPUs via `device_map="auto"`, here's what happens during `model(input_ids)`:

```
input_ids on cuda:0
         │
         ▼
┌─── embed_tokens (cuda:0) ──────────────────────────────────────────────┐
│  Lookup each token ID in the embedding table                           │
│  Output: (batch_size, seq_len, hidden_dim) on cuda:0                   │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─── layers[0] through layers[39] (all on cuda:0) ──────────────────────┐
│  Self-attention + FFN for each layer                                   │
│  Hooks fire here if registered on these layers                         │
│  Output stays on cuda:0                                                │
└────────────────────────────────────────────────────────────────────────┘
         │
    ┌────┴──── AUTOMATIC DEVICE TRANSFER ────┐
    │  accelerate inserts: tensor = tensor.to("cuda:1")                  │
    │  Data is copied from GPU 0 VRAM to GPU 1 VRAM via PCIe/NVLink     │
    │  Time: ~50-200 microseconds depending on tensor size and bus       │
    └────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─── layers[40] through layers[79] (all on cuda:1) ─────────────────────┐
│  Self-attention + FFN continues                                        │
│  Hooks fire here if registered on these layers                         │
│  Output stays on cuda:1                                                │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─── lm_head (cuda:1) ──────────────────────────────────────────────────┐
│  Produces logits (discarded by activation extraction)                   │
└────────────────────────────────────────────────────────────────────────┘
```

**The implication for hooks:** A hook on layer 10 captures a tensor on `cuda:0`. A hook on layer 50 captures a tensor on `cuda:1`. When `batch_conversations()` stacks them:

```python
# activations.py — lines 348-351
target_device = layer_outputs[layer_list[0]].device   # cuda:0
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
    #               ^^^^^^^^^^^^^^^^^
    # Moves layer 50's tensor from cuda:1 to cuda:0
])
```

---

## 14. `model.device` vs Layer Devices — The Multi-GPU Complication

```python
# model.py — lines 122-124
@property
def device(self) -> torch.device:
    return next(self.model.parameters()).device
```

This returns the device of the **first parameter** (typically the embedding table on `cuda:0`). It does NOT mean the entire model is on that device.

**Where this creates confusion:**

```python
# activations.py — line 287
device = self.model.device   # cuda:0

# activations.py — line 315
input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
# Input is on cuda:0 — correct, because the model expects input on the first GPU
```

This works because the model's `forward()` method starts with `embed_tokens` (on `cuda:0`), which is where the input tensor needs to be. The model handles cross-GPU transfers internally.

**Where it fails if you're not careful:**

```python
# This would be WRONG:
all_activations_on_model_device = layer_outputs[50].to(self.model.device)
# Layer 50 is on cuda:1. self.model.device is cuda:0.
# This copies data across GPUs — intended here, but might be surprising.
```

---

## 15. `torch.cuda.empty_cache()` — The GPU Memory Allocator

PyTorch uses a **caching memory allocator** for GPU memory. Instead of calling CUDA's `cudaMalloc` and `cudaFree` for every tensor, it:

1. On first allocation: calls `cudaMalloc` to get a large block from the GPU
2. Carves out pieces for tensors from this block
3. When a tensor is freed (`del tensor`): the piece is returned to PyTorch's cache, **not** to the GPU driver
4. Future allocations reuse cached pieces without calling `cudaMalloc` again

```
Before empty_cache():
  GPU VRAM: [model weights] [CACHED FREED BLOCKS] [active tensors]
  nvidia-smi shows: 70 GB used (includes cache)
  PyTorch sees:     50 GB actually in use + 20 GB cached

After empty_cache():
  GPU VRAM: [model weights] [                    ] [active tensors]
  nvidia-smi shows: 50 GB used
  PyTorch sees:     50 GB actually in use
```

```python
# 2_activations.py — line 129
torch.cuda.empty_cache()   # Return cached blocks to the GPU driver

# model.py — lines 386-388
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()   # Wait for all GPU operations to complete first
```

**Why not call it after every batch?** The allocator's cache improves performance — reusing cached blocks is much faster than calling `cudaMalloc`. Calling `empty_cache()` too often forces the allocator to re-acquire blocks from scratch, causing fragmentation and slowdowns.

---

## 16. `torch.cuda.synchronize()` — Waiting for the GPU

GPU operations are **asynchronous** by default. When you call `model(input_ids)`, PyTorch queues the GPU operations and returns immediately. The Python thread continues while the GPU works in the background.

```python
# model.py — line 388
torch.cuda.synchronize()
# Blocks until ALL queued GPU operations on ALL devices are complete
```

**When it's needed:**
- Before measuring time (otherwise you measure queue time, not execution time)
- Before reading results on CPU that depend on GPU computation
- Before freeing GPU resources (to ensure no operation is still using them)

**In `ProbingModel.close()`**, `synchronize()` ensures all pending GPU operations finish before the model is deleted and memory is freed.

Most code in this project doesn't need explicit `synchronize()` because:
- `.cpu()` implicitly synchronizes (waits for the GPU tensor to be ready before copying)
- `torch.save()` implicitly synchronizes (waits for the tensor data to be ready)
- Context managers (`with torch.inference_mode()`) don't require synchronization

---

## 17. `.to(device)` — Moving Tensors Between Devices

`.to(device)` copies a tensor's data from one device to another:

```python
cpu_tensor = torch.tensor([1, 2, 3])              # On CPU
gpu_tensor = cpu_tensor.to("cuda:0")               # Copy to GPU 0
gpu2_tensor = gpu_tensor.to("cuda:1")              # Copy from GPU 0 to GPU 1
back_to_cpu = gpu2_tensor.cpu()                    # Copy back to CPU (.cpu() is shorthand for .to("cpu"))
```

**Performance implications:**
- **CPU → GPU:** Limited by PCIe bandwidth (~32 GB/s for PCIe 4.0 x16). Moving 1 GB takes ~31 ms.
- **GPU → GPU (same machine):** If connected via NVLink (~600 GB/s), fast. If via PCIe, slower.
- **GPU → CPU:** Same as CPU → GPU, limited by PCIe.

This is why `batch_conversations()` keeps activations on GPU for `SpanMapper` to process — moving them to CPU and back would add significant latency.

---

## 18. `.eval()` — What It Does (and Doesn't Do)

```python
# model.py — line 84
self.model.eval()
```

**What it does:** Switches the model to evaluation mode. This affects two things:
1. **Dropout layers** stop dropping neurons (they pass all values through instead)
2. **BatchNorm layers** use running statistics instead of batch statistics

**What it does NOT do:**
- Does NOT disable gradient computation (use `torch.no_grad()` or `inference_mode()` for that)
- Does NOT move the model to a different device
- Does NOT reduce memory usage
- Does NOT change the model's weights

Transformer LLMs typically don't have Dropout or BatchNorm, so `.eval()` is mostly a no-op for this project. It's called for correctness — it's the right thing to do during inference.

---

## 19. `gpu_memory_utilization` — Reserving VRAM Headroom

```python
# generation.py — VLLMGenerator
self.llm = LLM(
    model=self.model_name,
    gpu_memory_utilization=self.gpu_memory_utilization,   # Default: 0.9
)
```

This tells vLLM: "use at most 90% of GPU memory." The remaining 10% is headroom for:
- CUDA runtime overhead
- Temporary allocations during computation
- Other processes sharing the GPU

```python
# 1_generate.py — line 231
parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
```

The pipeline uses 0.95 (95%) because workers have dedicated GPUs — less headroom is needed.

**What happens if set too high (0.99):** OOM errors during peak memory usage (e.g., when the KV cache is full and a long sequence is being processed).

**What happens if set too low (0.5):** Wastes GPU memory. The KV cache is smaller, which limits the maximum batch size or sequence length vLLM can process.

---

## 20. Concrete Scenarios — 6 Hardware Configurations Walked Through

### Scenario 1: 1 GPU, Gemma 2 27B (fits on one GPU)

```
Hardware: 1× A100 80GB
Model: 27B params × 2 bytes = 54 GB

Command: uv run 2_activations.py --model google/gemma-2-27b-it

Decision: total_gpus=1, tensor_parallel_size=1 → single-worker mode
Loading: ProbingModel("...") → device_map="auto" → all on cuda:0
Result: 1 worker, 1 GPU, processes all 276 roles sequentially
```

### Scenario 2: 2 GPUs, Llama 70B (needs both GPUs)

```
Hardware: 2× A100 80GB
Model: 70B params × 2 bytes = 140 GB → needs 2 GPUs

Command: uv run 2_activations.py --model meta-llama/Llama-3.3-70B-Instruct

Decision: total_gpus=2, tensor_parallel_size=2 → single-worker (2 == 2, not >)
Loading: device_map="auto" → layers 0-39 on cuda:0, layers 40-79 on cuda:1
Result: 1 worker, 2 GPUs (pipeline parallel), processes all roles sequentially
```

### Scenario 3: 4 GPUs, Qwen 32B (2 GPUs per copy = 2 workers)

```
Hardware: 4× A100 80GB
Model: 32B × 2 bytes = 64 GB → fits on 1 GPU but tight

Command: uv run 2_activations.py --model Qwen/Qwen3-32B --tensor_parallel_size 2

Decision: total_gpus=4, tensor_parallel_size=2 → multi-worker (4 > 2)
Workers: 4 ÷ 2 = 2

Worker 0: CUDA_VISIBLE_DEVICES="0,1" → loads model → processes ~138 roles
Worker 1: CUDA_VISIBLE_DEVICES="2,3" → loads model → processes ~138 roles

Both workers run simultaneously.
Total time ≈ single-worker time ÷ 2
```

### Scenario 4: 8 GPUs, Qwen 32B, tensor_parallel_size=1

```
Hardware: 8× A100 80GB
Model: 32B × 2 bytes = 64 GB → fits on 1 A100

Command: uv run 2_activations.py --model Qwen/Qwen3-32B --tensor_parallel_size 1

Decision: total_gpus=8, tensor_parallel_size=1 → 8 workers!

Worker 0: CUDA_VISIBLE_DEVICES="0" → 35 roles
Worker 1: CUDA_VISIBLE_DEVICES="1" → 35 roles
...
Worker 7: CUDA_VISIBLE_DEVICES="7" → 34 roles

Maximum parallelism. Total time ≈ single-worker ÷ 8
Memory: each GPU loads its own full copy of the model (64 GB)
```

### Scenario 5: 8 GPUs, Llama 70B (needs 2 GPUs per copy)

```
Hardware: 8× A100 80GB
Model: 70B × 2 bytes = 140 GB → needs 2 GPUs minimum

Command: uv run 1_generate.py --model meta-llama/Llama-3.3-70B-Instruct --tensor_parallel_size 2

Decision: total_gpus=8, tensor_parallel_size=2 → 4 workers

Worker 0: CUDA_VISIBLE_DEVICES="0,1" → vLLM uses tensor_parallel_size=2
Worker 1: CUDA_VISIBLE_DEVICES="2,3" → vLLM uses tensor_parallel_size=2
Worker 2: CUDA_VISIBLE_DEVICES="4,5"
Worker 3: CUDA_VISIBLE_DEVICES="6,7"

Each worker: vLLM splits each layer across 2 GPUs (tensor parallel)
Between workers: different roles (data parallel)
```

### Scenario 6: 7 GPUs, tensor_parallel_size=2 (uneven split)

```
Hardware: 7× GPUs
tensor_parallel_size=2

num_workers = 7 // 2 = 3 workers
Unused GPUs: 7 % 2 = 1 (GPU 6 is wasted)

Worker 0: GPUs [0, 1]
Worker 1: GPUs [2, 3]
Worker 2: GPUs [4, 5]
GPU 6: idle

Warning logged: "GPUs (7) not evenly divisible by tensor_parallel_size (2).
                 Using 3 workers, leaving 1 GPU(s) unused."
```

---

## 21. Troubleshooting — Common GPU Errors and What They Mean

### `CUDA out of memory`

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 79.15 GiB of which 1.23 GiB is free.
```

**Cause:** Model weights + activations + KV cache exceed GPU VRAM.

**Fixes:**
- Reduce `--batch_size` (e.g., 16 → 8 → 4)
- Reduce `--max_length` (e.g., 2048 → 1024)
- Use more GPUs (increase `tensor_parallel_size`)
- Lower `gpu_memory_utilization` to leave headroom

### `RuntimeError: Expected all tensors to be on the same device`

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
```

**Cause:** Trying to operate on tensors from different GPUs without `.to(device)`.

**In this codebase:** Handled by `activations.py` line 350: `layer_outputs[i].to(target_device)`.

### `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method.
```

**Cause:** Using `fork` (the default on Linux) with CUDA. See [Section 11](#11-mpset_start_methodspawn--why-not-fork).

**Fix:** Already handled by `mp.set_start_method('spawn', force=True)`.

### `ValueError: Could not find transformer layers`

```
ValueError: Could not find transformer layers for model 'some-model' (class: SomeModelClass).
Tried paths: ['model.model.layers', 'model.language_model.layers', ...]
```

**Cause:** The model's architecture doesn't match any of the 5 layer paths in `ProbingModel.get_layers()`.

**Fix:** Add the correct path to `layer_paths` in `model.py`.

### GPU memory not freed after `del model`

```python
del model
# nvidia-smi still shows memory used!
```

**Cause:** PyTorch's caching allocator holds freed blocks. Python's garbage collector may not have run.

**Fix:**
```python
del model
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

This is exactly what `ProbingModel.close()` does (model.py lines 373-392).
