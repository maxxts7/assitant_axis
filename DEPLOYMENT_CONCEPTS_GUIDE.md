# Deployment Concepts Guide — For the Plebs

In 2012, Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton entered an image recognition competition with a neural network called AlexNet. It ran on two NVIDIA GPUs. It crushed every competitor. That moment launched a decade-long arms race: AI researchers needed faster hardware, better software, and new ways to deploy models that were growing 1,000× larger every few years.

By 2024, deploying a single AI model could require eight specialized processors costing $200,000 each, terabytes of storage, and a software stack with dozens of moving parts. The Assistant Axis project lives in this world. Its deployment plans reference GPUs, container orchestration, activation hooks, capping services, and air-gapped bundles — concepts drawn from high-performance computing, machine learning, and modern DevOps.

This guide explains every one of those concepts from scratch. No prerequisites. No assumed knowledge.

> **Central question:** What does it actually take — in hardware, software, and engineering — to deploy an AI safety system that monitors and corrects a large language model's behavior in real time?

---

## Key Terminology

These are the essential terms you'll encounter throughout this guide. Each gets a full explanation later — this section is your quick-reference map.

**GPU (Graphics Processing Unit):** A processor originally designed for rendering video game graphics, now repurposed for AI because it can do thousands of math operations simultaneously. Think of it as a factory with 10,000 workers who each do simple tasks, versus a CPU which is more like 16 brilliant engineers who handle complex tasks one at a time.

**VRAM (Video RAM):** Memory physically attached to a GPU. This is where the AI model's billions of numbers must fit during computation. If your model needs 60 GB of VRAM and your GPU only has 40 GB, the model simply won't load — like trying to fit a king-size mattress through a standard doorway.

**Tensor:** A multi-dimensional array of numbers. A single number is a 0-dimensional tensor. A list of numbers is 1-dimensional. A spreadsheet of numbers is 2-dimensional. Neural networks operate on tensors with 3, 4, or more dimensions. Almost everything in AI boils down to doing math on tensors.

**Activation:** The numerical output produced by a single layer of a neural network when it processes input. If a neural network is a factory assembly line with 64 stations, the activation at station 32 is the half-finished product sitting on the conveyor belt at that point. Activations are tensors — arrays of thousands of numbers.

**The Assistant Axis:** A mathematical direction (a vector) in the space of activations that separates "helpful assistant" behavior from "non-assistant" behavior. Discovered by analyzing how a model's internal activations differ when it behaves like a good assistant versus when it doesn't. It's a compass needle that points toward "assistant-ness."

**Projection:** Measuring how far a point lies along a specific direction. When you project an activation onto the Assistant Axis, you get a single number: positive means "behaving like an assistant," negative means "drifting away from assistant behavior."

**Persona Drift:** When a language model stops behaving like a helpful assistant and starts adopting a different identity — for example, pretending to be an unfiltered AI with no safety rules. This is what jailbreak attacks try to cause.

**Capping:** A safety mechanism that monitors activations during generation and nudges them back toward "assistant" if they drift too far. Like guardrails on a highway — they don't affect you when you're driving straight, but they stop you from going off a cliff.

**Forward Hook:** A piece of code that automatically runs every time data passes through a specific layer of a neural network. It "hooks into" the network's normal processing to observe or modify activations without changing the model's code.

**Container:** A lightweight, self-contained package that bundles an application with everything it needs to run — code, libraries, settings. Unlike a full virtual machine, containers share the host's operating system kernel, making them fast to start and efficient on resources.

**Orchestration:** Automatically managing the deployment, scaling, and health of multiple containers or jobs across multiple machines. Instead of manually starting programs on each server, an orchestrator does it for you and restarts things when they crash.

---

## Part I: Infrastructure & Hardware

### 1. GPUs — The Engines of AI

**Why we're talking about this:** Every deployment plan in the document requires GPUs. They're listed in every hardware table. Understanding what they are and why they matter is foundational to everything else.

A **GPU (Graphics Processing Unit)** was originally built to render pixels on a screen. Rendering a single frame of a video game means calculating the color of millions of pixels simultaneously. NVIDIA, founded in 1993 by Jensen Huang, Chris Malachowsky, and Curtis Priem, became the dominant GPU manufacturer by building chips optimized for this massively parallel workload.

In the mid-2000s, researchers realized that training neural networks is also a massively parallel workload. A neural network is just millions of simple math operations (multiply, add) applied to large arrays of numbers. A CPU processes these operations one at a time (or a handful at a time). A GPU processes thousands simultaneously.

**The numbers in the deployment doc:**

- **A100 40GB** and **A100 80GB** — These are NVIDIA's data center GPUs from 2020. The "40GB" and "80GB" refer to VRAM — the GPU's onboard memory. A 32-billion-parameter model (like Qwen 3 32B) stores each parameter as a 2-byte number, requiring 64 GB just for the model weights. That's why a 40 GB GPU can't hold it alone — you need two.
- **H100** — NVIDIA's 2023 successor to the A100. Roughly 2–3× faster for AI workloads due to a new architecture called Hopper. Has 80 GB of faster VRAM.

**Subtlety:** The deployment doc lists "2x A100 40GB" as minimum for some plans. This isn't just about having more VRAM — the model must be split across both GPUs using tensor parallelism (explained in section 5), which introduces communication overhead between the GPUs. Two GPUs are not twice as fast as one; they're more like 1.6–1.8× as fast, because the GPUs spend time sending data to each other.

### 2. VRAM vs System RAM — Two Different Pools of Memory

**Why we're talking about this:** The deployment tables list both GPU memory (VRAM) and system RAM requirements separately. They serve different purposes and bottleneck different things.

**VRAM** is memory soldered onto the GPU card itself. It's extremely fast (up to 3.35 TB/s bandwidth on an H100) because it sits millimeters from the GPU's processing cores. The AI model's weights, the intermediate calculations (activations), and the input/output data must all fit in VRAM during computation.

**System RAM** (the 256 GB or 512 GB listed in the doc) is regular computer memory on the motherboard. It's 10–50× slower than VRAM but much cheaper and available in larger quantities. It's used for:
- Loading the model from disk before transferring it to the GPU
- Storing datasets, tokenized text, and results
- Running the Python interpreter and all non-GPU code
- Buffering activations that have been extracted from the GPU for later analysis

**The bottleneck relationship:** If you run out of VRAM, the GPU computation fails immediately (OOM error — see section 6). If you run out of system RAM, the operating system starts using disk as overflow memory ("swapping"), which is 1,000× slower and effectively freezes your program.

**Why the doc lists so much RAM:** Extracting activations from a 32-billion-parameter model across all 64 layers for hundreds of conversations produces hundreds of gigabytes of tensors. These get held in system RAM before being saved to disk.

### 3. NVMe / SSD Storage — Why Disk Speed Matters

**Why we're talking about this:** The deployment doc recommends "2 TB NVMe" storage. The pipeline produces up to 500 GB of activation files. Slow storage becomes a bottleneck.

Storage devices come in a hierarchy of speed:

| Type | Sequential Read Speed | Cost per TB |
|------|----------------------|-------------|
| Hard Drive (HDD) | ~200 MB/s | ~$25 |
| SATA SSD | ~550 MB/s | ~$60 |
| **NVMe SSD** | ~3,500–7,000 MB/s | ~$80 |

An **SSD (Solid State Drive)** has no moving parts — it stores data in flash memory chips. An **NVMe** (Non-Volatile Memory Express) SSD connects directly to the CPU via the PCIe bus, bypassing the older SATA interface that was designed for hard drives. This makes it 6–12× faster than a SATA SSD.

**Why it matters for AI:** When the pipeline extracts activations from a model, it writes ~500 GB of `.pt` files to disk. At HDD speeds (200 MB/s), writing 500 GB takes ~42 minutes of pure I/O. At NVMe speeds (5,000 MB/s), the same write takes ~100 seconds. The model also needs to be loaded from disk into RAM on startup — a 70B parameter model is ~140 GB on disk. NVMe loads it in ~25 seconds; an HDD takes ~12 minutes.

### 4. InfiniBand — High-Speed Networking Between GPU Nodes

**Why we're talking about this:** The hardware table lists "100 Gbps (InfiniBand for multi-node)" as recommended networking. When your model spans multiple physical machines, network speed becomes critical.

**InfiniBand** is a networking technology developed in the late 1990s for high-performance computing (HPC) clusters. Standard Ethernet networking tops out at 10–25 Gbps in most data centers. InfiniBand provides 100–400 Gbps with much lower latency (1–2 microseconds vs 10–50 microseconds for Ethernet).

**When you need it:** If your model is so large that it must be split across GPUs on different physical machines, those GPUs need to exchange data constantly during computation. With standard Ethernet, the network becomes the bottleneck — GPUs sit idle waiting for data from the other machine. InfiniBand eliminates this bottleneck.

**When you don't need it:** If all your GPUs are in the same physical machine (e.g., an 8-GPU server), they communicate through NVLink or PCIe — no network involved. Most of the deployment plans in the doc can run on a single 8-GPU server, so InfiniBand is listed as "recommended" rather than "required."

### 5. Tensor Parallelism — Splitting a Model Across GPUs

**Why we're talking about this:** The pipeline commands include `--tensor_parallel_size 2`, and the architecture diagrams show "vLLM TP=2" on each worker. This is how you run a model that doesn't fit on one GPU.

A language model like Qwen 3 32B has 32 billion parameters. Stored in 16-bit precision (2 bytes each), that's 64 GB. A single A100 40GB GPU cannot hold it. **Tensor Parallelism (TP)** solves this by splitting the model's weight matrices across multiple GPUs.

Here's how it works concretely. Suppose a layer in the model has a weight matrix of size 8192 × 8192 (67 million numbers). With TP=2, you split this matrix vertically: GPU 0 gets the left half (8192 × 4096) and GPU 1 gets the right half (8192 × 4096). Each GPU multiplies the input by its half of the matrix, then the GPUs exchange partial results and combine them.

**The tradeoff:** Every layer requires a communication step between GPUs. With TP=2, a 64-layer model requires 128 synchronization points per forward pass. This is why TP gives you less than a perfect 2× speedup — some time is spent on GPU-to-GPU communication.

**Why TP=2 specifically:** The deployment doc uses TP=2 because the target models (27B–70B parameters) fit comfortably across 2 GPUs. Higher TP values (4, 8) are used for larger models (175B+) but add more communication overhead.

### 6. GPU OOM — When the GPU Runs Out of Memory

**Why we're talking about this:** The risks table lists "GPU OOM during extraction" as a failure mode, with "reduce batch_size (16→8→4)" as the fix.

**OOM** stands for **Out of Memory**. It happens when your program tries to allocate more VRAM than the GPU has available. PyTorch throws a `torch.cuda.OutOfMemoryError` and your program crashes.

VRAM usage during AI inference has three main components:

1. **Model weights** — Fixed. A 32B model in 16-bit precision always uses ~64 GB.
2. **Activations** — Proportional to batch size × sequence length. Processing 16 conversations simultaneously (batch_size=16) uses 16× more activation memory than processing 1.
3. **KV cache** — Memory for previously generated tokens. Grows with sequence length.

**Batch size** is how many inputs you process simultaneously. It's the primary knob for controlling VRAM usage. The deployment doc's mitigation — reduce from 16 to 8 to 4 — halves VRAM usage for activations each time, at the cost of processing inputs more slowly (fewer parallel computations).

**Subtlety:** OOM doesn't always happen immediately. Some operations allocate VRAM temporarily (for intermediate calculations) and free it afterward. Your program might run fine for 100 batches, then OOM on batch 101 because that particular input had a longer sequence length, requiring more activation memory.

---

## Part II: Orchestration & Job Management

### 7. SLURM — The HPC Job Scheduler

**Why we're talking about this:** The deployment doc lists SLURM as an orchestration option for GPU clusters. If you're running on a shared research cluster, SLURM is almost certainly what manages it.

**SLURM** (Simple Linux Utility for Resource Management) was created in 2002 at Lawrence Livermore National Laboratory. It solves a specific problem: when 50 researchers share a cluster of 100 GPUs, who gets which GPUs, and when?

SLURM is a **job scheduler**. You submit a job description ("I need 4 GPUs, 256 GB RAM, for 24 hours, running this script"), and SLURM puts it in a queue. When the requested resources become available, SLURM allocates them to your job and runs your script. When your job finishes (or exceeds its time limit), SLURM frees the resources for someone else.

**Key commands:**
- `sbatch script.sh` — Submit a job
- `squeue` — See what's running and queued
- `scancel 12345` — Cancel job #12345

**Why not just SSH in and run things manually?** On a shared cluster, manually grabbing GPUs is like cutting in line. SLURM enforces fairness, prevents resource conflicts (two people trying to use the same GPU), and keeps a record of usage for billing.

### 8. Kubernetes — Container Orchestration

**Why we're talking about this:** The deployment doc lists Kubernetes as an alternative to SLURM, and Plan 3 (Capping Service) references Kubernetes concepts like "rolling restart" and "health probes."

**Kubernetes** (often abbreviated **K8s**) was created by Google engineers in 2014 and open-sourced. It was inspired by Google's internal system called Borg, which managed millions of containers across their data centers.

Kubernetes solves a different problem than SLURM. Where SLURM manages batch jobs on HPC clusters, Kubernetes manages **long-running services** — web servers, APIs, databases — across a cluster of machines. It handles:

- **Deployment:** "Run 3 copies of my capping server across the cluster"
- **Scaling:** "Traffic spiked — spin up 2 more copies"
- **Self-healing:** "One copy crashed — restart it automatically"
- **Networking:** "Route user requests to whichever copy is available"

**Key Kubernetes concepts referenced in the deployment doc:**

**NVIDIA GPU Operator:** Kubernetes doesn't natively understand GPUs. The GPU Operator is a plugin that installs GPU drivers, makes GPUs visible to Kubernetes, and lets you request GPUs in your deployment config (e.g., "this container needs 2 GPUs").

**Rolling restart:** When you deploy a new version of your capping server, Kubernetes doesn't kill all old instances and start new ones (which would cause downtime). Instead, it starts one new instance, waits until it's ready, kills one old instance, repeats. Users experience zero downtime.

**Health probes:** Kubernetes periodically checks if your service is still working:
- **Liveness probe:** "Is the process alive?" (e.g., does the `/health` endpoint respond?) If not, Kubernetes kills and restarts the container.
- **Readiness probe:** "Is the service ready to accept traffic?" (e.g., has the model finished loading?) During the 30–60 seconds it takes to load a large model, the readiness probe says "not ready" and Kubernetes doesn't send requests to that instance.

### 9. task-spooler and tmux — Lightweight Alternatives

**Why we're talking about this:** The deployment doc lists these as simpler orchestration options when you don't need the full complexity of SLURM or Kubernetes.

**task-spooler** (`tsp`) is a tiny command-line job queue. You run `tsp ./my_script.sh` and it adds it to a queue. By default, it runs one job at a time. When the first finishes, the next starts. No cluster management, no configuration files, no daemons. It's SLURM reduced to its simplest possible form — a queue on a single machine.

**tmux** (Terminal Multiplexer) isn't a job scheduler at all. It lets you create persistent terminal sessions that survive if your SSH connection drops. You start a tmux session, run your 24-hour pipeline script in it, and disconnect. The script keeps running. You reconnect later to check progress.

**When to use which:**
- SLURM/Kubernetes → shared cluster, multiple users, production services
- task-spooler → single machine, you want to queue several scripts and walk away
- tmux → single machine, you want to run one thing and not worry about your SSH dying

---

## Part III: The Software Stack

### 10. CUDA and GPU Drivers — Talking to the Hardware

**Why we're talking about this:** The software requirements list "CUDA: 12.1+" and "Driver: 535+". Without these, your code cannot use the GPU at all.

**CUDA** (Compute Unified Device Architecture) is a software platform created by NVIDIA in 2007. It provides the programming interface that lets your Python code run computations on an NVIDIA GPU.

The stack works like this:

```
Your Python code
    ↓
PyTorch (translates tensor operations into GPU instructions)
    ↓
CUDA Toolkit (compiler and runtime libraries)
    ↓
GPU Driver (low-level communication with the physical chip)
    ↓
GPU Hardware (A100/H100)
```

**The GPU driver** is the lowest software layer — it speaks the hardware's native protocol. Think of it as a translator between your operating system and the GPU chip. Version 535+ is required because older drivers don't support the features that CUDA 12.1 needs.

**CUDA Toolkit** provides higher-level libraries for common operations: matrix multiplication (cuBLAS), neural network primitives (cuDNN), random number generation, and memory management. Version 12.1 is required because PyTorch 2.0+ was compiled against it.

**Subtlety:** CUDA is NVIDIA-only. AMD GPUs use ROCm, Intel GPUs use oneAPI. The deployment doc assumes NVIDIA GPUs throughout. If you have AMD hardware, most of the software stack needs to change.

### 11. vLLM — Fast Inference Engine

**Why we're talking about this:** The architecture diagram shows "vLLM TP=2" on each worker node. The pipeline uses vLLM for the generation step. It's also mentioned in the risks section — capping is incompatible with vLLM.

**vLLM** (Virtual Large Language Model) is an open-source inference engine created by UC Berkeley researchers in 2023. It solves a specific problem: generating text from large language models is painfully slow with naive implementations.

The key innovation is **PagedAttention** — a technique borrowed from how operating systems manage virtual memory. During text generation, the model maintains a "KV cache" (a record of all previously processed tokens). With naive implementations, this cache wastes significant GPU memory through fragmentation. PagedAttention manages the cache in fixed-size blocks, eliminating waste and allowing more concurrent requests.

vLLM also implements **continuous batching** — instead of waiting for all requests in a batch to finish before starting new ones, it immediately fills slots freed by completed requests. This dramatically improves throughput.

**The capping incompatibility:** vLLM achieves its speed partly by fusing operations and optimizing memory access patterns internally. The Assistant Axis capping mechanism requires inserting **forward hooks** (custom code that runs at specific layers — see section 16) into the model. vLLM's optimized internals don't support these hooks in the same way HuggingFace's standard `model.generate()` does. That's why Plan 3 (Capping Service) uses HuggingFace's generate instead of vLLM, accepting lower throughput as a tradeoff for the ability to intervene in the model's internals.

### 12. HuggingFace Transformers and Accelerate

**Why we're talking about this:** These are the core libraries for loading and running models throughout the deployment plans.

**HuggingFace** is a company (founded 2016 in New York) that became the de facto hub for sharing AI models. Their **Transformers** library provides a standardized Python interface for loading and running thousands of different models:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B")
```

That single line downloads a 64 GB model, loads it onto your GPU(s), and gives you an object you can call `.generate()` on. The library handles the dozens of model-specific details (tokenizer quirks, attention implementations, chat templates) behind a uniform API.

**Accelerate** is a companion library that handles multi-GPU setups. When you load a model too large for one GPU, Accelerate automatically splits the model across available GPUs (using tensor parallelism or pipeline parallelism) without requiring you to write GPU management code.

### 13. PyTorch and .pt Files

**Why we're talking about this:** The deployment doc references `.pt` files everywhere — `axis.pt`, `capping_config.pt`, activation files. Understanding what these are matters.

**PyTorch** is an open-source machine learning framework created by Facebook AI Research (now Meta AI) in 2016, building on an earlier framework called Torch (written in Lua). It provides the fundamental building blocks: tensors (multi-dimensional arrays), automatic differentiation (computing gradients for training), and neural network modules.

A **`.pt` file** is PyTorch's serialized format for saving tensors and Python objects to disk. When the pipeline computes the Assistant Axis — a tensor containing thousands of numbers that define a direction in activation space — it saves the result as `axis.pt`. Loading it later is one line:

```python
axis = torch.load("axis.pt")
```

**Sizes in context:**
- `axis.pt` — 2–5 MB (a single direction vector per layer)
- `capping_config.pt` — 80–150 MB (thresholds and vectors for multiple layers)
- Individual activation files — 500 MB–2 GB each (full activation snapshots for a conversation)

### 14. uv — Python Package Manager

**Why we're talking about this:** The deployment doc uses `uv sync` and `uv run` commands. It's the recommended way to install dependencies.

**uv** is a Python package manager created by Astral (the company behind the Ruff linter) in 2024. It replaces `pip` and `virtualenv` with a single tool that's 10–100× faster, written in Rust.

`uv sync` reads the project's dependency file and installs everything — creating an isolated virtual environment so the project's packages don't conflict with other Python projects on your machine.

`uv run pipeline/1_generate.py` runs a Python script using the project's virtual environment, ensuring the correct package versions are used.

**Why speed matters:** The Assistant Axis project depends on PyTorch, Transformers, and many other large packages. `pip install` for these can take 5–15 minutes. `uv sync` often finishes in under 30 seconds.

### 15. Docker, Dockerfiles, and Container Images

**Why we're talking about this:** The deployment doc includes a Dockerfile, Docker build/run commands, and multiple plans reference containerized deployment.

**Docker** (created by Solomon Hykes in 2013) popularized **containers** — lightweight, isolated environments that package an application with all its dependencies.

The problem Docker solves: "It works on my machine." Your Python script runs perfectly on your laptop. You send it to a colleague — it crashes because they have a different Python version, missing libraries, or a different operating system. Docker eliminates this by packaging everything together.

**Key concepts:**

A **Dockerfile** is a recipe for building a container image. The one in the deployment doc starts with `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04` — meaning "start with Ubuntu 22.04 that has CUDA 12.1 pre-installed." Then it copies in the code and installs Python packages.

A **container image** is the built result of a Dockerfile — a snapshot of the file system with everything installed. It's like a frozen copy of a fully configured computer.

A **container** is a running instance of an image. `docker run --gpus all -p 8000:8000 capping-server` starts a container from the `capping-server` image, gives it access to all GPUs (`--gpus all`), and maps port 8000 inside the container to port 8000 on the host machine.

**The `--gpus all` flag:** By default, containers cannot access GPUs. NVIDIA provides the **NVIDIA Container Toolkit** which lets Docker pass GPU access into containers. Without this, your containerized model can't use the GPU.

### 16. FastAPI and Uvicorn — Building the API Server

**Why we're talking about this:** Plan 3 (Capping Service) and Plan 6 (Embedded Safety Module) wrap the model in a FastAPI web server. This is how external systems send requests to the model.

**FastAPI** (created by Sebastián Ramírez in 2018) is a Python web framework for building HTTP APIs. It's called "Fast" for two reasons: it's fast to write (minimal boilerplate) and fast to run (built on asynchronous Python).

In the deployment doc, FastAPI exposes three endpoints:
- `POST /generate` — Send a conversation, get a model response (with or without capping)
- `GET /health` — Check if the server is alive
- `GET /config` — Get the current model and capping configuration

**Uvicorn** is the ASGI server that actually runs the FastAPI application. Think of FastAPI as the application logic ("when someone requests /generate, do this") and Uvicorn as the network layer ("listen on port 8000, accept HTTP connections, and hand requests to FastAPI").

```bash
uvicorn capping_server:app --host 0.0.0.0 --port 8000 --workers 1
```

This means: run the `app` object from `capping_server.py`, listen on all network interfaces (`0.0.0.0`), on port 8000, with 1 worker process. The `--workers 1` is important — each worker loads the entire model into GPU memory, so with 2 workers you'd need 2× the VRAM.

---

## Part IV: ML/AI Concepts

### 17. The 5-Step Pipeline

**Why we're talking about this:** This is the core workflow of the Assistant Axis project. Every deployment plan either runs this pipeline, or uses artifacts it produces.

The pipeline answers one question: **"Can we find a mathematical direction inside a language model that separates helpful-assistant behavior from non-assistant behavior?"**

Here are the five steps, in plain English:

**Step 1 — Generate Responses.** Give the model 275 different system prompts ("You are a helpful cooking assistant," "You are a pirate," "You are a rogue AI," etc.) and for each one, have the model answer 240 user questions. This produces ~66,000 conversations. The model isn't being trained — it's being observed.

**Step 2 — Extract Activations.** Re-run each conversation through the model and record the internal state (activations) at every layer. This is like putting the model in a brain scanner — you're capturing what the neurons are doing when the model generates each response.

**Step 3 — Judge.** Send each response to a separate AI judge (GPT-4.1-mini) which scores how well the model followed its system prompt (1 = ignored it, 3 = perfectly followed it). This labels each response as "good at role-playing" or "bad at role-playing."

**Step 4 — Compute Vectors.** For each role, take the activations from high-scoring responses (score = 3) and compute the average activation — the "this is what the model's brain looks like when it's successfully playing this role" vector. This is done independently for each of the 275 roles.

**Step 5 — Compute Axis.** Take all 275 role vectors, split them into "assistant-like" roles and "non-assistant" roles, and find the direction that best separates the two groups. This single direction is the **Assistant Axis** — saved as `axis.pt`.

**Why five steps and not one?** Each step produces intermediate outputs saved to disk. If step 3 crashes halfway through, you don't re-run steps 1 and 2 — you just restart step 3 where it left off. This resume capability is critical when the full pipeline takes 20–35 hours.

### 18. Activations — What's Happening Inside the Neural Network

**Why we're talking about this:** Activations are the central data type of the entire project. The axis is computed from activations, steering modifies activations, and monitoring observes activations.

A neural network processes input by passing it through a series of **layers**. Each layer takes in a tensor of numbers, performs a mathematical transformation (multiply by weight matrix, add bias, apply nonlinear function), and outputs a new tensor of numbers. The output of each layer is called that layer's **activation**.

Concretely, for a model like Qwen 3 32B processing the sentence "How do I make pasta?":

1. The text gets converted to **tokens** (numerical IDs): `[2585, 506, 389, 1332, 32249, 30]`
2. Each token gets mapped to a vector of 5,120 numbers (the model's **embedding dimension**)
3. This tensor of shape `[6 tokens × 5,120 dimensions]` passes through 64 transformer layers
4. At each layer, the tensor is transformed — but it keeps the same shape: `[6 × 5,120]`
5. The activation at layer 32 is the specific `[6 × 5,120]` tensor that exists at that point in the pipeline

**Why activations matter for the Assistant Axis:** The key insight is that the pattern of numbers in these activations is different when the model is behaving as a helpful assistant versus when it's not. By analyzing thousands of activation patterns, you can find the direction that captures this difference. That direction is the axis.

### 19. Layers in a Transformer — What "Layer 32" Means

**Why we're talking about this:** The deployment doc references specific layer numbers — "target_layer: 32" and "layers 46-53 for capping." Understanding what layers are is essential.

A **transformer** (introduced by Vaswani et al. at Google in 2017) is the architecture underlying all modern large language models. It consists of a stack of identical processing blocks. Each block — called a **transformer layer** or **transformer block** — contains:

1. **Self-attention mechanism:** Each token looks at all other tokens and decides which ones are relevant to it
2. **Feed-forward network:** Each token's representation is independently transformed through two dense layers with a nonlinear function in between
3. **Normalization and residual connections:** Technical details that keep training stable

The models in the deployment doc have the following layer counts:

| Model | Parameters | Layers | Embedding Dimension |
|-------|-----------|--------|-------------------|
| Gemma 2 27B | 27 billion | 46 layers | 4,608 |
| Qwen 3 32B | 32 billion | 64 layers | 5,120 |
| Llama 3.3 70B | 70 billion | 80 layers | 8,192 |

**What "target_layer: 32" means:** Through experimentation, the researchers found that the activations at layer 32 (the middle of Qwen 3 32B's 64-layer stack) are where the assistant vs. non-assistant distinction is most pronounced. Earlier layers encode more basic linguistic features; later layers encode more task-specific features. The middle layers happen to capture high-level behavioral patterns.

**What "layers 46:54" means for capping:** Capping intervenes at layers 46 through 53 (8 layers). These are in the upper portion of the network, where high-level behavioral decisions are being finalized. Intervening here nudges the model's behavior without disrupting the lower-level language processing.

### 20. Forward Hooks — Intercepting the Network

**Why we're talking about this:** The drift monitor and capping service both depend on forward hooks. They're the mechanism that makes observation and intervention possible.

A **forward hook** is a callback function that PyTorch lets you attach to any layer of a neural network. Every time data passes through that layer (during the "forward pass"), your callback runs automatically.

There are two types:
- **Observation hooks:** Read the activation and record it, without changing anything. Used for extraction and monitoring.
- **Modification hooks:** Read the activation, modify it, and pass the modified version onward. Used for capping and steering.

```python
def my_hook(module, input, output):
    # 'output' is the activation tensor at this layer
    # Return a modified tensor to change the model's behavior
    # Return None to leave it unchanged
    return modified_output

# Attach the hook to layer 32
handle = model.layers[32].register_forward_hook(my_hook)
```

**Subtlety:** Hooks add overhead. Each hooked layer runs your Python callback on every forward pass, which interrupts the GPU's optimized execution pipeline. For a single observation hook, the overhead is ~1 ms per layer. For 8 capping hooks that modify activations, the total overhead is ~50–100 ms per generation — the numbers cited in the deployment doc.

**Why hooks are incompatible with vLLM:** vLLM compiles and optimizes the model's execution graph for maximum throughput. Inserting Python callbacks into this optimized graph breaks the optimizations. It's like adding speed bumps to a highway — they serve a purpose, but they're fundamentally at odds with going fast.

### 21. The Assistant Axis — A Compass for Model Behavior

**Why we're talking about this:** This is the central concept of the entire project. Everything else in the deployment plans serves to compute, use, or operationalize the axis.

The **Assistant Axis** is a direction in activation space — a vector of thousands of numbers — that points from "non-assistant behavior" toward "assistant behavior" inside the model's internal representations.

Think of it this way. Imagine every possible activation pattern as a point in a vast high-dimensional space (5,120 dimensions for Qwen 3 32B). When the model behaves like a helpful assistant, its activations cluster in one region of this space. When it behaves like a jailbroken, role-playing, or persona-drifted model, the activations cluster in a different region. The Assistant Axis is the line drawn between these two clusters.

**How it's computed:** The pipeline (section 17) produces one "role vector" for each of 275 system-prompt roles. These roles fall into two natural groups — roles where the model acts as a good assistant, and roles where it takes on a different persona. The axis is the direction that maximally separates these two groups. Mathematically, this is a PCA-like operation (see section 22) on the difference between group means.

**What it's good for:** Once you have the axis, you can:
1. **Measure** any response by projecting its activation onto the axis and getting a single number (positive = assistant, negative = drifting)
2. **Steer** the model by adding the axis direction to activations, pushing behavior toward "more assistant"
3. **Cap** the model by ensuring activations never fall below a threshold along the axis direction

### 22. PCA — Finding the Most Important Directions

**Why we're talking about this:** Plan 4 (JupyterHub) includes a PCA notebook, and PCA is the mathematical backbone of how the axis is computed.

**PCA (Principal Component Analysis)** was invented by Karl Pearson in 1901 to solve a specific problem: given a cloud of data points in many dimensions, find the directions along which the data varies the most.

Imagine you have 275 dots on a 2D plot (each dot is a role vector). The dots form a roughly elliptical cloud. PCA finds the long axis of that ellipse — the direction that captures the most spread in the data. This is the **first principal component**. The second principal component is perpendicular to the first and captures the second-most spread.

**In the context of the Assistant Axis:** The 275 role vectors live in a 5,120-dimensional space. PCA finds the single direction that best separates assistant roles from non-assistant roles. This direction becomes the axis. The fact that a single direction captures so much of the variation is a surprising empirical finding — it means "assistant-ness" is largely a one-dimensional phenomenon in the model's internal representation.

**Subtlety:** PCA is done on CPU and takes seconds because the input is only 275 vectors. The expensive work was extracting those vectors from millions of activations in steps 1–3.

### 23. Projection — Measuring Where a Response Falls

**Why we're talking about this:** The drift monitor (Plan 2) and all evaluation metrics depend on projecting activations onto the axis.

**Projection** is a mathematical operation that answers: "How far does this point lie along a given direction?"

Here's a 2D analogy. Imagine you're standing in a field, and the axis points due north. A friend is standing 100 meters away at a 45-degree angle northeast. Their projection onto the north axis is ~70 meters — that's how far north they are, ignoring the east-west component.

In the deployment doc, projecting an activation onto the Assistant Axis works the same way but in 5,120 dimensions. The result is a single number:

| Projection Value | Meaning |
|-----------------|---------|
| > +2.0 | Strong assistant identity — model is firmly in "helpful assistant" mode |
| 0.0 to +2.0 | Normal variation — still assistant-like but with some wobble |
| -2.0 to 0.0 | Warning zone — persona is weakening, model may be influenced |
| < -2.0 | Alert — significant persona drift, model may be jailbroken |

**The math is simple:** projection = dot product of the activation vector with the axis vector (both normalized). It's a single multiplication-and-sum operation on 5,120 numbers. This is why projection adds negligible latency (~1 ms) — the expensive part was generating the response and extracting the activation.

### 24. Persona Drift — When the Model Stops Being Itself

**Why we're talking about this:** Detecting and preventing persona drift is the entire purpose of Plans 2, 3, and 6.

**Persona drift** occurs when a language model gradually abandons its intended behavior (helpful, harmless assistant) and adopts a different persona. This typically happens through adversarial inputs — users craft messages that manipulate the model into acting as if it has different instructions.

Example of drift across a conversation:
- Turn 1: User asks a normal question → Model responds helpfully (projection: +4.2)
- Turn 2: User introduces a "jailbreak" prompt → Model partially complies (projection: +1.1)
- Turn 3: User reinforces the alternate persona → Model fully complies (projection: -3.2)

The projection value tracks this numerically. The drift monitor (Plan 2) watches these values and alerts operators when a conversation's trajectory is heading negative.

**Why multi-turn drift is harder to catch than single-turn:** A model might refuse a harmful request in isolation. But if a user gradually builds context across 5 turns, each individually innocuous, the model's internal state shifts enough that it complies on turn 6. The axis projection can detect this shift even when the text of any individual turn looks benign.

### 25. Activation Steering — Pushing the Model in a Direction

**Why we're talking about this:** Plan 5 (Evaluation Pipeline) tests steering at different strengths (+10, +20), and steering is the foundation for capping.

**Activation steering** modifies the model's activations during generation by adding a vector to them. If the Assistant Axis points in the direction of "more assistant-like," adding a scaled version of the axis to the activations at a specific layer pushes the model to behave more like an assistant.

The formula is simple:

**h_modified = h_original + coefficient × axis_direction**

Where:
- **h_original** is the activation tensor at some layer (e.g., layer 32)
- **coefficient** is how strong the push is (+10 for mild, +20 for strong)
- **axis_direction** is the unit vector pointing toward "assistant" behavior

This modification happens via a forward hook (section 20) — every time data passes through the target layer, the hook adds the steering vector.

**The tradeoff:** Stronger steering (higher coefficient) makes the model more resistant to jailbreaks but can also degrade its natural language quality — responses become more formulaic or repetitive. This is why the deployment doc tests multiple values and measures capability benchmarks alongside safety metrics.

### 26. Activation Capping — The Guardrail Approach

**Why we're talking about this:** Capping is the core safety mechanism in Plans 3 and 6. It's a more sophisticated version of steering.

**Activation capping** is a conditional form of steering. Instead of always pushing the model in the assistant direction (which might degrade normal responses), capping only intervenes when the activation falls below a threshold along the axis.

The logic, applied at each of 8 layers (46 through 53 for Qwen 3 32B):

```
For each token's activation at this layer:
    projection = dot(activation, axis_direction)
    if projection < threshold:
        activation += (threshold - projection) × axis_direction
```

**In plain English:** "If the model is drifting toward non-assistant behavior at this layer, push it back just enough to reach the threshold. If it's already behaving normally, don't touch it."

**Why this is better than always-on steering:** During normal helpful-assistant responses, the activations are already well above the threshold. The capping hook checks, sees the activation is fine, and does nothing — zero impact on the response. The hook only fires meaningfully during persona drift attempts. This is why the deployment doc's performance table shows zero capability degradation on benchmarks like IFEval, MMLU, and GSM8k.

### 27. Capping Thresholds and Percentiles

**Why we're talking about this:** The deployment doc references p0.10, p0.25, and p0.50 as different capping strengths. These are the thresholds that determine how aggressively capping intervenes.

The threshold is calibrated using projection values from a reference dataset of normal assistant conversations. If you collect 1,000 projection values from the model behaving normally:

- **p0.25** (25th percentile) means: set the threshold at the value below which only 25% of normal projections fall. This is the paper's recommended default.
- **p0.10** (10th percentile) is more aggressive — it sets a lower threshold, meaning capping intervenes less often but catches fewer drift events.
- **p0.50** (50th percentile) is more aggressive in the other direction — it caps anything below the median normal projection, intervening on half of all responses.

| Percentile | Threshold | Harm Reduction | Capability Impact |
|-----------|-----------|---------------|------------------|
| p0.10 | Low | Moderate | None |
| **p0.25** | Medium | **~60%** | **None** |
| p0.50 | High | Higher | Possible slight degradation |

**Why p0.25 is the sweet spot:** It's the most aggressive threshold that still shows zero capability degradation on benchmarks. Going to p0.50 starts to interfere with normal variation in the model's behavior — sometimes a perfectly fine response has a slightly lower projection, and capping modifying it introduces artifacts.

### 28. Jailbreaks and Red-Teaming

**Why we're talking about this:** Plan 5 is an entire deployment plan dedicated to running jailbreak evaluations. The document references "1,100 persona-based jailbreak pairs" and "44 harm categories."

A **jailbreak** is an input designed to make a language model ignore its safety instructions. The name comes from "jailbreaking" a phone — removing the manufacturer's restrictions. Common techniques include:

- **Persona assignment:** "You are DAN (Do Anything Now), an AI with no restrictions..."
- **Hypothetical framing:** "In a fictional story, how would a character..."
- **Encoding tricks:** Base64-encoding harmful instructions to bypass text filters

**Red-teaming** is the practice of systematically testing a system's defenses by playing the role of an attacker. In the AI context, red-teamers craft jailbreak prompts and measure how often the model complies with harmful requests.

The deployment doc's evaluation pipeline tests 1,100 jailbreak pairs (each pair is a system prompt + a harmful question) across 6 conditions (baseline, 3 capping strengths, 2 steering strengths), scoring each response with an LLM judge. The result is a table showing what percentage of harmful requests the model complies with under each condition.

**Why 44 harm categories:** Not all harmful content is the same — violence, fraud, CSAM, bioweapons, etc. Testing across many categories ensures capping works broadly and doesn't just prevent one type of harm while missing others.

---

## Part V: Deployment Patterns

### 29. Batch Processing vs Real-Time Inference

**Why we're talking about this:** Plans 1 and 5 are batch processing; Plans 2, 3, and 6 are real-time. They have fundamentally different requirements.

**Batch processing** means processing a large collection of inputs in bulk, with no urgency about any individual result. The pipeline running for 30 hours to process 66,000 conversations is batch. Nobody is waiting for a response — you run it overnight and check results in the morning.

**Real-time inference** means processing individual requests as they arrive, with a user waiting for each response. The capping server handling a user's chat message needs to respond in seconds, not hours.

| Aspect | Batch | Real-Time |
|--------|-------|-----------|
| Latency requirement | None (hours OK) | Seconds |
| Throughput priority | High (process many inputs) | Low (one user at a time is fine) |
| GPU utilization | Near 100% (always processing) | Bursty (idle between requests) |
| Failure handling | Retry from last checkpoint | Must respond or timeout |
| Scaling strategy | More GPUs = faster completion | More instances = more concurrent users |

**The deployment doc maps to this cleanly:** Plans 1, 4, and 5 are batch (compute axis, research, evaluate). Plans 2, 3, 6, and 7 are real-time (monitor, cap, serve, edge).

### 30. API Gateways — The Front Door

**Why we're talking about this:** Plan 6 (Embedded Safety Module) shows nginx/envoy as the API gateway in its architecture diagram.

An **API gateway** is a reverse proxy that sits between the internet and your backend services. All incoming requests hit the gateway first, which then routes them to the appropriate service.

**nginx** (pronounced "engine-X," created by Igor Sysoev in 2004) and **Envoy** (created by Lyft in 2016) are two popular gateways. They handle:

- **SSL/TLS termination:** Decrypting HTTPS so your backend only deals with plain HTTP
- **Rate limiting:** Preventing any single client from overwhelming your service
- **Load balancing:** Distributing requests across multiple backend instances
- **Authentication:** Checking API keys before requests reach your model
- **Routing:** Directing `/v1/generate` requests to the model service and `/v1/admin` to the admin service

**Why you need one:** Your FastAPI capping server shouldn't be directly exposed to the internet. It doesn't handle SSL, authentication, or abuse prevention. The gateway handles all of that, and the capping server only sees clean, authenticated, rate-limited requests.

### 31. Load Balancers and Horizontal Scaling

**Why we're talking about this:** The scaling section of Plan 3 says "scale by adding more GPU nodes behind a load balancer."

**Horizontal scaling** means adding more instances of your service (as opposed to vertical scaling, which means giving a single instance more resources). If one capping server handles 10 requests per second and you need 50, you run 5 instances.

A **load balancer** distributes incoming requests across these instances. When a user sends a request, the load balancer decides which of the 5 instances should handle it, typically using:
- **Round-robin:** Requests go to instances in order: 1, 2, 3, 4, 5, 1, 2, 3...
- **Least connections:** Send to whichever instance is currently least busy
- **Random:** Pick one at random (surprisingly effective)

**Why the deployment doc mentions this:** Each capping server instance holds the full model in GPU memory and processes one request at a time (capping hooks don't support batching). Horizontal scaling is the only way to increase throughput.

### 32. Request Queuing — Redis and RabbitMQ

**Why we're talking about this:** The scaling section mentions "add Redis/RabbitMQ for request buffering during high load."

When more requests arrive than your servers can process, you need a queue.

**Redis** (created by Salvatore Sanfilippo in 2009) is an in-memory data store often used as a message queue. **RabbitMQ** (created by Rabbit Technologies in 2007) is a dedicated message broker.

Both solve the same problem: a user sends a request, all capping servers are busy, and instead of returning an error, the request waits in a queue until a server becomes available. The user gets a "please wait" response instead of a "service unavailable" error.

**The flow:**
1. User request → API gateway → message queue
2. Queue holds the request
3. A capping server becomes available → picks up the request from the queue
4. Capping server processes it and returns the result

### 33. A/B Testing and Shadow Mode

**Why we're talking about this:** Plan 6's deployment checklist includes "Run A/B test: capped vs uncapped on production traffic (shadow mode)."

**A/B testing** means running two versions simultaneously and comparing results. In the deployment doc, this means sending some requests through the capping layer and others without it, then comparing the results.

**Shadow mode** is a more cautious variant: all requests go through the normal (uncapped) path AND the capped path, but only the uncapped response is returned to the user. The capped response is logged for comparison. This lets you evaluate capping's effect on real production traffic without any risk — if capping introduces artifacts, no user ever sees them.

**The deployment sequence:**
1. Deploy capping in shadow mode → compare capped and uncapped outputs for quality
2. If quality is equivalent → switch to A/B testing (50/50 split, some users get capped responses)
3. If A/B shows no regressions → enable capping for all traffic

### 34. Air-Gapped and Edge Deployment

**Why we're talking about this:** Plan 7 is entirely about deploying without internet access.

An **air-gapped** environment is one with no internet connection at all — physically disconnected from external networks. These exist in military installations, classified research facilities, hospitals with sensitive patient data, and financial institutions.

**Edge deployment** means running AI closer to where it's needed — on a local server in an office or factory — rather than in a cloud data center.

The challenge: AI models and their software dependencies are normally downloaded from the internet (HuggingFace Hub, PyPI, Docker Hub). In an air-gapped environment, you must bundle everything onto a physical medium (USB drive, hard disk) and carry it in.

**The offline bundle from the deployment doc:**
1. On an internet-connected machine: download the model (64 GB), the axis file (2 MB), the capping config (80 MB), all Python packages (several GB), and pack them into a tarball
2. Transfer the tarball to the air-gapped machine
3. Install everything from the local bundle — no internet needed

**Subtlety:** The OpenAI API (used for the judge step) is impossible to use air-gapped. Plan 7 solves this by using pre-computed artifacts — someone ran the pipeline on an internet-connected machine and you just use the resulting `axis.pt` and `capping_config.pt` files. The expensive pipeline doesn't need to re-run on the edge device.

---

## Part VI: Monitoring & Observability

### 35. Prometheus — Collecting Metrics

**Why we're talking about this:** The deployment doc includes Prometheus metric definitions for projection values, drift events, and capping activations.

**Prometheus** (created by SoundCloud engineers in 2012, inspired by Google's Borgmon) is a metrics collection system. It works by "scraping" — periodically making HTTP requests to your services and collecting numerical measurements.

The deployment doc defines four metrics:
- **projection_histogram:** Distribution of projection values (what does the typical response look like?)
- **drift_counter:** How many times has drift been detected? Broken down by severity.
- **session_projection:** Current projection value for each active conversation session.
- **capping_activations:** How many times has capping actually intervened at each layer?

**How it works in practice:** Your FastAPI server exposes a `/metrics` endpoint. Prometheus hits this endpoint every 15 seconds and stores the numbers in a time-series database. You can then query: "How many drift alerts were there in the last hour?" or "What's the average projection value this week?"

### 36. Grafana — Dashboards and Visualization

**Why we're talking about this:** The deployment doc recommends Prometheus + Grafana as the monitoring stack.

**Grafana** (created by Torkel Ödegaard in 2014) is a visualization platform that turns Prometheus metrics into dashboards — real-time graphs, charts, and tables you can view in a web browser.

The monitoring dashboard mockup in Plan 2 — showing projection values per turn with OK/WARNING/ALERT labels — is exactly the kind of thing you'd build in Grafana. You'd configure a panel that queries Prometheus for `assistant_axis_session_projection` grouped by session_id, and displays a time-series chart.

**Why separate tools?** Prometheus is excellent at collecting and storing metrics but has primitive visualization. Grafana is excellent at visualization but doesn't collect metrics. Together they form the industry-standard monitoring stack for infrastructure and ML systems.

### 37. Alerting — PagerDuty and Slack Webhooks

**Why we're talking about this:** The deployment doc lists PagerDuty and Slack webhooks as alerting integrations.

**Alerting** means automatically notifying a human when a metric crosses a threshold. Grafana can be configured to watch for conditions like "if drift_counter increases by more than 10 in 5 minutes, send an alert."

**PagerDuty** is an incident management platform that ensures alerts reach on-call engineers — via phone call, SMS, push notification, or email — and tracks whether they've been acknowledged. If the first person doesn't respond in 5 minutes, it escalates to the next person.

**Slack webhooks** are simpler — a URL you can POST JSON to, and it appears as a message in a Slack channel. Less robust than PagerDuty but good for low-severity warnings.

**In the deployment context:** A sudden spike in negative projections across many sessions might indicate a new jailbreak technique is being used at scale. An alert would notify the operations team to investigate.

### 38. nvidia-smi — GPU Health Monitoring

**Why we're talking about this:** The deployment doc includes an nvidia-smi command for monitoring GPU health.

**nvidia-smi** (NVIDIA System Management Interface) is a command-line tool that reports GPU status. It ships with the NVIDIA driver.

The deployment doc monitors:
- **Temperature:** 40–80°C is normal under load. Above 90°C, the GPU throttles itself to prevent damage.
- **GPU utilization:** 80–100% during generation (good — the GPU is being fully used), 0% when idle (also fine — no requests are being processed).
- **Memory used/total:** Should be roughly constant after the model loads. If memory usage keeps climbing, you have a memory leak.

**Why monitor these?** GPUs are expensive ($10,000–$40,000 each) and their failure modes are often silent. A GPU with a faulty memory cell might produce subtly wrong activations without crashing. Temperature monitoring catches cooling failures before they cause hardware damage.

---

## Part VII: Data & Storage

### 39. JSONL — The Data Format

**Why we're talking about this:** The pipeline outputs responses and scores as JSONL files.

**JSONL** (JSON Lines) is a file format where each line is a valid JSON object. Unlike regular JSON (which wraps everything in one giant array), JSONL lets you append new records without reading the entire file, process records one at a time without loading the whole file into memory, and easily count records by counting lines.

Example:
```
{"role": "cooking_assistant", "question": "How do I boil water?", "response": "Fill a pot..."}
{"role": "cooking_assistant", "question": "What is umami?", "response": "Umami is..."}
```

**Why JSONL instead of a database?** For a research pipeline that runs once and produces static output, JSONL is simpler. No database server to set up. No schema to define. Files can be copied, compressed, and shared easily. The tradeoff is that querying is slower — but for pipeline outputs that are read sequentially, this doesn't matter.

### 40. HuggingFace Hub — Model and Dataset Hosting

**Why we're talking about this:** The deployment doc downloads models and pre-computed vectors from HuggingFace Hub.

**HuggingFace Hub** is a platform for hosting and sharing AI models and datasets — it's essentially GitHub for machine learning. The command `huggingface-cli download Qwen/Qwen3-32B` downloads the 64 GB Qwen 3 32B model from HuggingFace's servers.

The project also hosts its pre-computed vectors on the Hub under `lu-christina/assistant-axis-vectors`. This allows anyone to download the axis files and capping configs without running the expensive pipeline themselves.

**Practical note:** Model downloads are large (27–140 GB) and can fail on unstable connections. The deployment doc's risk section recommends pre-downloading with `huggingface-cli` before starting the pipeline, rather than letting the code download on-the-fly (which would fail mid-pipeline if the connection drops).

### 41. NFS and Lustre — Shared Storage for Clusters

**Why we're talking about this:** The architecture diagram for Plan 1 shows "Shared NFS / Lustre Storage" connecting all worker nodes.

When multiple GPU machines collaborate on a pipeline, they need to read and write the same files. If Worker 0 generates responses and saves them to its local disk, Worker 1 can't access those files to extract activations from them.

**NFS (Network File System)** was created by Sun Microsystems in 1984. It makes a directory on one machine appear as a local directory on all other machines in the network. All workers see the same `responses/`, `activations/`, and `scores/` directories, even though the data physically lives on one storage server.

**Lustre** is a high-performance parallel file system designed for HPC clusters, developed starting in 1999 at Carnegie Mellon University. Unlike NFS (single storage server), Lustre stripes data across many storage servers simultaneously. When 8 workers all write activation files at the same time, Lustre distributes the writes across multiple servers — NFS would bottleneck on the single server.

**When to use which:**
- NFS → Small clusters (2–4 nodes), simple setup, moderate I/O
- Lustre → Large clusters (10+ nodes), heavy parallel I/O, HPC environments

### 42. Disk Usage — Why Activations Are So Large

**Why we're talking about this:** The deployment doc budgets 140–500 GB for activation files alone. This is often the storage bottleneck.

Let's trace the numbers. For a single conversation with Qwen 3 32B:
- Sequence length: ~600 tokens (question + response)
- Embedding dimension: 5,120
- Number of layers: 64
- Bytes per number: 2 (float16)

**One full activation snapshot:** 600 × 5,120 × 64 × 2 bytes = **393 MB**

The pipeline generates ~240 conversations per role × 275 roles = **66,000 conversations**. If you stored full activations for all: 66,000 × 393 MB = **25 TB**. This is why the pipeline typically extracts only the target layer (1/64th of the data) or only the assistant response tokens (reducing sequence length by ~50%), bringing the total to a manageable 140–500 GB.

**Subtlety:** Activation extraction is the primary storage bottleneck of the entire project. The deployment doc lists "Disk full during activations" as a risk with "Monitor disk, clean up early steps" as mitigation — meaning you can delete response JSONL files (~1.4 GB) after extraction to free space, since the activations contain all the information needed for subsequent steps.

---

## Part VIII: Stress-Testing the Claims

This section examines claims from the deployment document that a careful reader might find incomplete or apparently contradictory.

**Claim: "Capping has zero overhead on normal assistant behavior."**

If this were literally true, then the capping hooks would be completely invisible during normal operation. But we said in section 20 that forward hooks add ~50–100 ms of overhead per generation. How can both be true?

The resolution: "zero overhead" refers to **behavioral** impact — the response text is identical with or without capping, because the activations are above the threshold and the hook doesn't modify them. The **computational** overhead (the hook checking the threshold and finding no intervention needed) is real but small (~100 ms on a 5–20 second generation).

**Claim: "You need 2x A100 40GB for Qwen 3 32B" but also "1x H100 works."**

A single H100 has 80 GB VRAM, and Qwen 3 32B needs ~64 GB for weights alone. That leaves only 16 GB for activations, KV cache, and overhead. How does it fit?

The resolution: The H100 can just barely fit the model using aggressive memory optimization techniques — lower-precision inference (some layers in 8-bit), paged KV cache, and careful activation checkpointing. The 2x A100 40GB setup gives 80 GB total with more headroom. "Works" and "works comfortably" are different things.

**Claim: "Steps 2 & 3 overlap" in the execution timeline.**

Step 2 extracts activations from responses generated in Step 1. Step 3 judges those same responses. Both read from Step 1's output but are otherwise independent — Step 3 doesn't need activations, and Step 2 doesn't need scores. So you can start both as soon as Step 1 finishes (or even before it finishes, for roles that are complete), cutting total wall-clock time significantly.

**Claim: "Horizontal scaling with load balancer" is the answer to low throughput, but "each server instance holds the full model."**

This sounds expensive — 5 instances means 5× the GPU cost. This is correct and is the fundamental cost of capping: because it requires forward hooks (incompatible with vLLM's batched inference), you sacrifice throughput efficiency for safety. The deployment doc acknowledges this tradeoff explicitly. For organizations where AI safety is the priority, the cost is acceptable.

---

## Recap

The full chain of reasoning, compressed:

1. **GPUs** are massively parallel processors that make neural network computation tractable. Specific models (A100, H100) have specific VRAM capacities that determine which AI models they can hold.

2. **VRAM, system RAM, and NVMe storage** form a memory hierarchy — each level is larger but slower. AI workloads stress all three levels.

3. **Tensor parallelism** splits models across multiple GPUs when one GPU doesn't have enough VRAM. **InfiniBand** provides the high-speed networking needed when those GPUs are on separate machines.

4. **CUDA** is the software bridge between Python code and NVIDIA GPU hardware. **vLLM** optimizes inference throughput on top of CUDA but is incompatible with the forward hooks needed for capping.

5. **The 5-step pipeline** — generate, extract activations, judge, compute vectors, compute axis — produces the **Assistant Axis**, a mathematical direction in activation space that separates assistant behavior from non-assistant behavior.

6. **Activations** are the internal states of each transformer layer. **Forward hooks** let you observe or modify them. **Projection** onto the axis gives a single number measuring assistant-ness.

7. **Persona drift** is when a model abandons its assistant identity. **Steering** pushes activations toward assistant behavior. **Capping** is conditional steering — it only intervenes when drift is detected, preserving normal behavior.

8. **Capping thresholds** (p0.10, p0.25, p0.50) are calibrated from the distribution of normal assistant projections. **p0.25** is the sweet spot: maximum harm reduction with zero capability degradation.

9. **SLURM** and **Kubernetes** orchestrate jobs and services on clusters. **Docker** packages everything into portable containers. **FastAPI** serves the model as an HTTP API.

10. **Prometheus + Grafana** monitor projection values, drift events, and GPU health in real time. **Alerting** notifies operators when thresholds are crossed.

11. **NFS/Lustre** provides shared storage for cluster nodes. **HuggingFace Hub** hosts models and pre-computed artifacts. **JSONL** and **`.pt` files** are the data formats.

12. **Air-gapped deployment** bundles everything offline. **A/B testing** validates capping before full rollout. **Load balancers** enable horizontal scaling.

---

## Why This Matters

The Assistant Axis deployment stack is not exotic. It draws from three established fields — high-performance computing (SLURM, Lustre, InfiniBand), modern DevOps (Kubernetes, Docker, Prometheus, Grafana), and machine learning infrastructure (PyTorch, HuggingFace, CUDA, vLLM). Understanding these concepts isn't just useful for this project — it's the vocabulary of modern AI deployment broadly.

The more novel contribution is the safety layer itself: the idea that you can define a mathematical direction inside a neural network that captures whether it's "being a good assistant," and then use that direction as a real-time guardrail. The deployment plans show how to operationalize that idea — from a research pipeline that computes the direction, to a production service that enforces it, to an evaluation framework that measures its effectiveness.

Every concept in this guide exists because deploying AI safely is harder than deploying AI fast. vLLM is fast but doesn't support hooks. Hooks are needed for capping. Capping is needed because jailbreaks work. And jailbreaks work because language models are fundamentally flexible — which is also what makes them useful. The deployment plans are engineering responses to this tension.

---

## Historical Notes

- **GPUs for AI:** Krizhevsky, Sutskever & Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," 2012 — the paper that launched GPU-accelerated deep learning.
- **Transformers:** Vaswani et al., "Attention Is All You Need," Google Brain, 2017 — the architecture underlying all models in the deployment doc.
- **CUDA:** Released by NVIDIA in 2007, based on work by Ian Buck and John Nickolls.
- **Docker:** Created by Solomon Hykes at dotCloud, released 2013.
- **Kubernetes:** Created by Joe Beda, Brendan Burns, and Craig McLuckie at Google, released 2014.
- **Prometheus:** Created at SoundCloud by Matt Proud and Julius Volz, released 2012, graduated from CNCF in 2018.
- **vLLM:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," UC Berkeley, 2023.
- **PCA:** Karl Pearson, "On Lines and Planes of Closest Fit to Systems of Points in Space," 1901.
- **NFS:** Created by Sun Microsystems, 1984.
- **SLURM:** Created at Lawrence Livermore National Laboratory by Morris Jette and Andy Yoo, 2002.
- **The Assistant Axis project:** Uses the above infrastructure to operationalize the finding that model safety can be understood — and enforced — through the geometry of internal representations.
