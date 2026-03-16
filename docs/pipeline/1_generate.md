# `pipeline/1_generate.py`

## Overview

This script is the first stage of the pipeline. It generates model responses for a set of "roles" using **vLLM batch inference**. Each role has a system prompt that steers the model's persona. The script reads role definition files (JSON) and a set of questions (JSONL), then produces model responses for every role-question pair.

Key features:
- Supports **automatic multi-worker parallelization** when the number of available GPUs exceeds the tensor parallel size (i.e., it can run multiple model instances simultaneously, each spanning a subset of GPUs).
- **Idempotent / restartable**: it skips roles whose output files already exist, so it can be safely re-run.

---

## Line-by-Line Explanation

### Shebang and Module Docstring (Lines 1-24)

```python
#!/usr/bin/env python3
"""
Generate model responses for all roles using vLLM batch inference.

This script loads role files and generates model responses for each role
using the role-specific system prompts. It can be restarted and won't overwrite existing roles.

Supports automatic multi-worker parallelization when total GPUs > tensor_parallel_size.
Number of workers = total_gpus // tensor_parallel_size

Usage:
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --roles_dir data/prompts/roles \
        --questions_file data/prompts/questions.jsonl \
        --output_dir outputs/gemma-2-27b/responses \
        --question_count 240

    # With explicit tensor parallelism (will auto-parallelize across workers)
    uv run scripts/1_generate.py \
        --model google/gemma-2-27b-it \
        --tensor_parallel_size 2 \
        ...
"""
```

The shebang line (`#!/usr/bin/env python3`) allows the script to be executed directly on Unix-like systems. The docstring explains the purpose, the multi-worker parallelization strategy, and provides example command-line usage.

---

### Imports (Lines 26-31)

```python
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
```

- `argparse`: Parses command-line arguments.
- `logging`: Provides structured log output with timestamps and severity levels.
- `os`: Used to read and set environment variables (e.g., `CUDA_VISIBLE_DEVICES`).
- `sys`: Used for `sys.path` manipulation and `sys.exit()`.
- `Path`: Object-oriented filesystem paths.
- `List`, `Optional`: Type hints for function signatures.

---

### PyTorch Imports (Lines 33-34)

```python
import torch
import torch.multiprocessing as mp
```

- `torch`: Used here primarily for `torch.cuda.device_count()` to detect available GPUs.
- `torch.multiprocessing`: A drop-in replacement for Python's `multiprocessing` that handles CUDA tensor sharing between processes. Used to spawn worker processes.

---

### Path Setup and Project Import (Lines 36-38)

```python
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import RoleResponseGenerator
```

Line 36 inserts the project root directory (one level above `pipeline/`) into `sys.path` so that the `assistant_axis` package can be imported regardless of where the script is run from.

Line 38 imports `RoleResponseGenerator`, the core class that wraps vLLM for generating responses. This class handles model loading, prompt formatting, and batch inference.

---

### Logger Setup (Lines 40-41)

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

Configures the root logger to output `INFO`-level messages with a timestamp, severity, and message. A module-level logger is created for use throughout the script.

---

### `process_roles_on_worker` Function (Lines 44-116)

```python
def process_roles_on_worker(worker_id: int, gpu_ids: List[int], role_names: List[str], args):
    """Process a subset of roles on a worker with tensor parallelism support."""
```

This function is the **entry point for each spawned worker process** in multi-worker mode. Each worker receives its own ID, a list of GPU IDs to use, a list of role names to process, and the parsed command-line arguments.

#### Setting CUDA_VISIBLE_DEVICES (Lines 46-48)

```python
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
```

Restricts this worker process to only see its assigned GPUs. For example, if a worker is assigned GPUs `[2, 3]`, this sets `CUDA_VISIBLE_DEVICES=2,3`. This ensures the vLLM model loaded in this process only uses those GPUs.

#### Worker Logger Setup (Lines 50-56)

```python
    worker_logger = logging.getLogger(f"Worker-{worker_id}")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'%(asctime)s - Worker-{worker_id}[GPUs:{gpu_ids_str}] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)
```

Creates a dedicated logger for this worker process. The log format includes the worker ID and assigned GPU IDs, making it easy to distinguish output from different workers when they run concurrently.

#### Logging Start (Line 58)

```python
    worker_logger.info(f"Starting processing on Worker {worker_id} with GPUs {gpu_ids} and {len(role_names)} roles")
```

Logs a startup message indicating how many roles this worker will handle.

#### Creating the Generator (Lines 60-74)

```python
    try:
        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
```

Instantiates the `RoleResponseGenerator` with all the relevant configuration parameters. This object wraps vLLM and knows how to load questions, format prompts with role-specific system messages, and perform batch inference.

#### Loading the Model (Lines 76-77)

```python
        generator.generator.load()
```

Explicitly triggers model loading. The inner `.generator` attribute is the vLLM engine; calling `.load()` initializes it and loads model weights onto the assigned GPUs.

#### Loading and Filtering Role Files (Lines 79-90)

```python
        role_files = {}
        roles_dir = Path(args.roles_dir)
        for file_path in sorted(roles_dir.glob("*.json")):
            role_name = file_path.stem
            if role_name in role_names:
                try:
                    role_data = generator.load_role(file_path)
                    if 'instruction' in role_data:
                        role_files[role_name] = role_data
                except Exception as e:
                    worker_logger.error(f"Error loading {file_path}: {e}")
```

Iterates over all `.json` files in the roles directory (sorted alphabetically). For each file whose stem (filename without extension) matches one of the role names assigned to this worker, it loads the JSON data. Only roles that contain an `'instruction'` key (the system prompt) are kept. Errors during loading are logged but do not crash the worker.

#### Processing Roles with Progress Bar (Lines 92-108)

```python
        completed_count = 0
        failed_count = 0

        from tqdm import tqdm
        for role_name, role_data in tqdm(role_files.items(), desc=f"Worker-{worker_id}", position=worker_id):
            try:
                responses = generator.generate_role_responses(role_name, role_data)
                if responses:
                    generator.save_responses(role_name, responses)
                    completed_count += 1
                else:
                    failed_count += 1
                    worker_logger.warning(f"No responses generated for role '{role_name}'")
            except Exception as e:
                failed_count += 1
                worker_logger.error(f"Exception processing role {role_name}: {e}")
```

Iterates over each loaded role with a `tqdm` progress bar. The `position` parameter offsets the progress bar vertically so multiple workers' bars do not overwrite each other.

For each role:
1. `generate_role_responses` runs vLLM batch inference across all questions for this role.
2. If responses are returned, they are saved to disk via `save_responses`.
3. Failures are counted and logged.

#### Completion Logging (Lines 110-116)

```python
        worker_logger.info(f"Worker {worker_id} completed: {completed_count} successful, {failed_count} failed")

    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")

    finally:
        worker_logger.info(f"Worker {worker_id} cleanup completed")
```

After all roles are processed, a summary is logged. The outer `try/except` catches any fatal errors (e.g., GPU out-of-memory during model loading). The `finally` block always runs, logging cleanup completion.

---

### `run_multi_worker` Function (Lines 119-216)

```python
def run_multi_worker(args) -> int:
    """Run multi-worker processing with tensor parallelism support."""
```

Orchestrates the multi-worker strategy. It partitions GPUs and roles across workers, spawns processes, and waits for them to complete.

#### GPU Detection (Lines 121-127)

```python
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    total_gpus = len(gpu_ids)
```

Determines which GPUs are available. If the user has pre-set `CUDA_VISIBLE_DEVICES`, those are parsed. Otherwise, PyTorch's `device_count()` is used to discover all GPUs on the system.

#### Validation (Lines 129-137)

```python
    if total_gpus == 0:
        logger.error("No GPUs available.")
        return 1

    tensor_parallel_size = args.tensor_parallel_size

    if tensor_parallel_size > total_gpus:
        logger.error(f"tensor_parallel_size ({tensor_parallel_size}) cannot be greater than available GPUs ({total_gpus})")
        return 1
```

Guards against running with no GPUs or requesting more tensor parallel GPUs than are available. Returns exit code `1` on failure.

#### Worker Count Calculation (Lines 139-147)

```python
    num_workers = total_gpus // tensor_parallel_size

    if total_gpus % tensor_parallel_size != 0:
        logger.warning(f"Total GPUs ({total_gpus}) not evenly divisible by tensor_parallel_size ({tensor_parallel_size}). "
                      f"Using {num_workers} workers, leaving {total_gpus % tensor_parallel_size} GPU(s) unused.")

    logger.info(f"Available GPUs: {gpu_ids}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Number of workers: {num_workers}")
```

Calculates how many independent model instances (workers) can fit across the available GPUs. For example, 8 GPUs with `tensor_parallel_size=2` yields 4 workers, each using 2 GPUs. If the division is uneven, leftover GPUs are unused and a warning is logged.

#### Collecting Roles to Process (Lines 149-165)

```python
    roles_dir = Path(args.roles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    role_names = []
    for file_path in sorted(roles_dir.glob("*.json")):
        role_name = file_path.stem
        if args.roles and role_name not in args.roles:
            continue
        output_file = output_dir / f"{role_name}.jsonl"
        if output_file.exists():
            logger.info(f"Skipping role '{role_name}' (already exists)")
            continue
        role_names.append(role_name)
```

Scans the roles directory for JSON files. Roles are filtered in two ways:
1. If `--roles` was specified on the command line, only those roles are included.
2. If the output JSONL file already exists, the role is skipped (idempotency).

The output directory is created if it does not exist.

#### Early Exit (Lines 167-169)

```python
    if not role_names:
        logger.info("No roles to process")
        return 0
```

If all roles have already been processed (or none matched the filter), the function returns immediately with success.

#### GPU Partitioning (Lines 173-179)

```python
    gpu_chunks = []
    for i in range(num_workers):
        start_gpu_idx = i * tensor_parallel_size
        end_gpu_idx = start_gpu_idx + tensor_parallel_size
        worker_gpus = gpu_ids[start_gpu_idx:end_gpu_idx]
        gpu_chunks.append(worker_gpus)
```

Divides the flat list of GPU IDs into contiguous chunks, one per worker. Worker 0 gets the first `tensor_parallel_size` GPUs, worker 1 gets the next chunk, and so on.

#### Role Distribution (Lines 181-194)

```python
    roles_per_worker = len(role_names) // num_workers
    remainder = len(role_names) % num_workers

    role_chunks = []
    start_idx = 0

    for i in range(num_workers):
        chunk_size = roles_per_worker + (1 if i < remainder else 0)
        end_idx = start_idx + chunk_size
        chunk = role_names[start_idx:end_idx]
        role_chunks.append(chunk)
        logger.info(f"Worker {i} (GPUs {gpu_chunks[i]}): {len(chunk)} roles")
        start_idx = end_idx
```

Distributes roles across workers as evenly as possible. If there are 10 roles and 3 workers, the first worker gets 4 roles and the other two get 3 each (the remainder is spread across the first `remainder` workers).

#### Spawning Worker Processes (Lines 196-213)

```python
    mp.set_start_method('spawn', force=True)

    processes = []
    for worker_id in range(num_workers):
        if role_chunks[worker_id]:
            p = mp.Process(
                target=process_roles_on_worker,
                args=(worker_id, gpu_chunks[worker_id], role_chunks[worker_id], args)
            )
            p.start()
            processes.append(p)

    logger.info(f"Launched {len(processes)} worker processes")
    for p in processes:
        p.join()
```

Sets the multiprocessing start method to `'spawn'` (required for CUDA -- `fork` does not work reliably with CUDA contexts). Each worker with at least one role is started as a separate process. The main process then blocks on `p.join()` for each, waiting for all workers to finish.

#### Return (Lines 215-216)

```python
    logger.info("Multi-worker processing completed!")
    return 0
```

Logs completion and returns exit code `0` (success).

---

### `main` Function (Lines 219-292)

```python
def main():
    parser = argparse.ArgumentParser(
        description='Generate role responses using vLLM batch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
```

The main entry point. Sets up the argument parser with `RawDescriptionHelpFormatter` so that the help text preserves formatting.

#### Argument Definitions (Lines 225-236)

```python
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name')
    parser.add_argument('--roles_dir', type=str, default="../data/roles/instructions", help='Directory containing role JSON files')
    parser.add_argument('--questions_file', type=str, default="../data/extraction_questions.jsonl", help='Path to questions JSONL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSONL files')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum model context length')
    parser.add_argument('--tensor_parallel_size', type=int, default=None, help='Number of GPUs (auto-detect if None)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU memory utilization')
    parser.add_argument('--question_count', type=int, default=240, help='Number of questions per role')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--roles', nargs='+', help='Specific roles to process')
```

Defines all command-line arguments:
- `--model` (required): The HuggingFace model identifier (e.g., `google/gemma-2-27b-it`).
- `--roles_dir`: Where role JSON files live.
- `--questions_file`: Path to the JSONL file containing questions.
- `--output_dir` (required): Where to write generated response JSONL files.
- `--max_model_len`: Maximum context window length for vLLM.
- `--tensor_parallel_size`: How many GPUs each model instance should span; `None` means auto-detect.
- `--gpu_memory_utilization`: Fraction of GPU memory vLLM is allowed to use (default 95%).
- `--question_count`: How many questions to sample per role.
- `--temperature`: Sampling temperature (higher = more random).
- `--max_tokens`: Maximum number of tokens to generate per response.
- `--top_p`: Nucleus sampling threshold.
- `--roles`: Optional whitelist of specific role names to process.

#### GPU Detection and Mode Decision (Lines 238-264)

```python
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
        total_gpus = len(available_gpus)
    else:
        total_gpus = torch.cuda.device_count()

    tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus

    use_multi_worker = (
        total_gpus > 1 and
        tensor_parallel_size > 0 and
        total_gpus > tensor_parallel_size
    )
```

After parsing arguments, the script detects how many GPUs are available. The `tensor_parallel_size` defaults to using all GPUs for a single model instance if not explicitly set. Multi-worker mode is activated only when there are more GPUs than needed for a single model instance, meaning multiple instances can run in parallel.

#### Multi-Worker Branch (Lines 257-264)

```python
    if use_multi_worker:
        logger.info(f"Multi-worker mode: {total_gpus} GPUs with tensor_parallel_size={tensor_parallel_size}")
        logger.info(f"Number of workers: {total_gpus // tensor_parallel_size}")
        args.tensor_parallel_size = tensor_parallel_size
        exit_code = run_multi_worker(args)
        if exit_code != 0:
            sys.exit(exit_code)
```

In multi-worker mode, it delegates to `run_multi_worker()`, which handles GPU partitioning, role distribution, and process spawning. If that function returns a non-zero exit code, the script exits with the same code.

#### Single-Worker Branch (Lines 265-286)

```python
    else:
        logger.info(f"Single-worker mode: Using {tensor_parallel_size} GPU(s)")

        generator = RoleResponseGenerator(
            model_name=args.model,
            roles_dir=args.roles_dir,
            output_dir=args.output_dir,
            questions_file=args.questions_file,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            question_count=args.question_count,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )

        generator.process_all_roles(
            skip_existing=True,
            roles=args.roles
        )
```

In single-worker mode (either 1 GPU total, or tensor parallel uses all GPUs), it creates a single `RoleResponseGenerator` and calls `process_all_roles()`. The `skip_existing=True` flag ensures idempotency.

#### Final Log and Entry Point (Lines 288-292)

```python
    logger.info("Done!")


if __name__ == "__main__":
    main()
```

Logs a completion message. The `if __name__ == "__main__"` guard ensures `main()` runs only when the script is executed directly, not when imported as a module.
