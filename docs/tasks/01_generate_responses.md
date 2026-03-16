# Task 1: Generate Model Responses

## Overview

This task generates model responses for a set of **roles** by running vLLM batch inference. Each role is defined by a JSON file containing multiple system-prompt variants ("instructions"). For every role, the pipeline pairs each instruction variant with a pool of questions, sends the resulting conversations through a vLLM-hosted language model, and writes the completed conversations to per-role JSONL output files.

The pipeline supports automatic multi-worker parallelization when the number of available GPUs exceeds the tensor-parallel size. It is idempotent: roles whose output files already exist are skipped on re-run.

**Entry point:** `pipeline/1_generate.py`
**Core library:** `assistant_axis/generation.py`

---

## Sub-Tasks

### Sub-Task 1.1: Parse Command-Line Arguments and Detect GPU Topology

#### Input

Command-line arguments passed to `pipeline/1_generate.py`:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | *required* | HuggingFace model identifier (e.g. `google/gemma-2-27b-it`) |
| `--roles_dir` | `str` | `../data/roles/instructions` | Directory containing role JSON files |
| `--questions_file` | `str` | `../data/extraction_questions.jsonl` | Path to questions JSONL file |
| `--output_dir` | `str` | *required* | Output directory for JSONL response files |
| `--max_model_len` | `int` | `2048` | Maximum model context length |
| `--tensor_parallel_size` | `int` | `None` | Number of GPUs for tensor parallelism (auto-detect if `None`) |
| `--gpu_memory_utilization` | `float` | `0.95` | Fraction of GPU memory vLLM may use |
| `--question_count` | `int` | `240` | Number of questions to use per role |
| `--temperature` | `float` | `0.7` | Sampling temperature |
| `--max_tokens` | `int` | `512` | Maximum tokens to generate per response |
| `--top_p` | `float` | `0.9` | Top-p (nucleus) sampling threshold |
| `--roles` | `List[str]` | `None` | Optional whitelist of specific role names to process |

#### Processing

1. **Argument parsing** -- standard `argparse` setup:

```python
parser = argparse.ArgumentParser(
    description='Generate role responses using vLLM batch inference',
    formatter_class=argparse.RawDescriptionHelpFormatter
)

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

2. **GPU detection** -- determines available GPUs either from `CUDA_VISIBLE_DEVICES` or `torch.cuda.device_count()`:

```python
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    total_gpus = len(available_gpus)
else:
    total_gpus = torch.cuda.device_count()
```

3. **Tensor-parallel size resolution** -- if the user did not set `--tensor_parallel_size`, it defaults to `total_gpus`:

```python
tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus
```

4. **Single-worker vs. multi-worker decision** -- multi-worker mode activates when there are more GPUs than the tensor-parallel size (i.e. multiple independent model replicas can be hosted):

```python
use_multi_worker = (
    total_gpus > 1 and
    tensor_parallel_size > 0 and
    total_gpus > tensor_parallel_size
)
```

#### Output

- A populated `argparse.Namespace` object (`args`).
- A boolean decision (`use_multi_worker`) that gates the execution path.
- `tensor_parallel_size` is written back onto `args` for downstream use.

---

### Sub-Task 1.2: Resolve Model Short Name

#### Input

- `model_name: str` -- the HuggingFace model identifier (e.g. `google/gemma-2-27b-it`).

#### Processing

When `RoleResponseGenerator.__init__` is called, it resolves a human-readable short name for the model. This short name is later substituted into instruction templates via the `{model_name}` placeholder.

```python
if short_name is None:
    from .models import get_config
    config = get_config(model_name)
    self.short_name = config["short_name"]
else:
    self.short_name = short_name
```

The `get_config` function (from `assistant_axis/models.py`) first checks a hardcoded lookup table:

```python
MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
        "capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
    },
}
```

If the model is not in the table, it infers the config from the HuggingFace model architecture:

```python
def get_config(model_name: str) -> dict:
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()

    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        total_layers = config.num_hidden_layers
        target_layer = total_layers // 2

        model_lower = model_name.lower()
        if "gemma" in model_lower:
            short_name = "Gemma"
        elif "qwen" in model_lower:
            short_name = "Qwen"
        elif "llama" in model_lower:
            short_name = "Llama"
        elif "mistral" in model_lower:
            short_name = "Mistral"
        else:
            short_name = model_name.split("/")[-1].split("-")[0]

        return {
            "target_layer": target_layer,
            "total_layers": total_layers,
            "short_name": short_name,
        }
    except Exception as e:
        raise ValueError(f"Could not infer config for model {model_name}: {e}")
```

#### Output

- `self.short_name: str` -- e.g. `"Gemma"`, `"Qwen"`, `"Llama"`.

---

### Sub-Task 1.3: Initialize VLLMGenerator

#### Input

Parameters forwarded from `RoleResponseGenerator.__init__` (or directly when used standalone):

| Parameter | Type | Default |
|---|---|---|
| `model_name` | `str` | *required* |
| `max_model_len` | `int` | `2048` |
| `tensor_parallel_size` | `Optional[int]` | `None` |
| `gpu_memory_utilization` | `float` | `0.9` |
| `temperature` | `float` | `0.7` |
| `max_tokens` | `int` | `512` |
| `top_p` | `float` | `0.9` |

Note: `RoleResponseGenerator` passes `0.95` for `gpu_memory_utilization` from the pipeline script's default; the `VLLMGenerator` class default is `0.9` but the pipeline overrides it.

#### Processing

`VLLMGenerator.__init__` stores all parameters and sets `self.llm = None` (lazy loading):

```python
class VLLMGenerator:
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 2048,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.llm = None
        self.sampling_params = None
```

`RoleResponseGenerator.__init__` creates the `VLLMGenerator` and also prepares the output directory:

```python
self.generator = VLLMGenerator(
    model_name=model_name,
    max_model_len=max_model_len,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
)

self.questions = None
self.output_dir.mkdir(parents=True, exist_ok=True)
```

#### Output

- A `VLLMGenerator` instance with all sampling parameters stored, model not yet loaded.
- A `RoleResponseGenerator` instance wrapping it, with `self.prompt_indices` defaulting to `list(range(5))` (i.e. `[0, 1, 2, 3, 4]`).

---

### Sub-Task 1.4: Load vLLM Model

#### Input

- `self.model_name: str`
- `self.max_model_len: int`
- `self.tensor_parallel_size: Optional[int]`
- `self.gpu_memory_utilization: float`

#### Processing

`VLLMGenerator.load()` is called lazily (on first generate call, or explicitly). It is idempotent -- returns immediately if already loaded:

```python
def load(self):
    """Load the vLLM model."""
    if self.llm is not None:
        return

    from vllm import LLM, SamplingParams

    logger.info(f"Loading vLLM model: {self.model_name}")

    self.llm = LLM(
        model=self.model_name,
        max_model_len=self.max_model_len,
        tensor_parallel_size=self.tensor_parallel_size,
        gpu_memory_utilization=self.gpu_memory_utilization,
        trust_remote_code=True,
    )

    self.sampling_params = SamplingParams(
        temperature=self.temperature,
        max_tokens=self.max_tokens,
        top_p=self.top_p,
    )

    logger.info("Model loaded successfully")
```

In multi-worker mode the load is triggered explicitly via `generator.generator.load()` after per-worker `CUDA_VISIBLE_DEVICES` is set. In single-worker mode it is triggered inside `process_all_roles()`.

#### Output

- `self.llm: vllm.LLM` -- loaded model engine.
- `self.sampling_params: vllm.SamplingParams` -- configured sampling parameters.

---

### Sub-Task 1.5: Load Questions from JSONL

#### Input

- `self.questions_file: str` -- path to a JSONL file where each line is a JSON object with a `"question"` field.
- `self.question_count: int` -- how many questions to use (default 240).

**Questions file format** (`data/extraction_questions.jsonl`):
```json
{"question": "What is the relationship between law and morality?", "id": 0}
{"question": "What principles should guide human action?", "id": 1}
{"question": "Can you explain how facial recognition software identifies people?", "id": 2}
```

#### Processing

```python
def load_questions(self) -> List[str]:
    """Load questions from JSONL file."""
    if self.questions is not None:
        return self.questions

    import jsonlines

    questions = []
    with jsonlines.open(self.questions_file, 'r') as reader:
        for entry in reader:
            questions.append(entry['question'])

    self.questions = questions[:self.question_count]
    logger.info(f"Loaded {len(self.questions)} questions")
    return self.questions
```

The method reads all questions, extracts just the `"question"` string from each line, then truncates to `question_count`. Results are cached on `self.questions`.

#### Output

- `self.questions: List[str]` -- list of up to `question_count` question strings.

---

### Sub-Task 1.6: Load and Validate Role Files

#### Input

- `self.roles_dir: Path` -- directory containing `*.json` files.
- Optional `roles: List[str]` -- whitelist filter.
- Optional `skip_existing: bool` -- whether to skip roles with existing output files.

**Role JSON file format** (e.g. `data/roles/instructions/assistant.json`):

```json
{
  "instruction": [
    {
      "pos": "You are an assistant dedicated to helping users with their various tasks and queries."
    },
    {
      "pos": "Please be an assistant who provides comprehensive support across different topics and needs."
    },
    {
      "pos": "Act as an assistant that prioritizes being helpful, informative, and supportive to users."
    },
    {
      "pos": "You are an assistant focused on understanding user needs and providing appropriate assistance."
    },
    {
      "pos": "Be an assistant who approaches every request with the goal of being genuinely useful and responsive."
    }
  ],
  "questions": ["...role-specific questions (not used by this pipeline)..."],
  "eval_prompt": "...evaluation template (not used by this pipeline)..."
}
```

Each role file has an `"instruction"` array. Each element contains a `"pos"` key with the system-prompt text for that variant.

#### Processing

**In single-worker mode** (`process_all_roles`):

```python
def process_all_roles(
    self,
    skip_existing: bool = True,
    roles: Optional[List[str]] = None,
):
    self.generator.load()
    self.load_questions()

    role_files = {}
    for file_path in sorted(self.roles_dir.glob("*.json")):
        role_name = file_path.stem
        try:
            role_data = self.load_role(file_path)
            if 'instruction' not in role_data:
                logger.warning(f"Skipping {role_name}: missing 'instruction' field")
                continue
            role_files[role_name] = role_data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Found {len(role_files)} role files")

    if roles:
        role_files = {k: v for k, v in role_files.items() if k in roles}

    if skip_existing:
        role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}

    logger.info(f"Processing {len(role_files)} roles")
```

Where `load_role` and `should_skip_role` are:

```python
def load_role(self, role_file: Path) -> dict:
    """Load a role JSON file."""
    with open(role_file, 'r') as f:
        return json.load(f)

def should_skip_role(self, role_name: str) -> bool:
    """Check if role output already exists."""
    output_file = self.output_dir / f"{role_name}.jsonl"
    return output_file.exists()
```

**In multi-worker mode** (`run_multi_worker`), role discovery and skip-checking happen before spawning workers:

```python
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

Each worker then re-loads its assigned subset of role files in `process_roles_on_worker`:

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

#### Output

- `role_files: Dict[str, dict]` -- mapping from role name (filename stem) to parsed JSON data, filtered to only roles that need processing.

---

### Sub-Task 1.7: Format Instructions (Template Substitution)

#### Input

- `instruction: str` -- raw instruction text from the role JSON, which may contain the `{model_name}` placeholder.
- `self.short_name: str` -- resolved model short name (e.g. `"Gemma"`).

#### Processing

```python
def format_instruction(self, instruction: str) -> str:
    """Format instruction, replacing {model_name} placeholder."""
    return instruction.replace("{model_name}", self.short_name)
```

Called within `generate_role_responses` for each instruction variant:

```python
def generate_role_responses(self, role_name: str, role_data: dict) -> List[dict]:
    instructions = role_data.get('instruction', [])
    if not instructions:
        return []

    questions = self.load_questions()

    formatted_instructions = []
    for inst in instructions:
        raw = inst.get('pos', '')
        formatted_instructions.append(self.format_instruction(raw))
```

#### Output

- `formatted_instructions: List[str]` -- list of instruction strings with `{model_name}` replaced by the model's short name.

---

### Sub-Task 1.8: Build Conversations (Instruction x Question Cross Product)

#### Input

- `instructions: List[str]` -- formatted instruction strings (one per prompt variant).
- `questions: List[str]` -- loaded question strings.
- `prompt_indices: Optional[List[int]]` -- which instruction indices to use (default `[0, 1, 2, 3, 4]`).

#### Processing

`VLLMGenerator.generate_for_role` builds the full cross product of (instruction variant) x (question):

```python
def generate_for_role(
    self,
    instructions: List[str],
    questions: List[str],
    prompt_indices: Optional[List[int]] = None,
) -> List[Dict]:
    self.load()
    tokenizer = self.llm.get_tokenizer()

    if prompt_indices is None:
        prompt_indices = list(range(len(instructions)))

    all_conversations = []
    all_metadata = []

    for prompt_idx in prompt_indices:
        if prompt_idx >= len(instructions):
            continue

        instruction = instructions[prompt_idx]

        for q_idx, question in enumerate(questions):
            conversation = format_conversation(instruction, question, tokenizer)
            all_conversations.append(conversation)
            all_metadata.append({
                "system_prompt": instruction,
                "prompt_index": prompt_idx,
                "question_index": q_idx,
                "question": question,
            })

    if not all_conversations:
        return []
```

Each conversation is formatted by `format_conversation`, which detects whether the tokenizer's chat template supports system messages:

```python
def format_conversation(
    instruction: Optional[str],
    question: str,
    tokenizer,
) -> List[Dict[str, str]]:
    test_message = "__SYSTEM_TEST__"
    test_conversation = [
        {"role": "system", "content": test_message},
        {"role": "user", "content": "hello"},
    ]

    try:
        output = tokenizer.apply_chat_template(
            test_conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        supports_system = test_message in output
    except Exception:
        supports_system = False

    if supports_system:
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    else:
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]
```

**Total conversations per role:** `len(prompt_indices) * len(questions)`. With defaults (5 prompt variants, 240 questions), this is **1,200 conversations per role**.

#### Output

- `all_conversations: List[List[Dict[str, str]]]` -- each element is a list of message dicts (either `[system, user]` or `[user]`).
- `all_metadata: List[Dict]` -- parallel list tracking `system_prompt`, `prompt_index`, `question_index`, `question` for each conversation.

---

### Sub-Task 1.9: Run vLLM Batch Inference

#### Input

- `all_conversations: List[List[Dict[str, str]]]` -- the conversations built in Sub-Task 1.8.
- `self.llm: vllm.LLM` -- loaded model.
- `self.sampling_params: vllm.SamplingParams` -- configured sampling parameters.

#### Processing

`generate_for_role` calls `generate_batch`, which templates every conversation into a prompt string and runs batch inference:

```python
def generate_batch(
    self,
    conversations: List[List[Dict[str, str]]],
) -> List[str]:
    self.load()

    tokenizer = self.llm.get_tokenizer()

    # Disable thinking for Qwen models (https://github.com/vllm-project/vllm/issues/18066)
    chat_template_kwargs = {}
    if "qwen" in self.model_name.lower():
        chat_template_kwargs["enable_thinking"] = False

    prompts = []
    for conv in conversations:
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
            **chat_template_kwargs
        )
        prompts.append(prompt)

    logger.info(f"Running batch inference for {len(prompts)} prompts...")
    outputs = self.llm.generate(prompts, self.sampling_params)

    responses = [output.outputs[0].text for output in outputs]
    return responses
```

Key details:
- `tokenizer.apply_chat_template(..., add_generation_prompt=True)` appends the model's generation start token.
- For Qwen models, `enable_thinking=False` is passed to disable chain-of-thought.
- `self.llm.generate(prompts, self.sampling_params)` runs all prompts in a single vLLM batch.
- Only the first output sequence (`outputs[0]`) is taken from each result (no beam search).

#### Output

- `responses: List[str]` -- one generated text string per conversation, in the same order as the input.

---

### Sub-Task 1.10: Assemble Result Records

#### Input

- `all_conversations: List[List[Dict[str, str]]]` -- input conversations.
- `all_metadata: List[Dict]` -- metadata for each conversation.
- `responses: List[str]` -- generated response texts (from Sub-Task 1.9).

#### Processing

After batch inference, `generate_for_role` zips conversations with metadata and responses, appending the assistant reply to each conversation:

```python
    responses = self.generate_batch(all_conversations)

    results = []
    for conv, meta, response in zip(all_conversations, all_metadata, responses):
        result = {
            "system_prompt": meta["system_prompt"],
            "prompt_index": meta["prompt_index"],
            "question_index": meta["question_index"],
            "question": meta["question"],
            "conversation": conv + [{"role": "assistant", "content": response}],
        }
        results.append(result)

    return results
```

Then back in `generate_role_responses`, a `"label"` field is added to every result:

```python
    results = self.generator.generate_for_role(
        instructions=formatted_instructions,
        questions=questions,
        prompt_indices=self.prompt_indices,
    )

    for r in results:
        r["label"] = "pos"

    return results
```

#### Output

Each result dict has this structure:

```python
{
    "system_prompt": str,      # The formatted instruction text used
    "prompt_index": int,       # Index of the instruction variant (0-4)
    "question_index": int,     # Index of the question (0-239)
    "question": str,           # The question text
    "conversation": [          # Full conversation including the generated response
        {"role": "system", "content": "..."},   # (present if model supports system role)
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "label": "pos"             # Always "pos" (positive label)
}
```

---

### Sub-Task 1.11: Save Responses to JSONL

#### Input

- `role_name: str` -- the role identifier (filename stem).
- `responses: List[dict]` -- the assembled result records from Sub-Task 1.10.
- `self.output_dir: Path` -- the output directory.

#### Processing

```python
def save_responses(self, role_name: str, responses: List[dict]):
    """Save responses to JSONL file."""
    import jsonlines

    output_file = self.output_dir / f"{role_name}.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        for response in responses:
            writer.write(response)
    logger.info(f"Saved {len(responses)} responses to {output_file}")
```

#### Output

- One JSONL file per role at `{output_dir}/{role_name}.jsonl`.
- Each line is a JSON object with the structure described in Sub-Task 1.10.
- With default settings (5 instruction variants, 240 questions), each file contains **1,200 lines**.

**Example output line** (pretty-printed):
```json
{
  "system_prompt": "You are an assistant dedicated to helping users with their various tasks and queries.",
  "prompt_index": 0,
  "question_index": 0,
  "question": "What is the relationship between law and morality?",
  "conversation": [
    {"role": "system", "content": "You are an assistant dedicated to helping users with their various tasks and queries."},
    {"role": "user", "content": "What is the relationship between law and morality?"},
    {"role": "assistant", "content": "Law and morality are deeply intertwined concepts that..."}
  ],
  "label": "pos"
}
```

---

### Sub-Task 1.12: Orchestrate Single-Worker Execution

#### Input

- `args` namespace with all CLI parameters.
- Condition: `use_multi_worker == False`.

#### Processing

In single-worker mode, `main()` instantiates one `RoleResponseGenerator` and calls `process_all_roles`:

```python
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

`process_all_roles` iterates over all roles sequentially with a tqdm progress bar:

```python
for role_name, role_data in tqdm(role_files.items(), desc="Processing roles"):
    try:
        responses = self.generate_role_responses(role_name, role_data)
        if responses:
            self.save_responses(role_name, responses)
    except Exception as e:
        logger.error(f"Error processing {role_name}: {e}")
```

#### Output

- All role JSONL files written to `output_dir`.
- Roles with existing output files are skipped.

---

### Sub-Task 1.13: Orchestrate Multi-Worker Execution

#### Input

- `args` namespace with all CLI parameters.
- Condition: `use_multi_worker == True` (i.e. `total_gpus > tensor_parallel_size`).

#### Processing

**Step 1: Partition GPUs into worker groups.**

```python
num_workers = total_gpus // tensor_parallel_size

gpu_chunks = []
for i in range(num_workers):
    start_gpu_idx = i * tensor_parallel_size
    end_gpu_idx = start_gpu_idx + tensor_parallel_size
    worker_gpus = gpu_ids[start_gpu_idx:end_gpu_idx]
    gpu_chunks.append(worker_gpus)
```

Example: 8 GPUs with `tensor_parallel_size=2` yields 4 workers, each with 2 GPUs.

**Step 2: Distribute roles evenly across workers.**

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

The first `remainder` workers each get one extra role, ensuring balanced distribution.

**Step 3: Spawn worker processes via `torch.multiprocessing`.**

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

for p in processes:
    p.join()
```

**Step 4: Each worker process runs `process_roles_on_worker`.**

Each worker:
1. Sets `CUDA_VISIBLE_DEVICES` to its assigned GPU subset.
2. Creates its own `RoleResponseGenerator` and loads the model.
3. Iterates over its assigned roles, generating and saving responses.

```python
def process_roles_on_worker(worker_id: int, gpu_ids: List[int], role_names: List[str], args):
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    # ... logging setup ...

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

        generator.generator.load()

        # Load and filter role files
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

        # Process assigned roles
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
            except Exception as e:
                failed_count += 1
                worker_logger.error(f"Exception processing role {role_name}: {e}")
    except Exception as e:
        worker_logger.error(f"Fatal error on Worker {worker_id}: {e}")
    finally:
        worker_logger.info(f"Worker {worker_id} cleanup completed")
```

#### Output

- Same as single-worker mode: one JSONL file per role in `output_dir`.
- Processing is parallelized across `num_workers` independent model instances, each on its own GPU subset.

---

### Sub-Task 1.14: HuggingFace Single-Response Generation (Standalone Utility)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | HuggingFace model object | *required* | Loaded HF model |
| `tokenizer` | HuggingFace tokenizer | *required* | Corresponding tokenizer |
| `conversation` | `List[Dict[str, str]]` | *required* | List of message dicts |
| `max_new_tokens` | `int` | `512` | Max tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `top_p` | `float` | `0.9` | Nucleus sampling |
| `do_sample` | `bool` | `True` | `False` for greedy decoding |

#### Processing

This standalone function is not used by the pipeline script but is available in the `generation` module for single-inference use cases (e.g. interactive probing):

```python
def generate_response(
    model,
    tokenizer,
    conversation: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    chat_template_kwargs = {}
    if hasattr(tokenizer, 'name_or_path') and "qwen" in tokenizer.name_or_path.lower():
        chat_template_kwargs["enable_thinking"] = False

    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )

    return response
```

#### Output

- `str` -- the generated response text (decoded, special tokens stripped).

---

## Data Flow Summary

```
roles_dir/*.json ----+
                     |
                     v
              Load & Validate Roles
                     |
                     v
questions.jsonl ---> Load Questions (truncate to question_count)
                     |
                     v
              Format Instructions (replace {model_name})
                     |
                     v
              Build Conversations (instruction x question cross product)
                     |
                     v
              Tokenize via chat template (detect system prompt support)
                     |
                     v
              vLLM Batch Inference
                     |
                     v
              Assemble Results (conversation + metadata + label)
                     |
                     v
              Save to output_dir/{role_name}.jsonl
```

## Key Constants and Defaults

| Setting | Default | Effect |
|---|---|---|
| `prompt_indices` | `[0, 1, 2, 3, 4]` | Use all 5 instruction variants per role |
| `question_count` | `240` | Use first 240 questions from the JSONL |
| Conversations per role | `5 * 240 = 1,200` | Cross product of variants and questions |
| `max_model_len` | `2048` | vLLM context window |
| `max_tokens` | `512` | Maximum generation length |
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `gpu_memory_utilization` | `0.95` (pipeline) / `0.9` (class default) | GPU memory fraction for vLLM |
| `skip_existing` | `True` | Idempotent re-runs |
| Multiprocessing start method | `spawn` | Required for CUDA in child processes |
