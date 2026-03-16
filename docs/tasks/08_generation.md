# Task 8: Response Generation (generation.py)

## Overview

The `generation.py` module provides the response-generation layer of the assistant-axis pipeline. It turns a set of role-based system prompts and user questions into model-generated responses, using **vLLM** for high-throughput batch inference. The module is organised into three layers:

| Layer | Purpose |
|---|---|
| **Standalone helpers** (`generate_response`, `format_conversation`) | Low-level functions for formatting conversations and generating a single response via HuggingFace. |
| **`VLLMGenerator`** | Mid-level class that wraps a vLLM `LLM` instance, handling model loading, prompt templating, and batch generation. |
| **`RoleResponseGenerator`** | High-level orchestrator that iterates over role JSON files, loads questions, delegates generation to `VLLMGenerator`, and persists results as JSONL. |

---

## Sub-Tasks

---

### Sub-Task 8.1: Format a Conversation (`format_conversation`)

Builds the message list that will be fed to a model's chat template, gracefully handling models that do **not** support a `system` role.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `instruction` | `Optional[str]` | *(required)* | System-level instruction. May be `None`. |
| `question` | `str` | *(required)* | The user's question. |
| `tokenizer` | HuggingFace tokenizer | *(required)* | Tokenizer whose `apply_chat_template` is probed for system-prompt support. |

#### Processing

**Step 1 -- Probe system-prompt support.**
A test conversation containing a sentinel string is rendered through the tokenizer. If the sentinel survives, the model's chat template supports the `system` role.

```python
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
```

**Step 2 -- Build the message list.**

*If the model supports a system role:*

```python
if supports_system:
    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": question})
    return messages
```

*If it does not:* the instruction is prepended to the question inside a single `user` message.

```python
else:
    if instruction:
        formatted = f"{instruction}\n\n{question}"
    else:
        formatted = question
    return [{"role": "user", "content": formatted}]
```

#### Output

| Type | Shape | Description |
|---|---|---|
| `List[Dict[str, str]]` | 1--2 elements | A list of message dicts (`{"role": ..., "content": ...}`). Contains a `system` message only when both `instruction` is truthy and the tokenizer supports it. |

---

### Sub-Task 8.2: Initialise the vLLM Generator (`VLLMGenerator.__init__`)

Stores configuration for the vLLM engine. The model is **not** loaded yet (lazy loading happens in `load()`).

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | HuggingFace model identifier (e.g. `"google/gemma-2-27b-it"`). |
| `max_model_len` | `int` | `2048` | Maximum context length the vLLM engine will use. |
| `tensor_parallel_size` | `Optional[int]` | `None` | Number of GPUs for tensor parallelism. `None` lets vLLM auto-detect. |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory vLLM is allowed to use. |
| `temperature` | `float` | `0.7` | Sampling temperature. |
| `max_tokens` | `int` | `512` | Maximum number of tokens to generate per response. |
| `top_p` | `float` | `0.9` | Nucleus (top-p) sampling threshold. |

#### Processing

All parameters are stored as instance attributes. Two additional attributes are initialised to `None` to mark that the model has not yet been loaded.

```python
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

#### Output

A fully configured but **unloaded** `VLLMGenerator` instance. `self.llm` and `self.sampling_params` are both `None`.

---

### Sub-Task 8.3: Load the vLLM Model (`VLLMGenerator.load`)

Lazily instantiates the vLLM `LLM` engine and its `SamplingParams`. This method is idempotent -- calling it multiple times is safe.

#### Input

None (operates on `self`).

#### Processing

**Step 1 -- Guard against re-loading.**

```python
if self.llm is not None:
    return
```

**Step 2 -- Import vLLM (deferred to avoid top-level dependency).**

```python
from vllm import LLM, SamplingParams
```

**Step 3 -- Create the LLM engine.**

```python
self.llm = LLM(
    model=self.model_name,
    max_model_len=self.max_model_len,
    tensor_parallel_size=self.tensor_parallel_size,
    gpu_memory_utilization=self.gpu_memory_utilization,
    trust_remote_code=True,
)
```

**Step 4 -- Create the sampling parameters.**

```python
self.sampling_params = SamplingParams(
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    top_p=self.top_p,
)
```

#### Output

No return value. Side effects: `self.llm` is now a live `vllm.LLM` instance and `self.sampling_params` is a configured `SamplingParams` object.

---

### Sub-Task 8.4: Batch Generation (`VLLMGenerator.generate_batch`)

Takes a batch of pre-built conversations, templates them into raw prompt strings, and runs vLLM batch inference.

#### Input

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `conversations` | `List[List[Dict[str, str]]]` | `(N, *, 2 keys)` | A list of `N` conversations. Each conversation is a list of message dicts with `"role"` and `"content"` keys. |

#### Processing

**Step 1 -- Ensure the model is loaded.**

```python
self.load()
```

**Step 2 -- Obtain the tokenizer from the loaded engine.**

```python
tokenizer = self.llm.get_tokenizer()
```

**Step 3 -- Disable thinking for Qwen models.** Qwen's chat template has an `enable_thinking` parameter that must be set to `False` to avoid extraneous output (see [vllm#18066](https://github.com/vllm-project/vllm/issues/18066)).

```python
chat_template_kwargs = {}
if "qwen" in self.model_name.lower():
    chat_template_kwargs["enable_thinking"] = False
```

**Step 4 -- Template each conversation into a prompt string.**

```python
prompts = []
for conv in conversations:
    prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True,
        **chat_template_kwargs
    )
    prompts.append(prompt)
```

**Step 5 -- Run vLLM batch inference.**

```python
outputs = self.llm.generate(prompts, self.sampling_params)
```

**Step 6 -- Extract the generated text from each output.**

```python
responses = [output.outputs[0].text for output in outputs]
return responses
```

#### Output

| Type | Shape | Description |
|---|---|---|
| `List[str]` | `(N,)` | One generated response string per input conversation, in the same order. |

---

### Sub-Task 8.5: Generate Responses for a Role (`VLLMGenerator.generate_for_role`)

Generates the full cross-product of (instruction variants x questions) for a single role, returning structured result dicts.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `instructions` | `List[str]` | *(required)* | System prompt variants for the role. |
| `questions` | `List[str]` | *(required)* | User questions to pair with each instruction. |
| `prompt_indices` | `Optional[List[int]]` | `None` | Which indices into `instructions` to use. Defaults to all (`range(len(instructions))`). |

#### Processing

**Step 1 -- Load model and tokenizer.**

```python
self.load()
tokenizer = self.llm.get_tokenizer()
```

**Step 2 -- Default prompt_indices to all instruction indices.**

```python
if prompt_indices is None:
    prompt_indices = list(range(len(instructions)))
```

**Step 3 -- Build the cross-product of conversations and metadata.** For each selected instruction variant, pair it with every question via `format_conversation`.

```python
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
```

**Step 4 -- Early return if no conversations were built.**

```python
if not all_conversations:
    return []
```

**Step 5 -- Run batch generation.**

```python
responses = self.generate_batch(all_conversations)
```

**Step 6 -- Assemble result dicts.** Each result contains the full conversation (including the assistant's response appended) and its metadata.

```python
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

#### Output

| Type | Shape | Description |
|---|---|---|
| `List[Dict]` | `(len(prompt_indices) * len(questions),)` | Each dict contains: `system_prompt` (str), `prompt_index` (int), `question_index` (int), `question` (str), `conversation` (List[Dict] -- the full message list including the generated assistant turn). |

---

### Sub-Task 8.6: Initialise the Role Response Generator (`RoleResponseGenerator.__init__`)

Configures the high-level orchestrator that maps role JSON files + questions to generated responses.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | HuggingFace model identifier. |
| `roles_dir` | `str` | *(required)* | Directory containing role JSON files. |
| `output_dir` | `str` | *(required)* | Directory where JSONL output files are written. |
| `questions_file` | `str` | *(required)* | Path to a JSONL file of questions. |
| `max_model_len` | `int` | `2048` | Maximum context length. |
| `tensor_parallel_size` | `Optional[int]` | `None` | Number of GPUs. |
| `gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction. |
| `question_count` | `int` | `240` | Maximum number of questions to use per role. |
| `temperature` | `float` | `0.7` | Sampling temperature. |
| `max_tokens` | `int` | `512` | Max tokens per generated response. |
| `top_p` | `float` | `0.9` | Top-p sampling. |
| `prompt_indices` | `Optional[List[int]]` | `None` | Instruction indices to use (defaults to `[0, 1, 2, 3, 4]`). |
| `short_name` | `Optional[str]` | `None` | Short model name for the `{model_name}` placeholder in instructions. Auto-detected from `.models.get_config` when `None`. |

#### Processing

**Step 1 -- Store basic parameters.**

```python
self.model_name = model_name
self.roles_dir = Path(roles_dir)
self.output_dir = Path(output_dir)
self.questions_file = questions_file
self.question_count = question_count
self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))
```

**Step 2 -- Resolve the short model name.** If not provided explicitly, it is looked up via the project's model config registry.

```python
if short_name is None:
    from .models import get_config
    config = get_config(model_name)
    self.short_name = config["short_name"]
else:
    self.short_name = short_name
```

**Step 3 -- Create the underlying `VLLMGenerator`.**

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
```

**Step 4 -- Prepare the output directory and initialise questions cache.**

```python
self.questions = None
self.output_dir.mkdir(parents=True, exist_ok=True)
```

#### Output

A fully configured `RoleResponseGenerator` instance. The vLLM model is **not** loaded yet (lazy loading occurs when generation begins). The questions cache (`self.questions`) is `None`.

---

### Sub-Task 8.7: Load Questions (`RoleResponseGenerator.load_questions`)

Reads user questions from a JSONL file and caches them, truncated to `question_count`.

#### Input

None (operates on `self`). Uses `self.questions_file` and `self.question_count`.

#### Processing

**Step 1 -- Return early if already cached.**

```python
if self.questions is not None:
    return self.questions
```

**Step 2 -- Read the JSONL file.** Each line is expected to have a `"question"` field.

```python
import jsonlines

questions = []
with jsonlines.open(self.questions_file, 'r') as reader:
    for entry in reader:
        questions.append(entry['question'])
```

**Step 3 -- Truncate to the configured count and cache.**

```python
self.questions = questions[:self.question_count]
logger.info(f"Loaded {len(self.questions)} questions")
return self.questions
```

#### Output

| Type | Shape | Description |
|---|---|---|
| `List[str]` | `(min(total_questions, question_count),)` | Question strings, at most `question_count` elements. Result is also cached in `self.questions`. |

---

### Sub-Task 8.8: Load a Role File (`RoleResponseGenerator.load_role`)

Reads and parses a single role JSON file from disk.

#### Input

| Parameter | Type | Description |
|---|---|---|
| `role_file` | `Path` | Absolute or relative path to a `.json` role file. |

#### Processing

```python
with open(role_file, 'r') as f:
    return json.load(f)
```

#### Output

| Type | Description |
|---|---|
| `dict` | The parsed JSON object. Expected to contain an `"instruction"` key holding a list of instruction variant objects (each with a `"pos"` key). |

---

### Sub-Task 8.9: Format an Instruction String (`RoleResponseGenerator.format_instruction`)

Performs placeholder substitution on a raw instruction template.

#### Input

| Parameter | Type | Description |
|---|---|---|
| `instruction` | `str` | Raw instruction text, potentially containing `{model_name}`. |

#### Processing

```python
return instruction.replace("{model_name}", self.short_name)
```

The `{model_name}` placeholder is replaced with the resolved `self.short_name` (e.g. `"Gemma 2 27B"`).

#### Output

| Type | Description |
|---|---|
| `str` | The instruction with all `{model_name}` occurrences replaced. |

---

### Sub-Task 8.10: Generate Responses for a Single Role (`RoleResponseGenerator.generate_role_responses`)

Orchestrates the full generation pipeline for one role: loads the role's instruction variants, formats them, generates responses via `VLLMGenerator.generate_for_role`, and labels the results.

#### Input

| Parameter | Type | Description |
|---|---|---|
| `role_name` | `str` | Human-readable role identifier (used for logging). |
| `role_data` | `dict` | Parsed role JSON (as returned by `load_role`). Must contain an `"instruction"` list. |

#### Processing

**Step 1 -- Extract instruction variants from the role data.** Return early if none exist.

```python
instructions = role_data.get('instruction', [])
if not instructions:
    return []
```

**Step 2 -- Load questions (cached).**

```python
questions = self.load_questions()
```

**Step 3 -- Format each instruction variant.** The `"pos"` field of each instruction object is extracted and run through placeholder substitution.

```python
formatted_instructions = []
for inst in instructions:
    raw = inst.get('pos', '')
    formatted_instructions.append(self.format_instruction(raw))
```

**Step 4 -- Delegate to `VLLMGenerator.generate_for_role`.**

```python
results = self.generator.generate_for_role(
    instructions=formatted_instructions,
    questions=questions,
    prompt_indices=self.prompt_indices,
)
```

**Step 5 -- Add a `"label"` field to every result.**

```python
for r in results:
    r["label"] = "pos"
```

#### Output

| Type | Shape | Description |
|---|---|---|
| `List[dict]` | `(len(prompt_indices) * len(questions),)` | Result dicts from `generate_for_role`, each augmented with `"label": "pos"`. |

---

### Sub-Task 8.11: Save Responses to Disk (`RoleResponseGenerator.save_responses`)

Writes a list of response dicts to a JSONL file named after the role.

#### Input

| Parameter | Type | Description |
|---|---|---|
| `role_name` | `str` | Used to derive the output filename (`{role_name}.jsonl`). |
| `responses` | `List[dict]` | The response dicts to persist. |

#### Processing

```python
import jsonlines

output_file = self.output_dir / f"{role_name}.jsonl"
with jsonlines.open(output_file, mode='w') as writer:
    for response in responses:
        writer.write(response)
logger.info(f"Saved {len(responses)} responses to {output_file}")
```

#### Output

No return value. Side effect: creates/overwrites `{output_dir}/{role_name}.jsonl`.

---

### Sub-Task 8.12: Process All Roles (`RoleResponseGenerator.process_all_roles`)

Top-level entry point that discovers role files, filters them, and generates + saves responses for each.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip_existing` | `bool` | `True` | If `True`, roles whose output JSONL already exists are skipped. |
| `roles` | `Optional[List[str]]` | `None` | Whitelist of role names. `None` means process every `.json` file in `roles_dir`. |

#### Processing

**Step 1 -- Eagerly load the model and questions** so that all role iterations share the same loaded engine.

```python
self.generator.load()
self.load_questions()
```

**Step 2 -- Discover and parse all role JSON files.** Files without an `"instruction"` key or files that fail to parse are skipped with a warning/error log.

```python
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
```

**Step 3 -- Apply filters.**

*Whitelist filter:*

```python
if roles:
    role_files = {k: v for k, v in role_files.items() if k in roles}
```

*Skip-existing filter:*

```python
if skip_existing:
    role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}
```

The helper `should_skip_role` checks whether the output file already exists:

```python
def should_skip_role(self, role_name: str) -> bool:
    output_file = self.output_dir / f"{role_name}.jsonl"
    return output_file.exists()
```

**Step 4 -- Iterate over remaining roles, generate, and save.** A `tqdm` progress bar wraps the loop. Errors for individual roles are caught and logged, allowing the remaining roles to proceed.

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

No return value. Side effects:

- One JSONL file per successfully processed role is written to `self.output_dir`.
- The vLLM model and questions remain loaded in memory after completion.
