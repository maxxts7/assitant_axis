# `assistant_axis/generation.py`

## Overview

This file provides response generation utilities for transformer-based language models. It contains two main pathways for generating text:

1. **HuggingFace generation** -- a single-response function (`generate_response`) that runs inference through a standard HuggingFace model and tokenizer.
2. **vLLM batch generation** -- the `VLLMGenerator` and `RoleResponseGenerator` classes, which use the vLLM library for high-throughput batch inference.

There is also a helper function, `format_conversation`, that builds the chat-format message list a model expects, handling the case where a model does or does not support a system prompt role.

---

## Line-by-line explanation

### Module docstring (lines 1--14)

```python
"""
Response generation utilities for transformer models.

This module provides functions for generating model responses using vLLM
for batch inference.

For HuggingFace generation, use ProbingModel.generate() from assistant_axis.internals.

Example (vLLM - batch inference):
    from assistant_axis.generation import VLLMGenerator

    generator = VLLMGenerator("google/gemma-2-27b-it")
    responses = generator.generate_batch(conversations)
"""
```

The module-level docstring describes the purpose of the file: providing generation utilities. It directs the reader to `ProbingModel.generate()` for HuggingFace-based generation and gives a quick example of the vLLM workflow.

---

### Imports (lines 16--22)

```python
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch
from tqdm import tqdm
```

- `json` -- used later to load role JSON files from disk.
- `logging` -- standard Python logging; the module creates its own logger below.
- `Path` (from `pathlib`) -- object-oriented filesystem paths, used in `RoleResponseGenerator` for directory and file handling.
- `List`, `Dict`, `Optional` (from `typing`) -- type hints used throughout function signatures.
- `torch` -- PyTorch, needed for `torch.no_grad()` during HuggingFace inference.
- `tqdm` -- progress bar library, used when iterating over roles in `process_all_roles`.

---

### Logger setup (line 24)

```python
logger = logging.getLogger(__name__)
```

Creates a module-level logger named after the module (`assistant_axis.generation`). All log messages in this file go through this logger so that callers can configure logging level and handlers from the outside.

---

## `generate_response` function (lines 27--81)

### Signature (lines 27--35)

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
```

Defines a function that takes a HuggingFace `model` and `tokenizer`, a `conversation` (list of message dictionaries with `"role"` and `"content"` keys), and several generation hyperparameters. It returns a single string -- the model's generated response.

- `max_new_tokens` -- caps how many tokens the model may generate (default 512).
- `temperature` -- controls randomness; higher values produce more varied output.
- `top_p` -- nucleus sampling threshold; only the smallest set of tokens whose cumulative probability exceeds `top_p` is considered.
- `do_sample` -- when `True`, the model samples from the distribution; when `False`, it uses greedy decoding.

---

### Docstring (lines 36--50)

```python
    """
    Generate a single response for a conversation using HuggingFace.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversation: List of message dicts
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample (False = greedy)

    Returns:
        Generated response text
    """
```

Standard docstring documenting every parameter and the return value.

---

### Disable thinking for Qwen models (lines 51--54)

```python
    # Disable thinking for Qwen models
    chat_template_kwargs = {}
    if hasattr(tokenizer, 'name_or_path') and "qwen" in tokenizer.name_or_path.lower():
        chat_template_kwargs["enable_thinking"] = False
```

Some Qwen models have a "thinking" mode baked into their chat template. This block detects whether the tokenizer belongs to a Qwen model (by checking the `name_or_path` attribute for the substring `"qwen"`) and, if so, passes `enable_thinking=False` to suppress it. The extra keyword arguments are stored in `chat_template_kwargs` so they can be unpacked into `apply_chat_template` later.

---

### Apply chat template (lines 56--61)

```python
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs
    )
```

Converts the list of message dicts into the raw text prompt the model expects.

- `tokenize=False` -- returns a string instead of token IDs.
- `add_generation_prompt=True` -- appends the assistant turn prefix so the model knows it should start generating.
- `**chat_template_kwargs` -- passes any extra keyword arguments (e.g., `enable_thinking=False` for Qwen).

---

### Tokenize and move to device (lines 63--64)

```python
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
```

- The prompt string is tokenized into PyTorch tensors (`return_tensors="pt"`) and moved to the same device as the model (CPU or GPU).
- `input_length` records the number of input tokens so the generated tokens can be isolated later.

---

### Generate with no gradient tracking (lines 66--74)

```python
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
```

- `torch.no_grad()` disables gradient computation, which saves memory and speeds up inference since backpropagation is not needed.
- `model.generate()` runs autoregressive generation.
- `temperature` and `top_p` are only passed when `do_sample` is `True`; passing `None` lets the model use its defaults (and avoids warnings when using greedy decoding).
- `pad_token_id` is set explicitly to avoid a common HuggingFace warning when the tokenizer does not set it automatically.

---

### Decode the response (lines 76--79)

```python
    response = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    )
```

- `outputs[0]` is the full sequence of token IDs (input + generated).
- `[input_length:]` slices off the input tokens, keeping only the newly generated ones.
- `skip_special_tokens=True` strips special tokens (e.g., `<eos>`, `<pad>`) from the decoded string.

---

### Return (line 81)

```python
    return response
```

Returns the decoded response string.

---

## `format_conversation` function (lines 84--129)

### Signature (lines 84--88)

```python
def format_conversation(
    instruction: Optional[str],
    question: str,
    tokenizer,
) -> List[Dict[str, str]]:
```

Takes an optional system `instruction`, a user `question`, and a HuggingFace `tokenizer`. Returns a list of message dicts suitable for `apply_chat_template`.

---

### Docstring (lines 89--99)

```python
    """
    Format a conversation for model input.

    Args:
        instruction: Optional system instruction
        question: User question
        tokenizer: HuggingFace tokenizer (to check system prompt support)

    Returns:
        List of message dicts for the conversation
    """
```

Documents the purpose, parameters, and return value.

---

### System prompt support detection (lines 100--115)

```python
    # Check system prompt support by testing the chat template
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

Not all models support the `"system"` role in their chat template. This block tests support empirically:

1. A dummy conversation is created with a unique sentinel string (`"__SYSTEM_TEST__"`) in the system role.
2. `apply_chat_template` is called on it.
3. If the sentinel string appears in the rendered output, the template kept the system message, so `supports_system` is set to `True`.
4. If the template raises an exception (some templates reject the `"system"` role outright), `supports_system` is set to `False`.

---

### Build message list -- system-prompt-capable models (lines 117--122)

```python
    if supports_system:
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
```

When the model supports system prompts, the instruction (if provided) is placed in a `"system"` role message, followed by the user's question in a `"user"` role message.

---

### Build message list -- models without system prompt support (lines 123--129)

```python
    else:
        # Concatenate instruction with question for models without system support
        if instruction:
            formatted = f"{instruction}\n\n{question}"
        else:
            formatted = question
        return [{"role": "user", "content": formatted}]
```

When the model does not support a system role, the instruction is prepended to the question text (separated by two newlines) and everything is sent as a single `"user"` message.

---

## Section separator comment (lines 132--134)

```python
# =============================================================================
# vLLM Generation (for batch inference / pipeline)
# =============================================================================
```

A visual separator indicating that the code below deals with vLLM-based batch inference.

---

## `VLLMGenerator` class (lines 136--237)

### Class docstring (lines 136--143)

```python
class VLLMGenerator:
    """
    Generator for batch inference using vLLM.

    Example:
        generator = VLLMGenerator("google/gemma-2-27b-it")
        responses = generator.generate_batch(conversations)
    """
```

Describes the class and gives a minimal usage example.

---

### `__init__` (lines 145--176)

```python
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
```

Constructor parameters:

- `model_name` -- HuggingFace model identifier (e.g., `"google/gemma-2-27b-it"`).
- `max_model_len` -- maximum context length the vLLM engine allocates for (default 2048).
- `tensor_parallel_size` -- number of GPUs to split the model across; `None` means auto-detect.
- `gpu_memory_utilization` -- fraction of GPU memory vLLM is allowed to use (default 0.9).
- `temperature`, `max_tokens`, `top_p` -- generation hyperparameters.

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

All parameters are stored as instance attributes. `self.llm` and `self.sampling_params` are initialized to `None` -- they will be populated when `load()` is called. This deferred loading pattern lets you create the object cheaply and only load the (large) model when you actually need it.

---

### `load` method (lines 178--201)

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

- **Guard clause** (`if self.llm is not None: return`) -- ensures the model is only loaded once, even if `load()` is called multiple times.
- **Lazy import** -- `vllm` is imported inside the method rather than at module level. This means code that imports `generation.py` but does not use vLLM will not fail if vLLM is not installed.
- An `LLM` instance is created with `trust_remote_code=True`, which allows executing model code from the HuggingFace Hub (required by some model architectures).
- A `SamplingParams` object is created to hold the generation settings.

---

### `generate_batch` method (lines 203--237)

```python
    def generate_batch(
        self,
        conversations: List[List[Dict[str, str]]],
    ) -> List[str]:
```

Takes a list of conversations (each conversation is itself a list of message dicts) and returns a list of generated response strings -- one per conversation.

```python
        self.load()
```

Ensures the model is loaded before proceeding.

```python
        tokenizer = self.llm.get_tokenizer()
```

Retrieves the tokenizer from the loaded vLLM engine.

```python
        # Disable thinking for Qwen models (https://github.com/vllm-project/vllm/issues/18066)
        chat_template_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_template_kwargs["enable_thinking"] = False
```

Same Qwen thinking-mode suppression as in `generate_response`, but this time checking `self.model_name` instead of the tokenizer's `name_or_path`. A link to the relevant vLLM GitHub issue is included as context.

```python
        prompts = []
        for conv in conversations:
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True,
                **chat_template_kwargs
            )
            prompts.append(prompt)
```

Each conversation is rendered into a prompt string using the tokenizer's chat template, and all prompts are collected into a list.

```python
        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)
```

The full batch of prompts is submitted to vLLM in one call. vLLM handles scheduling and batching internally for maximum throughput.

```python
        responses = [output.outputs[0].text for output in outputs]
        return responses
```

Each `output` object contains a list of generated sequences in `output.outputs`. Since no `n > 1` parameter was set, there is exactly one output per prompt (`outputs[0]`). The `.text` attribute holds the decoded string. The list of response strings is returned.

---

### `generate_for_role` method (lines 239--300)

```python
    def generate_for_role(
        self,
        instructions: List[str],
        questions: List[str],
        prompt_indices: Optional[List[int]] = None,
    ) -> List[Dict]:
```

A higher-level method that generates responses for every combination of instruction variant and question. Returns a list of result dictionaries.

```python
        self.load()
        tokenizer = self.llm.get_tokenizer()
```

Ensures the model is loaded and gets the tokenizer.

```python
        if prompt_indices is None:
            prompt_indices = list(range(len(instructions)))
```

If no subset of instruction indices is specified, all instructions are used.

```python
        # Build all conversations
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

A nested loop iterates over each selected instruction index and each question. For every (instruction, question) pair:

- `format_conversation` builds the message list (handling system prompt support).
- The conversation is appended to `all_conversations`.
- A metadata dict recording the instruction text, prompt index, question index, and question text is appended to `all_metadata`.

An out-of-bounds check (`if prompt_idx >= len(instructions)`) silently skips invalid indices.

```python
        if not all_conversations:
            return []
```

Early return if there are no conversations to process (e.g., all prompt indices were out of bounds).

```python
        # Generate
        responses = self.generate_batch(all_conversations)
```

All conversations are sent to `generate_batch` in one call for efficient batch inference.

```python
        # Build results
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

The conversations, metadata, and responses are zipped together. Each result dict includes the original metadata plus the full conversation with the assistant's response appended. The list of result dicts is returned.

---

## `RoleResponseGenerator` class (lines 303--501)

### Class docstring (lines 303--317)

```python
class RoleResponseGenerator:
    """
    Generator for role-based model responses using vLLM batch inference.

    Processes role JSON files and generates responses for all roles.

    Example:
        generator = RoleResponseGenerator(
            model_name="google/gemma-2-27b-it",
            roles_dir="data/prompts/roles",
            output_dir="outputs/responses",
            questions_file="data/prompts/questions.jsonl"
        )
        generator.process_all_roles()
    """
```

Describes the class as a pipeline for processing role JSON files and generating responses using vLLM, with a usage example.

---

### `__init__` (lines 319--382)

```python
    def __init__(
        self,
        model_name: str,
        roles_dir: str,
        output_dir: str,
        questions_file: str,
        max_model_len: int = 2048,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        question_count: int = 240,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
        prompt_indices: Optional[List[int]] = None,
        short_name: Optional[str] = None,
    ):
```

Constructor parameters on top of the vLLM ones:

- `roles_dir` -- directory containing role definition JSON files.
- `output_dir` -- where generated JSONL output files will be saved.
- `questions_file` -- path to a JSONL file containing questions.
- `question_count` -- how many questions to use per role (default 240).
- `prompt_indices` -- which instruction indices from each role file to process (default: indices 0 through 4).
- `short_name` -- a short model name used for string interpolation in prompts; auto-detected from a config if not provided.

```python
        self.model_name = model_name
        self.roles_dir = Path(roles_dir)
        self.output_dir = Path(output_dir)
        self.questions_file = questions_file
        self.question_count = question_count
        self.prompt_indices = prompt_indices if prompt_indices is not None else list(range(5))
```

Stores config values. `roles_dir` and `output_dir` are wrapped in `Path` objects. If `prompt_indices` is not supplied, it defaults to `[0, 1, 2, 3, 4]`.

```python
        # Get short name for {model_name} placeholder
        if short_name is None:
            from .models import get_config
            config = get_config(model_name)
            self.short_name = config["short_name"]
        else:
            self.short_name = short_name
```

If no `short_name` is given, the code lazily imports `get_config` from the sibling `models` module and looks up the model's short name from the config. This short name is used later to replace `{model_name}` placeholders in system instructions.

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

Creates a `VLLMGenerator` instance (but does not load the model yet -- that happens lazily).

```python
        self.questions = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
```

`self.questions` is initialized to `None` (loaded lazily). The output directory is created if it does not already exist, including any necessary parent directories.

```python
        logger.info(f"Initialized RoleResponseGenerator with model: {model_name}")
        logger.info(f"Output directory: {self.output_dir}")
```

Logs initialization details.

---

### `load_questions` method (lines 384--398)

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

- Guard clause returns the cached list if questions are already loaded.
- `jsonlines` is imported lazily (only when needed).
- The JSONL file is opened and each entry's `"question"` field is extracted.
- The list is truncated to `self.question_count` items.
- The result is cached in `self.questions` for subsequent calls.

---

### `load_role` method (lines 400--403)

```python
    def load_role(self, role_file: Path) -> dict:
        """Load a role JSON file."""
        with open(role_file, 'r') as f:
            return json.load(f)
```

Opens a JSON file and returns its parsed contents as a dictionary.

---

### `format_instruction` method (lines 405--407)

```python
    def format_instruction(self, instruction: str) -> str:
        """Format instruction, replacing {model_name} placeholder."""
        return instruction.replace("{model_name}", self.short_name)
```

Replaces the literal placeholder `{model_name}` in an instruction string with the model's short name (e.g., `"Gemma 2 27B"`).

---

### `generate_role_responses` method (lines 409--436)

```python
    def generate_role_responses(self, role_name: str, role_data: dict) -> List[dict]:
        """Generate responses for a single role."""
        instructions = role_data.get('instruction', [])
        if not instructions:
            return []
```

Extracts the `"instruction"` list from the role data. If there are no instructions, returns an empty list immediately.

```python
        questions = self.load_questions()
```

Loads (or retrieves cached) questions.

```python
        # Get and format instructions
        formatted_instructions = []
        for inst in instructions:
            raw = inst.get('pos', '')
            formatted_instructions.append(self.format_instruction(raw))
```

Each instruction entry is expected to be a dict containing a `"pos"` key (for the positive/desired instruction text). The raw text is formatted via `format_instruction` to fill in the `{model_name}` placeholder.

```python
        logger.info(f"Processing role '{role_name}' with {len(questions)} questions")

        # Generate
        results = self.generator.generate_for_role(
            instructions=formatted_instructions,
            questions=questions,
            prompt_indices=self.prompt_indices,
        )
```

Calls `VLLMGenerator.generate_for_role` to produce responses for all (instruction, question) combinations.

```python
        # Add label
        for r in results:
            r["label"] = "pos"

        return results
```

Each result dict is annotated with `"label": "pos"` (indicating these are positive-class samples). The results list is returned.

---

### `save_responses` method (lines 438--446)

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

Writes a list of response dicts to a JSONL file named `<role_name>.jsonl` in the output directory. Each dict becomes one JSON line. `jsonlines` is lazily imported.

---

### `should_skip_role` method (lines 448--451)

```python
    def should_skip_role(self, role_name: str) -> bool:
        """Check if role output already exists."""
        output_file = self.output_dir / f"{role_name}.jsonl"
        return output_file.exists()
```

Returns `True` if the output JSONL file for a given role already exists on disk. Used to skip re-processing roles that have already been generated.

---

### `process_all_roles` method (lines 453--501)

```python
    def process_all_roles(
        self,
        skip_existing: bool = True,
        roles: Optional[List[str]] = None,
    ):
```

The main entry point for batch processing. Parameters:

- `skip_existing` -- when `True` (default), roles that already have output files are skipped.
- `roles` -- an optional list of specific role names to process; `None` means process all roles found in the directory.

```python
        # Load model
        self.generator.load()
        self.load_questions()
```

Eagerly loads the vLLM model and questions up front, before entering the processing loop.

```python
        # Get role files
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

Scans the `roles_dir` for all `*.json` files (sorted alphabetically). Each file is loaded and validated -- it must contain an `"instruction"` key. Files that fail to load or lack the required field are logged and skipped. Valid roles are stored in a `role_files` dict keyed by the role name (the file stem, i.e., filename without extension).

```python
        logger.info(f"Found {len(role_files)} role files")
```

Logs how many valid role files were found.

```python
        # Filter
        if roles:
            role_files = {k: v for k, v in role_files.items() if k in roles}

        if skip_existing:
            role_files = {k: v for k, v in role_files.items() if not self.should_skip_role(k)}
```

Two filtering passes:

1. If a specific list of role names was provided, only those are kept.
2. If `skip_existing` is `True`, roles whose output files already exist are removed.

```python
        logger.info(f"Processing {len(role_files)} roles")
```

Logs how many roles remain after filtering.

```python
        # Process
        for role_name, role_data in tqdm(role_files.items(), desc="Processing roles"):
            try:
                responses = self.generate_role_responses(role_name, role_data)
                if responses:
                    self.save_responses(role_name, responses)
            except Exception as e:
                logger.error(f"Error processing {role_name}: {e}")
```

Iterates over the remaining roles with a `tqdm` progress bar. For each role:

1. `generate_role_responses` produces all (instruction, question) responses.
2. If there are any responses, they are saved to disk via `save_responses`.
3. Any exception during processing is caught, logged, and the loop continues to the next role -- so one failing role does not halt the entire pipeline.
