# Task 2: Extract Activations

## Overview

This pipeline step takes the response JSONL files produced by Task 1 (one file per role, each line containing a conversation and metadata) and extracts mean hidden-state activations from the model's transformer layers for every assistant response. The output is one `.pt` file per role, containing a dictionary mapping structured keys to activation tensors.

The script supports automatic multi-worker parallelization when the number of available GPUs exceeds the `tensor_parallel_size`, spawning independent processes that each load a copy of the model on their assigned GPU subset.

**Entry point:** `pipeline/2_activations.py`
**Core classes used:**
- `ProbingModel` (`assistant_axis/internals/model.py`) -- loads the model and tokenizer, provides `get_layers()` for hook registration
- `ConversationEncoder` (`assistant_axis/internals/conversation.py`) -- applies chat templates and computes per-turn token spans
- `ActivationExtractor` (`assistant_axis/internals/activations.py`) -- registers forward hooks and captures hidden states
- `SpanMapper` (`assistant_axis/internals/spans.py`) -- maps token spans to activation slices, computes per-turn means

---

## Sub-Tasks

### Sub-Task 2.1: Parse Arguments and Detect GPU Topology

#### Input

Command-line arguments parsed by `argparse`:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | **required** | HuggingFace model identifier (e.g. `google/gemma-2-27b-it`) |
| `--responses_dir` | `str` | **required** | Directory containing per-role `.jsonl` response files |
| `--output_dir` | `str` | **required** | Directory where `.pt` activation files will be saved |
| `--layers` | `str` | `"all"` | Layers to extract -- either `"all"` or comma-separated indices (e.g. `"0,5,10"`) |
| `--batch_size` | `int` | `16` | Number of conversations per forward pass |
| `--max_length` | `int` | `2048` | Maximum token sequence length (sequences are truncated beyond this) |
| `--tensor_parallel_size` | `int` | `None` | GPUs per model instance; `None` means use all available GPUs for one instance |
| `--roles` | `List[str]` | `None` | If given, only process these role stems (e.g. `--roles helpful harmless`) |
| `--thinking` | `bool` | `False` | Enable thinking mode for Qwen models (`true`/`1`/`yes` accepted) |

Also reads the environment variable `CUDA_VISIBLE_DEVICES` if set.

#### Processing

GPU detection and mode selection:

```python
# Detect GPUs for multi-worker decision
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    available_gpus = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    total_gpus = len(available_gpus)
else:
    total_gpus = torch.cuda.device_count()

# Determine tensor parallel size
tensor_parallel_size = args.tensor_parallel_size if args.tensor_parallel_size else total_gpus

# Use multi-worker mode if we have more GPUs than tensor_parallel_size
use_multi_worker = (
    total_gpus > 1 and
    tensor_parallel_size > 0 and
    total_gpus > tensor_parallel_size
)
```

#### Output

- A boolean decision: single-worker mode vs. multi-worker mode.
- `tensor_parallel_size` (int): GPUs allocated per model instance.
- `num_workers = total_gpus // tensor_parallel_size` (only relevant in multi-worker mode).

---

### Sub-Task 2.2: Discover and Filter Role Files

#### Input

- `responses_dir` (`Path`): directory containing `*.jsonl` files (one per role).
- `output_dir` (`Path`): target directory for `.pt` output files.
- `args.roles` (`Optional[List[str]]`): optional list of role stems to include.

#### Processing

```python
role_files = []
for f in sorted(responses_dir.glob("*.jsonl")):
    # Filter by --roles if specified
    if args.roles and f.stem not in args.roles:
        continue
    # Skip existing
    output_file = output_dir / f"{f.stem}.pt"
    if output_file.exists():
        logger.info(f"Skipping {f.stem} (already exists)")
        continue
    role_files.append(f)
```

Steps:
1. Glob all `*.jsonl` files in `responses_dir`, sorted alphabetically.
2. If `--roles` was specified, skip any file whose stem is not in the list.
3. If the corresponding `.pt` output file already exists, skip it (resume support).
4. Collect the remaining paths into `role_files`.

#### Output

- `role_files` (`List[Path]`): ordered list of response files to process.
- Log messages for each skipped file.

---

### Sub-Task 2.3: Load Model and Determine Layers

#### Input

- `args.model` (`str`): HuggingFace model name.
- `args.layers` (`str`): `"all"` or comma-separated layer indices.

#### Processing

Model loading via `ProbingModel`:

```python
pm = ProbingModel(args.model)
```

Inside `ProbingModel.__init__`, the model and tokenizer are loaded:

```python
self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
self.tokenizer.padding_side = "left"

self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
self.model.eval()
```

Layer resolution uses `ProbingModel.get_layers()`, which searches common architecture paths:

```python
def get_layers(self) -> nn.ModuleList:
    layer_paths = [
        ('model.model.layers', lambda m: m.model.layers),        # Llama, Gemma 2, Qwen
        ('model.language_model.layers', lambda m: m.language_model.layers),  # Gemma 3, LLaVA
        ('model.transformer.h', lambda m: m.transformer.h),      # GPT-style
        ('model.transformer.layers', lambda m: m.transformer.layers),
        ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),
    ]
    for path_name, path_func in layer_paths:
        try:
            layers = path_func(self.model)
            if layers is not None and hasattr(layers, '__len__') and len(layers) > 0:
                self._layers = layers
                return self._layers
        except AttributeError:
            continue
```

Layer index parsing in the pipeline script:

```python
n_layers = len(pm.get_layers())

if args.layers == "all":
    layers = list(range(n_layers))
else:
    layers = [int(x.strip()) for x in args.layers.split(",")]
```

#### Output

- `pm` (`ProbingModel`): loaded model wrapper with `.model`, `.tokenizer`, `.get_layers()`.
- `layers` (`List[int]`): zero-based layer indices to extract (e.g. `[0, 1, ..., 31]` for a 32-layer model).

---

### Sub-Task 2.4: Load Responses from a Role File

#### Input

- `role_file` (`Path`): path to a single JSONL file such as `responses/helpful.jsonl`.

Each line in the JSONL file is a dict with at least these keys:

| Key | Type | Description |
|---|---|---|
| `conversation` | `List[Dict[str, str]]` | Chat messages, e.g. `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]` |
| `prompt_index` | `int` | Index of the system prompt variant |
| `question_index` | `int` | Index of the question |
| `label` | `str` | Ground-truth label for this sample |

#### Processing

```python
def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses
```

Conversations and metadata are then separated:

```python
conversations = []
metadata = []
for resp in responses:
    conversations.append(resp["conversation"])
    metadata.append({
        "prompt_index": resp["prompt_index"],
        "question_index": resp["question_index"],
        "label": resp["label"],
    })
```

#### Output

- `conversations` (`List[List[Dict[str, str]]]`): list of conversations, each a list of role/content dicts.
- `metadata` (`List[Dict]`): parallel list with keys `prompt_index`, `question_index`, `label`.

---

### Sub-Task 2.5: Batch Tokenization and Turn Span Construction

#### Input

- `conversations` (`List[List[Dict[str, str]]]`): a batch of conversations (size up to `batch_size`).
- `pm.tokenizer`: the HuggingFace tokenizer.
- `pm.model_name` (`str`): model identifier used to detect model family.
- `chat_kwargs` (`dict`): e.g. `{"enable_thinking": False}` for Qwen models.

#### Processing

An `ActivationExtractor` and a `ConversationEncoder` are instantiated:

```python
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
extractor = ActivationExtractor(pm, encoder)
span_mapper = SpanMapper(pm.tokenizer)
```

Qwen-specific chat kwargs:

```python
chat_kwargs = {}
if 'qwen' in pm.model_name.lower():
    chat_kwargs['enable_thinking'] = enable_thinking
```

Turn span construction is delegated to `ConversationEncoder.build_batch_turn_spans`:

```python
def build_batch_turn_spans(
    self,
    conversations: List[List[Dict[str, str]]],
    **chat_kwargs,
) -> Tuple[List[List[int]], List[Dict[str, Any]], Dict[str, Any]]:
    batch_full_ids = []
    batch_spans = []
    batch_metadata = {
        'conversation_lengths': [],
        'total_conversations': len(conversations),
        'conversation_offsets': []
    }

    global_offset = 0

    for conv_id, conversation in enumerate(conversations):
        full_ids, spans = self.build_turn_spans(conversation, **chat_kwargs)

        batch_full_ids.append(full_ids)
        batch_metadata['conversation_lengths'].append(len(full_ids))
        batch_metadata['conversation_offsets'].append(global_offset)

        for span in spans:
            enhanced_span = span.copy()
            enhanced_span['conversation_id'] = conv_id
            enhanced_span['local_start'] = span['start']
            enhanced_span['local_end'] = span['end']
            enhanced_span['global_start'] = global_offset + span['start']
            enhanced_span['global_end'] = global_offset + span['end']
            batch_spans.append(enhanced_span)

        global_offset += len(full_ids)

    return batch_full_ids, batch_spans, batch_metadata
```

For each individual conversation, `build_turn_spans` dispatches by model family. For Qwen models, it uses the `_build_turn_spans_qwen` method that locates `<|im_start|>role` / `<|im_end|>` markers in the token IDs and optionally filters out `<think>...</think>` blocks. For Gemma/Llama and other models, it uses a differential-tokenization approach:

```python
def build_turn_spans(self, conversation, **chat_kwargs):
    full_ids = self.tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, **chat_kwargs
    )

    if self._is_qwen():
        return self._build_turn_spans_qwen(conversation, full_ids, **chat_kwargs)

    spans = []
    msgs_before = []
    turn_idx = 0

    for msg in conversation:
        role = msg["role"]
        text = msg.get("content", "")

        if role == "system":
            msgs_before.append(msg)
            continue

        content_ids, start_in_delta = self._content_only_ids_and_offset(
            msgs_before, role, text, **chat_kwargs
        )

        msgs_empty_for_this = msgs_before + [{"role": role, "content": ""}]
        ids_empty_full = self.tokenizer.apply_chat_template(
            msgs_empty_for_this, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
        ids_full_for_this = self.tokenizer.apply_chat_template(
            msgs_before + [{"role": role, "content": text}],
            tokenize=True, add_generation_prompt=False, **chat_kwargs
        )

        pref_len = self._longest_common_prefix_len(ids_full_for_this, ids_empty_full)
        abs_start = pref_len + start_in_delta
        abs_end = abs_start + len(content_ids)

        spans.append({
            "turn": turn_idx,
            "role": role,
            "start": abs_start,
            "end": abs_end,
            "n_tokens": len(content_ids),
            "text": text,
        })
        msgs_before.append(msg)
        turn_idx += 1

    return full_ids, spans
```

Each span dict contains:

| Field | Type | Description |
|---|---|---|
| `turn` | `int` | Zero-based turn index (user=0, assistant=1, ...) |
| `role` | `str` | `"user"` or `"assistant"` |
| `start` | `int` | Start token index (inclusive) within the conversation |
| `end` | `int` | End token index (exclusive) within the conversation |
| `n_tokens` | `int` | Number of content tokens in this span |
| `text` | `str` | Original text content of this turn |
| `conversation_id` | `int` | Index of the conversation within the batch |
| `local_start` / `local_end` | `int` | Same as `start`/`end` |
| `global_start` / `global_end` | `int` | Offsets relative to the concatenation of all conversations |

#### Output

- `batch_full_ids` (`List[List[int]]`): tokenized IDs for each conversation.
- `batch_spans` (`List[Dict[str, Any]]`): flat list of span dicts across all conversations in the batch.
- `batch_metadata` (`Dict[str, Any]`): contains `conversation_lengths`, `total_conversations`, `conversation_offsets`.

---

### Sub-Task 2.6: Forward Pass with Hook-Based Activation Capture

#### Input

- `conversations` (`List[List[Dict[str, str]]]`): batch of conversations.
- `layer` (`List[int]`): layer indices to capture.
- `max_length` (`int`): maximum token sequence length.
- `chat_kwargs` (`dict`): model-specific template arguments.

#### Processing

This is performed by `ActivationExtractor.batch_conversations`. The method:

1. **Tokenizes and pads** the batch:

```python
batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
    conversations, **chat_kwargs
)

batch_size = len(batch_full_ids)
actual_max_len = max(len(ids) for ids in batch_full_ids)
max_seq_len = min(max_length, actual_max_len)

input_ids_batch = []
attention_mask_batch = []

for ids in batch_full_ids:
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]

    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

    input_ids_batch.append(padded_ids)
    attention_mask_batch.append(attention_mask)

input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)
```

2. **Registers forward hooks** on each target layer to capture outputs:

```python
layer_outputs = {}
handles = []

def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        layer_outputs[layer_idx] = act_tensor
    return hook_fn

model_layers = self.probing_model.get_layers()
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)
```

3. **Runs a single forward pass** with `torch.inference_mode()`:

```python
try:
    with torch.inference_mode():
        _ = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
finally:
    for handle in handles:
        handle.remove()
```

4. **Stacks and normalizes** the captured activations:

```python
target_device = layer_outputs[layer_list[0]].device
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
])  # (num_layers, batch_size, seq_len, hidden_size)

if selected_activations.dtype != torch.bfloat16:
    selected_activations = selected_activations.to(torch.bfloat16)
```

5. **Builds batch metadata**:

```python
batch_metadata = {
    'conversation_lengths': span_metadata['conversation_lengths'],
    'total_conversations': span_metadata['total_conversations'],
    'conversation_offsets': span_metadata['conversation_offsets'],
    'max_seq_len': max_seq_len,
    'attention_mask': attention_mask_tensor,
    'actual_lengths': [len(ids) for ids in batch_full_ids],
    'truncated_lengths': [min(len(ids), max_seq_len) for ids in batch_full_ids]
}
```

#### Output

- `selected_activations` (`torch.Tensor`): shape `(num_layers, batch_size, max_seq_len, hidden_size)`, dtype `bfloat16`.
- `batch_metadata` (`Dict[str, Any]`): contains `conversation_lengths`, `total_conversations`, `conversation_offsets`, `max_seq_len`, `attention_mask`, `actual_lengths`, `truncated_lengths`.

---

### Sub-Task 2.7: Map Spans to Per-Turn Mean Activations

#### Input

- `batch_activations` (`torch.Tensor`): shape `(num_layers, batch_size, max_seq_len, hidden_size)`.
- `batch_spans` (`List[Dict[str, Any]]`): flat list of span dicts from Sub-Task 2.5.
- `batch_metadata` (`Dict[str, Any]`): from Sub-Task 2.6.

#### Processing

`SpanMapper.map_spans` groups spans by `conversation_id`, sorts them by turn index, and extracts the mean activation over each span's token range:

```python
def map_spans(
    self,
    batch_activations: torch.Tensor,
    batch_spans: List[Dict[str, Any]],
    batch_metadata: Dict[str, Any],
) -> List[torch.Tensor]:
    num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
    device = batch_activations.device
    dtype = batch_activations.dtype

    conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]

    # Group spans by conversation
    spans_by_conversation = {}
    for span in batch_spans:
        conv_id = span['conversation_id']
        if conv_id not in spans_by_conversation:
            spans_by_conversation[conv_id] = []
        spans_by_conversation[conv_id].append(span)

    # Sort spans by turn within each conversation
    for conv_id in spans_by_conversation:
        spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])

    # Extract per-turn activations for each conversation
    for conv_id in range(batch_metadata['total_conversations']):
        if conv_id not in spans_by_conversation:
            conversation_activations[conv_id] = torch.empty(
                0, num_layers, hidden_size, dtype=dtype, device=device
            )
            continue

        spans = spans_by_conversation[conv_id]
        turn_activations = []

        for span in spans:
            start_idx = span['start']
            end_idx = span['end']

            # Check bounds to handle truncation
            actual_length = batch_metadata['truncated_lengths'][conv_id]
            if start_idx >= actual_length:
                continue

            end_idx = min(end_idx, actual_length)

            if start_idx >= end_idx:
                continue

            # Extract activations for this span
            # shape: (num_layers, span_length, hidden_size)
            span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

            # Compute mean across tokens in this span
            span_length = span_activations.size(1)
            if span_length > 0:
                if span_length == 1:
                    mean_activation = span_activations.squeeze(1)
                else:
                    mean_activation = span_activations.mean(dim=1)
                turn_activations.append(mean_activation)

        if turn_activations:
            conversation_activations[conv_id] = torch.stack(turn_activations)
        else:
            conversation_activations[conv_id] = torch.empty(
                0, num_layers, hidden_size, dtype=dtype, device=device
            )

    return conversation_activations
```

Key bounds-checking logic:
- If `start_idx >= truncated_length`, the span was entirely truncated and is skipped.
- `end_idx` is clamped to `min(end_idx, truncated_length)` to handle partial truncation.

#### Output

- `conv_activations_list` (`List[torch.Tensor]`): one tensor per conversation, each with shape `(num_turns, num_layers, hidden_size)`. If a conversation has no valid spans, the tensor has shape `(0, num_layers, hidden_size)`.

---

### Sub-Task 2.8: Select Assistant Turn Activations and Aggregate

#### Input

- `conv_activations_list` (`List[torch.Tensor]`): per-conversation tensors from Sub-Task 2.7, each of shape `(num_turns, num_layers, hidden_size)`.

Turns are indexed as: turn 0 = user, turn 1 = assistant, turn 2 = user, turn 3 = assistant, etc.

#### Processing

For each conversation, extract only the assistant turns (odd-indexed) and compute their mean:

```python
for conv_acts in conv_activations_list:
    if conv_acts.numel() == 0:
        all_activations.append(None)
        continue

    # conv_acts shape: (num_turns, num_layers, hidden_size)
    if conv_acts.shape[0] >= 2:
        # Take the last assistant turn (index 1 for single-turn)
        assistant_act = conv_acts[1::2]  # All assistant turns
        if assistant_act.shape[0] > 0:
            # Take mean across all assistant turns
            mean_act = assistant_act.mean(dim=0).cpu()  # (num_layers, hidden_size)
            all_activations.append(mean_act)
        else:
            all_activations.append(None)
    else:
        all_activations.append(None)
```

Indexing logic:
- `conv_acts[1::2]` selects indices 1, 3, 5, ... (all assistant turns).
- `.mean(dim=0)` averages across all assistant turns, yielding `(num_layers, hidden_size)`.
- `.cpu()` moves the result to CPU memory for saving.

#### Output

- `all_activations` (`List[Optional[torch.Tensor]]`): one entry per conversation. Each is either `None` (if no valid assistant activations) or a tensor of shape `(num_layers, hidden_size)`.

---

### Sub-Task 2.9: Build Keyed Dictionary and Save as `.pt`

#### Input

- `activations_list` (`List[Optional[torch.Tensor]]`): from Sub-Task 2.8.
- `metadata` (`List[Dict]`): parallel list with `label`, `prompt_index`, `question_index`.
- `output_dir` (`Path`): target directory.
- `role` (`str`): the role stem (e.g. `"helpful"`), derived from `role_file.stem`.

#### Processing

```python
activations_dict = {}
for i, (act, meta) in enumerate(zip(activations_list, metadata)):
    if act is not None:
        key = f"{meta['label']}_p{meta['prompt_index']}_q{meta['question_index']}"
        activations_dict[key] = act

if activations_dict:
    torch.save(activations_dict, output_file)
    logger.info(f"Saved {len(activations_dict)} activations for {role}")
```

Key format: `"{label}_p{prompt_index}_q{question_index}"` -- e.g. `"positive_p0_q3"`, `"negative_p2_q1"`.

After saving, GPU memory is cleaned up:

```python
gc.collect()
torch.cuda.empty_cache()
```

#### Output

- One file per role: `{output_dir}/{role}.pt`
- The file contains a Python dict saved with `torch.save`, structured as:

```
{
    "positive_p0_q0": tensor(num_layers, hidden_size),
    "positive_p0_q1": tensor(num_layers, hidden_size),
    "negative_p1_q0": tensor(num_layers, hidden_size),
    ...
}
```

- Each value is a `torch.Tensor` of shape `(num_layers, hidden_size)` in float32 (moved to CPU).
- Example: for a 32-layer model with hidden_size 3584, each tensor is `(32, 3584)`.

---

### Sub-Task 2.10: Multi-Worker Parallelization

#### Input

- `args` (argparse namespace): all CLI arguments.
- `gpu_ids` (`List[int]`): available GPU device IDs.
- `tensor_parallel_size` (`int`): GPUs per model instance.
- `role_files` (`List[Path]`): role files to process.

#### Processing

1. **Partition GPUs** into chunks, one per worker:

```python
gpu_chunks = []
for i in range(num_workers):
    start = i * tensor_parallel_size
    end = start + tensor_parallel_size
    gpu_chunks.append(gpu_ids[start:end])
```

2. **Distribute roles** across workers with round-robin assignment:

```python
role_chunks = [[] for _ in range(num_workers)]
for i, role_file in enumerate(role_files):
    role_chunks[i % num_workers].append(role_file)
```

3. **Spawn worker processes** using `torch.multiprocessing`:

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

4. Each worker **sets `CUDA_VISIBLE_DEVICES`** to restrict its GPU visibility, then loads its own model instance and processes its assigned roles sequentially:

```python
def process_roles_on_worker(worker_id, gpu_ids, role_files, args):
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

    pm = ProbingModel(args.model)

    n_layers = len(pm.get_layers())
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    for role_file in tqdm(role_files, desc=f"Worker-{worker_id}", position=worker_id):
        try:
            success = process_role(pm, role_file, output_dir, layers,
                                   args.batch_size, args.max_length, args.thinking)
        except Exception as e:
            worker_logger.error(f"Exception processing {role_file.stem}: {e}")
```

#### Output

- Same as Sub-Task 2.9 but produced in parallel: one `.pt` file per role, each written independently by the worker that processed it.
- Log output per worker with format: `Worker-{id}[GPUs:{ids}] - {level} - {message}`.

---

## End-to-End Data Flow Summary

```
responses/{role}.jsonl
    |
    |  load_responses()
    v
List[dict] with "conversation", "prompt_index", "question_index", "label"
    |
    |  separate conversations & metadata
    v
conversations: List[List[Dict[str,str]]]     metadata: List[Dict]
    |
    |  ConversationEncoder.build_batch_turn_spans()
    v
batch_full_ids: List[List[int]]
batch_spans: List[Dict]  (with conversation_id, start, end, role, turn)
span_metadata: Dict
    |
    |  ActivationExtractor.batch_conversations()
    |    - pad/truncate to max_length
    |    - register forward hooks on target layers
    |    - single forward pass
    v
batch_activations: Tensor (num_layers, batch_size, max_seq_len, hidden_size)
batch_metadata: Dict (with truncated_lengths, attention_mask, etc.)
    |
    |  SpanMapper.map_spans()
    |    - group spans by conversation_id
    |    - slice activations by [start:end] per span
    |    - mean over token dimension per span
    v
conv_activations_list: List[Tensor]  each (num_turns, num_layers, hidden_size)
    |
    |  select assistant turns (odd indices)
    |  mean across assistant turns
    v
all_activations: List[Optional[Tensor]]  each (num_layers, hidden_size)
    |
    |  build keyed dict: "{label}_p{prompt}_q{question}" -> tensor
    |  torch.save()
    v
{output_dir}/{role}.pt
```
