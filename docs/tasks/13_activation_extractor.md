# Task 13: Activation Extractor (internals/activations.py)

## Overview

The `ActivationExtractor` class provides the primary interface for extracting hidden-state activations from transformer model layers. It uses PyTorch forward hooks to intercept intermediate representations during the model's forward pass, avoiding the need for `output_hidden_states=True` (which forces the model to retain all layers in memory simultaneously).

Four public methods cover the main extraction scenarios:

| Method | Purpose | Position(s) Extracted |
|---|---|---|
| `full_conversation` | All tokens across all (or selected) layers | Every token in the sequence |
| `at_newline` | Single activation vector at the newline token | Last `\n\n` (or `\n`) token position |
| `for_prompts` | Batch of prompts, each extracted at newline | One vector per prompt |
| `batch_conversations` | Padded batch of multi-turn conversations | All tokens, with metadata for span indexing |

One private helper is also documented:

| Helper | Purpose |
|---|---|
| `_find_newline_position` | Locate the newline token index in a 1-D token sequence |

The class also interacts with:

- **`ProbingModel`** (`internals/model.py`) -- wraps the HuggingFace model; provides `get_layers()`, `model`, and `tokenizer`.
- **`ConversationEncoder`** (`internals/conversation.py`) -- handles chat-template formatting, tokenization, and turn-span extraction.
- **`StopForward`** (`internals/exceptions.py`) -- a custom exception defined for early-stopping forward passes (declared in the codebase but not used directly by `ActivationExtractor`).

---

## Dependencies and Related Types

### StopForward (internals/exceptions.py)

```python
class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass
```

Defined in `internals/exceptions.py`. It exists to allow other parts of the codebase to raise an exception inside a forward hook and abort the forward pass early (saving compute when only early layers are needed). `ActivationExtractor` itself does **not** raise `StopForward` -- it always runs a full forward pass so that hooks on any requested layer fire.

---

## Sub-Tasks

### Sub-Task 13.1: Constructor (`__init__`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `probing_model` | `ProbingModel` | (required) | Loaded model wrapper; provides `.model`, `.tokenizer`, and `.get_layers()` |
| `encoder` | `ConversationEncoder` | (required) | Conversation formatter that knows model-specific chat templates |

#### Processing

The constructor stores references to the underlying model, tokenizer, and both wrapper objects. No heavy work is performed here -- model loading and tokenizer initialisation happen in `ProbingModel.__init__`.

```python
def __init__(self, probing_model: 'ProbingModel', encoder: 'ConversationEncoder'):
    self.model = probing_model.model
    self.tokenizer = probing_model.tokenizer
    self.probing_model = probing_model
    self.encoder = encoder
```

#### Output

A fully initialised `ActivationExtractor` instance with four attributes:

| Attribute | Type | Source |
|---|---|---|
| `self.model` | `transformers.PreTrainedModel` | `probing_model.model` |
| `self.tokenizer` | `transformers.AutoTokenizer` | `probing_model.tokenizer` |
| `self.probing_model` | `ProbingModel` | passed directly |
| `self.encoder` | `ConversationEncoder` | passed directly |

---

### Sub-Task 13.2: Full Conversation Extraction (`full_conversation`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `conversation` | `Union[str, List[Dict[str, str]]]` | (required) | A raw string **or** a list of `{"role": ..., "content": ...}` message dicts |
| `layer` | `Optional[Union[int, List[int]]]` | `None` | Single layer index, list of layer indices, or `None` for all layers |
| `chat_format` | `bool` | `True` | Whether to apply the chat template before tokenisation |
| `**chat_kwargs` | | | Forwarded to `tokenizer.apply_chat_template` (e.g. `enable_thinking`) |

#### Processing

**Step 1 -- Determine layer mode and layer list.**

```python
if isinstance(layer, int):
    single_layer_mode = True
    layer_list = [layer]
elif isinstance(layer, list):
    single_layer_mode = False
    layer_list = layer
else:
    single_layer_mode = False
    layer_list = list(range(len(self.probing_model.get_layers())))
```

`single_layer_mode` controls whether the return value is squeezed (2-D) or kept stacked (3-D).

**Step 2 -- Format the conversation.**

If `chat_format=True`, a bare string is wrapped as a single user message. The chat template is applied with `add_generation_prompt=False`.

```python
if chat_format:
    if isinstance(conversation, str):
        conversation = [{"role": "user", "content": conversation}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )
else:
    formatted_prompt = conversation
```

**Step 3 -- Tokenize.**

```python
tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
input_ids = tokens["input_ids"].to(self.model.device)
```

Produces a `(1, num_tokens)` tensor on the model device.

**Step 4 -- Register forward hooks on every requested layer.**

A closure captures each layer index. When the hook fires during the forward pass, it extracts the activation tensor. For tuple outputs (common in HuggingFace models where the layer returns `(hidden_states, attention, ...)`), element `[0]` is taken. The batch dimension is removed (`[0, :, :]`) and the tensor is moved to CPU.

```python
activations = []
handles = []

def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu())
    return hook_fn

model_layers = self.probing_model.get_layers()
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)
```

**Step 5 -- Run the forward pass.**

A single full forward pass fires every registered hook. `torch.inference_mode()` disables gradient tracking for efficiency. The `finally` block guarantees hook cleanup even if the forward pass raises an error.

```python
try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    for handle in handles:
        handle.remove()
```

**Step 6 -- Stack and return.**

```python
activations = torch.stack(activations)

if single_layer_mode:
    return activations[0]  # Return single layer
else:
    return activations
```

#### Output

| `layer` argument | Return type | Shape |
|---|---|---|
| `int` (single layer) | `torch.Tensor` | `(num_tokens, hidden_size)` |
| `list` or `None` (multi-layer) | `torch.Tensor` | `(num_layers, num_tokens, hidden_size)` |

All tensors are on CPU with the model's native dtype (typically `bfloat16`).

---

### Sub-Task 13.3: Activation at Newline Position (`at_newline`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | (required) | Text prompt (single string, not a conversation) |
| `layer` | `Union[int, List[int]]` | `15` | Layer index or list of layer indices |
| `swap` | `bool` | `False` | Whether to use swapped chat format (see `ConversationEncoder.format_chat`) |
| `**chat_kwargs` | | | Forwarded to `encoder.format_chat` |

#### Processing

**Step 1 -- Determine layer mode.**

```python
if isinstance(layer, int):
    single_layer_mode = True
    layer_list = [layer]
else:
    single_layer_mode = False
    layer_list = layer
```

**Step 2 -- Format and tokenize.**

Uses `ConversationEncoder.format_chat` which applies the model's chat template (with optional role swapping for contrastive probing setups).

```python
formatted_prompt = self.encoder.format_chat(prompt, swap=swap, **chat_kwargs)

tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
input_ids = tokens["input_ids"].to(self.model.device)
```

**Step 3 -- Locate the newline position.**

Delegates to `_find_newline_position` (see Sub-Task 13.6).

```python
newline_pos = self._find_newline_position(input_ids[0])
```

**Step 4 -- Register hooks that extract a single position.**

Unlike `full_conversation`, each hook only captures the activation at `newline_pos` -- a single vector per layer.

```python
activations = {}
handles = []

def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()
    return hook_fn

model_layers = self.probing_model.get_layers()
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)
```

**Step 5 -- Forward pass with cleanup.**

```python
try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    for handle in handles:
        handle.remove()
```

**Step 6 -- Validate and return.**

After the forward pass, every requested layer must have captured an activation. If any is missing, a `ValueError` is raised.

```python
for layer_idx in layer_list:
    if layer_idx not in activations:
        raise ValueError(f"Failed to extract activation for layer {layer_idx} with prompt: {prompt[:50]}...")

if single_layer_mode:
    return activations[layer_list[0]]
else:
    return activations
```

#### Output

| `layer` argument | Return type | Shape |
|---|---|---|
| `int` (single layer) | `torch.Tensor` | `(hidden_size,)` -- a single activation vector |
| `list` (multi-layer) | `Dict[int, torch.Tensor]` | `{layer_idx: tensor of shape (hidden_size,)}` |

---

### Sub-Task 13.4: Batch Prompt Extraction (`for_prompts`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompts` | `List[str]` | (required) | List of text prompts |
| `layer` | `Union[int, List[int]]` | `15` | Layer index or list of layer indices |
| `swap` | `bool` | `False` | Whether to use swapped chat format |
| `**chat_kwargs` | | | Forwarded to `at_newline` |

#### Processing

This method iterates over prompts and calls `at_newline` for each one. It does **not** use batched inference -- each prompt triggers a separate forward pass. This is because variable-length prompts with different newline positions make true batching impractical at this abstraction level.

**Single-layer mode (`layer` is `int`):**

```python
single_layer_mode = isinstance(layer, int)

if single_layer_mode:
    activations = []
    for prompt in prompts:
        try:
            activation = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
            activations.append(activation)
            print(f"✓ Extracted activation for: {prompt[:50]}...")
        except Exception as e:
            print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")

    return torch.stack(activations) if activations else None
```

Each successful extraction appends a `(hidden_size,)` tensor. Failed prompts are logged and skipped. The results are stacked into a single 2-D tensor.

**Multi-layer mode (`layer` is `list`):**

```python
else:
    layer_activations = {layer_idx: [] for layer_idx in layer}

    for prompt in prompts:
        try:
            activation_dict = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
            for layer_idx in layer:
                layer_activations[layer_idx].append(activation_dict[layer_idx])
            print(f"✓ Extracted activations for: {prompt[:50]}...")
        except Exception as e:
            print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")

    result = {}
    for layer_idx in layer:
        if layer_activations[layer_idx]:
            result[layer_idx] = torch.stack(layer_activations[layer_idx])
        else:
            result[layer_idx] = None

    return result
```

Each layer's collected vectors are stacked independently.

#### Output

| `layer` argument | Return type | Shape |
|---|---|---|
| `int` (single layer) | `torch.Tensor` or `None` | `(num_prompts, hidden_size)` -- `None` if all prompts failed |
| `list` (multi-layer) | `Dict[int, Optional[torch.Tensor]]` | `{layer_idx: tensor of shape (num_prompts, hidden_size)}` -- individual entries may be `None` |

Note: if a prompt fails, it is silently skipped. The resulting tensor may have fewer rows than the length of `prompts`. Errors are printed to stdout.

---

### Sub-Task 13.5: Batch Conversation Extraction (`batch_conversations`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `conversations` | `List[List[Dict[str, str]]]` | (required) | List of conversations; each conversation is a list of `{"role", "content"}` dicts |
| `layer` | `Optional[Union[int, List[int]]]` | `None` | Layer index, list of indices, or `None` for all layers |
| `max_length` | `int` | `4096` | Maximum sequence length; longer conversations are truncated |
| `**chat_kwargs` | | | Forwarded to `encoder.build_batch_turn_spans` |

#### Processing

**Step 1 -- Build turn spans for the entire batch.**

Delegates to `ConversationEncoder.build_batch_turn_spans`, which tokenizes every conversation, computes per-turn spans, and assigns global offsets.

```python
batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
    conversations, **chat_kwargs
)
```

Returns:
- `batch_full_ids`: `List[List[int]]` -- raw token ID lists (variable length).
- `batch_spans`: `List[Dict]` -- per-turn span metadata with local and global start/end indices.
- `span_metadata`: `Dict` -- batching metadata (conversation lengths, offsets, total count).

**Step 2 -- Resolve layer list.**

```python
if isinstance(layer, int):
    layer_list = [layer]
elif isinstance(layer, list):
    layer_list = layer
else:
    layer_list = list(range(len(self.probing_model.get_layers())))
```

**Step 3 -- Pad and truncate sequences into a uniform-length batch.**

The maximum sequence length is capped by `max_length`. If any conversation exceeds this, a warning is logged and it is truncated.

```python
batch_size = len(batch_full_ids)
device = self.model.device

actual_max_len = max(len(ids) for ids in batch_full_ids)
max_seq_len = min(max_length, actual_max_len)

if actual_max_len > max_length:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Truncating sequences: max conversation length {actual_max_len} > max_length {max_length}")

input_ids_batch = []
attention_mask_batch = []

for ids in batch_full_ids:
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]

    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

    input_ids_batch.append(padded_ids)
    attention_mask_batch.append(attention_mask)
```

Padding uses `tokenizer.pad_token_id` (set to `eos_token_id` in `ProbingModel.__init__` if not natively available). Padding is added on the **right** side (note: `ProbingModel` sets `padding_side="left"` on the tokenizer, but this manual padding is right-side).

**Step 4 -- Convert to tensors.**

```python
input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)
```

Shapes: `(batch_size, max_seq_len)`.

**Step 5 -- Register hooks and run batched forward pass.**

Unlike the single-sequence methods, hooks here do **not** strip the batch dimension and do **not** move to CPU -- they store the full batch tensor on-device.

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

**Step 6 -- Stack layer outputs and normalise dtype.**

All layer tensors are moved to a common device and stacked. The result is cast to `bfloat16` if it is not already.

```python
target_device = layer_outputs[layer_list[0]].device
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
])  # (num_layers, batch_size, seq_len, hidden_size)

if selected_activations.dtype != torch.bfloat16:
    selected_activations = selected_activations.to(torch.bfloat16)
```

**Step 7 -- Build batch metadata.**

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

return selected_activations, batch_metadata
```

#### Output

A `tuple` of two elements:

**Element 1 -- `batch_activations`:**

| Field | Type | Shape |
|---|---|---|
| `batch_activations` | `torch.Tensor` (dtype `bfloat16`) | `(num_layers, batch_size, max_seq_len, hidden_size)` |

**Element 2 -- `batch_metadata`:**

| Key | Type | Description |
|---|---|---|
| `conversation_lengths` | `List[int]` | Original (untruncated) token count per conversation |
| `total_conversations` | `int` | Number of conversations in the batch |
| `conversation_offsets` | `List[int]` | Cumulative token offsets for global indexing |
| `max_seq_len` | `int` | The padded sequence length used (`min(max_length, longest_conversation)`) |
| `attention_mask` | `torch.Tensor` | Shape `(batch_size, max_seq_len)`, dtype `long`; `1` for real tokens, `0` for padding |
| `actual_lengths` | `List[int]` | Untruncated token count per conversation (same as `conversation_lengths`) |
| `truncated_lengths` | `List[int]` | Effective token count per conversation after truncation |

---

### Sub-Task 13.6: Newline Position Finder (`_find_newline_position`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `torch.Tensor` | (required) | 1-D tensor of token IDs (a single sequence, not batched) |

#### Processing

The method searches for the newline token in the sequence using a three-level fallback strategy. The goal is to find the position where the model transitions from the user/system prompt to the assistant response area -- typically marked by a newline.

**Attempt 1 -- Double newline (`\n\n`).**

```python
try:
    newline_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[0]
    newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
    if len(newline_positions) > 0:
        return newline_positions[-1].item()  # Use the last occurrence
except:
    pass
```

Encodes `"\n\n"` and takes the first resulting token ID. Searches for all positions in `input_ids` that match and returns the **last** occurrence (closest to the assistant response boundary).

**Attempt 2 -- Single newline (`\n`).**

```python
try:
    newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
    newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
    if len(newline_positions) > 0:
        return newline_positions[-1].item()
except:
    pass
```

Same logic, but with a single `\n`. Used when the tokenizer does not have a dedicated `\n\n` token.

**Attempt 3 -- Last token fallback.**

```python
return len(input_ids) - 1
```

If no newline token is found at all, the last token position is returned.

#### Output

| Return | Type | Description |
|---|---|---|
| position | `int` | Zero-based token index of the target newline (or last token as fallback) |

---

## Hook Mechanism Summary

All extraction methods follow the same hook pattern:

1. **Create a closure** that captures the layer index (and optionally a position index).
2. **Register `register_forward_hook`** on the target `nn.Module` (a transformer decoder layer).
3. **Run a single forward pass** under `torch.inference_mode()`.
4. **Remove all hooks** in a `finally` block to prevent leaks.
5. **Aggregate** the captured tensors (stack, index, or dict).

The hook function signature is always:

```python
def hook_fn(module, input, output):
    act_tensor = output[0] if isinstance(output, tuple) else output
    # ... store act_tensor slice ...
```

The `output[0]` guard handles models that return `(hidden_states, attention_weights, ...)` tuples from their layer forward methods.

---

## Data Flow Diagram

```
                          ActivationExtractor
                         /        |        \         \
                        /         |         \         \
            full_conversation  at_newline  for_prompts  batch_conversations
                 |                |            |              |
                 |         format_chat     (loops over    build_batch_turn_spans
                 |                |        at_newline)         |
                 v                v                            v
           apply_chat_template   _find_newline_position   pad + truncate
                 |                |                            |
                 v                v                            v
              tokenize         tokenize                    tokenize (batch)
                 |                |                            |
                 v                v                            v
           register hooks   register hooks              register hooks
                 |                |                            |
                 v                v                            v
           forward pass     forward pass                 forward pass
           (all tokens)     (newline pos)               (attention_mask)
                 |                |                            |
                 v                v                            v
        (L, T, H) tensor   (H,) vector(s)            (L, B, T, H) tensor
```

Where `L` = layers, `T` = tokens, `H` = hidden size, `B` = batch size.
