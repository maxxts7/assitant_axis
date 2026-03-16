# Task 14: Span Mapper (internals/spans.py)

## Overview

The `SpanMapper` class maps token-level span indices to their corresponding activation tensors extracted from a language model, then aggregates those activations into per-turn mean vectors. It provides three methods:

1. **`map_spans`** -- Standard mapping: takes batched activations and span metadata, returns per-conversation, per-turn mean activations.
2. **`map_spans_no_code`** -- Code-excluding mapping: same as `map_spans` but masks out code-block tokens before computing means.
3. **`mean_all_turn_activations`** -- Single-conversation convenience method: orchestrates the full pipeline (encoding, extraction, aggregation) for one conversation at a single layer.

All methods preserve dtype (bf16) and device (GPU/CPU) consistency throughout.

---

## Sub-Tasks

### Sub-Task 14.1: SpanMapper Construction

#### Input

| Parameter   | Type                      | Description                                    |
|-------------|---------------------------|------------------------------------------------|
| `tokenizer` | HuggingFace `PreTrainedTokenizer` | Tokenizer used for code-block detection in `map_spans_no_code` |

#### Processing

The constructor simply stores the tokenizer reference for later use:

```python
def __init__(self, tokenizer):
    self.tokenizer = tokenizer
```

#### Output

A `SpanMapper` instance with `self.tokenizer` set.

---

### Sub-Task 14.2: `map_spans` -- Map Span Indices to Per-Turn Mean Activations

#### Input

| Parameter           | Type                          | Shape / Format                                          | Description |
|---------------------|-------------------------------|---------------------------------------------------------|-------------|
| `batch_activations` | `torch.Tensor`                | `(num_layers, batch_size, max_seq_len, hidden_size)`    | Activation tensor from the model for the entire batch |
| `batch_spans`       | `List[Dict[str, Any]]`       | Each dict has keys: `conversation_id` (int), `turn` (int), `start` (int), `end` (int) | List of span descriptors with local token indices |
| `batch_metadata`    | `Dict[str, Any]`             | Keys: `total_conversations` (int), `truncated_lengths` (List[int]) | Batching metadata including per-conversation truncated lengths |

#### Processing

**Step 1: Extract tensor dimensions and device/dtype info.**

```python
num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
device = batch_activations.device
dtype = batch_activations.dtype  # Preserve bf16
```

**Step 2: Initialize per-conversation accumulator.**

```python
conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]
```

**Step 3: Group spans by `conversation_id`.**

```python
spans_by_conversation = {}
for span in batch_spans:
    conv_id = span['conversation_id']
    if conv_id not in spans_by_conversation:
        spans_by_conversation[conv_id] = []
    spans_by_conversation[conv_id].append(span)
```

**Step 4: Sort spans within each conversation by `turn` index.**

```python
for conv_id in spans_by_conversation:
    spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])
```

**Step 5: Iterate over each conversation and extract per-turn mean activations.**

For conversations with no spans, an empty tensor is created to maintain dtype/device consistency:

```python
if conv_id not in spans_by_conversation:
    conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
    continue
```

For each span in a conversation:

**Step 5a: Read local start/end indices.**

```python
start_idx = span['start']  # Local start within the conversation
end_idx = span['end']      # Local end within the conversation
```

**Step 5b: Bounds-check against truncated length. Skip the span entirely if it starts beyond the actual length; clamp `end_idx` if it exceeds the actual length.**

```python
actual_length = batch_metadata['truncated_lengths'][conv_id]
if start_idx >= actual_length:
    continue

end_idx = min(end_idx, actual_length)

if start_idx >= end_idx:
    continue
```

**Step 5c: Slice the activation tensor to extract the span.**

```python
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
```

This yields a tensor of shape `(num_layers, span_length, hidden_size)`.

**Step 5d: Compute mean across the token dimension (dim=1). Single-token spans skip the mean call by using `squeeze`.**

```python
span_length = span_activations.size(1)
if span_length > 0:
    if span_length == 1:
        mean_activation = span_activations.squeeze(1)  # (num_layers, hidden_size)
    else:
        mean_activation = span_activations.mean(dim=1)  # (num_layers, hidden_size)
    turn_activations.append(mean_activation)
```

**Step 6: Stack all turn activations for each conversation.**

```python
if turn_activations:
    conversation_activations[conv_id] = torch.stack(turn_activations)
else:
    conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
```

#### Output

| Type                   | Shape per element                            | Description |
|------------------------|----------------------------------------------|-------------|
| `List[torch.Tensor]`   | Each element: `(num_turns, num_layers, hidden_size)` or `(0, num_layers, hidden_size)` for empty conversations | List of length `total_conversations`, where each tensor holds the per-turn mean activations for that conversation |

---

### Sub-Task 14.3: `map_spans_no_code` -- Map Spans Excluding Code Blocks

#### Input

Identical to `map_spans`:

| Parameter           | Type                          | Shape / Format                                          | Description |
|---------------------|-------------------------------|---------------------------------------------------------|-------------|
| `batch_activations` | `torch.Tensor`                | `(num_layers, batch_size, max_seq_len, hidden_size)`    | Activation tensor from the model for the entire batch |
| `batch_spans`       | `List[Dict[str, Any]]`       | Each dict has keys: `conversation_id` (int), `turn` (int), `start` (int), `end` (int), `text` (str) | Span descriptors -- **must include `text` key** for code detection |
| `batch_metadata`    | `Dict[str, Any]`             | Keys: `total_conversations` (int), `truncated_lengths` (List[int]) | Batching metadata |

Note: Unlike `map_spans`, each span dict **must** contain a `text` key holding the raw text of that span (used for code-block detection).

#### Processing

Steps 1--5b are identical to `map_spans` (dimension extraction, grouping, sorting, bounds-checking). The differences begin after the span activation slice is extracted.

**Step 5c (same): Slice activations.**

```python
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
```

**Step 5d (new): Instantiate a `ConversationEncoder` and compute a code-block exclusion mask.**

A `ConversationEncoder` is instantiated (import deferred to runtime to avoid circular imports):

```python
from .conversation import ConversationEncoder
encoder = ConversationEncoder(self.tokenizer)
```

The code-block mask is obtained from the span text:

```python
text = span['text']
exclude_mask = encoder.code_block_token_mask(text)
```

**Step 5e (new): Reconcile mask length with actual span length.**

If the mask length differs from the span length (due to tokenization edge cases), it is truncated or zero-padded:

```python
span_length = span_activations.size(1)
if len(exclude_mask) != span_length:
    if len(exclude_mask) > span_length:
        exclude_mask = exclude_mask[:span_length]
    else:
        padding = torch.zeros(span_length - len(exclude_mask), dtype=torch.bool)
        exclude_mask = torch.cat([exclude_mask, padding])
```

**Step 5f (new): Invert mask to get the include mask and compute mean over non-code tokens only.**

```python
include_mask = ~exclude_mask

if include_mask.any():
    included_activations = span_activations[:, include_mask, :]  # (num_layers, included_tokens, hidden_size)

    if included_activations.size(1) == 1:
        mean_activation = included_activations.squeeze(1)  # (num_layers, hidden_size)
    else:
        mean_activation = included_activations.mean(dim=1)  # (num_layers, hidden_size)
    turn_activations.append(mean_activation)
else:
    # All tokens are code blocks - skip this turn
    continue
```

**Step 6 (same): Stack turn activations.**

```python
if turn_activations:
    conversation_activations[conv_id] = torch.stack(turn_activations)
else:
    conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
```

#### Output

Same format as `map_spans`:

| Type                   | Shape per element                            | Description |
|------------------------|----------------------------------------------|-------------|
| `List[torch.Tensor]`   | Each element: `(num_turns, num_layers, hidden_size)` or `(0, num_layers, hidden_size)` | Per-conversation mean activations with code-block tokens excluded from the mean computation. Turns that consist entirely of code are dropped. |

---

### Sub-Task 14.4: `mean_all_turn_activations` -- Single-Conversation End-to-End Mean Activations

#### Input

| Parameter      | Type                              | Default   | Description |
|----------------|-----------------------------------|-----------|-------------|
| `probing_model`| `ProbingModel`                    | (required)| Model instance used for activation extraction |
| `encoder`      | `ConversationEncoder`             | (required)| Encoder instance for building turn spans |
| `conversation` | `List[Dict[str, str]]`            | (required)| Conversation as a list of dicts with `role` and `content` keys |
| `layer`        | `int`                             | `15`      | Layer index to extract activations from |
| `chat_format`  | `bool`                            | `True`    | Whether to apply chat template formatting |
| `**chat_kwargs`| keyword arguments                 | --        | Additional arguments forwarded to `apply_chat_template` |

#### Processing

**Step 1: Build turn spans from the conversation.**

Uses the `ConversationEncoder.build_turn_spans` method to tokenize the conversation and identify the token-index boundaries for each turn:

```python
full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
```

`full_ids` is the full token ID sequence; `spans` is a list of dicts with `start` and `end` keys.

**Step 2: Import `ActivationExtractor` (deferred to avoid circular imports) and extract full-conversation activations.**

```python
from .activations import ActivationExtractor
extractor = ActivationExtractor(probing_model, encoder)

activations = extractor.full_conversation(
    conversation, layer=layer, chat_format=chat_format, **chat_kwargs
)
```

**Step 3: Handle multi-layer vs. single-layer activation format.**

If the extractor returns a 3D tensor `(num_layers, num_tokens, hidden_size)`, squeeze to the single requested layer:

```python
if activations.ndim == 3:  # (num_layers, num_tokens, hidden_size)
    activations = activations[0]  # Take the first (and only) layer
```

After this step, `activations` has shape `(num_tokens, hidden_size)`.

**Step 4: Iterate over spans and compute per-turn mean activations.**

For each span, slice the activation matrix and compute the mean over the token dimension:

```python
for span in spans:
    start_idx = span['start']
    end_idx = span['end']

    if start_idx < end_idx and end_idx <= activations.shape[0]:
        turn_activations = activations[start_idx:end_idx, :]  # (turn_tokens, hidden_size)
        mean_activation = turn_activations.mean(dim=0)  # (hidden_size,)
        turn_mean_activations.append(mean_activation)
```

**Step 5: Stack or return empty tensor.**

```python
if not turn_mean_activations:
    return torch.empty(0, activations.shape[1] if activations.ndim > 1 else 0)

return torch.stack(turn_mean_activations)
```

#### Output

| Type            | Shape                        | Description |
|-----------------|------------------------------|-------------|
| `torch.Tensor`  | `(num_turns, hidden_size)`   | Mean activation vector for each turn in chronological order |
| `torch.Tensor`  | `(0, hidden_size)` or `(0, 0)` | Empty tensor if no valid turns were found |

---

## Cross-Cutting Concerns

### Dtype and Device Consistency

All three methods preserve the original dtype (typically `bf16`) and device (GPU) of the input activations when creating empty fallback tensors:

```python
torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
```

This prevents downstream errors from dtype or device mismatches when results are concatenated or compared.

### Truncation Handling

`map_spans` and `map_spans_no_code` guard against spans that reference token positions beyond the actual (truncated) sequence length stored in `batch_metadata['truncated_lengths']`. Spans that start beyond the truncated length are skipped entirely; spans that extend past it are clamped.

### Circular Import Avoidance

Both `map_spans_no_code` and `mean_all_turn_activations` use deferred (in-function) imports:

- `map_spans_no_code` imports `ConversationEncoder` from `.conversation`
- `mean_all_turn_activations` imports `ActivationExtractor` from `.activations`

This avoids circular dependency chains between the `spans`, `conversation`, and `activations` modules.

### Code Block Masking (map_spans_no_code only)

The `encoder.code_block_token_mask(text)` call returns a boolean tensor where `True` marks tokens that belong to code blocks (e.g., content between triple-backtick fences). The mask is inverted to select only natural-language tokens for the mean computation. If all tokens in a span are code, that turn is dropped from the output.
