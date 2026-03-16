# `assistant_axis/internals/spans.py`

## Overview

This file defines the `SpanMapper` class, which is responsible for mapping token span indices to their corresponding activation tensors extracted from a language model, and computing per-turn mean activations within conversations. It provides three core methods:

1. **`map_spans`** -- Maps batch-level activations to per-conversation, per-turn mean activations.
2. **`map_spans_no_code`** -- Same as `map_spans`, but excludes code block tokens before computing means.
3. **`mean_all_turn_activations`** -- A higher-level convenience method that builds spans, extracts activations, and computes per-turn means for a single conversation.

---

## Line-by-Line Explanation

### Module Docstring and Imports (Lines 1--9)

```python
"""SpanMapper - Map token spans to activations and compute per-turn aggregates."""
```

The module-level docstring summarises the purpose of the file: mapping token spans to activations and aggregating them per conversation turn.

```python
from __future__ import annotations
```

Enables PEP 604-style postponed evaluation of annotations so that type hints referring to classes defined later (or in other modules) are treated as strings at runtime rather than being eagerly resolved. This avoids `NameError` issues for forward references.

```python
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch
```

Imports standard typing utilities (`Any`, `Dict`, `List`, `Optional`, `TYPE_CHECKING`) and PyTorch. `TYPE_CHECKING` is a special constant that is `True` only when a static type checker (e.g. mypy) is analysing the code, and `False` at runtime.

```python
if TYPE_CHECKING:
    from .conversation import ConversationEncoder
```

Conditionally imports `ConversationEncoder` only during static type checking. This prevents a circular import at runtime (since `conversation.py` may itself import from this module or a sibling), while still allowing type checkers and IDE autocompletion to understand the type.

---

### Class Definition and Constructor (Lines 12--29)

```python
class SpanMapper:
    """
    Maps token span indices to activations and computes per-turn aggregations.

    Handles:
    - Mapping spans to activation tensors
    - Excluding code blocks from aggregation
    - Computing mean activations per turn
    """
```

Declares the `SpanMapper` class with a docstring outlining its three responsibilities.

```python
    def __init__(self, tokenizer):
        """
        Initialize the span mapper.

        Args:
            tokenizer: HuggingFace tokenizer for code block detection
        """
        self.tokenizer = tokenizer
```

The constructor takes a single argument, `tokenizer`, which is a HuggingFace tokenizer instance. It is stored as an instance attribute so that the `map_spans_no_code` method can later use it to detect code block tokens.

---

### `map_spans` Method (Lines 31--117)

#### Signature and Docstring (Lines 31--48)

```python
    def map_spans(
        self,
        batch_activations: torch.Tensor,
        batch_spans: List[Dict[str, Any]],
        batch_metadata: Dict[str, Any],
    ) -> List[torch.Tensor]:
```

Defines `map_spans` with three parameters:

- `batch_activations`: A 4-D tensor of shape `(num_layers, batch_size, max_seq_len, hidden_size)` containing the raw model activations for an entire batch of conversations.
- `batch_spans`: A list of dictionaries, each describing one span. Each dict contains keys like `conversation_id`, `turn`, `start`, and `end`.
- `batch_metadata`: A dictionary carrying batching information such as `total_conversations` and `truncated_lengths`.

The method returns a list of tensors, one per conversation, each of shape `(num_turns, num_layers, hidden_size)`.

#### Unpacking Shape and Device (Lines 49--51)

```python
        num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
        device = batch_activations.device
        dtype = batch_activations.dtype  # Preserve bf16
```

Destructures the four dimensions of `batch_activations` into named variables. Captures the `device` (e.g. `cuda:0` or `cpu`) and `dtype` (e.g. `torch.bfloat16`) so that any newly created tensors will live on the same device and use the same data type, avoiding silent precision changes or device mismatches.

#### Initialising Output Structure (Line 53)

```python
        conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]
```

Creates a list with one empty sub-list per conversation. Each sub-list will later hold per-turn mean activation tensors (or be replaced by a stacked tensor).

#### Grouping Spans by Conversation (Lines 56--61)

```python
        spans_by_conversation = {}
        for span in batch_spans:
            conv_id = span['conversation_id']
            if conv_id not in spans_by_conversation:
                spans_by_conversation[conv_id] = []
            spans_by_conversation[conv_id].append(span)
```

Iterates through the flat list of span dicts and groups them into a dictionary keyed by `conversation_id`. This allows efficient lookup later when processing each conversation.

#### Sorting Spans by Turn (Lines 64--65)

```python
        for conv_id in spans_by_conversation:
            spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])
```

Within each conversation, sorts the spans by their `turn` index to ensure they are processed in chronological order.

#### Main Loop -- Iterating Over Conversations (Lines 68--72)

```python
        for conv_id in range(batch_metadata['total_conversations']):
            if conv_id not in spans_by_conversation:
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
                continue
```

Iterates over every conversation ID from `0` to `total_conversations - 1`. If a conversation has no spans (e.g. it was entirely filtered out), it assigns an empty tensor with shape `(0, num_layers, hidden_size)` to maintain a consistent output structure, preserving the correct `dtype` and `device`.

#### Setting Up Per-Turn Processing (Lines 74--75)

```python
            spans = spans_by_conversation[conv_id]
            turn_activations = []
```

Retrieves the sorted list of spans for this conversation and initialises an empty list to accumulate per-turn mean activations.

#### Processing Each Span (Lines 77--108)

```python
            for span in spans:
                start_idx = span['start']
                end_idx = span['end']
```

For each span, retrieves the `start` and `end` token indices. These are local indices within the conversation's portion of the batch.

```python
                actual_length = batch_metadata['truncated_lengths'][conv_id]
                if start_idx >= actual_length:
                    continue
```

Looks up the actual (possibly truncated) token length of this conversation from metadata. If the span starts beyond the truncated length, the span is entirely outside the available data and is skipped.

```python
                end_idx = min(end_idx, actual_length)
```

Clamps the end index to the actual length so that no out-of-bounds indexing occurs if the span was only partially truncated.

```python
                if start_idx >= end_idx:
                    continue
```

After clamping, if the span has zero or negative length, it is invalid and skipped.

```python
                span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
```

Slices the 4-D activation tensor to extract the activations for this specific span. The result has shape `(num_layers, span_length, hidden_size)`.

```python
                span_length = span_activations.size(1)
                if span_length > 0:
                    if span_length == 1:
                        mean_activation = span_activations.squeeze(1)
                    else:
                        mean_activation = span_activations.mean(dim=1)
                    turn_activations.append(mean_activation)
```

Computes the mean activation across all tokens in the span (dimension 1). There is an optimisation: if the span contains exactly one token, `squeeze(1)` is used instead of `mean(dim=1)` to avoid the overhead of a reduction operation. The result in both cases has shape `(num_layers, hidden_size)` and is appended to `turn_activations`.

#### Finalising Per-Conversation Output (Lines 110--115)

```python
            if turn_activations:
                conversation_activations[conv_id] = torch.stack(turn_activations)
            else:
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
```

If any valid turns were collected, `torch.stack` combines them into a single tensor of shape `(num_turns, num_layers, hidden_size)`. Otherwise, an empty tensor with the correct shape, dtype, and device is assigned.

#### Return (Line 117)

```python
        return conversation_activations
```

Returns the list of per-conversation activation tensors.

---

### `map_spans_no_code` Method (Lines 119--232)

#### Signature and Docstring (Lines 119--136)

```python
    def map_spans_no_code(
        self,
        batch_activations: torch.Tensor,
        batch_spans: List[Dict[str, Any]],
        batch_metadata: Dict[str, Any],
    ) -> List[torch.Tensor]:
```

Identical signature to `map_spans`. The difference is that this method excludes tokens that fall within code blocks before computing the per-turn mean activations.

#### Shape Unpacking and Initialisation (Lines 137--153)

```python
        num_layers, batch_size, max_seq_len, hidden_size = batch_activations.shape
        device = batch_activations.device
        dtype = batch_activations.dtype

        conversation_activations = [[] for _ in range(batch_metadata['total_conversations'])]

        spans_by_conversation = {}
        for span in batch_spans:
            conv_id = span['conversation_id']
            if conv_id not in spans_by_conversation:
                spans_by_conversation[conv_id] = []
            spans_by_conversation[conv_id].append(span)

        for conv_id in spans_by_conversation:
            spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])
```

This block is identical to the corresponding block in `map_spans`: it unpacks the tensor dimensions, captures device/dtype, initialises the output list, groups spans by conversation, and sorts them by turn.

#### Runtime Import of ConversationEncoder (Lines 155--157)

```python
        from .conversation import ConversationEncoder
        encoder = ConversationEncoder(self.tokenizer)
```

Imports `ConversationEncoder` at runtime (inside the method body) to avoid a circular import that would occur if it were imported at module level. A new `ConversationEncoder` instance is created using the tokenizer stored during `__init__`. This encoder provides the `code_block_token_mask` method needed to identify which tokens belong to code blocks.

#### Main Loop -- Same Structure as `map_spans` (Lines 160--185)

```python
        for conv_id in range(batch_metadata['total_conversations']):
            if conv_id not in spans_by_conversation:
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)
                continue

            spans = spans_by_conversation[conv_id]
            turn_activations = []

            for span in spans:
                start_idx = span['start']
                end_idx = span['end']

                actual_length = batch_metadata['truncated_lengths'][conv_id]
                if start_idx >= actual_length:
                    continue

                end_idx = min(end_idx, actual_length)

                if start_idx >= end_idx:
                    continue

                span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
```

This section mirrors `map_spans` exactly: iterating over conversations, skipping empty ones, iterating over spans, performing bounds checking, clamping, and slicing activations.

#### Code Block Exclusion Logic (Lines 192--223)

```python
                text = span['text']
                exclude_mask = encoder.code_block_token_mask(text)
```

Retrieves the raw text of the span and passes it to `code_block_token_mask`, which returns a boolean tensor where `True` indicates a token that is part of a code block and should be excluded.

```python
                span_length = span_activations.size(1)
                if len(exclude_mask) != span_length:
                    if len(exclude_mask) > span_length:
                        exclude_mask = exclude_mask[:span_length]
                    else:
                        padding = torch.zeros(span_length - len(exclude_mask), dtype=torch.bool)
                        exclude_mask = torch.cat([exclude_mask, padding])
```

Handles a potential mismatch between the length of the exclude mask and the actual span length. This can happen due to tokenization differences (e.g. the text was tokenized independently by `code_block_token_mask` versus how it was tokenized in the batch). If the mask is longer, it is truncated. If shorter, it is padded with `False` values (meaning those extra tokens are included, not excluded).

```python
                include_mask = ~exclude_mask
```

Inverts the mask so that `True` now marks tokens to include (non-code tokens).

```python
                if include_mask.any():
                    included_activations = span_activations[:, include_mask, :]

                    if included_activations.size(1) == 1:
                        mean_activation = included_activations.squeeze(1)
                    else:
                        mean_activation = included_activations.mean(dim=1)
                    turn_activations.append(mean_activation)
                else:
                    continue
```

If there are any non-code tokens, the activations for those tokens are selected using boolean indexing. The mean is then computed across only these tokens, using the same single-token optimisation as `map_spans`. If every token in the span is a code block token, the entire span is skipped.

#### Finalising and Returning (Lines 225--232)

```python
            if turn_activations:
                conversation_activations[conv_id] = torch.stack(turn_activations)
            else:
                conversation_activations[conv_id] = torch.empty(0, num_layers, hidden_size, dtype=dtype, device=device)

        return conversation_activations
```

Identical to `map_spans`: stacks per-turn activations into a single tensor per conversation, or assigns an empty tensor if no valid turns exist.

---

### `mean_all_turn_activations` Method (Lines 234--291)

#### Signature and Docstring (Lines 234--255)

```python
    def mean_all_turn_activations(
        self,
        probing_model,
        encoder: 'ConversationEncoder',
        conversation: List[Dict[str, str]],
        layer: int = 15,
        chat_format: bool = True,
        **chat_kwargs,
    ) -> torch.Tensor:
```

A higher-level method for computing per-turn mean activations from a single conversation. Parameters:

- `probing_model`: A `ProbingModel` instance used to run the language model and extract activations.
- `encoder`: A `ConversationEncoder` instance for building turn spans.
- `conversation`: The conversation as a list of dicts, each with `'role'` and `'content'` keys.
- `layer`: Which transformer layer to extract activations from (defaults to 15).
- `chat_format`: Whether to apply the chat template (defaults to `True`).
- `**chat_kwargs`: Any additional keyword arguments forwarded to `apply_chat_template`.

Returns a tensor of shape `(num_turns, hidden_size)`.

#### Building Turn Spans (Line 257)

```python
        full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
```

Calls `build_turn_spans` on the encoder, which tokenizes the conversation and returns:
- `full_ids`: The full token ID sequence for the conversation.
- `spans`: A list of span dicts indicating the start and end token indices for each turn.

#### Extracting Full Activations (Lines 259--266)

```python
        from .activations import ActivationExtractor
        extractor = ActivationExtractor(probing_model, encoder)

        activations = extractor.full_conversation(
            conversation, layer=layer, chat_format=chat_format, **chat_kwargs
        )
```

Imports `ActivationExtractor` at runtime (again to avoid circular imports) and creates an instance. Then calls `full_conversation` to run the entire conversation through the model and extract activations at the specified layer. The result is a tensor of activations for all tokens.

#### Handling Multi-Layer Format (Lines 269--270)

```python
        if activations.ndim == 3:  # (num_layers, num_tokens, hidden_size)
            activations = activations[0]  # Take the first (and only) layer
```

If `full_conversation` returns a 3-D tensor (which would be the case if it returns activations with a leading layer dimension), this collapses it to 2-D by selecting the first (and presumably only) layer, resulting in shape `(num_tokens, hidden_size)`.

#### Computing Per-Turn Means (Lines 273--283)

```python
        turn_mean_activations = []

        for span in spans:
            start_idx = span['start']
            end_idx = span['end']

            if start_idx < end_idx and end_idx <= activations.shape[0]:
                turn_activations = activations[start_idx:end_idx, :]
                mean_activation = turn_activations.mean(dim=0)
                turn_mean_activations.append(mean_activation)
```

Iterates over each span. For each valid span (where `start < end` and the end does not exceed the number of tokens), slices the activations to get a `(turn_tokens, hidden_size)` tensor, computes the mean across tokens (dimension 0) to get a single `(hidden_size,)` vector, and appends it.

#### Returning Results (Lines 285--290)

```python
        if not turn_mean_activations:
            return torch.empty(0, activations.shape[1] if activations.ndim > 1 else 0)

        return torch.stack(turn_mean_activations)
```

If no valid turns were found, returns an empty tensor with shape `(0, hidden_size)`. Otherwise, stacks all per-turn mean vectors into a `(num_turns, hidden_size)` tensor and returns it.
