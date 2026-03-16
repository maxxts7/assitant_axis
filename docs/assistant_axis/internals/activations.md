# `assistant_axis/internals/activations.py`

## Overview

This file defines the `ActivationExtractor` class, which extracts hidden-state activations from the internal layers of a transformer model. It uses PyTorch forward hooks to intercept intermediate outputs during a model's forward pass. The class supports several extraction modes: extracting activations for an entire conversation, extracting the activation at a specific newline-token position, processing batches of prompts one at a time, and efficiently processing batches of conversations with padding and truncation.

---

## Line-by-Line Explanation

### Line 1: Module docstring

```python
"""ActivationExtractor - Extract hidden state activations from model layers."""
```

A module-level docstring summarising the purpose of the file.

---

### Line 3: Future annotations import

```python
from __future__ import annotations
```

Enables PEP 604 / PEP 563 postponed evaluation of annotations so that type hints are treated as strings at runtime. This allows forward references (such as `'ProbingModel'`) without raising `NameError` and also permits the use of newer annotation syntax on older Python versions.

---

### Line 5: Standard library and third-party imports

```python
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import torch
```

- `Dict, List, Optional, Union` -- generic type-hint aliases used throughout the class signatures.
- `TYPE_CHECKING` -- a constant that is `True` only when a static type checker (e.g. mypy) is analysing the code, and `False` at runtime. Used on the next lines to avoid circular imports.
- `torch` -- PyTorch, the tensor library used for model inference and activation storage.

---

### Lines 8-10: Conditional type-checking imports

```python
if TYPE_CHECKING:
    from .model import ProbingModel
    from .conversation import ConversationEncoder
```

These imports are only executed during static analysis, not at runtime. This avoids circular-import issues between this module and `model.py` / `conversation.py` while still giving type checkers the information they need.

---

### Lines 13-22: Class definition and docstring

```python
class ActivationExtractor:
    """
    Extract activations from model layers using forward hooks.

    This class handles:
    - Full conversation activation extraction
    - Activation at specific positions (e.g., newline)
    - Batch prompt processing
    - Efficient batch conversation processing
    """
```

Declares the `ActivationExtractor` class and documents its four main responsibilities.

---

### Lines 24-35: `__init__`

```python
    def __init__(self, probing_model: 'ProbingModel', encoder: 'ConversationEncoder'):
        """
        Initialize the activation extractor.

        Args:
            probing_model: ProbingModel instance with loaded model and tokenizer
            encoder: ConversationEncoder for formatting conversations
        """
        self.model = probing_model.model
        self.tokenizer = probing_model.tokenizer
        self.probing_model = probing_model
        self.encoder = encoder
```

- **`probing_model`** -- a wrapper object that holds the loaded transformer model and its tokenizer. The constructor unpacks `probing_model.model` and `probing_model.tokenizer` into instance attributes for convenience.
- **`self.probing_model`** -- kept as a reference so that helper methods like `get_layers()` can be called later.
- **`self.encoder`** -- a `ConversationEncoder` used to format raw conversations into the chat-template strings the model expects.

---

### Lines 37-43: `full_conversation` signature

```python
    def full_conversation(
        self,
        conversation: Union[str, List[Dict[str, str]]],
        layer: Optional[Union[int, List[int]]] = None,
        chat_format: bool = True,
        **chat_kwargs,
    ) -> torch.Tensor:
```

Method for extracting activations over every token in a conversation. Parameters:

- **`conversation`** -- either a raw string or a list of role/content message dicts.
- **`layer`** -- which layer(s) to extract from. `None` means all layers.
- **`chat_format`** -- whether to wrap the input through the tokenizer's chat template.
- **`**chat_kwargs`** -- forwarded to `apply_chat_template` (e.g. special tokens, system prompts).

Returns a tensor whose shape depends on whether a single layer or multiple layers were requested.

---

### Lines 57-66: Layer specification normalisation

```python
        # Handle backward compatibility
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

Normalises the `layer` argument into a uniform list (`layer_list`) regardless of how it was provided:

- **Single int** -- wrapped into a one-element list; `single_layer_mode` is set so the return value can be squeezed back to a single-layer tensor.
- **List of ints** -- used directly.
- **`None`** -- expanded to every layer index by querying `get_layers()`.

---

### Lines 68-75: Conversation formatting

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

- If `chat_format` is `True` and a bare string was given, it is first wrapped into a single-turn user message.
- `apply_chat_template` converts the message list into the model-specific prompt string (e.g. with `<|user|>` / `<|assistant|>` markers), returning raw text (`tokenize=False`).
- If `chat_format` is `False`, the input is used as-is (assumed to be already formatted).

---

### Lines 77-79: Tokenisation

```python
        # Tokenize
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self.model.device)
```

- Tokenises the formatted string into PyTorch tensors (`return_tensors="pt"`).
- `add_special_tokens=False` prevents the tokenizer from prepending BOS / appending EOS tokens (the chat template already handles those).
- The resulting `input_ids` tensor is moved to the same device (CPU or GPU) as the model.

---

### Lines 81-83: Activation and handle storage

```python
        # Dictionary to store activations from multiple layers
        activations = []
        handles = []
```

- `activations` -- a list that will collect one tensor per hooked layer during the forward pass.
- `handles` -- stores the hook handles returned by `register_forward_hook` so they can be removed later.

---

### Lines 86-91: Hook factory

```python
        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                activations.append(act_tensor[0, :, :].cpu())
            return hook_fn
```

A closure factory that produces a hook function for each layer:

- **`output`** -- the raw output from the layer module. Some architectures return a tuple (hidden states, attention weights, ...); taking `output[0]` handles both cases.
- **`act_tensor[0, :, :]`** -- selects batch element 0 (there is only one), keeping all tokens and all hidden dimensions. The result has shape `(num_tokens, hidden_size)`.
- **`.cpu()`** -- moves the tensor to CPU immediately to free GPU memory.

The outer function `create_hook_fn` exists to capture `layer_idx` by value in each closure (avoiding the common late-binding pitfall with loops), although in this particular implementation `layer_idx` is not used inside the hook -- the ordering in the `activations` list implicitly tracks it.

---

### Lines 93-98: Registering hooks

```python
        # Register hooks for all target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)
```

- `get_layers()` returns the list of transformer layer modules (e.g. `model.layers` for LLaMA-style architectures).
- For each requested layer index, a forward hook is attached. The hook will fire automatically when data flows through that layer during `model(input_ids)`.

---

### Lines 100-106: Forward pass with cleanup

```python
        try:
            with torch.inference_mode():
                _ = self.model(input_ids)  # Full forward pass to capture all layers
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()
```

- **`torch.inference_mode()`** -- disables gradient tracking and certain checks, making the forward pass faster and more memory-efficient.
- The model output itself is discarded (`_ =`); only the side-effects captured by the hooks matter.
- The `finally` block guarantees hooks are removed even if the forward pass raises an exception, preventing leaked hooks from corrupting future runs.

---

### Lines 108-114: Stack and return

```python
        activations = torch.stack(activations)

        # Return format based on input type
        if single_layer_mode:
            return activations[0]  # Return single layer
        else:
            return activations
```

- `torch.stack` combines the list of per-layer tensors into a single tensor of shape `(num_layers, num_tokens, hidden_size)`.
- If the caller originally passed a single integer for `layer`, the leading dimension is removed and a 2-D tensor `(num_tokens, hidden_size)` is returned for convenience.

---

### Lines 116-122: `at_newline` signature

```python
    def at_newline(
        self,
        prompt: str,
        layer: Union[int, List[int]] = 15,
        swap: bool = False,
        **chat_kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
```

Extracts the activation vector at the position of the newline token within the formatted prompt. Parameters:

- **`prompt`** -- the raw text prompt.
- **`layer`** -- defaults to layer 15.
- **`swap`** -- passed to `format_chat`; when `True`, the roles of user and assistant are swapped.
- Returns either a single tensor or a dictionary mapping layer indices to tensors.

---

### Lines 136-142: Layer normalisation (at_newline)

```python
        # Handle backward compatibility
        if isinstance(layer, int):
            single_layer_mode = True
            layer_list = [layer]
        else:
            single_layer_mode = False
            layer_list = layer
```

Same pattern as `full_conversation`: normalises the layer argument into a list and records whether the caller asked for a single layer.

---

### Lines 144-149: Format and tokenise

```python
        # Format as chat
        formatted_prompt = self.encoder.format_chat(prompt, swap=swap, **chat_kwargs)

        # Tokenize
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self.model.device)
```

- `format_chat` uses the `ConversationEncoder` to build a properly templated prompt string, optionally with swapped roles.
- The result is tokenised and moved to the model device, just like in `full_conversation`.

---

### Lines 151-152: Find newline position

```python
        # Find newline position
        newline_pos = self._find_newline_position(input_ids[0])
```

Delegates to `_find_newline_position` (defined at the bottom of the class) to locate the token index of the last newline in the sequence. This position is where the activation will be sampled.

---

### Lines 154-164: Hook factory for newline extraction

```python
        # Dictionary to store activations from multiple layers
        activations = {}
        handles = []

        # Create hooks for all requested layers
        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()
            return hook_fn
```

Similar to the `full_conversation` hook factory, but with two differences:

1. `activations` is a **dict** keyed by layer index (not a list), because the results are returned as a dict when multiple layers are requested.
2. Instead of slicing all token positions (`[:, :]`), only the single position at `newline_pos` is extracted: `act_tensor[0, newline_pos, :]` yields a 1-D vector of shape `(hidden_size,)`.

---

### Lines 166-179: Hook registration and forward pass

```python
        # Register hooks for all target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)

        try:
            with torch.inference_mode():
                _ = self.model(input_ids)  # Full forward pass to capture all layers
        finally:
            # Clean up all hooks
            for handle in handles:
                handle.remove()
```

Identical pattern to `full_conversation`: register hooks, run the forward pass under `inference_mode`, and clean up in a `finally` block.

---

### Lines 181-190: Validation and return

```python
        # Check that we captured all requested activations
        for layer_idx in layer_list:
            if layer_idx not in activations:
                raise ValueError(f"Failed to extract activation for layer {layer_idx} with prompt: {prompt[:50]}...")

        # Return format based on input type
        if single_layer_mode:
            return activations[layer_list[0]]
        else:
            return activations
```

- A safety check ensures every requested layer actually produced an activation. If a hook did not fire (e.g. due to an invalid layer index), a `ValueError` is raised with a diagnostic message.
- Single-layer callers get a plain tensor; multi-layer callers get the full dictionary.

---

### Lines 192-198: `for_prompts` signature

```python
    def for_prompts(
        self,
        prompts: List[str],
        layer: Union[int, List[int]] = 15,
        swap: bool = False,
        **chat_kwargs,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
```

A convenience method that iterates over a list of prompts and calls `at_newline` on each one, collecting the results. Returns either a stacked tensor (single-layer mode) or a dictionary of stacked tensors (multi-layer mode).

---

### Lines 212-226: Single-layer prompt loop

```python
        # Handle backward compatibility
        single_layer_mode = isinstance(layer, int)

        if single_layer_mode:
            # Single layer mode - maintain original behavior
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

- Iterates over every prompt, calling `at_newline` for the single requested layer.
- Prints a progress indicator for each prompt (success or failure).
- Failed prompts are skipped (their activation is not appended), so the returned tensor may have fewer rows than the input list.
- If every prompt failed, `None` is returned instead of an empty tensor.

---

### Lines 228-249: Multi-layer prompt loop

```python
        else:
            # Multi-layer mode - extract all layers in single forward passes
            layer_activations = {layer_idx: [] for layer_idx in layer}

            for prompt in prompts:
                try:
                    activation_dict = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
                    for layer_idx in layer:
                        layer_activations[layer_idx].append(activation_dict[layer_idx])
                    print(f"✓ Extracted activations for: {prompt[:50]}...")
                except Exception as e:
                    print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")

            # Convert lists to tensors for each layer
            result = {}
            for layer_idx in layer:
                if layer_activations[layer_idx]:
                    result[layer_idx] = torch.stack(layer_activations[layer_idx])
                else:
                    result[layer_idx] = None

            return result
```

- Pre-allocates a dictionary of empty lists, one per requested layer.
- Each call to `at_newline` returns a dict of activations; the individual tensors are distributed into the per-layer lists.
- After the loop, each list is stacked into a tensor of shape `(num_prompts, hidden_size)`, producing a dict like `{15: tensor, 20: tensor, ...}`.
- Layers for which every prompt failed get `None`.

---

### Lines 251-257: `batch_conversations` signature

```python
    def batch_conversations(
        self,
        conversations: List[List[Dict[str, str]]],
        layer: Optional[Union[int, List[int]]] = None,
        max_length: int = 4096,
        **chat_kwargs,
    ) -> tuple[torch.Tensor, Dict]:
```

An efficient batched extraction method. Instead of running one forward pass per conversation (as `for_prompts` does), this method pads all conversations to the same length and processes them in a single batched forward pass.

- **`conversations`** -- a list of conversations, each being a list of message dicts.
- **`max_length`** -- the maximum sequence length; longer sequences are truncated.
- Returns a tuple of the activation tensor and metadata about the batch.

---

### Lines 272-275: Build batch turn spans

```python
        # Get tokenized conversations and spans
        batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
            conversations, **chat_kwargs
        )
```

Delegates to the encoder's `build_batch_turn_spans` method, which:

1. Tokenises every conversation.
2. Identifies the token spans corresponding to each conversational turn.
3. Returns the raw token ID lists (`batch_full_ids`), span boundaries (`batch_spans`), and aggregate metadata (`span_metadata`).

---

### Lines 277-283: Layer normalisation (batch)

```python
        # Handle layer specification
        if isinstance(layer, int):
            layer_list = [layer]
        elif isinstance(layer, list):
            layer_list = layer
        else:
            layer_list = list(range(len(self.probing_model.get_layers())))
```

Same normalisation as the other methods, converting the `layer` argument into a list.

---

### Lines 285-297: Compute max sequence length and check for truncation

```python
        # Prepare batch tensors
        batch_size = len(batch_full_ids)
        device = self.model.device

        # Find max length and pad sequences - ALWAYS respect max_length limit
        actual_max_len = max(len(ids) for ids in batch_full_ids)
        max_seq_len = min(max_length, actual_max_len)

        # Log warning if truncation will occur
        if actual_max_len > max_length:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Truncating sequences: max conversation length {actual_max_len} > max_length {max_length}")
```

- `actual_max_len` is the length of the longest conversation in the batch.
- `max_seq_len` is capped by `max_length` to prevent excessive memory use.
- If any conversation will be truncated, a warning is logged.

---

### Lines 299-312: Padding and attention mask construction

```python
        input_ids_batch = []
        attention_mask_batch = []

        for ids in batch_full_ids:
            # Truncate if too long
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]

            # Pad to max length
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
            attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(attention_mask)
```

For each conversation's token IDs:

1. **Truncate** if the sequence exceeds `max_seq_len`.
2. **Right-pad** with the tokenizer's `pad_token_id` to reach `max_seq_len`.
3. Build an **attention mask** with `1` for real tokens and `0` for padding, so the model ignores pad tokens during self-attention.

---

### Lines 314-316: Convert to tensors

```python
        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)
```

The Python lists of lists are converted to 2-D `torch.long` tensors and placed on the model's device.

---

### Lines 318-334: Hook setup for batch extraction

```python
        # Extract activations using hooks (more reliable than output_hidden_states)
        layer_outputs = {}  # Will store {layer_idx: tensor} after forward pass
        handles = []

        def create_hook_fn(layer_idx):
            def hook_fn(module, input, output):
                # Extract the activation tensor (handle tuple output)
                act_tensor = output[0] if isinstance(output, tuple) else output
                layer_outputs[layer_idx] = act_tensor
            return hook_fn

        # Register hooks for target layers
        model_layers = self.probing_model.get_layers()
        for layer_idx in layer_list:
            target_layer = model_layers[layer_idx]
            handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
            handles.append(handle)
```

Same hook pattern, but this time the **full batch tensor** is stored (not just element 0), because the batch dimension is meaningful.

---

### Lines 336-345: Batched forward pass

```python
        try:
            with torch.inference_mode():
                _ = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                )
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()
```

Runs the model with both `input_ids` and `attention_mask` so that padding tokens are properly masked. Hooks capture the layer outputs; the model's final output is discarded.

---

### Lines 347-355: Stack and cast activations

```python
        # Stack activations in layer order, moving to consistent device (first layer's device)
        target_device = layer_outputs[layer_list[0]].device
        selected_activations = torch.stack([
            layer_outputs[i].to(target_device) for i in layer_list
        ])  # (num_layers, batch_size, seq_len, hidden_size)

        # Ensure consistent bf16 dtype
        if selected_activations.dtype != torch.bfloat16:
            selected_activations = selected_activations.to(torch.bfloat16)
```

- All layer tensors are moved to the same device (in case pipeline parallelism places layers on different devices) and stacked into a 4-D tensor.
- The tensor is cast to `bfloat16` for consistent downstream processing and reduced memory usage.

---

### Lines 357-367: Build metadata and return

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

Returns a tuple:

1. **`selected_activations`** -- shape `(num_layers, batch_size, max_seq_len, hidden_size)`.
2. **`batch_metadata`** -- a dictionary containing:
   - `conversation_lengths` -- how many turns each conversation has.
   - `total_conversations` -- the total number of conversations in the batch.
   - `conversation_offsets` -- index offsets for locating each conversation within the batch.
   - `max_seq_len` -- the padded/truncated sequence length.
   - `attention_mask` -- the attention mask tensor (useful for downstream masking).
   - `actual_lengths` -- pre-truncation token counts.
   - `truncated_lengths` -- post-truncation token counts.

---

### Lines 369-398: `_find_newline_position`

```python
    def _find_newline_position(self, input_ids: torch.Tensor) -> int:
        """
        Find the position of the newline token in the assistant section.

        Args:
            input_ids: 1D tensor of token IDs

        Returns:
            Index of newline token (or last token as fallback)
        """
        # Try to find '\n\n' token first
        try:
            newline_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[0]
            newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
            if len(newline_positions) > 0:
                return newline_positions[-1].item()  # Use the last occurrence
        except:
            pass

        # Fallback to single '\n' token
        try:
            newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
            newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
            if len(newline_positions) > 0:
                return newline_positions[-1].item()
        except:
            pass

        # Final fallback to last token
        return len(input_ids) - 1
```

A private helper that locates the position of the newline token in a tokenised sequence, using a three-tier fallback strategy:

1. **Try `"\n\n"`** -- encodes the double-newline string and takes the first resulting token ID. Searches `input_ids` for all occurrences and returns the **last** one (closest to the assistant's response area).
2. **Try `"\n"`** -- same approach with a single newline, in case the tokenizer does not have a dedicated double-newline token.
3. **Last token** -- if no newline token is found at all, returns the index of the final token as a safe default.

The bare `except` clauses catch any error during encoding or searching (e.g. if the tokenizer produces an unexpected output), ensuring the method always returns a valid index.
