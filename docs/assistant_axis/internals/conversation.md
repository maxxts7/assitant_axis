# `assistant_axis/internals/conversation.py`

This file defines the `ConversationEncoder` class, which provides a unified interface for formatting multi-turn chat conversations, tokenizing them, and extracting the token indices that correspond to assistant (or user) responses. It handles model-specific quirks for Qwen, LLaMA, and Gemma model families and falls back to a generic approach for other architectures. It also includes a utility for building a boolean mask over tokens that fall inside code blocks.

---

## Imports

```python
"""ConversationEncoder - Handles chat formatting and token indexing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer
import re
```

- The module docstring summarises the purpose of the file.
- `from __future__ import annotations` enables PEP 604-style postponed evaluation of type annotations, allowing forward references and `X | Y` union syntax in all Python 3.7+ versions.
- `typing` imports bring in the generic type aliases used throughout the class signatures.
- `torch` is imported for creating boolean tensors (used in `code_block_token_mask`).
- `AutoTokenizer` from HuggingFace `transformers` is referenced as the expected type for the tokenizer parameter.
- `re` is the standard regular-expression module, used for finding code-block regions in text.

---

## Class Definition and Constructor

```python
class ConversationEncoder:
    """
    Handles conversation formatting, tokenization, and response index extraction.

    This class knows about model-specific quirks (Qwen vs LLaMA vs Gemma) and
    provides a unified interface for working with chat templates.
    """

    def __init__(self, tokenizer: AutoTokenizer, model_name: Optional[str] = None):
        self.tokenizer = tokenizer
        self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()
```

- `ConversationEncoder` wraps a HuggingFace tokenizer and exposes higher-level methods for chat-oriented workflows.
- `__init__` takes two arguments:
  - `tokenizer` -- any HuggingFace tokenizer that supports `apply_chat_template`.
  - `model_name` -- an optional string used to detect which model family is being used.
- `self.tokenizer` stores the tokenizer for later use.
- `self.model_name` is resolved with a fallback chain: if `model_name` is `None` (or empty), it reads `tokenizer.name_or_path` (a standard HuggingFace attribute that stores the model identifier used when loading the tokenizer). The result is lowercased so that all subsequent substring checks are case-insensitive.

---

## Model Detection Helpers

```python
def _is_qwen(self) -> bool:
    """Check if this is a Qwen model."""
    return 'qwen' in self.model_name

def _is_llama(self) -> bool:
    """Check if this is a Llama model."""
    return 'llama' in self.model_name or 'meta-llama' in self.model_name

def _is_gemma(self) -> bool:
    """Check if this is a Gemma model."""
    return 'gemma' in self.model_name
```

- `_is_qwen` returns `True` when the lowercased model name contains `"qwen"`.
- `_is_llama` returns `True` when the model name contains `"llama"` or `"meta-llama"`. The second check is redundant (since `"meta-llama"` contains `"llama"`), but it makes the intent explicit.
- `_is_gemma` returns `True` when the model name contains `"gemma"`.

These three predicates are used throughout the class to dispatch to model-specific implementations.

---

## `format_chat`

```python
def format_chat(
    self,
    conversation: Union[str, List[Dict[str, str]]],
    swap: bool = False,
    **chat_kwargs,
) -> str:
```

This method converts a conversation (or a single prompt string) into a fully formatted chat string using the tokenizer's chat template.

```python
    if isinstance(conversation, str):
        conversation = [{"role": "user", "content": conversation}]
```

If `conversation` is a plain string, it is wrapped in a single-message list with the `"user"` role so that the rest of the method can treat it uniformly.

```python
    if swap:
        messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": conversation[0]["content"]}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        parts = formatted_prompt.rsplit('model', 1)
        if len(parts) == 2:
            formatted_prompt = 'user'.join(parts)
        return formatted_prompt
```

When `swap=True`, the method builds a two-message conversation where:
1. A dummy user message `"Hello."` is added first.
2. The original user content is placed in a `"model"` role message.

After formatting with the chat template, the last occurrence of the literal string `"model"` in the formatted output is replaced with `"user"`. This effectively swaps the role label in the rendered template so the content appears under the user role while retaining the structural position of a model response. This is useful for special evaluation or training scenarios.

```python
    else:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
```

In the standard (non-swap) path, the tokenizer's `apply_chat_template` is called with `tokenize=False` (returning a string rather than token IDs) and `add_generation_prompt=True` (appending the model's expected response prefix so the model knows to start generating).

---

## `token_ids`

```python
def token_ids(
    self,
    conversation: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    **chat_kwargs,
) -> List[int]:
    return self.tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **chat_kwargs,
    )
```

A thin wrapper around `apply_chat_template` with `tokenize=True`. It accepts a conversation (list of role/content dicts) and returns the corresponding list of integer token IDs. The `add_generation_prompt` parameter controls whether the generation-prompting suffix is appended; it defaults to `False` here (unlike `format_chat`, which defaults to `True`).

---

## `response_indices`

```python
def response_indices(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool = False,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
```

This is the main public method for finding which token positions in the tokenized conversation correspond to assistant responses.

```python
    if self._is_qwen():
        return self._get_response_indices_qwen(conversation, per_turn, **chat_kwargs)
    elif self._is_llama() or self._is_gemma():
        return self._get_response_indices_gemma(conversation, per_turn, **chat_kwargs)
    else:
        return self._get_response_indices_simple(conversation, per_turn, **chat_kwargs)
```

It dispatches to one of three model-specific implementations:
- Qwen models get `_get_response_indices_qwen`, which uses Qwen's `<|im_start|>` / `<|im_end|>` special tokens.
- LLaMA and Gemma models get `_get_response_indices_gemma`, which uses character-level offset mapping.
- Everything else gets `_get_response_indices_simple`, which uses a range-based approach comparing token counts.

When `per_turn=False`, all methods return a single flat list of token indices across all assistant turns. When `per_turn=True`, they return a list of lists, one per assistant turn.

---

## `_get_response_indices_qwen`

```python
def _get_response_indices_qwen(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
```

This method extracts assistant response token indices for Qwen models.

```python
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
```

Initialises either a list-of-lists (`all_turn_indices`) or a flat list (`response_indices`) depending on the `per_turn` flag.

```python
    enable_thinking = chat_kwargs.get('enable_thinking', False)
```

Checks whether "thinking mode" is enabled. Qwen models can produce `<think>...</think>` blocks before their actual response; when thinking is disabled, those tokens should be excluded from the response indices.

```python
    full_formatted = self.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )
    full_tokens = self.tokenizer(full_formatted, add_special_tokens=False)
    all_token_ids = full_tokens['input_ids']
```

The entire conversation is formatted into a string and then tokenized. `add_special_tokens=False` prevents the tokenizer from adding its own BOS/EOS tokens on top of the ones already embedded in the chat template. The result is a flat list of token IDs representing the whole conversation.

```python
    try:
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        assistant_token_id = self.tokenizer.convert_tokens_to_ids('assistant')

        try:
            think_start_id = self.tokenizer.convert_tokens_to_ids('<think>')
            think_end_id = self.tokenizer.convert_tokens_to_ids('</think>')
        except (KeyError, ValueError):
            think_start_id = None
            think_end_id = None

    except (KeyError, ValueError):
        return self._get_response_indices_simple(conversation, per_turn, **chat_kwargs)
```

Looks up the integer IDs for Qwen's special tokens:
- `<|im_start|>` and `<|im_end|>` are the delimiters that wrap each turn in Qwen's chat format.
- `assistant` is the role token that follows `<|im_start|>` for assistant turns.
- `<think>` and `</think>` are optional thinking-block delimiters (not present in all Qwen variants).

If the primary special tokens are not found, the method falls back to the simple implementation.

```python
    i = 0
    while i < len(all_token_ids):
        if (i + 1 < len(all_token_ids) and
            all_token_ids[i] == im_start_id and
            all_token_ids[i + 1] == assistant_token_id):
```

Scans through the token sequence looking for the two-token pattern `<|im_start|>assistant`, which marks the beginning of every assistant turn in Qwen's format.

```python
            response_start = i + 2
```

The actual response content begins two tokens after the pattern (skipping `<|im_start|>` and `assistant`).

```python
            response_end = None
            for j in range(response_start, len(all_token_ids)):
                if all_token_ids[j] == im_end_id:
                    response_end = j
                    break
```

Searches forward from `response_start` for the closing `<|im_end|>` token. When found, `response_end` is set to its position (the `<|im_end|>` token itself is not included in the response range).

```python
            if response_end is not None:
                raw_turn_indices = list(range(response_start, response_end))
```

Builds a list of all token indices from the start of the assistant content up to (but not including) `<|im_end|>`.

```python
                if not enable_thinking and think_start_id is not None and think_end_id is not None:
                    filtered_indices = []
                    skip_until_think_end = False

                    for idx in raw_turn_indices:
                        token_id = all_token_ids[idx]

                        if token_id == think_start_id:
                            skip_until_think_end = True
                            continue

                        if token_id == think_end_id:
                            skip_until_think_end = False
                            continue

                        if skip_until_think_end:
                            continue

                        filtered_indices.append(idx)
```

When thinking is disabled, this block filters out all tokens that fall between `<think>` and `</think>` (inclusive of the delimiter tokens themselves). It uses a boolean flag `skip_until_think_end` that is set to `True` when `<think>` is encountered and reset to `False` when `</think>` is encountered.

```python
                    if filtered_indices:
                        extracted_token_ids = [all_token_ids[i] for i in filtered_indices]
                        extracted_text = self.tokenizer.decode(extracted_token_ids)

                        if extracted_text.strip() != extracted_text:
                            while (filtered_indices and
                                   self.tokenizer.decode([all_token_ids[filtered_indices[0]]]).strip() == ''):
                                filtered_indices.pop(0)

                            while (filtered_indices and
                                   self.tokenizer.decode([all_token_ids[filtered_indices[-1]]]).strip() == ''):
                                filtered_indices.pop()

                    turn_indices = filtered_indices
```

After filtering out thinking tokens, the remaining tokens may start or end with whitespace-only tokens (e.g., newlines that separated the `</think>` block from the real content). This block:
1. Decodes the filtered tokens back to text.
2. If the decoded text has leading or trailing whitespace, it trims whitespace-only tokens from the front and back of the index list by decoding each boundary token individually and checking if it is entirely whitespace.

```python
                else:
                    turn_indices = raw_turn_indices
```

If thinking is enabled (or thinking tokens do not exist), the raw indices are used without filtering.

```python
                if per_turn:
                    all_turn_indices.append(turn_indices)
                else:
                    response_indices.extend(turn_indices)

                i = response_end + 1
            else:
                i += 1
        else:
            i += 1

    return all_turn_indices if per_turn else response_indices
```

The collected indices for each turn are either appended as a sub-list (per-turn mode) or merged into the flat list. The scan pointer `i` advances past the `<|im_end|>` token if a complete turn was found, or advances by one if no match was found. Finally the accumulated indices are returned.

---

## `_get_response_indices_gemma`

```python
def _get_response_indices_gemma(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
```

This implementation is used for Gemma and LLaMA models. It uses character-level offset mapping to precisely locate assistant content within the formatted string.

```python
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []
```

Same initialisation pattern as the Qwen method.

```python
    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue
```

Iterates through the conversation and only processes assistant turns.

```python
        conversation_before = conversation[:i]
        conversation_including = conversation[:i+1]
```

For each assistant turn, two slices of the conversation are prepared:
- `conversation_before` -- everything up to but not including this assistant turn.
- `conversation_including` -- everything up to and including this assistant turn.

```python
        if conversation_before:
            before_formatted = self.tokenizer.apply_chat_template(
                conversation_before, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            before_tokens = self.tokenizer(before_formatted, add_special_tokens=False)
            before_length = len(before_tokens['input_ids'])
        else:
            before_length = 0
```

The "before" conversation is formatted and tokenized to determine how many tokens precede this assistant turn. If there are no messages before, the length is zero. The `add_generation_prompt=True` is used here because the generation prompt is what would appear right before the assistant's response.

```python
        including_formatted = self.tokenizer.apply_chat_template(
            conversation_including, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
        including_tokens = self.tokenizer(including_formatted, add_special_tokens=False)
        including_length = len(including_tokens['input_ids'])
```

The conversation including the assistant turn is also formatted and tokenized. `add_generation_prompt=False` is used here because the assistant has already responded.

```python
        assistant_content = turn['content'].strip()
```

The raw text of the assistant's response, with leading/trailing whitespace removed.

```python
        turn_indices = []

        content_start_in_formatted = including_formatted.find(assistant_content)
        if content_start_in_formatted != -1:
            content_end_in_formatted = content_start_in_formatted + len(assistant_content)
```

Searches for the assistant content as a substring within the full formatted string. If found, the character-level start and end positions are recorded.

```python
            tokens_with_offsets = self.tokenizer(including_formatted, return_offsets_mapping=True, add_special_tokens=False)
            offset_mapping = tokens_with_offsets['offset_mapping']

            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                if (start_char >= content_start_in_formatted and start_char < content_end_in_formatted) or \
                   (end_char > content_start_in_formatted and end_char <= content_end_in_formatted) or \
                   (start_char < content_start_in_formatted and end_char > content_end_in_formatted):
                    turn_indices.append(token_idx)
```

The formatted string is re-tokenized with `return_offsets_mapping=True`, which provides the character-level `(start, end)` span for each token. Each token is checked for overlap with the assistant content region using three conditions:
1. The token starts within the content region.
2. The token ends within the content region.
3. The token completely spans the content region (the token is larger than the content, which can happen with subword tokenization at boundaries).

Any token that overlaps with the assistant content in any of these ways is included.

```python
        else:
            assistant_start = before_length
            assistant_end = including_length
            turn_indices.extend(range(assistant_start, assistant_end))
```

If the assistant content cannot be found as a substring (e.g., the template modifies the content), the method falls back to using the difference in token counts between the "before" and "including" tokenizations as the range of assistant tokens.

```python
        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)

    return all_turn_indices if per_turn else response_indices
```

Results are accumulated and returned in the same pattern as the Qwen method.

---

## `_get_response_indices_simple`

```python
def _get_response_indices_simple(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
```

This is the generic fallback for models not specifically handled. It uses a purely range-based approach.

```python
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []

    for i, turn in enumerate(conversation):
        if turn['role'] != 'assistant':
            continue

        conversation_before = conversation[:i]
        conversation_including = conversation[:i+1]

        if conversation_before:
            before_formatted = self.tokenizer.apply_chat_template(
                conversation_before, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
            before_tokens = self.tokenizer(before_formatted, add_special_tokens=False)
            before_length = len(before_tokens['input_ids'])
        else:
            before_length = 0

        including_formatted = self.tokenizer.apply_chat_template(
            conversation_including, tokenize=False, add_generation_prompt=False, **chat_kwargs
        )
        including_tokens = self.tokenizer(including_formatted, add_special_tokens=False)
        including_length = len(including_tokens['input_ids'])
```

The logic is identical to the first part of `_get_response_indices_gemma`: for each assistant turn, it computes how many tokens exist before the turn and how many exist after including the turn.

```python
        assistant_start = before_length
        assistant_end = including_length

        turn_indices = list(range(assistant_start, assistant_end))
```

The difference between `including_length` and `before_length` gives the token range for this assistant turn. This includes any formatting/template tokens that wrap the assistant content (e.g., role markers, end-of-turn tokens), making this less precise than the Qwen or Gemma methods but universally applicable.

```python
        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)

    return all_turn_indices if per_turn else response_indices
```

Accumulation and return follow the standard pattern.

---

## `build_turn_spans`

```python
def build_turn_spans(
    self,
    conversation: List[Dict[str, str]],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
```

Builds structured metadata about every non-system turn in a conversation, identifying the exact token span (start and end indices) for each turn's content within the fully tokenized conversation.

```python
    full_ids = self.tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, **chat_kwargs
    )
```

Tokenizes the entire conversation into a single list of token IDs. This is the "ground truth" sequence that the spans will index into.

```python
    if self._is_qwen():
        return self._build_turn_spans_qwen(conversation, full_ids, **chat_kwargs)
```

Qwen models are dispatched to a specialised implementation that uses pattern matching on `<|im_start|>` / `<|im_end|>` markers.

```python
    spans = []
    msgs_before = []
    turn_idx = 0

    for msg in conversation:
        role = msg["role"]
        text = msg.get("content", "")

        if role == "system":
            msgs_before.append(msg)
            continue
```

For non-Qwen models, the method iterates through each message. System messages are accumulated in `msgs_before` but are skipped for span creation (they do not get their own span entry).

```python
        content_ids, start_in_delta = self._content_only_ids_and_offset(
            msgs_before, role, text, **chat_kwargs
        )
```

Calls a helper to extract just the content token IDs (without template/role tokens) and the offset of those content tokens within the "delta" (the new tokens added by this message).

```python
        msgs_empty_for_this = msgs_before + [{"role": role, "content": ""}]
        ids_empty_full = self.tokenizer.apply_chat_template(
            msgs_empty_for_this, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
```

Creates a version of the conversation where this turn has empty content. This is used as a reference point: the tokens in this version represent the template structure without the actual message content.

```python
        ids_full_for_this = self.tokenizer.apply_chat_template(
            msgs_before + [{"role": role, "content": text}], tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
```

Also tokenizes the version with the actual content, so the two can be compared.

```python
        pref_len = self._longest_common_prefix_len(ids_full_for_this, ids_empty_full)
        abs_start = pref_len + start_in_delta
        abs_end = abs_start + len(content_ids)
```

- `pref_len` is the number of tokens that are identical between the empty and full versions. This is the point where the content diverges from the template.
- `abs_start` adds the `start_in_delta` offset (which accounts for any template tokens between the divergence point and the actual content).
- `abs_end` is `abs_start` plus the number of content tokens.

```python
        spans.append({
            "turn": turn_idx,
            "role": role,
            "start": abs_start,
            "end": abs_end,   # exclusive
            "n_tokens": len(content_ids),
            "text": text,
        })
        msgs_before.append(msg)
        turn_idx += 1

    return full_ids, spans
```

Each span is recorded as a dictionary with:
- `turn` -- zero-based index among non-system turns.
- `role` -- `"user"` or `"assistant"`.
- `start` -- inclusive start token index in `full_ids`.
- `end` -- exclusive end token index in `full_ids`.
- `n_tokens` -- number of content tokens.
- `text` -- the original text content.

The current message is appended to `msgs_before` so the next iteration has the correct prefix. The method returns the full token ID list and the list of span dicts.

---

## `_build_turn_spans_qwen`

```python
def _build_turn_spans_qwen(
    self,
    conversation: List[Dict[str, str]],
    full_ids: List[int],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
```

Qwen-specific span builder that uses the `<|im_start|>` / `<|im_end|>` pattern to precisely locate each turn.

```python
    spans = []

    enable_thinking = chat_kwargs.get('enable_thinking', False)
```

Initialises the span list and checks the thinking flag, just as in `_get_response_indices_qwen`.

```python
    try:
        im_start_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        user_token_id = self.tokenizer.convert_tokens_to_ids('user')
        assistant_token_id = self.tokenizer.convert_tokens_to_ids('assistant')

        try:
            think_start_id = self.tokenizer.convert_tokens_to_ids('<think>')
            think_end_id = self.tokenizer.convert_tokens_to_ids('</think>')
        except (KeyError, ValueError):
            think_start_id = None
            think_end_id = None

    except (KeyError, ValueError):
        return self._build_turn_spans_fallback(conversation, full_ids, **chat_kwargs)
```

Looks up the special token IDs for Qwen's format markers. Unlike the response-indices method, this one also looks up `user_token_id` because it builds spans for both user and assistant turns. Falls back to `_build_turn_spans_fallback` if the tokens are not found.

```python
    expected_turns = []
    for msg in conversation:
        if msg["role"] != "system":
            expected_turns.append((msg["role"], msg.get("content", "")))
```

Builds an ordered list of expected non-system turns. This list is used to match detected spans against the original conversation messages, ensuring role alignment.

```python
    turn_idx = 0
    i = 0

    while i < len(full_ids):
        if i + 1 < len(full_ids) and full_ids[i] == im_start_id:
            role_token = full_ids[i + 1]

            if role_token == user_token_id:
                role = "user"
            elif role_token == assistant_token_id:
                role = "assistant"
            else:
                i += 1
                continue
```

Scans the full token sequence for `<|im_start|>` followed by either `user` or `assistant`. System turns (or any other role) are skipped.

```python
            content_start = i + 2

            content_end = None
            for j in range(content_start, len(full_ids)):
                if full_ids[j] == im_end_id:
                    content_end = j
                    break
```

The content of the turn starts two positions after `<|im_start|>` (skipping the role token). The method then searches forward for `<|im_end|>` to find the end of the content.

```python
            if content_end is not None and turn_idx < len(expected_turns):
                expected_role, expected_text = expected_turns[turn_idx]

                if role == expected_role:
                    raw_indices = list(range(content_start, content_end))
```

If a complete turn boundary is found and the detected role matches the expected role from the conversation, the raw content indices (between `<|im_start|>role` and `<|im_end|>`) are collected.

```python
                    if role == "assistant" and not enable_thinking and think_start_id is not None and think_end_id is not None:
                        filtered_indices = []
                        skip_until_think_end = False

                        for idx in raw_indices:
                            token_id = full_ids[idx]

                            if token_id == think_start_id:
                                skip_until_think_end = True
                                continue

                            if token_id == think_end_id:
                                skip_until_think_end = False
                                continue

                            if skip_until_think_end:
                                continue

                            filtered_indices.append(idx)
```

For assistant turns with thinking disabled, the same thinking-token filtering logic from `_get_response_indices_qwen` is applied: tokens between `<think>` and `</think>` (inclusive of the delimiters) are removed.

```python
                        if filtered_indices:
                            extracted_token_ids = [full_ids[idx] for idx in filtered_indices]
                            extracted_text = self.tokenizer.decode(extracted_token_ids)

                            if extracted_text.strip() != extracted_text:
                                while (filtered_indices and
                                       self.tokenizer.decode([full_ids[filtered_indices[0]]]).strip() == ''):
                                    filtered_indices.pop(0)

                                while (filtered_indices and
                                       self.tokenizer.decode([full_ids[filtered_indices[-1]]]).strip() == ''):
                                    filtered_indices.pop()

                        final_indices = filtered_indices
                    else:
                        final_indices = raw_indices
```

After filtering, leading and trailing whitespace-only tokens are trimmed, following the same procedure as in `_get_response_indices_qwen`. For non-assistant turns or when thinking is enabled, the raw indices are used as-is.

```python
                    if final_indices:
                        spans.append({
                            "turn": turn_idx,
                            "role": role,
                            "start": min(final_indices),
                            "end": max(final_indices) + 1,  # exclusive
                            "n_tokens": len(final_indices),
                            "text": expected_text,
                        })
                    turn_idx += 1

                i = content_end + 1
            else:
                i += 1
        else:
            i += 1

    return full_ids, spans
```

If there are valid indices remaining after filtering, a span dict is created. The `start` is the minimum index and `end` is one past the maximum (exclusive), matching Python's range convention. The `text` field uses the original message text from the conversation, not the decoded tokens. The scan pointer advances past the `<|im_end|>` token.

---

## `_build_turn_spans_fallback`

```python
def _build_turn_spans_fallback(
    self,
    conversation: List[Dict[str, str]],
    full_ids: List[int],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
```

A fallback method for when Qwen's special tokens cannot be resolved. It uses subsequence searching to locate content within the full token sequence.

```python
    spans = []
    msgs_before = []
    turn_idx = 0

    for msg in conversation:
        role = msg["role"]
        text = msg.get("content", "")

        if role == "system":
            msgs_before.append(msg)
            continue
```

Standard iteration with system message skipping.

```python
        content_ids, start_in_delta = self._content_only_ids_and_offset(
            msgs_before, role, text, **chat_kwargs
        )
```

Extracts the content-only token IDs using the helper method.

```python
        abs_start = self._find_subsequence(full_ids, content_ids)
        if abs_start == -1:
            msgs_before.append(msg)
            continue
        abs_end = abs_start + len(content_ids)
```

Searches for the content tokens as a subsequence within the full conversation token list. If not found (`-1`), the turn is skipped. Otherwise, the start and end positions are computed.

```python
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

The span is recorded and the message is added to the prefix for the next iteration.

---

## `build_batch_turn_spans`

```python
def build_batch_turn_spans(
    self,
    conversations: List[List[Dict[str, str]]],
    **chat_kwargs,
) -> Tuple[List[List[int]], List[Dict[str, Any]], Dict[str, Any]]:
```

Processes multiple conversations at once, building spans for each and tracking global (cross-conversation) token positions.

```python
    batch_full_ids = []
    batch_spans = []
    batch_metadata = {
        'conversation_lengths': [],
        'total_conversations': len(conversations),
        'conversation_offsets': []
    }

    global_offset = 0
```

Initialises the output containers:
- `batch_full_ids` will hold one token ID list per conversation.
- `batch_spans` will hold all spans from all conversations with additional metadata.
- `batch_metadata` tracks per-conversation lengths and cumulative offsets.
- `global_offset` is a running counter for the total number of tokens seen so far.

```python
    for conv_id, conversation in enumerate(conversations):
        full_ids, spans = self.build_turn_spans(conversation, **chat_kwargs)

        batch_full_ids.append(full_ids)
        batch_metadata['conversation_lengths'].append(len(full_ids))
        batch_metadata['conversation_offsets'].append(global_offset)
```

Each conversation is processed by `build_turn_spans`. The token IDs and metadata are stored.

```python
        for span in spans:
            enhanced_span = span.copy()
            enhanced_span['conversation_id'] = conv_id
            enhanced_span['local_start'] = span['start']
            enhanced_span['local_end'] = span['end']
            enhanced_span['global_start'] = global_offset + span['start']
            enhanced_span['global_end'] = global_offset + span['end']
            batch_spans.append(enhanced_span)

        global_offset += len(full_ids)
```

Each span is enhanced with:
- `conversation_id` -- which conversation this span belongs to.
- `local_start` / `local_end` -- the token positions within the individual conversation.
- `global_start` / `global_end` -- the token positions as if all conversations were concatenated end-to-end.

The global offset advances by the length of the current conversation.

```python
    return batch_full_ids, batch_spans, batch_metadata
```

Returns all three structures.

---

## `code_block_token_mask`

```python
def code_block_token_mask(self, text: str) -> torch.Tensor:
```

Creates a boolean tensor indicating which tokens in a piece of text fall inside code blocks (either single-backtick inline code or triple-backtick fenced code blocks).

```python
    tokenized = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = tokenized['input_ids']
    offset_mapping = tokenized['offset_mapping']

    n_tokens = len(token_ids)
    exclude_mask = torch.zeros(n_tokens, dtype=torch.bool)

    if n_tokens == 0:
        return exclude_mask
```

The text is tokenized with offset mapping enabled, providing character-level positions for each token. A boolean mask of the same length is initialised to all `False`. If the text is empty, the empty mask is returned immediately.

```python
    code_regions = []

    triple_pattern = r'```[\s\S]*?```'
    for match in re.finditer(triple_pattern, text):
        code_regions.append((match.start(), match.end()))
```

First pass: finds all triple-backtick code blocks. The pattern `[\s\S]*?` matches any character (including newlines) non-greedily, so it captures everything between opening and closing triple backticks.

```python
    single_pattern = r'`[^`\n]*?`'
    for match in re.finditer(single_pattern, text):
        start, end = match.start(), match.end()
        overlaps = any(triple_start <= start < triple_end or triple_start < end <= triple_end
                      for triple_start, triple_end in code_regions)
        if not overlaps:
            code_regions.append((start, end))
```

Second pass: finds single-backtick inline code spans. The pattern `[^`\n]*?` matches any character except backticks and newlines, non-greedily. Before adding a match, it checks whether the single-backtick region overlaps with any previously found triple-backtick region -- if it does, the match is discarded (triple backticks take precedence).

```python
    for char_start, char_end in code_regions:
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if (token_start < char_end and token_end > char_start):
                exclude_mask[i] = True

    return exclude_mask
```

Maps each code region to token indices using overlap detection: a token is marked `True` in the mask if its character span overlaps with any code region. The overlap check `(token_start < char_end and token_end > char_start)` is the standard interval overlap formula. The completed mask is returned.

---

## Private Helper Methods

### `_content_only_ids_and_offset`

```python
def _content_only_ids_and_offset(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
```

Dispatcher that extracts just the content token IDs from a turn (excluding template/role tokens) and the offset of those tokens within the turn's "delta" (the new tokens added by appending this message).

```python
    if self._is_qwen() and role == "assistant":
        return self._content_only_ids_and_offset_qwen(messages_before, role, content, **chat_kwargs)
    else:
        return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)
```

Qwen assistant turns are handled specially because thinking tokens can interfere with the standard approach. All other cases use the standard method.

---

### `_content_only_ids_and_offset_qwen`

```python
def _content_only_ids_and_offset_qwen(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
```

Qwen-specific content extraction for assistant turns.

```python
    if role == "assistant":
        msgs_full = messages_before + [{"role": role, "content": content}]
        ids_full = self.tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )

        plain = self.tokenizer(content, add_special_tokens=False).input_ids
        content_start = self._find_subsequence(ids_full, plain)
```

For assistant turns: tokenizes the full conversation including this message, then tokenizes the raw content in isolation. It searches for the raw content tokens as a subsequence within the full tokenized conversation.

```python
        if content_start != -1:
            if messages_before:
                ids_before = self.tokenizer.apply_chat_template(
                    messages_before, tokenize=True, add_generation_prompt=False, **chat_kwargs
                )
                prefix_len = len(ids_before)
            else:
                prefix_len = 0

            start_in_delta = content_start - prefix_len
            return plain, max(0, start_in_delta)
```

If the raw content is found, the offset within the delta is calculated by subtracting the number of prefix tokens (tokens from all preceding messages). `max(0, ...)` ensures the offset is never negative (which could happen if tokenization merges tokens at boundaries).

```python
    return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)
```

Falls back to the standard method if the content is not found or for non-assistant roles.

---

### `_content_only_ids_and_offset_standard`

```python
def _content_only_ids_and_offset_standard(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
```

The standard content extraction method, used for most models and for user turns on Qwen.

```python
    msgs_empty = messages_before + [{"role": role, "content": ""}]
    msgs_full  = messages_before + [{"role": role, "content": content}]
```

Creates two versions of the conversation: one with empty content for this turn, and one with the actual content.

```python
    if messages_before:
        ids_before = self.tokenizer.apply_chat_template(
            messages_before, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )
    else:
        ids_before = []
    ids_empty = self.tokenizer.apply_chat_template(
        msgs_empty, tokenize=True, add_generation_prompt=False, **chat_kwargs
    )
    ids_full  = self.tokenizer.apply_chat_template(
        msgs_full,  tokenize=True, add_generation_prompt=False, **chat_kwargs
    )
```

Tokenizes three versions: just the prefix, the prefix with an empty turn, and the prefix with the full turn.

```python
    pref = self._longest_common_prefix_len(ids_full, ids_empty)
    delta = ids_full[pref:]
    delta = self._strip_trailing_special(delta, set(self.tokenizer.all_special_ids))
```

- `pref` identifies where `ids_full` and `ids_empty` diverge -- this is where the actual content begins.
- `delta` is everything in `ids_full` after that divergence point.
- Trailing special tokens (like EOS) are stripped from the delta, since they are template artifacts rather than content.

```python
    plain = self.tokenizer(content, add_special_tokens=False).input_ids
    sp    = self.tokenizer(" " + content, add_special_tokens=False).input_ids
```

The raw content is tokenized two ways: as-is, and with a leading space. Some tokenizers produce different token sequences depending on whether the text appears at the start of input or mid-sentence (the leading space variant handles the latter case).

```python
    start = self._find_subsequence(delta, plain)
    use = plain
    if start == -1:
        start = self._find_subsequence(delta, sp)
        use = sp if start != -1 else plain
```

Searches for the raw content tokens within the delta. If not found, tries the space-prefixed version. Tracks which variant was actually found.

```python
    if start == -1:
        return delta, 0
    else:
        return delta[start:start+len(use)], start
```

If neither variant is found, the entire delta is returned as the content (a conservative fallback). Otherwise, the precisely located content tokens and their offset within the delta are returned.

---

### `_longest_common_prefix_len`

```python
@staticmethod
def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
```

Compares two integer lists element-by-element from the start and returns the number of elements that are identical. This is used to find where two tokenized sequences diverge.

---

### `_strip_trailing_special`

```python
@staticmethod
def _strip_trailing_special(ids: List[int], special_ids: set) -> List[int]:
    i = len(ids)
    while i > 0 and ids[i-1] in special_ids:
        i -= 1
    return ids[:i]
```

Removes special tokens from the end of a token ID list by walking backwards from the end and stopping at the first non-special token. Returns a new (possibly shorter) list.

---

### `_find_subsequence`

```python
@staticmethod
def _find_subsequence(hay: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return i
    return -1
```

A brute-force substring search over integer lists. Returns the starting index of the first occurrence of `needle` in `hay`, or `-1` if not found. Returns `-1` immediately if `needle` is empty or longer than `hay`.
