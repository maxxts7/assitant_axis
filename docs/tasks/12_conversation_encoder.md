# Task 12: Conversation Encoder (internals/conversation.py)

## Overview

`ConversationEncoder` is a class that wraps a HuggingFace tokenizer and provides a unified interface for:

1. **Formatting** multi-turn conversations into model-specific chat template strings.
2. **Tokenizing** conversations into token ID sequences.
3. **Extracting response indices** -- identifying exactly which token positions correspond to assistant (or user) content, with model-specific logic for Qwen, Llama, and Gemma.
4. **Building turn spans** -- producing per-turn `[start, end)` token ranges for every non-system message in a conversation.
5. **Batch span building** -- processing multiple conversations at once with global offset tracking.
6. **Code-block masking** -- producing a boolean mask that flags tokens falling inside inline or fenced code blocks.

Model dispatch is handled by three private predicates (`_is_qwen`, `_is_llama`, `_is_gemma`) that inspect `self.model_name`.

---

## Sub-Tasks

---

### Sub-Task 12.1: Constructor (`__init__`)

#### Input

| Parameter    | Type                         | Default | Description |
|--------------|------------------------------|---------|-------------|
| `tokenizer`  | `transformers.AutoTokenizer` | --      | A HuggingFace tokenizer that supports `apply_chat_template`. |
| `model_name` | `Optional[str]`              | `None`  | Human-readable model identifier used for dispatch. Falls back to `tokenizer.name_or_path`. |

#### Processing

```python
def __init__(self, tokenizer: AutoTokenizer, model_name: Optional[str] = None):
    self.tokenizer = tokenizer
    self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()
```

1. Store the tokenizer reference.
2. Resolve `model_name`: if the caller supplies one, lower-case it; otherwise pull `name_or_path` from the tokenizer object and lower-case that. The empty string is the final fallback.

#### Output

A fully initialised `ConversationEncoder` instance with two attributes: `self.tokenizer` and `self.model_name` (a lower-cased `str`).

---

### Sub-Task 12.2: Model Detection Predicates (`_is_qwen`, `_is_llama`, `_is_gemma`)

#### Input

None (operates on `self.model_name`).

#### Processing

```python
def _is_qwen(self) -> bool:
    return 'qwen' in self.model_name

def _is_llama(self) -> bool:
    return 'llama' in self.model_name or 'meta-llama' in self.model_name

def _is_gemma(self) -> bool:
    return 'gemma' in self.model_name
```

Simple substring checks on the lower-cased model name.

#### Output

`bool` -- whether the current model matches the given family.

**Dispatch priority** (used in `response_indices` and `build_turn_spans`):

1. Qwen is checked first.
2. Llama OR Gemma share the same code path.
3. Everything else falls through to the "simple" / standard path.

---

### Sub-Task 12.3: Format Chat (`format_chat`)

#### Input

| Parameter      | Type                                          | Default | Description |
|----------------|-----------------------------------------------|---------|-------------|
| `conversation` | `Union[str, List[Dict[str, str]]]`            | --      | Either a bare string prompt or a list of `{"role": ..., "content": ...}` message dicts. |
| `swap`         | `bool`                                        | `False` | When `True`, the first message's content is placed into a synthetic `model` role then the word `"model"` is replaced with `"user"` in the output. |
| `**chat_kwargs`| --                                            | --      | Forwarded to `tokenizer.apply_chat_template`. |

#### Processing

```python
def format_chat(
    self,
    conversation: Union[str, List[Dict[str, str]]],
    swap: bool = False,
    **chat_kwargs,
) -> str:
    if isinstance(conversation, str):
        conversation = [{"role": "user", "content": conversation}]

    if swap:
        messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": conversation[0]["content"]}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        parts = formatted_prompt.rsplit('model', 1)
        if len(parts) == 2:
            formatted_prompt = 'user'.join(parts)
        return formatted_prompt
    else:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
```

**Normal path** (`swap=False`):

1. If `conversation` is a bare string, wrap it as `[{"role": "user", "content": conversation}]`.
2. Call `tokenizer.apply_chat_template` with `tokenize=False` and `add_generation_prompt=True`.
3. Return the formatted string.

**Swapped path** (`swap=True`):

1. Build a synthetic two-message conversation: `[{"role": "user", "content": "Hello."}, {"role": "model", "content": <original content>}]`.
2. Format via the chat template.
3. Find the **last** occurrence of the literal string `"model"` in the output (`rsplit('model', 1)`) and replace it with `"user"`.
4. Return the modified string.

#### Output

`str` -- the fully formatted chat string, ready for tokenization or display.

---

### Sub-Task 12.4: Token IDs (`token_ids`)

#### Input

| Parameter               | Type                       | Default | Description |
|-------------------------|----------------------------|---------|-------------|
| `conversation`          | `List[Dict[str, str]]`     | --      | Message list. |
| `add_generation_prompt` | `bool`                     | `False` | Whether to append the generation prompt. |
| `**chat_kwargs`         | --                         | --      | Forwarded to `apply_chat_template`. |

#### Processing

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

A thin wrapper: calls `apply_chat_template` with `tokenize=True`.

#### Output

`List[int]` -- the token IDs for the entire formatted conversation.

---

### Sub-Task 12.5: Response Indices -- Dispatcher (`response_indices`)

#### Input

| Parameter      | Type                       | Default | Description |
|----------------|----------------------------|---------|-------------|
| `conversation` | `List[Dict[str, str]]`     | --      | Message list. |
| `per_turn`     | `bool`                     | `False` | If `True`, return a list-of-lists (one inner list per assistant turn). If `False`, return a single flat list. |
| `**chat_kwargs`| --                         | --      | Forwarded to `apply_chat_template`. |

#### Processing

```python
def response_indices(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool = False,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
    if self._is_qwen():
        return self._get_response_indices_qwen(conversation, per_turn, **chat_kwargs)
    elif self._is_llama() or self._is_gemma():
        return self._get_response_indices_gemma(conversation, per_turn, **chat_kwargs)
    else:
        return self._get_response_indices_simple(conversation, per_turn, **chat_kwargs)
```

Dispatches to one of three private implementations based on model family.

#### Output

- `per_turn=False`: `List[int]` -- flat list of token indices.
- `per_turn=True`: `List[List[int]]` -- one inner list per assistant turn.

---

### Sub-Task 12.5a: Response Indices -- Qwen Variant (`_get_response_indices_qwen`)

#### Input

Same as `response_indices` (delegated).

#### Processing

```python
def _get_response_indices_qwen(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
    if per_turn:
        all_turn_indices = []
    else:
        response_indices = []

    enable_thinking = chat_kwargs.get('enable_thinking', False)

    full_formatted = self.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )
    full_tokens = self.tokenizer(full_formatted, add_special_tokens=False)
    all_token_ids = full_tokens['input_ids']

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

    i = 0
    while i < len(all_token_ids):
        if (i + 1 < len(all_token_ids) and
            all_token_ids[i] == im_start_id and
            all_token_ids[i + 1] == assistant_token_id):

            response_start = i + 2

            response_end = None
            for j in range(response_start, len(all_token_ids)):
                if all_token_ids[j] == im_end_id:
                    response_end = j
                    break

            if response_end is not None:
                raw_turn_indices = list(range(response_start, response_end))

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
                else:
                    turn_indices = raw_turn_indices

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

**Algorithm (step by step)**:

1. Format the full conversation to text (`tokenize=False`, `add_generation_prompt=False`), then tokenize that text with `add_special_tokens=False` to get `all_token_ids`.
2. Resolve Qwen special token IDs: `<|im_start|>`, `<|im_end|>`, literal `assistant`. Also attempt to resolve `<think>` and `</think>` (may not exist on all Qwen variants).
3. If the special tokens cannot be resolved, fall back to `_get_response_indices_simple`.
4. Scan `all_token_ids` linearly. For every consecutive pair `[<|im_start|>, assistant]`:
   - Mark `response_start = i + 2` (the first content token).
   - Scan forward to find the matching `<|im_end|>`, which becomes `response_end` (exclusive).
   - Collect `raw_turn_indices = range(response_start, response_end)`.
5. **Thinking-token filtering** (when `enable_thinking=False` and thinking tokens exist):
   - Walk `raw_turn_indices`; set a `skip_until_think_end` flag on `<think>`, clear it on `</think>`, drop everything in between.
   - **Whitespace trimming**: if the decoded text of the surviving indices has leading or trailing whitespace, pop leading/trailing indices whose single-token decode is whitespace-only.
6. Append to `all_turn_indices` (per-turn) or extend `response_indices` (flat).

#### Output

- `per_turn=False`: `List[int]` -- flat indices into `all_token_ids`.
- `per_turn=True`: `List[List[int]]` -- one inner list per assistant turn.

---

### Sub-Task 12.5b: Response Indices -- Gemma / Llama Variant (`_get_response_indices_gemma`)

#### Input

Same as `response_indices` (delegated).

#### Processing

```python
def _get_response_indices_gemma(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
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

        assistant_content = turn['content'].strip()

        turn_indices = []

        content_start_in_formatted = including_formatted.find(assistant_content)
        if content_start_in_formatted != -1:
            content_end_in_formatted = content_start_in_formatted + len(assistant_content)

            tokens_with_offsets = self.tokenizer(including_formatted, return_offsets_mapping=True, add_special_tokens=False)
            offset_mapping = tokens_with_offsets['offset_mapping']

            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                if (start_char >= content_start_in_formatted and start_char < content_end_in_formatted) or \
                   (end_char > content_start_in_formatted and end_char <= content_end_in_formatted) or \
                   (start_char < content_start_in_formatted and end_char > content_end_in_formatted):
                    turn_indices.append(token_idx)
        else:
            assistant_start = before_length
            assistant_end = including_length
            turn_indices.extend(range(assistant_start, assistant_end))

        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)

    return all_turn_indices if per_turn else response_indices
```

**Algorithm (step by step)**:

1. Iterate over all messages; skip non-assistant turns.
2. For each assistant turn at index `i`:
   - Format and tokenize `conversation[:i]` (with `add_generation_prompt=True`) to get `before_length`.
   - Format and tokenize `conversation[:i+1]` (with `add_generation_prompt=False`) to get `including_length` and `including_formatted`.
3. Find the assistant content string inside `including_formatted` via `str.find`.
4. **Offset-mapping approach** (primary): tokenize `including_formatted` with `return_offsets_mapping=True`. For each token, check if its character span overlaps with the content character range. Three overlap conditions are checked (token starts inside content, token ends inside content, or token fully encloses content).
5. **Fallback** (content string not found literally): use the coarser range `[before_length, including_length)`.
6. Collect into per-turn or flat lists.

#### Output

Same shape as 12.5a.

---

### Sub-Task 12.5c: Response Indices -- Simple Fallback (`_get_response_indices_simple`)

#### Input

Same as `response_indices` (delegated).

#### Processing

```python
def _get_response_indices_simple(
    self,
    conversation: List[Dict[str, str]],
    per_turn: bool,
    **chat_kwargs,
) -> Union[List[int], List[List[int]]]:
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

        assistant_start = before_length
        assistant_end = including_length

        turn_indices = list(range(assistant_start, assistant_end))

        if per_turn:
            all_turn_indices.append(turn_indices)
        else:
            response_indices.extend(turn_indices)

    return all_turn_indices if per_turn else response_indices
```

**Algorithm**:

1. For each assistant turn at index `i`:
   - Compute `before_length` = number of tokens when formatting `conversation[:i]` (with generation prompt).
   - Compute `including_length` = number of tokens when formatting `conversation[:i+1]` (without generation prompt).
2. The assistant turn's tokens occupy `range(before_length, including_length)`.
3. This is a coarse range that includes any template/formatting tokens around the assistant content (role markers, newlines, etc.).

#### Output

Same shape as 12.5a.

---

### Sub-Task 12.6: Build Turn Spans -- Dispatcher (`build_turn_spans`)

#### Input

| Parameter      | Type                       | Default | Description |
|----------------|----------------------------|---------|-------------|
| `conversation` | `List[Dict[str, str]]`     | --      | Message list. |
| `**chat_kwargs`| --                         | --      | Forwarded to `apply_chat_template`. |

#### Processing

```python
def build_turn_spans(
    self,
    conversation: List[Dict[str, str]],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
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
            msgs_before + [{"role": role, "content": text}], tokenize=True, add_generation_prompt=False, **chat_kwargs
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

**Standard (non-Qwen) algorithm**:

1. Tokenize the full conversation to get `full_ids`.
2. For each non-system message:
   a. Call `_content_only_ids_and_offset` to get the pure content token IDs and their offset within the turn's delta.
   b. Build two template expansions: one with empty content and one with actual content for this turn (appended to `msgs_before`).
   c. Find the longest common prefix between the two to determine where the content diverges.
   d. Compute `abs_start = pref_len + start_in_delta` and `abs_end = abs_start + len(content_ids)`.
3. Build a span dict for each turn.

#### Output

`Tuple[List[int], List[Dict[str, Any]]]`:

- `full_ids`: `List[int]` -- token IDs for the entire conversation.
- `spans`: list of dicts, each with:

| Key        | Type   | Description |
|------------|--------|-------------|
| `turn`     | `int`  | Zero-based turn index (system messages excluded from counting). |
| `role`     | `str`  | `"user"` or `"assistant"`. |
| `start`    | `int`  | Inclusive start index into `full_ids`. |
| `end`      | `int`  | Exclusive end index into `full_ids`. |
| `n_tokens` | `int`  | Number of content tokens (`end - start`). |
| `text`     | `str`  | Original message content string. |

---

### Sub-Task 12.6a: Build Turn Spans -- Qwen Variant (`_build_turn_spans_qwen`)

#### Input

| Parameter      | Type                       | Description |
|----------------|----------------------------|-------------|
| `conversation` | `List[Dict[str, str]]`     | Message list. |
| `full_ids`     | `List[int]`                | Pre-tokenized full conversation IDs. |
| `**chat_kwargs`| --                         | Forwarded kwargs (notably `enable_thinking`). |

#### Processing

```python
def _build_turn_spans_qwen(
    self,
    conversation: List[Dict[str, str]],
    full_ids: List[int],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    spans = []
    enable_thinking = chat_kwargs.get('enable_thinking', False)

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

    expected_turns = []
    for msg in conversation:
        if msg["role"] != "system":
            expected_turns.append((msg["role"], msg.get("content", "")))

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

            content_start = i + 2

            content_end = None
            for j in range(content_start, len(full_ids)):
                if full_ids[j] == im_end_id:
                    content_end = j
                    break

            if content_end is not None and turn_idx < len(expected_turns):
                expected_role, expected_text = expected_turns[turn_idx]

                if role == expected_role:
                    raw_indices = list(range(content_start, content_end))

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

                    if final_indices:
                        spans.append({
                            "turn": turn_idx,
                            "role": role,
                            "start": min(final_indices),
                            "end": max(final_indices) + 1,
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

**Algorithm (step by step)**:

1. Resolve Qwen special token IDs (`<|im_start|>`, `<|im_end|>`, `user`, `assistant`, and optionally `<think>`/`</think>`). Fall back to `_build_turn_spans_fallback` if resolution fails.
2. Build an `expected_turns` list from non-system messages (preserving order).
3. Scan `full_ids` linearly for `[<|im_start|>, user_token_id]` or `[<|im_start|>, assistant_token_id]` patterns.
4. For each match, locate the corresponding `<|im_end|>` to define `[content_start, content_end)`.
5. Verify that the detected role matches the expected role in `expected_turns[turn_idx]`.
6. For assistant turns with `enable_thinking=False`: filter out `<think>...</think>` blocks and trim whitespace-only tokens from boundaries (identical logic to Sub-Task 12.5a).
7. Build span dict. Note: `start = min(final_indices)` and `end = max(final_indices) + 1`, which handles potential non-contiguous indices after thinking-token filtering.

#### Output

Same `Tuple[List[int], List[Dict[str, Any]]]` as the parent `build_turn_spans`.

---

### Sub-Task 12.6b: Build Turn Spans -- Fallback (`_build_turn_spans_fallback`)

#### Input

| Parameter      | Type                       | Description |
|----------------|----------------------------|-------------|
| `conversation` | `List[Dict[str, str]]`     | Message list. |
| `full_ids`     | `List[int]`                | Pre-tokenized full conversation IDs. |
| `**chat_kwargs`| --                         | Forwarded kwargs. |

#### Processing

```python
def _build_turn_spans_fallback(
    self,
    conversation: List[Dict[str, str]],
    full_ids: List[int],
    **chat_kwargs,
) -> Tuple[List[int], List[Dict[str, Any]]]:
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

        abs_start = self._find_subsequence(full_ids, content_ids)
        if abs_start == -1:
            msgs_before.append(msg)
            continue
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

**Algorithm**:

1. For each non-system message, extract content-only token IDs via `_content_only_ids_and_offset`.
2. Use `_find_subsequence` to locate those content IDs within `full_ids`.
3. If not found (`-1`), skip the turn.
4. Otherwise, build the span dict from the found position.

#### Output

Same `Tuple[List[int], List[Dict[str, Any]]]` as the parent.

---

### Sub-Task 12.7: Build Batch Turn Spans (`build_batch_turn_spans`)

#### Input

| Parameter       | Type                                  | Default | Description |
|-----------------|---------------------------------------|---------|-------------|
| `conversations` | `List[List[Dict[str, str]]]`          | --      | A list of conversations (each itself a list of message dicts). |
| `**chat_kwargs` | --                                    | --      | Forwarded to `build_turn_spans` / `apply_chat_template`. |

#### Processing

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

**Algorithm**:

1. Initialise `global_offset = 0` to track the running token position across concatenated conversations.
2. For each conversation:
   a. Call `build_turn_spans` to get `full_ids` and `spans`.
   b. Record the conversation's token length and its global offset.
   c. For each span, create an `enhanced_span` that copies all original fields and adds:
      - `conversation_id`: which conversation this span belongs to.
      - `local_start` / `local_end`: the span's position within its own conversation.
      - `global_start` / `global_end`: the span's position in the hypothetical concatenation of all conversations.
   d. Advance `global_offset` by the conversation's token count.

#### Output

`Tuple[List[List[int]], List[Dict[str, Any]], Dict[str, Any]]`:

- `batch_full_ids`: `List[List[int]]` -- token IDs for each conversation.
- `batch_spans`: flat `List[Dict]` of enhanced span dicts. Each dict contains all fields from `build_turn_spans` plus:

| Key               | Type  | Description |
|-------------------|-------|-------------|
| `conversation_id` | `int` | Index of the conversation in the batch. |
| `local_start`     | `int` | Same as `start` (position within conversation). |
| `local_end`       | `int` | Same as `end` (position within conversation). |
| `global_start`    | `int` | Position in concatenated batch. |
| `global_end`      | `int` | Position in concatenated batch (exclusive). |

- `batch_metadata`: `Dict[str, Any]` with:

| Key                      | Type         | Description |
|--------------------------|--------------|-------------|
| `conversation_lengths`   | `List[int]`  | Token count per conversation. |
| `total_conversations`    | `int`        | Number of conversations in the batch. |
| `conversation_offsets`   | `List[int]`  | Cumulative token offset for each conversation. |

---

### Sub-Task 12.8: Code Block Token Mask (`code_block_token_mask`)

#### Input

| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `text`    | `str` | --      | The text string to analyse for code blocks. |

#### Processing

```python
def code_block_token_mask(self, text: str) -> torch.Tensor:
    tokenized = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = tokenized['input_ids']
    offset_mapping = tokenized['offset_mapping']

    n_tokens = len(token_ids)
    exclude_mask = torch.zeros(n_tokens, dtype=torch.bool)

    if n_tokens == 0:
        return exclude_mask

    code_regions = []

    # Triple backticks first (take precedence)
    triple_pattern = r'```[\s\S]*?```'
    for match in re.finditer(triple_pattern, text):
        code_regions.append((match.start(), match.end()))

    # Single backticks, excluding those inside triple-backtick regions
    single_pattern = r'`[^`\n]*?`'
    for match in re.finditer(single_pattern, text):
        start, end = match.start(), match.end()
        overlaps = any(triple_start <= start < triple_end or triple_start < end <= triple_end
                      for triple_start, triple_end in code_regions)
        if not overlaps:
            code_regions.append((start, end))

    # Map character regions to token indices
    for char_start, char_end in code_regions:
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if (token_start < char_end and token_end > char_start):
                exclude_mask[i] = True

    return exclude_mask
```

**Algorithm (step by step)**:

1. Tokenize `text` with `return_offsets_mapping=True` and `add_special_tokens=False` to get token IDs and their `(start_char, end_char)` tuples.
2. Initialise a boolean mask of shape `(n_tokens,)` filled with `False`.
3. Find all **triple-backtick** fenced code blocks via regex `` ```[\s\S]*?``` `` (non-greedy, spanning newlines). Record their `(start, end)` character positions. These take precedence.
4. Find all **single-backtick** inline code spans via regex `` `[^`\n]*?` `` (non-greedy, no newlines). For each match, check whether it overlaps with any already-found triple-backtick region; if so, skip it. Otherwise, add it to `code_regions`.
5. For each code region, mark every token whose character span overlaps with the region (simple interval overlap check: `token_start < char_end and token_end > char_start`).

#### Output

`torch.Tensor` of shape `(n_tokens,)` with `dtype=torch.bool`. `True` at position `i` means token `i` falls inside a code block and should be excluded from downstream processing.

---

### Sub-Task 12.9: Content-Only IDs and Offset -- Dispatcher (`_content_only_ids_and_offset`)

#### Input

| Parameter        | Type                       | Description |
|------------------|----------------------------|-------------|
| `messages_before`| `List[Dict[str, str]]`     | All messages preceding the current turn. |
| `role`           | `str`                      | Role of the current turn (`"user"`, `"assistant"`, `"system"`). |
| `content`        | `str`                      | Content text of the current turn. |
| `**chat_kwargs`  | --                         | Forwarded to `apply_chat_template`. |

#### Processing

```python
def _content_only_ids_and_offset(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
    if self._is_qwen() and role == "assistant":
        return self._content_only_ids_and_offset_qwen(messages_before, role, content, **chat_kwargs)
    else:
        return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)
```

Dispatches to Qwen-specific logic only for assistant turns on Qwen models; everything else goes to the standard path.

#### Output

`Tuple[List[int], int]`:
- `content_ids`: `List[int]` -- token IDs representing **only** the message content (no template tokens).
- `start_in_delta`: `int` -- offset of `content_ids` within the "delta" (the suffix of tokens introduced by adding this message).

---

### Sub-Task 12.9a: Content-Only IDs -- Qwen Variant (`_content_only_ids_and_offset_qwen`)

#### Input

Same as `_content_only_ids_and_offset`.

#### Processing

```python
def _content_only_ids_and_offset_qwen(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
    if role == "assistant":
        msgs_full = messages_before + [{"role": role, "content": content}]
        ids_full = self.tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False, **chat_kwargs
        )

        plain = self.tokenizer(content, add_special_tokens=False).input_ids
        content_start = self._find_subsequence(ids_full, plain)

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

    return self._content_only_ids_and_offset_standard(messages_before, role, content, **chat_kwargs)
```

**Algorithm**:

1. Tokenize the full conversation (messages_before + this assistant turn).
2. Tokenize the raw content string directly (no template).
3. Find where the raw content tokens appear as a subsequence in the full token list.
4. If found, compute `start_in_delta` relative to the prefix length (tokens before this turn).
5. Return `(plain_content_ids, max(0, start_in_delta))`.
6. If the content tokens are not found as a subsequence, fall through to the standard method.

This avoids issues with Qwen's `<think>...</think>` tokens that get injected into assistant turns by the chat template even when thinking is disabled.

#### Output

Same as `_content_only_ids_and_offset`.

---

### Sub-Task 12.9b: Content-Only IDs -- Standard Variant (`_content_only_ids_and_offset_standard`)

#### Input

Same as `_content_only_ids_and_offset`.

#### Processing

```python
def _content_only_ids_and_offset_standard(
    self,
    messages_before: List[Dict[str, str]],
    role: str,
    content: str,
    **chat_kwargs,
) -> Tuple[List[int], int]:
    msgs_empty = messages_before + [{"role": role, "content": ""}]
    msgs_full  = messages_before + [{"role": role, "content": content}]

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

    pref = self._longest_common_prefix_len(ids_full, ids_empty)
    delta = ids_full[pref:]
    delta = self._strip_trailing_special(delta, set(self.tokenizer.all_special_ids))

    plain = self.tokenizer(content, add_special_tokens=False).input_ids
    sp    = self.tokenizer(" " + content, add_special_tokens=False).input_ids

    start = self._find_subsequence(delta, plain)
    use = plain
    if start == -1:
        start = self._find_subsequence(delta, sp)
        use = sp if start != -1 else plain

    if start == -1:
        return delta, 0
    else:
        return delta[start:start+len(use)], start
```

**Algorithm**:

1. Create two message lists: one with empty content and one with actual content for this turn.
2. Tokenize both via the chat template.
3. Find the longest common prefix between the two token lists. Everything after this prefix in `ids_full` is the "delta" introduced by the content.
4. Strip trailing special tokens from the delta.
5. Tokenize the raw content in two ways: bare (`plain`) and with a leading space (`sp`), because some tokenizers fuse a leading space into the first token.
6. Try to find `plain` as a subsequence in `delta`; if that fails, try `sp`.
7. If found, return the matched slice and its offset. If not found at all, return the entire `delta` with offset `0` as a fallback.

#### Output

Same as `_content_only_ids_and_offset`.

---

### Sub-Task 12.10: Static Helper -- Longest Common Prefix Length (`_longest_common_prefix_len`)

#### Input

| Parameter | Type         | Description |
|-----------|--------------|-------------|
| `a`       | `List[int]`  | First token sequence. |
| `b`       | `List[int]`  | Second token sequence. |

#### Processing

```python
@staticmethod
def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
```

Element-wise comparison from index 0 until a mismatch is found.

#### Output

`int` -- the number of tokens that match from the start.

---

### Sub-Task 12.11: Static Helper -- Strip Trailing Special Tokens (`_strip_trailing_special`)

#### Input

| Parameter     | Type         | Description |
|---------------|--------------|-------------|
| `ids`         | `List[int]`  | Token ID sequence. |
| `special_ids` | `set`        | Set of token IDs considered "special". |

#### Processing

```python
@staticmethod
def _strip_trailing_special(ids: List[int], special_ids: set) -> List[int]:
    i = len(ids)
    while i > 0 and ids[i-1] in special_ids:
        i -= 1
    return ids[:i]
```

Walks backward from the end, removing any token whose ID is in `special_ids`.

#### Output

`List[int]` -- the input with trailing special tokens removed.

---

### Sub-Task 12.12: Static Helper -- Find Subsequence (`_find_subsequence`)

#### Input

| Parameter | Type         | Description |
|-----------|--------------|-------------|
| `hay`     | `List[int]`  | The sequence to search in (haystack). |
| `needle`  | `List[int]`  | The sequence to search for. |

#### Processing

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

Brute-force sliding-window search. Returns the index of the first occurrence of `needle` in `hay`, or `-1` if not found. Note: returns `-1` for empty needles.

#### Output

`int` -- starting index of the first match, or `-1`.
