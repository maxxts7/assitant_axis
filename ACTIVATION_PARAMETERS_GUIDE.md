# `activations.py` — Every Parameter Explained

Every parameter in every function in `ActivationExtractor`, what it does, why it exists, what values it accepts, and what happens when you change it. When a parameter introduces a concept (chat templates, inference mode, attention masks, etc.), that concept is explained in-place.

---

## Table of Contents

- [1. The Constructor — `__init__`](#1-the-constructor--__init__)
- [2. `full_conversation()`](#2-full_conversation)
- [3. `at_newline()`](#3-at_newline)
- [4. `for_prompts()`](#4-for_prompts)
- [5. `batch_conversations()`](#5-batch_conversations)
- [6. `_find_newline_position()`](#6-_find_newline_position)
- [7. The Hook Function Parameters — `module`, `input`, `output`](#7-the-hook-function-parameters--module-input-output)
- [8. Parameters Used Inside the Tokenizer Calls](#8-parameters-used-inside-the-tokenizer-calls)
- [9. The `batch_metadata` Return Dict](#9-the-batch_metadata-return-dict)
- [10. Parameters in the Pipeline Caller (`2_activations.py`)](#10-parameters-in-the-pipeline-caller-2_activationspy)

---

## 1. The Constructor — `__init__`

```python
def __init__(self, probing_model: 'ProbingModel', encoder: 'ConversationEncoder'):
```

### `probing_model: ProbingModel`

**What it is:** A wrapper object (defined in `internals/model.py`) that holds a loaded HuggingFace model and its tokenizer together.

**Why it's not just a raw model:** The extractor needs three things from the model:
1. `probing_model.model` — the actual `nn.Module` to run forward passes on
2. `probing_model.tokenizer` — to convert text to token IDs
3. `probing_model.get_layers()` — to find the layer modules for hook registration

A raw HuggingFace model doesn't have `get_layers()`. Different architectures store layers in different places (`model.model.layers` for Llama, `model.transformer.h` for GPT-2, etc.). `ProbingModel.get_layers()` tries 5 paths and returns whichever one works.

**How it's created upstream:**
```python
pm = ProbingModel("Qwen/Qwen3-32B")
# This loads the model onto GPU(s) with bfloat16 precision,
# loads the tokenizer, and sets up padding configuration.
```

**What the constructor extracts from it:**
```python
self.model = probing_model.model          # nn.Module — for forward passes
self.tokenizer = probing_model.tokenizer  # AutoTokenizer — for tokenization
self.probing_model = probing_model        # Full wrapper — for get_layers()
```

### `encoder: ConversationEncoder`

**What it is:** A wrapper around the tokenizer that knows how to format conversations using the model's chat template and find token boundaries for each turn.

**Why it's separate from ProbingModel:** Separation of concerns. `ProbingModel` manages the model. `ConversationEncoder` manages text formatting. They have no dependency on each other — both just use the same tokenizer.

**How it's created upstream:**
```python
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
# model_name is used to detect Qwen/Llama/Gemma and dispatch
# to model-specific tokenization logic.
```

**Where it's used inside ActivationExtractor:**
- `at_newline()` calls `self.encoder.format_chat()` — to apply the chat template
- `batch_conversations()` calls `self.encoder.build_batch_turn_spans()` — to tokenize and find turn boundaries

---

## 2. `full_conversation()`

```python
def full_conversation(
    self,
    conversation: Union[str, List[Dict[str, str]]],
    layer: Optional[Union[int, List[int]]] = None,
    chat_format: bool = True,
    **chat_kwargs,
) -> torch.Tensor:
```

### `conversation: Union[str, List[Dict[str, str]]]`

**What it accepts:**

**Option A — a raw string:**
```python
extractor.full_conversation("What is the meaning of life?")
```
The string is auto-wrapped into `[{"role": "user", "content": "What is the meaning of life?"}]` before processing (line 70).

**Option B — a list of message dicts:**
```python
extractor.full_conversation([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the meaning of life?"},
    {"role": "assistant", "content": "The meaning of life is a deeply personal question..."},
])
```

Each dict must have a `"role"` key (one of `"system"`, `"user"`, `"assistant"`) and a `"content"` key (the text). This is the standard HuggingFace chat format used by `apply_chat_template`.

**Why both are supported:** Convenience. For quick probing you just pass a string. For analyzing full conversations (the pipeline use case), you pass the structured format.

**What happens to it:** The conversation gets formatted via the chat template into a single string with role markers, then tokenized into integer IDs that the model can process.

---

### `layer: Optional[Union[int, List[int]]] = None`

**What it controls:** Which transformer layers to capture activations from. This is the most important parameter because it determines both *what data you get* and *how much GPU memory you use*.

**The three modes:**

| Value | Example | Hooks registered on | Output shape | Memory usage |
|-------|---------|-------------------|--------------|-------------|
| `int` | `layer=22` | Layer 22 only | `(num_tokens, hidden_dim)` | Minimal |
| `list` | `layer=[20, 22, 24]` | Layers 20, 22, 24 | `(3, num_tokens, hidden_dim)` | 3x one layer |
| `None` (default) | `layer=None` | ALL layers | `(num_layers, num_tokens, hidden_dim)` | Maximum |

**Concept — what is a layer index?** A transformer has N stacked layers (e.g., 64 for Qwen 3 32B). Layer 0 processes the embedded input first. Layer 63 produces the final representation before the prediction head. The index is just the position in this stack.

```python
# What happens for each mode:

if isinstance(layer, int):        # layer=22
    single_layer_mode = True
    layer_list = [layer]          # Register 1 hook

elif isinstance(layer, list):     # layer=[20, 22, 24]
    single_layer_mode = False
    layer_list = layer            # Register 3 hooks

else:                             # layer=None
    single_layer_mode = False
    layer_list = list(range(len(self.probing_model.get_layers())))
    # For a 64-layer model: [0, 1, 2, ..., 63] — register 64 hooks
```

**Why `single_layer_mode` matters:** It changes the return shape. When you ask for one layer, you don't want an unnecessary leading dimension. `single_layer_mode=True` strips it:

```python
if single_layer_mode:
    return activations[0]   # (num_tokens, hidden_dim) — 2D
else:
    return activations      # (num_layers, num_tokens, hidden_dim) — 3D
```

**Memory implications:** For a 200-token conversation with `hidden_dim=4096` in bfloat16:
- `layer=22` → 200 × 4096 × 2 bytes = **1.6 MB**
- `layer=None` on a 64-layer model → 64 × 200 × 4096 × 2 = **100 MB**

For a batch of 16 conversations at 500 tokens with all 64 layers: **6.4 GB**. This is why `batch_conversations()` has a `max_length` parameter.

---

### `chat_format: bool = True`

**What it controls:** Whether to apply the model's chat template before tokenization.

**When `True` (default):** The conversation goes through `apply_chat_template`, which wraps it in model-specific structural tokens:

```
Input:  [{"role": "user", "content": "Hello"}]
Output: "<|im_start|>user\nHello<|im_end|>\n"  (Qwen)
   or:  "<start_of_turn>user\nHello<end_of_turn>\n"  (Gemma)
```

**When `False`:** The `conversation` parameter is treated as a raw string and tokenized directly. No role markers, no special tokens added. You'd use this if you've already formatted the text yourself or if you're extracting activations from non-chat text (e.g., a paragraph from a book).

```python
# chat_format=False — pass a pre-formatted string
extractor.full_conversation(
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi!<|im_end|>",
    chat_format=False
)
```

**Why it exists:** Flexibility. The pipeline always uses `True` (conversations need the template). Notebooks might use `False` for raw text probing.

---

### `**chat_kwargs`

**What it is:** A catch-all for extra keyword arguments passed through to `apply_chat_template()`.

**Concept — `**kwargs` (keyword arguments):** In Python, `**chat_kwargs` collects any extra named arguments into a dictionary. If you call:

```python
extractor.full_conversation(conv, enable_thinking=False)
```

Then inside the function, `chat_kwargs` is `{"enable_thinking": False}`. This dict is unpacked into `apply_chat_template(..., **chat_kwargs)`.

**The most important kwarg in this project — `enable_thinking`:**

Qwen 3 models have a "thinking" feature where the model produces `<think>...</think>` blocks before its actual response. When `enable_thinking=False`:
- The chat template omits the thinking block from the formatted output
- `ConversationEncoder` filters out thinking tokens from span boundaries

```python
# Used in the pipeline (2_activations.py, line 70):
chat_kwargs = {}
if 'qwen' in pm.model_name.lower():
    chat_kwargs['enable_thinking'] = False   # Don't include thinking tokens
```

**Why it's not a regular parameter:** Different models support different template options. Gemma might accept `add_bos=True`, Llama might accept something else. Using `**kwargs` makes the code model-agnostic — it just passes through whatever the caller provides.

---

## 3. `at_newline()`

```python
def at_newline(
    self,
    prompt: str,
    layer: Union[int, List[int]] = 15,
    swap: bool = False,
    **chat_kwargs,
) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
```

### `prompt: str`

**What it is:** A single text string (NOT a conversation list). This is different from `full_conversation()` which accepts both.

**Why only a string?** `at_newline()` is designed for quick single-prompt probing. The prompt is passed to `self.encoder.format_chat(prompt)`, which wraps it as `[{"role": "user", "content": prompt}]` and applies the chat template with `add_generation_prompt=True`. The newline it looks for is the one after the `assistant` role marker — the point where the model is about to respond.

```
<|im_start|>user
What is the meaning of life?<|im_end|>
<|im_start|>assistant
↑ ← This is the newline at_newline() captures the activation at
```

---

### `layer: Union[int, List[int]] = 15`

Same concept as in `full_conversation()`, but with two differences:

1. **Default is 15, not None.** `at_newline()` is for quick probing, so it defaults to a single layer instead of extracting all layers. Layer 15 is a reasonable middle-ground default for smaller models.

2. **Return type changes with the mode:**

| Value | Return type |
|-------|------------|
| `int` (e.g., `layer=22`) | `torch.Tensor` — shape `(hidden_dim,)`, a single vector |
| `list` (e.g., `layer=[20, 22]`) | `Dict[int, torch.Tensor]` — `{20: tensor, 22: tensor}` |

Note: `at_newline()` returns a single vector per layer (the activation at one token position), not a matrix. Compare with `full_conversation()` which returns a matrix (activations at all token positions).

---

### `swap: bool = False`

**What it controls:** An unusual formatting mode where the roles are swapped.

**When `False` (default):** Normal chat formatting.
```
<|im_start|>user
What is the meaning of life?<|im_end|>
<|im_start|>assistant
```

**When `True`:** The prompt text is placed in the model/assistant role, and the generation prompt is set for the user role instead. This is a special technique used in some probing experiments where you want to measure the model's representation of a text *as if it were the model's own output*, not a user's input.

```python
# encoder.format_chat() with swap=True does roughly:
messages = [
    {"role": "user", "content": "Hello."},
    {"role": "model", "content": prompt}  # Your text goes here
]
# Then swaps 'model' back to 'user' in the formatted string
```

**When you'd use it:** Rare. It's for specific probing experiments, not for the main pipeline.

---

### `**chat_kwargs`

Same as in `full_conversation()`. Passed through to `self.encoder.format_chat(prompt, swap=swap, **chat_kwargs)`, which eventually calls `apply_chat_template`.

---

## 4. `for_prompts()`

```python
def for_prompts(
    self,
    prompts: List[str],
    layer: Union[int, List[int]] = 15,
    swap: bool = False,
    **chat_kwargs,
) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
```

### `prompts: List[str]`

**What it is:** A list of prompt strings. Each one is processed individually via `at_newline()`.

```python
extractor.for_prompts([
    "You are a pirate captain.",
    "You are a helpful doctor.",
    "You are an ancient philosopher.",
])
```

**Why a list of strings instead of a list of conversations?** This method is designed for batch-probing many role instructions at the newline position. Each prompt is a short string, not a multi-turn conversation. For full conversations, use `batch_conversations()`.

---

### `layer`, `swap`, `**chat_kwargs`

All identical to `at_newline()` — they're passed directly through:

```python
activation = self.at_newline(prompt, layer, swap=swap, **chat_kwargs)
```

**Return type depends on `layer`:**

| `layer` value | Return |
|---------------|--------|
| `int` | `torch.Tensor` of shape `(num_prompts, hidden_dim)` — stacked vectors |
| `list` | `Dict[int, torch.Tensor]` — each entry shape `(num_prompts, hidden_dim)` |

---

## 5. `batch_conversations()`

```python
def batch_conversations(
    self,
    conversations: List[List[Dict[str, str]]],
    layer: Optional[Union[int, List[int]]] = None,
    max_length: int = 4096,
    **chat_kwargs,
) -> tuple[torch.Tensor, Dict]:
```

This is the most parameter-dense method. It's the one used by the pipeline.

### `conversations: List[List[Dict[str, str]]]`

**What it is:** A list of conversations, where each conversation is itself a list of message dicts.

```python
extractor.batch_conversations([
    # Conversation 0
    [
        {"role": "system", "content": "You are a pirate."},
        {"role": "user", "content": "How do you handle conflict?"},
        {"role": "assistant", "content": "Arrr! I draw me cutlass..."},
    ],
    # Conversation 1
    [
        {"role": "user", "content": "What is gravity?"},
        {"role": "assistant", "content": "Gravity is a fundamental force..."},
    ],
    # ... up to batch_size conversations
])
```

**Concept — why a list of lists?** Each inner list is one conversation. The outer list is the batch. GPUs process batches in parallel — 16 conversations take about the same time as 1. The pipeline splits 1200 conversations per role into batches of 16.

**Constraints:**
- Conversations can have different numbers of turns
- Conversations can have different total token lengths
- Both are handled by padding and span metadata

---

### `layer: Optional[Union[int, List[int]]] = None`

Same three modes as `full_conversation()`. In the pipeline, this is set to a list of all layer indices:

```python
# 2_activations.py — lines 210-214
n_layers = len(pm.get_layers())
if args.layers == "all":
    layers = list(range(n_layers))  # [0, 1, 2, ..., 63]
else:
    layers = [int(x.strip()) for x in args.layers.split(",")]

# Then passed as:
extractor.batch_conversations(batch, layer=layers, ...)
```

**Note:** Unlike `full_conversation()`, this method does NOT have a `single_layer_mode` that changes the return shape. The output is always `(num_layers, batch_size, max_seq_len, hidden_dim)` regardless of whether `layer` is an int, list, or None.

---

### `max_length: int = 4096`

**What it controls:** The maximum number of tokens any conversation can have. Conversations longer than this are truncated (tokens beyond `max_length` are chopped off). All conversations are then padded to the length of the longest remaining one (up to `max_length`).

**Concept — why truncation is necessary:**

In a batch, every conversation must be padded to the same length. If one conversation is 8000 tokens and the other 15 are ~200 tokens, you'd pad all 16 to 8000 tokens. That means:
- GPU processes 16 × 8000 = 128,000 token positions
- But only 16 × 200 + 8000 = 11,200 are real tokens
- 91% of the computation is wasted on padding

Worse, the activation tensor would be `64 layers × 16 batch × 8000 tokens × 4096 hidden × 2 bytes = 64 GB`. That won't fit on any GPU.

`max_length` prevents this by capping the sequence length.

```python
# activations.py — lines 290-291
actual_max_len = max(len(ids) for ids in batch_full_ids)
max_seq_len = min(max_length, actual_max_len)
```

**In the pipeline** it's set to 2048:
```python
# 2_activations.py — line 85
extractor.batch_conversations(batch, max_length=max_length, ...)
# where max_length comes from --max_length 2048 (default)
```

**What happens to truncated turns?** If truncation cuts off part of the assistant's response, `SpanMapper` handles it gracefully:

```python
# spans.py — lines 83-89
actual_length = batch_metadata['truncated_lengths'][conv_id]
if start_idx >= actual_length:
    continue   # Span is entirely in the truncated region — skip it
end_idx = min(end_idx, actual_length)   # Span is partially truncated — use what's left
```

---

### `**chat_kwargs`

Passed through to `self.encoder.build_batch_turn_spans(conversations, **chat_kwargs)`.

In this method, `chat_kwargs` flows through two paths:
1. Into `build_batch_turn_spans()` → which calls `build_turn_spans()` for each conversation → which calls `apply_chat_template()` for tokenization
2. The span computation uses it to handle model-specific tokens (e.g., Qwen's `enable_thinking`)

---

### Return value: `tuple[torch.Tensor, Dict]`

This method returns TWO things, not one. Let's explain both.

**`batch_activations: torch.Tensor`** — shape `(num_layers, batch_size, max_seq_len, hidden_dim)`

This is the raw activation data for every token at every requested layer. Includes padding positions (which contain garbage activations).

Example: for 64 layers, 16 conversations, max 512 tokens, hidden dim 4096:
```
(64, 16, 512, 4096) in bfloat16 = 64 × 16 × 512 × 4096 × 2 bytes = 4.3 GB
```

**`batch_metadata: Dict`** — explained in [Section 9](#9-the-batch_metadata-return-dict).

---

## 6. `_find_newline_position()`

```python
def _find_newline_position(self, input_ids: torch.Tensor) -> int:
```

### `input_ids: torch.Tensor`

**What it is:** A 1D tensor of integer token IDs. Note the `[0]` when this is called:

```python
newline_pos = self._find_newline_position(input_ids[0])
#                                                  ^^^
# input_ids has shape (1, num_tokens) — the [0] removes the batch dim
# So this receives a (num_tokens,) 1D tensor
```

**What the method does with it:** Scans for the token ID that represents `"\n\n"` (double newline) or `"\n"` (single newline). Returns the index of the last occurrence.

```python
# Step 1: What integer does "\n\n" encode to?
newline_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[0]
# e.g., for Qwen this might be token ID 198

# Step 2: Find all positions in input_ids where that token appears
newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
# e.g., tensor([15, 42, 87]) — newline appears at positions 15, 42, and 87

# Step 3: Take the last one
return newline_positions[-1].item()  # 87
```

**Why `.item()`?** Converts a single-element tensor to a plain Python `int`. PyTorch indexing expects an int, not a tensor.

**Why `add_special_tokens=False`?** When encoding `"\n\n"`, we want ONLY the newline token ID. With `add_special_tokens=True`, the encoder would prepend a BOS token, giving us `[BOS_id, newline_id]` — and `[0]` would grab the BOS instead.

**Fallback chain:**
1. Try `"\n\n"` → if found, return last position
2. Try `"\n"` → if found, return last position
3. Use `len(input_ids) - 1` → last token in the sequence

---

## 7. The Hook Function Parameters — `module`, `input`, `output`

Every hook function in this file has the same signature:

```python
def hook_fn(module, input, output):
```

These three parameters are provided by PyTorch automatically when the hook fires. You don't pass them yourself.

### `module`

**What it is:** The `nn.Module` object the hook is attached to — i.e., the specific transformer layer. For example, if you registered the hook on `model.model.layers[22]`, then `module` IS that layer 22 object.

**Not used in this code.** The hook knows which layer it's on via the closure over `layer_idx`. `module` would be useful if you needed to inspect the layer's weights or configuration inside the hook, but that's not needed here.

### `input`

**What it is:** A tuple containing whatever was passed as input to the layer's `forward()` method. Typically `(hidden_states, attention_mask, position_ids, ...)`.

**Not used in this code.** We only care about the output (what the layer produced), not the input (what it received).

### `output`

**What it is:** Whatever the layer's `forward()` method returned. This is the critical parameter — it contains the activations we want.

**Why the tuple check:**
```python
act_tensor = output[0] if isinstance(output, tuple) else output
```

Different model architectures return different things:

| Architecture | Layer `forward()` returns | `output` type |
|-------------|--------------------------|--------------|
| Llama, Qwen | `(hidden_states, self_attn_weights, present_key_value)` | tuple of 3 |
| Gemma 2 | `(hidden_states, self_attn_weights)` | tuple of 2 |
| Some models | `hidden_states` | tensor directly |

The hidden states are always at index 0 in the tuple. The check handles both cases.

**What `act_tensor` looks like:**
```
Shape: (batch_size, num_tokens, hidden_dim)
Dtype: bfloat16 (matches the model's dtype)
Device: whichever GPU this layer lives on
```

### The three different indexing patterns across methods

Each method's hook extracts a different slice:

```python
# full_conversation() — ALL positions for ONE conversation
activations.append(act_tensor[0, :, :].cpu())
#                            ↑  ↑↑↑
#                            │   └── all hidden dims
#                            └── batch index 0 (only one conversation)
# Result: (num_tokens, hidden_dim)

# at_newline() — ONE position for ONE conversation
activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()
#                                   ↑  ↑↑↑↑↑↑↑↑↑↑↑
#                                   │       └── one specific token position
#                                   └── batch index 0
# Result: (hidden_dim,)

# batch_conversations() — ALL positions for ALL conversations
layer_outputs[layer_idx] = act_tensor
# No indexing! Keep the full (batch_size, num_tokens, hidden_dim) tensor.
# Result: (batch_size, max_seq_len, hidden_dim)
```

---

## 8. Parameters Used Inside the Tokenizer Calls

These parameters appear inside the method bodies, not in the function signatures. They control how text becomes token IDs.

### `tokenize=False` (in `apply_chat_template`)

```python
formatted_prompt = self.tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
)
```

**What it does:** `apply_chat_template` can either:
- `tokenize=True` → return a list of integer token IDs
- `tokenize=False` → return the formatted string (with role markers etc.)

This code uses `False` because it does tokenization separately in the next line. Why separate? Because `apply_chat_template(tokenize=True)` returns a plain Python list, but the model needs a `torch.Tensor` with batch dimension. Doing tokenization separately via `self.tokenizer(formatted_prompt, return_tensors="pt")` gives us the tensor format directly.

### `add_generation_prompt=False` (in `full_conversation`)

**What it does:** Controls whether the template appends an "assistant is about to speak" marker at the end.

```
With add_generation_prompt=True:
  ...<|im_end|>\n<|im_start|>assistant\n     ← added at end

With add_generation_prompt=False:
  ...<|im_end|>                              ← nothing added
```

In `full_conversation()`, this is `False` because the conversation already includes the assistant's response — we're not asking the model to generate, we're reading activations from existing text.

In `at_newline()`, the encoder calls `apply_chat_template` with `add_generation_prompt=True` (via `format_chat`) because we WANT that trailing `assistant\n` marker — that's exactly the newline we're looking for.

### `return_tensors="pt"` (in `self.tokenizer(...)`)

```python
tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
```

**Concept — `return_tensors`:** The tokenizer can return different formats:
- `"pt"` → PyTorch tensors (shape `(1, num_tokens)`)
- `"tf"` → TensorFlow tensors
- `"np"` → NumPy arrays
- `None` (default) → plain Python lists

We use `"pt"` because PyTorch tensors can be moved to GPU with `.to(device)` and fed directly into the model.

### `add_special_tokens=False` (in `self.tokenizer(...)`)

**What it does:** Prevents the tokenizer from adding its own BOS/EOS tokens around the text.

**Why `False`?** The `apply_chat_template` call already inserted all necessary special tokens into the formatted string. If `add_special_tokens=True`, you'd get duplicate BOS tokens:

```
With add_special_tokens=True:  [BOS, BOS, system, \n, You, are, ...]  ← double BOS!
With add_special_tokens=False: [BOS, system, \n, You, are, ...]       ← correct
```

### `.to(self.model.device)`

```python
input_ids = tokens["input_ids"].to(self.model.device)
```

**Concept — device:** PyTorch tensors live on a specific device (CPU or a specific GPU). The tokenizer produces tensors on CPU. The model's parameters live on GPU (e.g., `cuda:0`). If you try to multiply a CPU tensor with a GPU tensor, PyTorch throws an error. `.to(device)` moves the tensor to the right device.

For multi-GPU models (where different layers are on different GPUs), `self.model.device` returns the device of the first parameter. The model's internal logic handles moving data between GPUs during the forward pass.

---

## 9. The `batch_metadata` Return Dict

`batch_conversations()` returns a metadata dict alongside the activations. Here's every key:

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

### `conversation_lengths: List[int]`

The token count of each conversation BEFORE any padding or truncation. Example: `[187, 243, 156, ...]`.

**Used by:** Nothing in this file directly. Available for debugging or downstream analysis.

### `total_conversations: int`

The number of conversations in the batch. Same as `len(conversations)`.

**Used by:** `SpanMapper.map_spans()` to iterate over conversations: `for conv_id in range(batch_metadata['total_conversations'])`.

### `conversation_offsets: List[int]`

If you conceptually concatenated all conversations end-to-end, these would be the starting positions of each one. Example: `[0, 187, 430, 586, ...]`. Conversation 0 starts at position 0, conversation 1 starts at 187, etc.

**Used by:** Not used in the current pipeline. Available for alternative processing modes that work with concatenated sequences.

### `max_seq_len: int`

The padded sequence length — all conversations in the batch are padded to this length. It's `min(max_length, max(actual lengths))`.

**Used by:** Gives context for interpreting the activation tensor's third dimension. `batch_activations[:, :, :max_seq_len, :]` is the full sequence including padding.

### `attention_mask: torch.Tensor`

Shape: `(batch_size, max_seq_len)`. Values are `1` for real tokens and `0` for padding tokens.

**Concept — attention mask:** In the self-attention mechanism, each token computes a weighted average over all other tokens. The attention mask prevents real tokens from attending to padding tokens. Without it, the model would mix padding noise into the real activations.

```
Conversation: [You] [are] [a] [pirate] [PAD] [PAD] [PAD]
Attn mask:    [ 1 ] [ 1 ] [1] [  1   ] [ 0 ] [ 0 ] [ 0 ]

Token "pirate" can attend to: [You, are, a, pirate]     ← only mask=1 positions
Token "pirate" IGNORES:       [PAD, PAD, PAD]             ← mask=0 positions
```

**Used by:** Passed to the model during the forward pass (`self.model(input_ids=..., attention_mask=...)`). Also available for downstream consumers who need to know which positions are real.

### `actual_lengths: List[int]`

The token count of each conversation BEFORE truncation but AFTER tokenization. This is the same as `conversation_lengths` in most cases.

**Used by:** Debugging — compare with `truncated_lengths` to see if truncation occurred.

### `truncated_lengths: List[int]`

The token count of each conversation AFTER truncation to `max_seq_len`. Formula: `min(actual_length, max_seq_len)`.

**Used by:** `SpanMapper.map_spans()` to check whether a span falls in the truncated region:

```python
actual_length = batch_metadata['truncated_lengths'][conv_id]
if start_idx >= actual_length:
    continue   # This span was truncated away — skip it
```

This is the most important metadata field for correctness. Without it, `SpanMapper` would try to read activations at positions that are actually padding, getting garbage data.

---

## 10. Parameters in the Pipeline Caller (`2_activations.py`)

The pipeline function `extract_activations_batch()` wraps the extractor with its own parameters:

```python
def extract_activations_batch(
    pm: ProbingModel,
    conversations: List[List[Dict[str, str]]],
    layers: List[int],
    batch_size: int = 16,
    max_length: int = 2048,
    enable_thinking: bool = False,
) -> List[Optional[torch.Tensor]]:
```

### `pm: ProbingModel`

The loaded model. Created once per worker and reused for all roles:
```python
pm = ProbingModel(args.model)   # e.g., ProbingModel("Qwen/Qwen3-32B")
```

### `conversations: List[List[Dict[str, str]]]`

All conversations for one role. Loaded from the JSONL file produced by Step 1:
```python
responses = load_responses(role_file)      # Load pirate.jsonl
conversations = [resp["conversation"] for resp in responses]  # Extract the conversation field
```

A typical role has 1200 conversations (5 prompts × 240 questions).

### `layers: List[int]`

Which layers to extract. The pipeline defaults to ALL layers:
```python
if args.layers == "all":
    layers = list(range(n_layers))   # [0, 1, 2, ..., 63]
```

**Why all layers?** The axis is computed at every layer (`axis` has shape `(n_layers, hidden_dim)`). Extracting all layers upfront means you only need to run the forward pass once per conversation, even though the axis might ultimately be used at just one target layer. Running 64 layers once is much cheaper than running 1 layer 64 times (each forward pass processes ALL layers regardless of hooks).

### `batch_size: int = 16`

How many conversations to process in a single forward pass. Directly passed to the batching loop:

```python
for batch_start in range(0, num_conversations, batch_size):
    batch_end = min(batch_start + batch_size, num_conversations)
    batch_conversations = conversations[batch_start:batch_end]
    # Process batch_conversations (up to 16 at a time)
```

**Trade-off:**
- **Larger batch (32, 64):** Faster throughput (GPU is more utilized), but uses more memory. May cause OOM (out of memory) errors.
- **Smaller batch (4, 8):** Slower, but fits in memory. Safer for long conversations or limited GPU memory.
- **16 (default):** Balances speed and memory for typical conversations (~200-500 tokens) on an 80GB GPU.

### `max_length: int = 2048`

Passed directly to `extractor.batch_conversations(batch, max_length=max_length)`. See the [max_length explanation in Section 5](#max_length-int--4096).

**Why 2048 in the pipeline but 4096 in the method default?** Pipeline conversations are single-turn (one user question + one assistant answer), rarely exceeding 1000 tokens. 2048 is generous enough to avoid truncation while keeping memory reasonable. The method default of 4096 is for general-purpose use (notebooks with longer conversations).

### `enable_thinking: bool = False`

Controls Qwen 3's thinking feature. Converted into a chat kwarg:

```python
chat_kwargs = {}
if 'qwen' in pm.model_name.lower():
    chat_kwargs['enable_thinking'] = enable_thinking

# Then passed through to:
extractor.batch_conversations(batch, **chat_kwargs)
```

**Why `False` by default?** The thinking block (`<think>I need to figure out how to be a pirate...</think>`) is internal reasoning, not the persona-expressing response. Including thinking tokens in the activation mean would dilute the persona signal with "planning" activations.

**CLI flag:**
```bash
uv run 2_activations.py --thinking false   # Default
uv run 2_activations.py --thinking true    # Include thinking tokens
```

---

## Summary: Parameter Quick Reference

### `full_conversation(conversation, layer, chat_format, **chat_kwargs)`

| Parameter | Type | Default | One-liner |
|-----------|------|---------|-----------|
| `conversation` | `str` or `List[Dict]` | required | The text to extract activations from |
| `layer` | `int`, `List[int]`, or `None` | `None` | Which layers to hook — `None` means all |
| `chat_format` | `bool` | `True` | Apply chat template or treat as raw text |
| `**chat_kwargs` | keyword args | none | Passed to `apply_chat_template` (e.g., `enable_thinking=False`) |

### `at_newline(prompt, layer, swap, **chat_kwargs)`

| Parameter | Type | Default | One-liner |
|-----------|------|---------|-----------|
| `prompt` | `str` | required | Text to probe at the newline position |
| `layer` | `int` or `List[int]` | `15` | Which layer(s) — returns vector(s) at one token position |
| `swap` | `bool` | `False` | Put prompt in assistant role instead of user role |
| `**chat_kwargs` | keyword args | none | Passed to `format_chat` → `apply_chat_template` |

### `for_prompts(prompts, layer, swap, **chat_kwargs)`

| Parameter | Type | Default | One-liner |
|-----------|------|---------|-----------|
| `prompts` | `List[str]` | required | List of prompts — loops `at_newline()` over each |
| `layer` | `int` or `List[int]` | `15` | Same as `at_newline` |
| `swap` | `bool` | `False` | Same as `at_newline` |
| `**chat_kwargs` | keyword args | none | Same as `at_newline` |

### `batch_conversations(conversations, layer, max_length, **chat_kwargs)`

| Parameter | Type | Default | One-liner |
|-----------|------|---------|-----------|
| `conversations` | `List[List[Dict]]` | required | Batch of conversations to process in parallel |
| `layer` | `int`, `List[int]`, or `None` | `None` | Which layers — returns full `(layers, batch, tokens, hidden)` tensor |
| `max_length` | `int` | `4096` | Cap sequence length to prevent memory explosion |
| `**chat_kwargs` | keyword args | none | Passed to `build_batch_turn_spans` → `apply_chat_template` |

### `_find_newline_position(input_ids)`

| Parameter | Type | Default | One-liner |
|-----------|------|---------|-----------|
| `input_ids` | `torch.Tensor` (1D) | required | Token IDs to scan for `\n\n` or `\n` |
