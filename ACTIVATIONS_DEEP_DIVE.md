# `activations.py` — A Line-by-Line Deep Dive in Q&A Form

This document dissects `assistant_axis/internals/activations.py` — every function, every design choice, every connection to the rest of the system — structured as the questions you'd naturally ask while reading this code for the first time.

---

## Table of Contents

- [Part 1: The Big Questions](#part-1-the-big-questions)
- [Part 2: The Constructor — `__init__`](#part-2-the-constructor--__init__)
- [Part 3: `full_conversation()` — Line by Line](#part-3-full_conversation--line-by-line)
- [Part 4: Forward Hooks — The Core Mechanism](#part-4-forward-hooks--the-core-mechanism)
- [Part 5: `at_newline()` — Single-Position Extraction](#part-5-at_newline--single-position-extraction)
- [Part 6: `for_prompts()` — Looping Over Many Prompts](#part-6-for_prompts--looping-over-many-prompts)
- [Part 7: `batch_conversations()` — The Pipeline Workhorse](#part-7-batch_conversations--the-pipeline-workhorse)
- [Part 8: `_find_newline_position()` — The Helper](#part-8-_find_newline_position--the-helper)
- [Part 9: How the Pipeline (`2_activations.py`) Uses All of This](#part-9-how-the-pipeline-2_activationspy-uses-all-of-this)
- [Part 10: The Full Data Journey — End to End](#part-10-the-full-data-journey--end-to-end)

---

## Part 1: The Big Questions

### Q: What is `ActivationExtractor` and why does it exist?

A transformer model is a stack of layers. When you feed text in, data flows through each layer and gets transformed. The output of each layer is called that layer's "hidden state" or "activation" — a tensor of shape `(num_tokens, hidden_dim)`. Usually you only see the final output (the generated text). But the research behind this project needs to *look inside* the model — to grab the intermediate activations at any layer and analyze them.

`ActivationExtractor` is the class that does this. It intercepts the data as it flows through the model using PyTorch's "forward hook" mechanism, copies out whatever it needs, and gives you the raw activation tensors.

Without this class, you'd have to modify the model's source code to return intermediate states, or use HuggingFace's `output_hidden_states=True` (which has its own problems — see [Part 4](#q-why-use-hooks-instead-of-output_hidden_statestrue)).

### Q: Where does `ActivationExtractor` sit in the overall project?

It's the third link in a four-link chain:

```
ProbingModel          ConversationEncoder        ActivationExtractor        SpanMapper
(loads model)    →    (formats text, finds    →  (hooks into model,     →  (maps token-level
                       token boundaries)          captures activations)      activations to
                                                                             per-turn means)
```

- **ProbingModel** gives it the model and layer access.
- **ConversationEncoder** gives it the formatted text and span information.
- **SpanMapper** (downstream) takes its raw output and computes per-turn averages.

### Q: What is the end goal? Why do we need activations at all?

The assistant axis is computed as: `axis = mean(default_activations) - mean(role_activations)`.

To get `default_activations` and `role_activations`, you need to:
1. Prompt the model with role instructions + questions (Step 1 of the pipeline)
2. Feed those conversations back through the model and **capture what the model's neurons are doing** at each layer (Step 2 — this is where `ActivationExtractor` lives)
3. Average those activations per role, filter by quality score, and subtract

So `ActivationExtractor` is the tool that turns text conversations into numerical vectors that can be analyzed mathematically.

### Q: What are the four methods and when do you use each one?

| Method | Input | Output shape | Use case |
|--------|-------|-------------|----------|
| `full_conversation()` | One conversation | `(num_tokens, hidden_dim)` or `(num_layers, num_tokens, hidden_dim)` | Notebooks, single conversation analysis |
| `at_newline()` | One prompt string | `(hidden_dim,)` single vector | Quick single-vector probing at a specific token position |
| `for_prompts()` | List of prompt strings | `(num_prompts, hidden_dim)` | Batch probing of many prompts (loops `at_newline`) |
| `batch_conversations()` | List of conversations | `(num_layers, batch_size, max_seq_len, hidden_dim)` | **Pipeline Step 2** — efficient batch extraction |

The first three are for interactive/notebook use. The last one is the heavy-duty method used by the pipeline.

---

## Part 2: The Constructor — `__init__`

```python
def __init__(self, probing_model: 'ProbingModel', encoder: 'ConversationEncoder'):
    self.model = probing_model.model          # The raw HF nn.Module
    self.tokenizer = probing_model.tokenizer  # The HF tokenizer
    self.probing_model = probing_model        # The full ProbingModel wrapper
    self.encoder = encoder                    # For formatting conversations
```

### Q: Why does the constructor store both `self.model` and `self.probing_model`? Isn't that redundant?

Yes, `self.model` is just `probing_model.model`. It's stored separately for convenience — most methods call `self.model(input_ids)` frequently, so `self.model` saves typing `self.probing_model.model` everywhere. But `self.probing_model` is also kept because some operations need it:
- `self.probing_model.get_layers()` — to find the layer list for hook registration
- `self.probing_model.get_layers().__len__()` — to know how many layers exist (when `layer=None`)

### Q: Why does it need a `ConversationEncoder` at all?

Only `at_newline()` uses it directly (via `self.encoder.format_chat()`). The `batch_conversations()` method also uses it indirectly — it calls `self.encoder.build_batch_turn_spans()` to tokenize conversations and find turn boundaries.

`full_conversation()` doesn't use the encoder — it calls `self.tokenizer.apply_chat_template()` directly. This is a slight inconsistency in the design, but it works because `full_conversation()` doesn't need span information.

### Q: What's the `TYPE_CHECKING` import at the top?

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ProbingModel
    from .conversation import ConversationEncoder
```

This is a Python pattern to avoid circular imports. At runtime, `activations.py` never imports `model.py` or `conversation.py` — it just receives those objects as constructor arguments. But for type checkers (mypy, IDE autocompletion), the imports exist so they can validate the type hints. The `if TYPE_CHECKING:` block is `False` at runtime but `True` when a type checker analyzes the code.

Why would there be a circular import? Because `spans.py` imports from `conversation.py`, `conversation.py` could potentially reference `activations.py`, etc. This pattern breaks the cycle.

---

## Part 3: `full_conversation()` — Line by Line

This is the simplest extraction method. Let's walk through every line.

### Q: What does the `layer` parameter's three modes mean?

```python
if isinstance(layer, int):        # "Give me just layer 22"
    single_layer_mode = True
    layer_list = [layer]
elif isinstance(layer, list):     # "Give me layers [20, 22, 24]"
    single_layer_mode = False
    layer_list = layer
else:                             # layer=None → "Give me ALL layers"
    single_layer_mode = False
    layer_list = list(range(len(self.probing_model.get_layers())))
```

`single_layer_mode` is tracked so the return type can differ:
- Single layer → `(num_tokens, hidden_dim)` — just the 2D matrix
- Multiple layers → `(num_layers, num_tokens, hidden_dim)` — 3D, with a layer dimension

This is a convenience for callers. If you ask for one layer, you probably don't want to deal with an extra dimension.

### Q: What does `apply_chat_template` do and why `add_generation_prompt=False`?

```python
formatted_prompt = self.tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
)
```

`apply_chat_template` converts a list of `{"role": "user", "content": "..."}` dicts into the model-specific format. For example, for Qwen it wraps each message in `<|im_start|>user\n...<|im_end|>` tags. For Gemma it uses `<start_of_turn>user\n...<end_of_turn>`.

`tokenize=False` means "give me the string, not token IDs" — because the next line does its own tokenization.

`add_generation_prompt=False` means "don't append the marker that says 'now generate a response.'" We don't want the model to generate here — we already have the full conversation including the assistant's response. We just want to read the activations from the existing text.

### Q: Why `add_special_tokens=False` during tokenization?

```python
tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
```

Because `apply_chat_template` already added all the special tokens (BOS, role markers, etc.) into the formatted string. If you also set `add_special_tokens=True`, you'd get duplicate BOS tokens or other model-specific duplicates.

### Q: What does `.to(self.model.device)` do?

```python
input_ids = tokens["input_ids"].to(self.model.device)
```

The tokenizer produces tensors on CPU. The model might be on `cuda:0` (or sharded across GPUs). This moves the input to wherever the model's first parameters live, so the forward pass doesn't crash with a device mismatch.

### Q: Walk me through the hook registration and forward pass.

```python
activations = []        # Will be filled by hook callbacks
handles = []            # Track hooks so we can remove them

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

try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    for handle in handles:
        handle.remove()
```

Here's the sequence of events:

1. **Before the forward pass:** Hooks are registered on specific layer modules. Each hook is a function that will be called automatically by PyTorch when data passes through that layer.

2. **During the forward pass (`self.model(input_ids)`):** Data flows through the model layer by layer. When it exits a hooked layer, PyTorch calls that layer's hook function with `(module, input, output)`. The hook grabs the output tensor and appends it to the `activations` list.

3. **After the forward pass:** All hooks are removed via `handle.remove()`. This is critical — if you forget, the hooks stay attached and fire on every future forward pass, leaking memory and corrupting behavior.

4. **The `try/finally` block** ensures hooks are removed even if the forward pass crashes (e.g., out of memory).

See [Part 4](#part-4-forward-hooks--the-core-mechanism) for much more on hooks.

---

## Part 4: Forward Hooks — The Core Mechanism

### Q: What exactly is a PyTorch forward hook?

A forward hook is a callback function you attach to any `nn.Module`. PyTorch guarantees that after the module's `forward()` method runs, your hook will be called with:

```python
hook_fn(module, input, output)
#        │       │       │
#        │       │       └── What the module produced (tensor or tuple of tensors)
#        │       └── What the module received (tuple of tensors)
#        └── The module itself (e.g., the TransformerDecoderLayer object)
```

You register it like this:

```python
handle = some_module.register_forward_hook(my_hook_fn)
```

And remove it like this:

```python
handle.remove()
```

The `handle` is a `RemovableHandle` — it's the only way to detach the hook later.

### Q: Why use hooks instead of `output_hidden_states=True`?

HuggingFace models support `output_hidden_states=True`, which returns all layer outputs. So why not use that?

The code itself has a comment on line 318: `"Extract activations using hooks (more reliable than output_hidden_states)"`. The reasons:

1. **Not all models support it identically.** Some models return hidden states in different formats, some include the embedding layer output, some don't. Hooks give you consistent access to exactly the layer module you target.

2. **Memory.** `output_hidden_states=True` materializes ALL layers' hidden states simultaneously in GPU memory. For a 70B model with 80 layers and a 4096-token sequence, that's 80 × 4096 × 8192 × 2 bytes = ~5 GB of extra GPU memory. With hooks, `full_conversation()` moves each layer's output to CPU immediately (`.cpu()`) and only keeps one layer in GPU memory at a time.

3. **Selectivity.** With hooks, you can extract just layers [22, 40] without paying the cost for all 80 layers.

4. **Works with any model architecture.** Hooks work on any `nn.Module`, not just HuggingFace models with the `output_hidden_states` flag.

### Q: Why does the hook do `output[0] if isinstance(output, tuple) else output`?

Different model architectures return different things from their layer modules:

- **Llama/Qwen:** Each layer returns a **tuple** `(hidden_states, attention_weights, ...)`. The hidden states are at index 0.
- **Some models:** Return just a tensor directly.

So the hook checks: "Did I get a tuple? Take the first element. Otherwise use it directly." This makes the code work across model architectures without knowing in advance what the layer returns.

### Q: Why does `full_conversation` do `act_tensor[0, :, :]` — what's the `[0]`?

```python
activations.append(act_tensor[0, :, :].cpu())
```

The `0` removes the **batch dimension**. Even though `full_conversation()` processes a single conversation, PyTorch tensors always have a batch dimension. `act_tensor` has shape `(1, num_tokens, hidden_dim)`. The `[0]` turns it into `(num_tokens, hidden_dim)`.

### Q: Why `.cpu()` — why not keep everything on GPU?

Two reasons:

1. **Memory.** If you're extracting activations from all 64 layers of a Qwen 3 32B model, keeping all of them on GPU would use gigabytes. Moving each to CPU immediately frees GPU memory for the next layer.

2. **Multi-GPU models.** When a model is sharded across GPUs (e.g., layers 0-31 on GPU 0, layers 32-63 on GPU 1), different layers' outputs live on different devices. You can't `torch.stack()` tensors from different devices. Moving everything to CPU gives a common ground.

Note: `batch_conversations()` does NOT do `.cpu()` — it keeps activations on GPU because `SpanMapper` needs to compute per-turn means on GPU for speed, and it handles the multi-device issue differently (via `.to(target_device)`).

### Q: Why use a factory function `create_hook_fn(layer_idx)` instead of a lambda?

```python
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        ...
    return hook_fn
```

This is a classic Python closure gotcha. If you wrote:

```python
for layer_idx in layer_list:
    handle = model_layers[layer_idx].register_forward_hook(
        lambda module, input, output: activations.append(...)  # BUG!
    )
```

All lambdas would capture the *same* `layer_idx` variable (which ends up as the last value after the loop). The factory function creates a new scope for each iteration, so each `hook_fn` captures its own copy of `layer_idx`.

In `full_conversation()`, the factory doesn't actually use `layer_idx` inside the hook (it just appends to a list in order), so the bug wouldn't manifest. But in `at_newline()`, the hook stores results in `activations[layer_idx]`, so the factory is essential there. The code uses the factory consistently across all methods for safety.

### Q: Why `torch.inference_mode()` instead of `torch.no_grad()`?

```python
with torch.inference_mode():
    _ = self.model(input_ids)
```

`torch.inference_mode()` is a stricter version of `torch.no_grad()`. Both disable gradient computation, but `inference_mode` also disables autograd's version counting and other bookkeeping, making it faster and more memory-efficient. Since we never need gradients (we're only reading activations, not training), `inference_mode` is the right choice.

### Q: Why `_ = self.model(input_ids)` — why assign to `_`?

The model's return value (logits, past_key_values, etc.) is not needed. We only care about the hook side effects. Assigning to `_` is Python convention for "I'm intentionally ignoring this value."

---

## Part 5: `at_newline()` — Single-Position Extraction

### Q: What is `at_newline()` for? Why the newline token specifically?

Unlike `full_conversation()` which captures ALL token positions, `at_newline()` captures just ONE position — the last newline token. This gives you a single vector instead of a matrix.

Why the newline? In chat templates, the model's response is typically preceded by a newline:

```
<start_of_turn>model
Hello! How can I help you today?<end_of_turn>
                                ^
                    This newline (before the response) contains
                    a compressed representation of "I'm about
                    to respond as [this persona]"
```

The activation at this token position encodes the model's "state of mind" right before it starts responding — which persona it's about to adopt. This makes it a good single-point probe for measuring persona-ness.

### Q: Why does `at_newline()` use `self.encoder.format_chat()` but `full_conversation()` uses `self.tokenizer.apply_chat_template()` directly?

```python
# at_newline:
formatted_prompt = self.encoder.format_chat(prompt, swap=swap, **chat_kwargs)

# full_conversation:
formatted_prompt = self.tokenizer.apply_chat_template(conversation, ...)
```

`at_newline()` supports the `swap` parameter (swapped role formatting, a special case), which is implemented in `ConversationEncoder.format_chat()`. `full_conversation()` doesn't need swap, so it goes straight to the tokenizer.

This is another slight inconsistency — `full_conversation()` could theoretically use the encoder too. It works because `encoder.format_chat()` calls `apply_chat_template` internally anyway.

### Q: Why does the hook in `at_newline()` index into a specific position?

```python
# full_conversation hook:
activations.append(act_tensor[0, :, :].cpu())        # ALL positions → (num_tokens, hidden_dim)

# at_newline hook:
activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()  # ONE position → (hidden_dim,)
```

The difference: `full_conversation` grabs the full `(num_tokens, hidden_dim)` matrix. `at_newline` grabs only the row at `newline_pos`, giving a single `(hidden_dim,)` vector.

Also note: `at_newline` uses a dict (`activations[layer_idx] = ...`) instead of a list (`activations.append(...)`), because with a dict you can verify that all requested layers were captured (the check at line 182-184).

---

## Part 6: `for_prompts()` — Looping Over Many Prompts

### Q: What does `for_prompts()` do?

It loops over a list of prompt strings, calling `at_newline()` on each one, and stacks the results into a single tensor of shape `(num_prompts, hidden_dim)`.

### Q: Why doesn't it batch them into a single forward pass?

Each prompt may have a different length. `at_newline()` needs to find the newline position in each prompt individually. Batching would require padding, attention masks, and careful position tracking — which is what `batch_conversations()` does. `for_prompts()` is the simpler, slower approach for notebook/interactive use where convenience matters more than speed.

### Q: Why does it print checkmarks and X marks?

```python
print(f"✓ Extracted activation for: {prompt[:50]}...")
print(f"✗ Error with prompt: {prompt[:50]}... | Error: {e}")
```

This is for interactive use in notebooks. When you're probing 200 prompts one by one, you want to see progress and catch failures without the whole thing crashing. The `try/except` swallows errors from individual prompts so the loop continues.

### Q: What's the multi-layer mode branch?

When `layer` is a list (e.g., `[20, 22, 24]`), `for_prompts()` calls `at_newline()` with the full list. `at_newline()` returns a dict `{20: tensor, 22: tensor, 24: tensor}`. `for_prompts()` collects these dicts across prompts and stacks them per layer, returning `{20: (num_prompts, hidden_dim), 22: (num_prompts, hidden_dim), ...}`.

---

## Part 7: `batch_conversations()` — The Pipeline Workhorse

This is the most complex method and the one used by the pipeline. Let's go through it piece by piece.

### Q: Why does `batch_conversations()` exist? Why not just loop `full_conversation()` over each conversation?

Performance. A GPU is a massively parallel processor — it's far more efficient to process 16 conversations in one forward pass than to do 16 separate passes. The pipeline processes 276 roles × 1200 conversations = 331,200 conversations. At ~0.5 seconds per forward pass, looping would take ~46 hours. Batching at 16 cuts that to ~3 hours.

### Q: Walk me through what happens step by step.

**Step 1: Get tokenized conversations and spans.**
```python
batch_full_ids, batch_spans, span_metadata = self.encoder.build_batch_turn_spans(
    conversations, **chat_kwargs
)
```

This calls `ConversationEncoder.build_batch_turn_spans()`, which:
- Formats each conversation using the chat template
- Tokenizes each one
- Finds where each turn (user/assistant) starts and ends in the token sequence
- Returns the token IDs per conversation, the span metadata, and batch-level metadata

At this point we have: "conversation 0 is 347 tokens long, its user turn is tokens 5-42, its assistant turn is tokens 43-347."

**Step 2: Figure out which layers to extract.**
```python
if isinstance(layer, int):
    layer_list = [layer]
elif isinstance(layer, list):
    layer_list = layer
else:
    layer_list = list(range(len(self.probing_model.get_layers())))
```

Same logic as `full_conversation()`.

**Step 3: Pad all conversations to the same length.**
```python
actual_max_len = max(len(ids) for ids in batch_full_ids)
max_seq_len = min(max_length, actual_max_len)

for ids in batch_full_ids:
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]     # Truncate long ones
    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))
```

Why padding? A GPU forward pass needs a rectangular tensor — all sequences in the batch must have the same length. Short conversations get padded with the pad token. The attention mask tells the model "these positions are padding, ignore them."

Why truncation? The `max_length` parameter (default 4096) caps memory usage. A single 8192-token conversation in a batch of 16 would waste GPU memory for the other 15 shorter conversations.

**Step 4: Register hooks, run forward pass, remove hooks.**

Same pattern as `full_conversation()`, but with one key difference:

```python
# full_conversation:
activations.append(act_tensor[0, :, :].cpu())     # .cpu() immediately

# batch_conversations:
layer_outputs[layer_idx] = act_tensor              # Stays on GPU!
```

In `batch_conversations()`, activations stay on GPU. Why? Because `SpanMapper` (the next step in the pipeline) needs to slice into these tensors to extract per-turn segments and compute means. Doing that on GPU is much faster than moving to CPU first.

**Step 5: Stack and normalize.**
```python
target_device = layer_outputs[layer_list[0]].device
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
])
```

For multi-GPU models, different layers' outputs might be on different devices (layer 0's output on GPU 0, layer 40's output on GPU 1). `.to(target_device)` moves everything to the same device so `torch.stack()` works.

The final shape is `(num_layers, batch_size, max_seq_len, hidden_dim)`.

**Step 6: Return activations + metadata.**
```python
batch_metadata = {
    'conversation_lengths': ...,    # How many tokens each conversation has (pre-padding)
    'total_conversations': ...,     # How many conversations in the batch
    'conversation_offsets': ...,    # Global token offsets (for if you concatenated all conversations)
    'max_seq_len': ...,             # The padded length
    'attention_mask': ...,          # 1 = real token, 0 = padding
    'actual_lengths': ...,          # Original lengths before truncation
    'truncated_lengths': ...,       # Lengths after truncation (min(actual, max_length))
}
```

The metadata is essential for `SpanMapper` to know which token positions are real vs. padding, and where each conversation's turns are.

### Q: Why does `batch_conversations()` force `bfloat16`?

```python
if selected_activations.dtype != torch.bfloat16:
    selected_activations = selected_activations.to(torch.bfloat16)
```

The model runs in `bfloat16` (set in `ProbingModel.__init__` with `dtype=torch.bfloat16`). But some operations might promote to `float32`. This ensures the output is consistently `bfloat16`, which matters for:
1. Memory — `bfloat16` is half the size of `float32`
2. Downstream consistency — `SpanMapper` checks and preserves the dtype

### Q: What does `batch_conversations()` NOT do?

It returns raw per-token activations for the full padded sequence. It does NOT:
- Separate user turns from assistant turns
- Compute any averages
- Filter out padding positions

All of that is `SpanMapper`'s job. `batch_conversations()` is purely about "run the model, capture the activations, give them back with enough metadata for someone else to slice them up."

---

## Part 8: `_find_newline_position()` — The Helper

```python
def _find_newline_position(self, input_ids: torch.Tensor) -> int:
```

### Q: Why try `\n\n` before `\n`?

Different models use different newline conventions. Some chat templates insert `\n\n` (double newline) between the role marker and the content. Others use single `\n`. By trying `\n\n` first, the method prefers the more distinctive marker, reducing false matches.

### Q: Why the last occurrence?

```python
return newline_positions[-1].item()  # Use the last occurrence
```

In a multi-turn conversation, there may be multiple newlines. The last one is typically right before the model's final response — the most recent "about to speak" moment, which is what we want to probe.

### Q: Why the fallback to `len(input_ids) - 1`?

Some tokenizers might not produce a separate newline token (e.g., the newline could be fused into another token). In that case, using the last token is a reasonable fallback — it still captures the model's state at the end of the input.

---

## Part 9: How the Pipeline (`2_activations.py`) Uses All of This

Now that you understand `ActivationExtractor`, here's how the pipeline script wires it all together.

### Q: What is the overall flow of `2_activations.py`?

```
For each role (e.g., "pirate", "doctor", "default"):
    1. Load the JSONL file of conversations from Step 1
    2. Break conversations into batches of 16
    3. For each batch:
        a. ActivationExtractor.batch_conversations() → raw activations
        b. ConversationEncoder.build_batch_turn_spans() → span boundaries
        c. SpanMapper.map_spans() → per-turn mean activations
        d. Extract only assistant turns, average them
    4. Save as {role}.pt
```

### Q: Show me the exact function call chain for one batch.

```python
# In extract_activations_batch():

# Step A: Create the three objects
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
extractor = ActivationExtractor(pm, encoder)
span_mapper = SpanMapper(pm.tokenizer)

# Step B: Run the model and capture activations
batch_activations, batch_metadata = extractor.batch_conversations(
    batch_conversations, layer=layers, max_length=max_length, **chat_kwargs
)
# batch_activations shape: (num_layers, batch_size, max_seq_len, hidden_size)

# Step C: Build spans (again — yes, this duplicates work from inside batch_conversations)
_, batch_spans, span_metadata = encoder.build_batch_turn_spans(
    batch_conversations, **chat_kwargs
)

# Step D: Map spans to per-turn means
conv_activations_list = span_mapper.map_spans(
    batch_activations, batch_spans, batch_metadata
)
# conv_activations_list: list of tensors, each (num_turns, num_layers, hidden_size)

# Step E: Extract assistant turns only
for conv_acts in conv_activations_list:
    assistant_act = conv_acts[1::2]  # Odd indices = assistant turns
    mean_act = assistant_act.mean(dim=0)  # Average across all assistant turns
    # mean_act shape: (num_layers, hidden_size)
```

### Q: Why does it call `build_batch_turn_spans()` TWICE — once inside `batch_conversations()` and once outside?

This is a design quirk. Inside `batch_conversations()`, `build_batch_turn_spans()` is called to tokenize the conversations (it needs the token IDs to build the input tensor). The spans are computed as a side effect but are NOT returned — `batch_conversations()` only returns activations + metadata.

The pipeline script needs those spans to call `SpanMapper.map_spans()`, so it calls `build_batch_turn_spans()` again. The tokenization work is duplicated. This could be optimized by having `batch_conversations()` return the spans too, but it would change the API.

### Q: What does `conv_acts[1::2]` mean — "odd indices = assistant turns"?

In a typical single-turn conversation:
```
Turn 0: user      → "What is the meaning of life?"
Turn 1: assistant  → "The meaning of life is..."
```

In a multi-turn conversation:
```
Turn 0: user       Turn 1: assistant       Turn 2: user       Turn 3: assistant
```

`SpanMapper` returns turns in chronological order. The convention is: system prompts are skipped (they're not separate turns in the span output), user turns are at even indices, assistant turns at odd indices. `[1::2]` is Python slice syntax for "start at index 1, take every 2nd element" — i.e., all the assistant turns.

### Q: What's the final output for one role?

A `.pt` file containing a dict like:

```python
{
    "pos_p0_q0":   tensor(shape=[64, 4096]),   # layers × hidden_dim
    "pos_p0_q1":   tensor(shape=[64, 4096]),
    "pos_p0_q2":   tensor(shape=[64, 4096]),
    ...
    "pos_p4_q239": tensor(shape=[64, 4096]),
}
```

The key format is `{label}_p{prompt_index}_q{question_index}`:
- `pos` = the instruction label (all roles use `"pos"`)
- `p0`-`p4` = which of the 5 instruction variants was used
- `q0`-`q239` = which of the 240 questions was asked

Each value is the mean activation across all assistant tokens in that conversation, at every layer.

### Q: How does multi-worker parallelism work?

The pipeline needs to process ~276 roles. With 8 GPUs and a model that needs 2 GPUs (tensor parallel):

```
8 GPUs ÷ 2 GPUs/worker = 4 workers

Worker 0 (GPU 0,1): roles [aberration, absurdist, ..., chef]     ~69 roles
Worker 1 (GPU 2,3): roles [chemist, chimera, ..., editor]        ~69 roles
Worker 2 (GPU 4,5): roles [egregore, elder, ..., librarian]      ~69 roles
Worker 3 (GPU 6,7): roles [linguist, machinist, ..., zealot]     ~69 roles
```

Each worker:
1. Sets `CUDA_VISIBLE_DEVICES` to its assigned GPUs
2. Loads its own copy of the model (ProbingModel)
3. Processes its assigned roles sequentially
4. They all run simultaneously as separate OS processes (`torch.multiprocessing`)

The key mechanism is `mp.set_start_method('spawn')` + `mp.Process(target=process_roles_on_worker, ...)`. `spawn` creates a fresh Python interpreter for each worker (unlike `fork`, which copies the parent process). This is required because CUDA contexts can't be forked.

### Q: Why is the encoder re-created per batch inside `extract_activations_batch()` instead of once at the top?

```python
def extract_activations_batch(pm, conversations, ...):
    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)   # Created here
    extractor = ActivationExtractor(pm, encoder)                  # And here
    span_mapper = SpanMapper(pm.tokenizer)                        # And here
```

These objects are lightweight (they just store references, no heavy computation in `__init__`). Creating them per call is cleaner than managing object lifetimes across the function boundary. There's no performance cost.

---

## Part 10: The Full Data Journey — End to End

Let's trace one conversation from raw text to final axis contribution.

### The conversation

```python
# From pirate.jsonl (output of Step 1):
{
    "system_prompt": "You are a pirate captain on the high seas...",
    "prompt_index": 2,
    "question_index": 47,
    "question": "How do you handle conflict with others?",
    "conversation": [
        {"role": "system", "content": "You are a pirate captain on the high seas..."},
        {"role": "user", "content": "How do you handle conflict with others?"},
        {"role": "assistant", "content": "Arrr! When some scurvy dog crosses me..."}
    ],
    "label": "pos"
}
```

### Step 2: What happens to this conversation

```
1. Pipeline loads pirate.jsonl, extracts the "conversation" field
                                    ↓
2. ConversationEncoder.build_batch_turn_spans() tokenizes it:
   "You are a pirate captain..." → [token IDs: 1, 4521, 892, ...]
   Finds spans:
     user turn:      tokens 23-31
     assistant turn: tokens 32-187
                                    ↓
3. ActivationExtractor.batch_conversations() runs a forward pass:
   - Pads to batch's max length
   - Registers hooks on all 64 layers
   - self.model(input_ids, attention_mask) executes
   - Hooks fire: layer 0 output captured, layer 1 output captured, ..., layer 63
   - Result: tensor shape (64, 16, 512, 4096)
             (layers, batch, tokens, hidden)
                                    ↓
4. SpanMapper.map_spans() uses the span info:
   - Extracts tokens 32-187 from the activation tensor (assistant turn)
   - Computes mean across those 155 tokens: (155, 4096) → (4096,)
   - Does this for all 64 layers
   - Result: (1 turn, 64 layers, 4096 hidden) → squeezed to (64, 4096)
                                    ↓
5. Pipeline takes assistant turns only, averages if multi-turn:
   mean_act shape: (64, 4096)
                                    ↓
6. Saved as activations_dict["pos_p2_q47"] = tensor(64, 4096)
```

### Step 4: This activation becomes part of a role vector

```
7. Step 4 loads pirate.pt and pirate_scores.json
   If scores["pos_p2_q47"] == 3 (fully role-playing):
     → This activation is included in the pirate's mean vector

   pirate_vector = mean of all score=3 activations
   Shape: (64, 4096)
```

### Step 5: The role vector contributes to the axis

```
8. Step 5 loads all role vectors:
   role_mean = mean([pirate_vector, doctor_vector, poet_vector, ...])
   default_mean = mean([default_vector])

   axis = default_mean - role_mean
   Shape: (64, 4096)
```

### Using the axis

```
9. project(some_activation, axis, layer=32) computes:
   dot_product = some_activation[32] · normalize(axis[32])
   → single scalar: "how Assistant-like is this activation?"
```

That scalar is what gets plotted over turns to show persona drift, and what activation capping prevents from going too low.

---

## Summary: The Key Insights

1. **`ActivationExtractor` is a hook-based interception layer.** It doesn't change the model — it eavesdrops on the data flowing through it.

2. **Four methods, one mechanism.** All four methods use the same hook pattern. They differ in what they capture (all positions vs. one position) and how they handle batching (one-at-a-time vs. padded batch).

3. **`batch_conversations()` is the pipeline method.** It's the only one optimized for throughput — it handles padding, attention masks, multi-device models, and returns structured metadata for `SpanMapper`.

4. **The output of `ActivationExtractor` is raw.** It gives you per-token, per-layer activations. Turning those into per-turn averages is `SpanMapper`'s job. Turning per-turn averages into per-role vectors is the pipeline's job. Turning per-role vectors into an axis is Step 5's job.

5. **Hooks are always cleaned up.** Every method uses `try/finally` to ensure hooks are removed, even on error. Leaked hooks would corrupt all future forward passes through the model.
