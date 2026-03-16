# Activation Extraction — Concepts Glossary

Every non-obvious concept used in `activations.py` and the pipeline that calls it, explained alongside the code where it appears. Ordered from most fundamental to most specialized.

---

## Table of Contents

- [1. Tensor Dimensions and the Batch Convention](#1-tensor-dimensions-and-the-batch-convention)
- [2. Tensor Slicing and Indexing](#2-tensor-slicing-and-indexing)
- [3. Device — CPU vs GPU and `.to(device)`](#3-device--cpu-vs-gpu-and-todevice)
- [4. Dtype — bfloat16 vs float32](#4-dtype--bfloat16-vs-float32)
- [5. Batch Extraction — Why and How](#5-batch-extraction--why-and-how)
- [6. Padding Tokens and `pad_token_id`](#6-padding-tokens-and-pad_token_id)
- [7. Attention Mask — How It Works Mechanically](#7-attention-mask--how-it-works-mechanically)
- [8. Truncation — Right-Truncation and Why Right Not Left](#8-truncation--right-truncation-and-why-right-not-left)
- [9. Right-Padding vs Left-Padding](#9-right-padding-vs-left-padding)
- [10. Forward Hook — The PyTorch Internals](#10-forward-hook--the-pytorch-internals)
- [11. `RemovableHandle` and Hook Cleanup](#11-removablehandle-and-hook-cleanup)
- [12. Tuple Output — Why Layers Return Tuples](#12-tuple-output--why-layers-return-tuples)
- [13. `torch.inference_mode()` — What It Disables](#13-torchinference_mode--what-it-disables)
- [14. `torch.stack()` vs `torch.cat()`](#14-torchstack-vs-torchcat)
- [15. `torch.nonzero()` and `as_tuple=True`](#15-torchnonzero-and-as_tupletrue)
- [16. `.cpu()` and `.item()` — Getting Data Off the GPU](#16-cpu-and-item--getting-data-off-the-gpu)
- [17. Chat Template — What `apply_chat_template` Does](#17-chat-template--what-apply_chat_template-does)
- [18. Span — Token Ranges for Conversation Turns](#18-span--token-ranges-for-conversation-turns)
- [19. Mean Reduction — `.mean(dim=N)`](#19-mean-reduction--meandimn)
- [20. Multi-GPU Sharding and `device_map="auto"`](#20-multi-gpu-sharding-and-device_mapauto)
- [21. `CUDA_VISIBLE_DEVICES` — Controlling GPU Visibility](#21-cuda_visible_devices--controlling-gpu-visibility)
- [22. Multiprocessing with `spawn` — Why Not `fork`](#22-multiprocessing-with-spawn--why-not-fork)
- [23. Tensor Parallelism vs Data Parallelism](#23-tensor-parallelism-vs-data-parallelism)
- [24. `torch.cuda.empty_cache()` and `gc.collect()`](#24-torchcudaempty_cache-and-gccollect)
- [25. Closure and the Factory Function Pattern](#25-closure-and-the-factory-function-pattern)

---

## 1. Tensor Dimensions and the Batch Convention

A PyTorch tensor is a multi-dimensional array. Each dimension has a name by convention:

```
Dimension 0: batch       — how many separate inputs
Dimension 1: sequence    — how many tokens
Dimension 2: hidden      — the activation vector at each token
```

In this code, you'll see shapes like `(1, 200, 4096)`. That's 1 conversation, 200 tokens, each represented by a 4096-dimensional vector.

**Where it appears:**

```python
# activations.py — line 90
activations.append(act_tensor[0, :, :].cpu())
#                             ^
#                             └── The [0] removes the batch dimension.
#                                 act_tensor is (1, 200, 4096).
#                                 act_tensor[0] is (200, 4096).
```

**Why it matters:** Every tensor flowing through a PyTorch model has a batch dimension, even when you're processing a single input. The model expects shape `(batch_size, ...)`. If you forget the batch dimension, operations crash. If you forget to remove it from the output, you get unexpected extra dimensions.

In `batch_conversations()`, the batch dimension is meaningful — it holds 16 conversations. In `full_conversation()`, it's always 1, so `[0]` strips it for a cleaner output.

---

## 2. Tensor Slicing and Indexing

PyTorch uses the same slicing syntax as NumPy. The three patterns in `activations.py`:

```python
# Pattern 1: act_tensor[0, :, :]
# "Take batch item 0. Take ALL tokens. Take ALL hidden dims."
# (1, 200, 4096) → (200, 4096)

# Pattern 2: act_tensor[0, newline_pos, :]
# "Take batch item 0. Take ONE specific token. Take ALL hidden dims."
# (1, 200, 4096) → (4096,)

# Pattern 3: batch_activations[:, conv_id, start_idx:end_idx, :]
# "Take ALL layers. Take ONE conversation. Take tokens from start to end. Take ALL hidden dims."
# (64, 16, 512, 4096) → (64, 142, 4096)    (if end-start = 142)
```

**The colon `:` by itself means "everything in this dimension."** It's equivalent to `0:length`. You can omit trailing colons — `act_tensor[0]` is the same as `act_tensor[0, :, :]`.

**`start:end` is a range slice** — positions `start` through `end-1` (end is exclusive). This is how `SpanMapper` cuts out just the assistant's tokens:

```python
# spans.py — line 97
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
# If start_idx=45 and end_idx=187, this grabs tokens 45, 46, ..., 186
```

---

## 3. Device — CPU vs GPU and `.to(device)`

Every PyTorch tensor lives on a specific device — either CPU RAM or a specific GPU's VRAM. You can't mix devices in operations.

```python
cpu_tensor = torch.tensor([1, 2, 3])           # Lives on CPU
gpu_tensor = cpu_tensor.to("cuda:0")            # Copied to GPU 0
result = gpu_tensor + gpu_tensor                 # Works — same device
result = cpu_tensor + gpu_tensor                 # CRASHES — different devices
```

**Where it appears in `activations.py`:**

```python
# Line 79 — Move tokenized input to the model's GPU
input_ids = tokens["input_ids"].to(self.model.device)

# Line 90 — Move captured activation back to CPU
activations.append(act_tensor[0, :, :].cpu())

# Lines 348-350 — Move activations from different GPUs to one GPU
target_device = layer_outputs[layer_list[0]].device
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
])
```

**Why the `.to(target_device)` in line 350?** In multi-GPU models, layer 0 might be on GPU 0 and layer 40 on GPU 1. Their outputs live on different GPUs. `torch.stack()` can't combine tensors from different devices, so everything is moved to one device first.

**Why `.cpu()` in line 90?** Activations are large. Keeping them on GPU would exhaust memory. Moving each layer's activation to CPU immediately frees GPU VRAM before the next layer's hook fires.

---

## 4. Dtype — bfloat16 vs float32

A tensor's dtype is its numerical precision — how many bits each number uses.

| Dtype | Bits | Range | Precision | Memory per number |
|-------|------|-------|-----------|-------------------|
| `float32` | 32 | ±3.4 × 10³⁸ | ~7 decimal digits | 4 bytes |
| `float16` | 16 | ±65504 | ~3 decimal digits | 2 bytes |
| `bfloat16` | 16 | ±3.4 × 10³⁸ | ~3 decimal digits | 2 bytes |

`bfloat16` ("brain float") is the sweet spot for large language models: it has the same range as `float32` (so values don't overflow) but half the memory. The slight loss in precision doesn't matter for inference.

**Where it appears:**

```python
# activations.py — lines 354-355
if selected_activations.dtype != torch.bfloat16:
    selected_activations = selected_activations.to(torch.bfloat16)
```

**Why force it?** The model runs in `bfloat16` (set in `ProbingModel.__init__` with `dtype=torch.bfloat16`). But certain PyTorch operations can silently promote to `float32`. This line ensures the output is consistently `bfloat16`, which matters for:
- **Memory:** A `(64, 16, 512, 4096)` tensor in float32 is **8.6 GB**. In bfloat16, it's **4.3 GB**.
- **Downstream consistency:** `SpanMapper` preserves the dtype, and later pipeline steps save tensors to disk. Mixed dtypes cause subtle bugs.

```python
# model.py — line 25
dtype: torch.dtype = torch.bfloat16,  # Default precision for the whole model
```

---

## 5. Batch Extraction — Why and How

**The problem:** The pipeline needs to extract activations from ~1200 conversations per role. Processing them one at a time means 1200 forward passes.

**The solution:** Process them in batches of 16. One forward pass handles all 16. A GPU is a parallel processor — 16 sequences take roughly the same time as 1.

**The constraint:** A GPU forward pass requires a single rectangular tensor. All 16 sequences must have the same length. This is where padding enters.

**The code:**

```python
# activations.py — batch_conversations(), lines 302-316

for ids in batch_full_ids:
    # Every conversation has a different number of tokens.
    # ids might be [151644, 8948, 198, ...] — say 187 tokens.

    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]    # Truncate if too long

    # Pad to the batch's maximum length
    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    # If max_seq_len=350 and len(ids)=187, append 163 padding tokens.

    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))
    # [1,1,1,...,1, 0,0,0,...,0]
    #  ← 187 ones → ← 163 zeros →

# Convert to 2D tensors
input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
# Shape: (16, 350) — 16 conversations, each 350 tokens (with padding)
```

**After the forward pass**, the output activation tensor has shape `(num_layers, 16, 350, hidden_dim)`. Positions beyond each conversation's real length contain garbage (the model processed padding tokens, producing meaningless activations). The metadata tracks which positions are real:

```python
'truncated_lengths': [min(len(ids), max_seq_len) for ids in batch_full_ids]
# e.g., [187, 243, 350, 156, ...] — each conversation's real token count
```

---

## 6. Padding Tokens and `pad_token_id`

A padding token is a special token that fills unused space in a batch. It has a token ID like any other token, but the model is told to ignore it via the attention mask.

```python
# model.py — lines 49-50
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

**Why reassign EOS as the pad token?** Some tokenizers (e.g., Llama) don't have a dedicated pad token. The code reuses the EOS token. This is safe because the attention mask tells the model "these positions are padding" regardless of what token ID fills them.

**What happens physically:** The padding token gets embedded, processed through layers, and produces activations — but those activations are garbage because:
1. The attention mask prevents real tokens from attending to padding positions
2. `SpanMapper` only reads positions within the span boundaries, which never include padding

```python
# activations.py — line 308
padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# e.g., token ID 151643 repeated 163 times
```

---

## 7. Attention Mask — How It Works Mechanically

The attention mask doesn't just "mark" which tokens are real. It physically prevents information flow during the self-attention computation.

**Self-attention in 30 seconds:** For each token, the model computes attention weights — a probability distribution over all other tokens. These weights determine how much each token's information is mixed into the current token's representation.

```
Without mask:
  Token "pirate" attends to: [You, are, a, pirate, PAD, PAD, PAD]
  Weights:                    [0.1, 0.1, 0.05, 0.3, 0.15, 0.15, 0.15]
  ← Padding tokens get real weight! This corrupts the "pirate" activation.

With mask:
  Token "pirate" attends to: [You, are, a, pirate, PAD, PAD, PAD]
  Pre-softmax scores:         [2.1, 1.8, 0.9, 3.5, -inf, -inf, -inf]
  After softmax weights:      [0.18, 0.14, 0.05, 0.63, 0.0, 0.0, 0.0]
  ← Padding tokens get zero weight. "pirate" activation is clean.
```

The mask works by adding `-infinity` to the attention scores at masked positions *before* the softmax. Since `softmax(-inf) = 0`, those positions contribute zero to the weighted sum.

**The code creates the mask:**

```python
# activations.py — line 309
attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))
# [1, 1, 1, ..., 1, 0, 0, 0, ..., 0]
#  ← real tokens →  ← padding →
```

**The code passes it to the model:**

```python
# activations.py — lines 338-341
_ = self.model(
    input_ids=input_ids_tensor,
    attention_mask=attention_mask_tensor,   # ← The model uses this internally
)
```

Inside the model, the attention computation multiplies the mask into the attention scores. The extractor never needs to handle masking itself — it's baked into the model's forward pass.

---

## 8. Truncation — Right-Truncation and Why Right Not Left

When a conversation exceeds `max_length`, it's truncated:

```python
# activations.py — lines 304-305
if len(ids) > max_seq_len:
    ids = ids[:max_seq_len]     # Keep first max_seq_len tokens, drop the rest
```

This is **right-truncation** — the end of the sequence is chopped off.

**Why not left-truncation?** Left-truncation (dropping the beginning) is used in generation, where the most recent tokens matter most. But for activation extraction:
1. **System prompts are at the beginning.** Cutting the start would lose the role instruction ("You are a pirate"), making the remaining activations meaningless.
2. **Span indices would break.** Spans say "the assistant's response starts at token 45." If you cut 100 tokens from the front, position 45 would point to the wrong place. Right-truncation keeps all positions stable.
3. **The assistant's response is usually at the end.** If anything gets cut, it's the end of the response — typically less important for the overall persona signal than the beginning.

**What `SpanMapper` does with truncated conversations:**

```python
# spans.py — lines 83-89
actual_length = batch_metadata['truncated_lengths'][conv_id]
if start_idx >= actual_length:
    continue   # Entire span was truncated — skip it

end_idx = min(end_idx, actual_length)   # Span partially truncated — use what's left
```

If truncation cuts into the middle of the assistant's response, `SpanMapper` only averages over the surviving tokens. If truncation cuts before the response starts, the conversation is skipped entirely.

---

## 9. Right-Padding vs Left-Padding

Two different strategies for where padding tokens go:

```
Right-padding (used here):
  [You] [are] [a] [pirate] [PAD] [PAD] [PAD]
  Position: 0    1    2      3      4      5      6

Left-padding (used in generation):
  [PAD] [PAD] [PAD] [You] [are] [a] [pirate]
  Position: 0    1    2      3      4    5      6
```

**Why this code uses right-padding:**

Spans reference absolute positions: "the assistant's response is at tokens 45-187." Right-padding preserves these positions — token 45 is still at position 45 regardless of padding. With left-padding, all positions shift by the padding length, breaking the span indices.

```python
# activations.py — line 308
padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
#            ^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            real tokens first                   then padding on the right
```

**Note:** `ProbingModel.__init__` sets `tokenizer.padding_side = "left"` for generation (where left-padding is standard). But `batch_conversations()` does its own manual right-padding, overriding that setting.

---

## 10. Forward Hook — The PyTorch Internals

`register_forward_hook` is a PyTorch API on every `nn.Module`. Here's what happens internally:

```python
# You call:
handle = layer_module.register_forward_hook(my_hook_fn)

# Internally, PyTorch does roughly:
layer_module._forward_hooks[unique_id] = my_hook_fn

# When layer_module.forward() runs:
output = layer_module._original_forward(input)        # Normal computation
for hook_fn in layer_module._forward_hooks.values():   # All registered hooks
    hook_result = hook_fn(layer_module, input, output)  # Your callback fires
    if hook_result is not None:
        output = hook_result                            # Hook can modify output
return output
```

**Key properties:**
- Hooks fire **after** the module's forward method completes
- Hooks fire in registration order
- If a hook returns something (not `None`), it **replaces** the output for downstream layers. The hooks in `activations.py` don't return anything — they're read-only observers. But `steering.py`'s hooks DO return modified outputs to alter the model's behavior.

**The code:**

```python
# activations.py — lines 93-98
model_layers = self.probing_model.get_layers()   # e.g., model.model.layers — a ModuleList of 64 layers
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]        # One specific nn.Module (a TransformerDecoderLayer)
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)
```

`model_layers[22]` is an `nn.Module` representing the 22nd transformer layer. Registering a hook on it means "every time data flows through layer 22, call my function."

---

## 11. `RemovableHandle` and Hook Cleanup

`register_forward_hook` returns a `RemovableHandle` object. Its only purpose is cleanup:

```python
handle = module.register_forward_hook(fn)   # Hook is registered
handle.remove()                              # Hook is unregistered
```

**Why cleanup is critical:** If you don't remove hooks, they persist. Every future forward pass through the model triggers the hook, even in unrelated code. In this codebase:
- **Extraction hooks** would accumulate activations forever, consuming memory
- **Steering hooks** would keep modifying the model's behavior after the `with` block ends

```python
# activations.py — lines 100-106
try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    for handle in handles:
        handle.remove()            # ALWAYS runs, even if forward pass crashes
```

The `try/finally` pattern guarantees cleanup. Without it, an out-of-memory error during the forward pass would leave hooks attached permanently.

---

## 12. Tuple Output — Why Layers Return Tuples

When you call a transformer layer's `forward()`, it can return multiple things:

```python
# What Llama/Qwen layers return:
output = (
    hidden_states,        # (batch, seq_len, hidden_dim) — the main output
    self_attn_weights,    # (batch, num_heads, seq_len, seq_len) — optional
    present_key_value,    # Cache for efficient autoregressive generation — optional
)

# What some simpler architectures return:
output = hidden_states    # Just the tensor directly
```

The hidden states (index 0) are always the "main product" — the tensor that flows into the next layer. Attention weights and key-value caches are auxiliary outputs used for visualization or fast generation.

**The code handles both:**

```python
# activations.py — line 89
act_tensor = output[0] if isinstance(output, tuple) else output
```

`isinstance(output, tuple)` checks the type at runtime. If it's a tuple, take the first element. If it's already a tensor, use it directly.

---

## 13. `torch.inference_mode()` — What It Disables

PyTorch has an automatic differentiation engine called **autograd**. It tracks every operation on tensors so gradients can be computed later (for training). This tracking has overhead:

| Feature | `torch.no_grad()` | `torch.inference_mode()` |
|---------|-------------------|--------------------------|
| Gradient computation | Disabled | Disabled |
| Autograd graph recording | Disabled | Disabled |
| Version counters on tensors | **Still tracked** | **Disabled** |
| Allow in-place ops on inputs | **No** | **Yes** |
| Speed | Fast | **Faster** |

**Version counters** are integers PyTorch increments every time a tensor is modified in-place. They exist to detect bugs where an in-place operation invalidates a saved computation graph. Since we never compute gradients, we don't need them.

```python
# activations.py — line 101
with torch.inference_mode():
    _ = self.model(input_ids)    # Everything inside is faster
```

**Why not just `no_grad`?** `inference_mode` was introduced in PyTorch 1.9 specifically for pure-inference workloads. It's strictly superior to `no_grad` when you don't need gradients.

---

## 14. `torch.stack()` vs `torch.cat()`

Both combine tensors, but they work differently:

```python
a = torch.tensor([1, 2, 3])   # shape: (3,)
b = torch.tensor([4, 5, 6])   # shape: (3,)

torch.stack([a, b])
# Creates a NEW dimension:
# tensor([[1, 2, 3],
#          [4, 5, 6]])
# Shape: (2, 3) ← new dimension of size 2

torch.cat([a, b])
# Concatenates along EXISTING dimension:
# tensor([1, 2, 3, 4, 5, 6])
# Shape: (6,) ← same number of dimensions, just longer
```

**Where `stack` is used in `activations.py`:**

```python
# Line 108 — Stack activations from different layers
activations = torch.stack(activations)
# If activations is a list of 64 tensors, each (200, 4096):
# Result: (64, 200, 4096) ← new "layer" dimension

# Line 226 — Stack activations from different prompts
return torch.stack(activations)
# If activations is a list of 100 tensors, each (4096,):
# Result: (100, 4096) ← new "prompt" dimension

# Lines 349-351 — Stack layer outputs into one tensor
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list
])
# List of 64 tensors, each (16, 512, 4096):
# Result: (64, 16, 512, 4096) ← new "layer" dimension
```

**Rule of thumb:** Use `stack` when combining items that should have a new leading dimension (like different layers, different samples). Use `cat` when extending an existing dimension (like concatenating two batches into one bigger batch).

---

## 15. `torch.nonzero()` and `as_tuple=True`

`nonzero()` finds the positions where a tensor has non-zero (or `True`) values.

```python
# activations.py — line 382
newline_positions = (input_ids == newline_token_id).nonzero(as_tuple=True)[0]
```

Breaking this down:

```python
input_ids = torch.tensor([10, 20, 198, 30, 198, 40])

# Step 1: Boolean comparison
input_ids == 198
# tensor([False, False, True, False, True, False])

# Step 2: nonzero(as_tuple=True)
(input_ids == 198).nonzero(as_tuple=True)
# Returns: (tensor([2, 4]),)
# A tuple containing one tensor (since input is 1D).
# The tensor holds the indices where the condition is True.

# Step 3: [0] extracts the tensor from the tuple
# tensor([2, 4]) — newline token appears at positions 2 and 4

# Step 4: [-1] takes the last one
newline_positions[-1].item()  # 4
```

**Why `as_tuple=True`?** Without it, `nonzero()` returns a 2D tensor where each row is a coordinate. For 1D input, `as_tuple=True` is simpler — it gives you a 1D tensor of indices directly.

---

## 16. `.cpu()` and `.item()` — Getting Data Off the GPU

```python
# .cpu() moves an entire tensor from GPU to CPU
gpu_tensor = torch.tensor([1, 2, 3], device="cuda:0")
cpu_tensor = gpu_tensor.cpu()   # Now on CPU

# .item() converts a single-element tensor to a Python number
t = torch.tensor(42)
n = t.item()    # n is the Python int 42
type(n)         # <class 'int'>
```

**Where they appear:**

```python
# activations.py — line 90: Move the captured activation to CPU
activations.append(act_tensor[0, :, :].cpu())

# activations.py — line 384: Convert a tensor index to a Python int
return newline_positions[-1].item()
```

**Why `.cpu()` instead of letting Python handle it?** Python can't directly work with GPU tensors for most operations outside PyTorch. And keeping large tensors on GPU when you're done with them wastes VRAM. `.cpu()` explicitly transfers the data.

**Why `.item()` and not just use the tensor?** A single-element tensor is NOT a Python int. Some APIs expect a plain int (like indexing a list). `.item()` does the conversion. For tensors with more than one element, `.item()` raises an error — it's a safety check that you're extracting exactly one number.

---

## 17. Chat Template — What `apply_chat_template` Does

A HuggingFace chat template is a Jinja2 template string stored in the tokenizer that converts a list of message dicts into the model-specific formatted string.

**Input:**
```python
[
    {"role": "system", "content": "You are a pirate."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Ahoy there!"},
]
```

**Output for Qwen:**
```
<|im_start|>system
You are a pirate.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Ahoy there!<|im_end|>
```

**Output for Gemma 2:**
```
<start_of_turn>user
You are a pirate.

Hello!<end_of_turn>
<start_of_turn>model
Ahoy there!<end_of_turn>
```

Note: Gemma 2 doesn't support system prompts, so the instruction is merged into the user message.

**Where it appears:**

```python
# activations.py — lines 71-73
formatted_prompt = self.tokenizer.apply_chat_template(
    conversation,
    tokenize=False,              # Return string, not token IDs
    add_generation_prompt=False,  # Don't add "assistant is about to speak" marker
    **chat_kwargs                 # e.g., enable_thinking=False for Qwen 3
)
```

**Why it matters for activation extraction:** The template inserts special tokens (BOS, EOS, role markers) that become part of the token sequence. The model's activations at these positions carry structural information (e.g., "I've just finished reading the system prompt"), which is why `SpanMapper` exists to select only the content tokens.

---

## 18. Span — Token Ranges for Conversation Turns

A **span** is this project's term for a contiguous range of token positions corresponding to one turn in a conversation. Spans are produced by `ConversationEncoder.build_turn_spans()`.

```python
# What a span dict looks like:
{
    "turn": 1,
    "role": "assistant",
    "start": 45,          # Token index where content starts (inclusive)
    "end": 187,           # Token index where content ends (exclusive)
    "n_tokens": 142,      # end - start
    "text": "Arrr! When some scurvy dog...",
    "conversation_id": 0  # (only in batch spans)
}
```

**What's included vs excluded from the span:**

```
Token sequence:
... <|im_start|> assistant \n Arrr ! When some scurvy dog ... <|im_end|> ...
    │           │          │  ↑                              ↑ │
    │           │          │  span start (45)        span end (187)
    │           │          │  ←── these tokens are IN the span ──→
    │           └──────────└── these tokens are OUTSIDE the span
    └── this token is OUTSIDE the span
```

The span boundaries are chosen to include only the **content** of the turn — not the role markers, not the BOS/EOS tokens. This is why spans are necessary: they tell `SpanMapper` exactly which token positions to average over.

**Where spans are used:**

```python
# spans.py — line 97 (SpanMapper.map_spans)
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
# Slices out ONLY the tokens within this span — no role markers, no padding
```

---

## 19. Mean Reduction — `.mean(dim=N)`

`.mean(dim=N)` computes the average along dimension N, collapsing that dimension.

```python
t = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
# Shape: (2, 3)

t.mean(dim=0)   # Average across rows → tensor([2.5, 3.5, 4.5])     shape: (3,)
t.mean(dim=1)   # Average across columns → tensor([2.0, 5.0])        shape: (2,)
```

**Where it appears:**

```python
# spans.py — line 107: Average across tokens in a span
mean_activation = span_activations.mean(dim=1)
# span_activations shape: (num_layers, span_length, hidden_size)
# dim=1 is the token dimension
# Result shape: (num_layers, hidden_size) — one vector per layer

# 2_activations.py — line 119: Average across assistant turns
mean_act = assistant_act.mean(dim=0).cpu()
# assistant_act shape: (num_assistant_turns, num_layers, hidden_size)
# dim=0 is the turn dimension
# Result shape: (num_layers, hidden_size) — one vector per layer
```

**Why mean and not sum?** Mean normalizes by count. If one conversation has 50 assistant tokens and another has 200, the mean produces comparable vectors. Sum would make longer responses dominate.

---

## 20. Multi-GPU Sharding and `device_map="auto"`

Large models (27B, 32B, 70B parameters) don't fit on a single GPU. `device_map="auto"` splits the model across multiple GPUs:

```python
# model.py — lines 62-64
elif device is None or device == "auto":
    model_kwargs["device_map"] = "auto"

self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
```

**What happens internally:** HuggingFace's `accelerate` library examines the model's size and each GPU's available memory, then assigns layers to GPUs:

```
GPU 0 (80GB): Layers 0-31 + embedding table
GPU 1 (80GB): Layers 32-63 + lm_head
```

During a forward pass, data starts on GPU 0, flows through layers 0-31, is automatically transferred to GPU 1, and flows through layers 32-63. This is transparent — you call `model(input_ids)` as if everything were on one device.

**Implications for activation extraction:** Hooks on layer 10 capture a tensor on GPU 0. Hooks on layer 50 capture a tensor on GPU 1. `batch_conversations()` handles this:

```python
# activations.py — lines 348-351
target_device = layer_outputs[layer_list[0]].device    # GPU of the first layer
selected_activations = torch.stack([
    layer_outputs[i].to(target_device) for i in layer_list  # Move all to same GPU
])
```

---

## 21. `CUDA_VISIBLE_DEVICES` — Controlling GPU Visibility

An environment variable that controls which GPUs a process can see.

```python
# 2_activations.py — lines 189-190
gpu_ids_str = ','.join(map(str, gpu_ids))   # e.g., "2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
```

**What this does:** After setting `CUDA_VISIBLE_DEVICES=2,3`:
- The process can only see 2 GPUs (physical GPUs 2 and 3)
- Inside the process, they're addressed as `cuda:0` and `cuda:1` (re-indexed from 0)
- Physical GPUs 0, 1, 4, 5, 6, 7 are invisible — the process can't accidentally use them

**Why it's needed:** In multi-worker mode, each worker loads its own copy of the model. Worker 0 uses GPUs 0-1, Worker 1 uses GPUs 2-3, etc. Without `CUDA_VISIBLE_DEVICES`, all workers would try to load onto GPU 0 simultaneously and crash.

---

## 22. Multiprocessing with `spawn` — Why Not `fork`

Python's `multiprocessing` module creates OS-level processes. There are two methods:

| Method | How it works | CUDA compatible? |
|--------|-------------|------------------|
| `fork` | Copies the parent process's memory (copy-on-write) | **No** |
| `spawn` | Creates a fresh Python interpreter, re-imports modules | **Yes** |

```python
# 2_activations.py — line 311
mp.set_start_method('spawn', force=True)
```

**Why `fork` doesn't work with CUDA:** When you fork a process, the child inherits the parent's CUDA context (GPU state, memory allocations, kernel handles). But CUDA contexts are not designed to be shared. The child's GPU operations silently corrupt the parent's state, causing crashes, wrong results, or hangs.

`spawn` avoids this by starting a clean process with no inherited CUDA state. Each process initializes its own CUDA context from scratch.

**Trade-off:** `spawn` is slower to start (it re-imports everything) and can't share in-memory objects (each process loads the model independently). But correctness beats speed.

---

## 23. Tensor Parallelism vs Data Parallelism

The pipeline uses the term `tensor_parallel_size`. This is a specific form of parallelism:

| Strategy | What's split | Example |
|----------|-------------|---------|
| **Data parallelism** | The batch. Each GPU processes different data with the same model. | 4 GPUs each process 4 conversations = 16 total per step |
| **Tensor parallelism** | The model. Each GPU holds part of each layer. All GPUs process the same data. | A single layer's weight matrix is split across 2 GPUs |
| **Pipeline parallelism** | The layers. Different GPUs hold different layers. | GPU 0 has layers 0-31, GPU 1 has layers 32-63 (this is what `device_map="auto"` does) |

The pipeline script uses `tensor_parallel_size` to determine how many GPUs one model copy needs, and then calculates how many independent workers can run:

```python
# 2_activations.py — line 262
num_workers = total_gpus // tensor_parallel_size
# 8 GPUs ÷ 2 GPUs per model = 4 workers, each processing different roles
```

Each worker is data-parallel (processes its own subset of roles). Within a worker, the model may use tensor or pipeline parallelism across its assigned GPUs.

---

## 24. `torch.cuda.empty_cache()` and `gc.collect()`

Two different memory cleanup mechanisms:

**`gc.collect()`** — Python's garbage collector. Reclaims Python objects (including tensor wrappers) that have no remaining references. Normally runs automatically, but large objects (like activation tensors) can linger. Explicit collection forces immediate cleanup.

**`torch.cuda.empty_cache()`** — PyTorch's GPU memory allocator keeps a cache of freed GPU memory blocks for fast reuse. `empty_cache()` returns this cache to the CUDA driver. It doesn't free memory that's still in use — only cached-but-freed blocks.

```python
# 2_activations.py — lines 127-129
del batch_activations                    # Drop the Python reference to the GPU tensor
if (batch_start // batch_size) % 5 == 0:
    torch.cuda.empty_cache()             # Return freed GPU blocks to the driver

# 2_activations.py — lines 180-181
gc.collect()                   # Reclaim all unreferenced Python objects
torch.cuda.empty_cache()       # Then return freed GPU blocks
```

**Why only every 5 batches?** `empty_cache()` has overhead and can cause memory fragmentation. Calling it after every batch hurts performance. Every 5 batches is a compromise — frequent enough to prevent OOM, rare enough to stay fast.

---

## 25. Closure and the Factory Function Pattern

A **closure** is a function that captures variables from its enclosing scope. The factory pattern creates a new scope for each iteration of a loop, avoiding a common Python bug.

**The bug (without factory):**

```python
hooks = []
for layer_idx in [0, 1, 2]:
    def hook_fn(module, input, output):
        print(f"Hook fired for layer {layer_idx}")  # BUG: layer_idx is shared
    hooks.append(hook_fn)

# When hooks fire later:
hooks[0](...)  # Prints "Hook fired for layer 2" ← WRONG! Should be 0
hooks[1](...)  # Prints "Hook fired for layer 2" ← WRONG! Should be 1
hooks[2](...)  # Prints "Hook fired for layer 2" ← Correct by coincidence
```

All three hooks see the same `layer_idx` variable, which has value `2` by the time the loop finishes.

**The fix (factory function):**

```python
# activations.py — lines 86-91
def create_hook_fn(layer_idx):       # Factory creates a new scope
    def hook_fn(module, input, output):
        # This layer_idx is the parameter of create_hook_fn,
        # captured when create_hook_fn was called — NOT the loop variable.
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu())
    return hook_fn

for layer_idx in layer_list:
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    #                                          ^^^^^^^^^^^^^^^^^^^^^^^^^
    # Each call to create_hook_fn creates a NEW function with its own
    # captured copy of layer_idx.
```

Each call to `create_hook_fn(22)` creates a new `hook_fn` that has `layer_idx=22` baked in. The loop variable can change freely — the closure captured the value at creation time, not a reference to the variable.
