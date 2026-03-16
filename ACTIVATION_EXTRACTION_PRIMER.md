# Activation Extraction Primer

**What physically happens when you "extract activations" from a transformer — explained concept-by-concept alongside the actual code that does it.**

---

## Table of Contents

- [1. What Is an Activation?](#1-what-is-an-activation)
- [2. The Journey of Text Through a Transformer](#2-the-journey-of-text-through-a-transformer)
- [3. Tokenization — Text Becomes Numbers](#3-tokenization--text-becomes-numbers)
- [4. The Embedding Layer — Numbers Become Vectors](#4-the-embedding-layer--numbers-become-vectors)
- [5. Transformer Layers — Where Activations Live](#5-transformer-layers--where-activations-live)
- [6. Intercepting Activations with Forward Hooks](#6-intercepting-activations-with-forward-hooks)
- [7. What Happens to Special Tokens (EOS, BOS, Padding)](#7-what-happens-to-special-tokens-eos-bos-padding)
- [8. Why Mean Over Tokens? Why Not Just Use One Token?](#8-why-mean-over-tokens-why-not-just-use-one-token)
- [9. Batched Extraction — Padding, Masking, and Why It's Tricky](#9-batched-extraction--padding-masking-and-why-its-tricky)
- [10. From Raw Activations to Per-Turn Means (The SpanMapper Step)](#10-from-raw-activations-to-per-turn-means-the-spanmapper-step)
- [11. Why Intermediate Layers? Why Not the Final Output?](#11-why-intermediate-layers-why-not-the-final-output)
- [12. Putting It All Together — The Complete Pipeline Trace](#12-putting-it-all-together--the-complete-pipeline-trace)

---

## 1. What Is an Activation?

A neural network is a series of mathematical transformations. Data enters, gets transformed by each layer, and exits as a prediction. An **activation** is the intermediate result — the data as it exists *between* two layers.

Concretely, for a transformer language model:
- Input: a sequence of token IDs, e.g., `[1, 4521, 892, 15, 7703]`
- Each layer transforms the sequence into a new sequence of vectors
- The vectors have the same count (one per token) but carry different information at each layer
- **An activation is one of those vectors** — a 1D array of floating-point numbers, typically 3,584 to 8,192 values long depending on the model

```
Layer 0 output:  [vec₀, vec₁, vec₂, vec₃, vec₄]   ← each vecᵢ is (hidden_dim,) e.g. 4096 floats
Layer 1 output:  [vec₀, vec₁, vec₂, vec₃, vec₄]   ← same shape, different values
...
Layer 63 output: [vec₀, vec₁, vec₂, vec₃, vec₄]   ← most "processed" representation
```

When this project says "extract activations," it means: run text through the model, intercept these intermediate vectors at specific layers, and save them.

---

## 2. The Journey of Text Through a Transformer

Here's the complete path from human-readable text to a usable activation tensor. Every step has corresponding code in this codebase.

```
"You are a pirate. How do you handle conflict?"
                    │
                    ▼
         ┌──── TOKENIZATION ────┐
         │  Text → integer IDs  │    ConversationEncoder + tokenizer
         └──────────┬───────────┘
                    │  [1, 4521, 892, 15, 2847, 73, ...]
                    ▼
         ┌──── EMBEDDING ──────┐
         │  IDs → dense vectors │    model.embed_tokens()
         └──────────┬──────────┘
                    │  shape: (1, num_tokens, hidden_dim)
                    ▼
         ┌──── LAYER 0 ────────┐
         │  Self-attention +    │    model.layers[0]
         │  feed-forward        │
         └──────────┬──────────┘
                    │  shape: (1, num_tokens, hidden_dim)  ← HOOK INTERCEPTS HERE
                    ▼
         ┌──── LAYER 1 ────────┐
         │  Self-attention +    │    model.layers[1]
         │  feed-forward        │
         └──────────┬──────────┘
                    │  ← HOOK INTERCEPTS HERE
                    ▼
                   ...
                    │
         ┌──── LAYER N-1 ─────┐
         │  Final layer         │    model.layers[N-1]
         └──────────┬──────────┘
                    │
                    ▼
         ┌──── LM HEAD ────────┐
         │  Vectors → logits    │    model.lm_head()
         │  (vocabulary probs)  │
         └─────────────────────┘
                    │
                    ▼
              Next token prediction (not used by us)
```

The key insight: **we don't care about the final prediction.** We intercept the data mid-flight, between layers, and save those intermediate vectors. The model's actual output (logits, predicted next token) is thrown away.

---

## 3. Tokenization — Text Becomes Numbers

### The concept

A model can't process text directly. It processes integer IDs that correspond to entries in a vocabulary table. "Tokenization" is the conversion from text to these IDs. A token might be a whole word (`"hello"` → `15339`), a subword (`"un"` + `"break"` + `"able"` → `3` tokens), or a single character.

Importantly, chat models have **special tokens** that structure the conversation:

| Token type | Example (Qwen) | Purpose |
|------------|----------------|---------|
| BOS (begin of sequence) | `<\|im_start\|>` | Marks the start of a role's turn |
| EOS (end of sequence) | `<\|im_end\|>` | Marks the end of a role's turn |
| Role markers | `user`, `assistant`, `system` | Identifies who is speaking |
| Pad token | `<\|endoftext\|>` | Fills unused space in batches |

### The code

The first thing `ActivationExtractor` does is convert text to token IDs. But it doesn't just call `tokenizer("some text")` — it first applies the **chat template**, which wraps the conversation in the model-specific structural tokens.

```python
# activations.py — full_conversation(), lines 68-79

if chat_format:
    if isinstance(conversation, str):
        conversation = [{"role": "user", "content": conversation}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
    )

# Tokenize
tokens = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
input_ids = tokens["input_ids"].to(self.model.device)
```

**Why two steps?** `apply_chat_template(tokenize=False)` produces a string with all the structural markup. Then `self.tokenizer(...)` converts that string to integer IDs. The `add_special_tokens=False` prevents double-adding BOS/EOS tokens (the chat template already inserted them).

**What `apply_chat_template` produces** (example for Qwen):

```
Input conversation:
[
    {"role": "system", "content": "You are a pirate captain."},
    {"role": "user", "content": "How do you handle conflict?"},
    {"role": "assistant", "content": "Arrr! When some scurvy dog crosses me..."}
]

Output string:
<|im_start|>system
You are a pirate captain.<|im_end|>
<|im_start|>user
How do you handle conflict?<|im_end|>
<|im_start|>assistant
Arrr! When some scurvy dog crosses me...<|im_end|>
```

After tokenization, this becomes something like:

```
Token IDs:    [151644, 8948, 198, 2610, 525, 264, 55066, ...]
               │       │     │    └── "You"  "are" "a" "pirate" ...
               │       │     └── newline character
               │       └── "system"
               └── <|im_start|>
```

Each ID is an integer. The sequence length (number of tokens) depends on how long the conversation is — typically 100-500 tokens for a single Q&A exchange.

### What `add_generation_prompt=False` means

When set to `True`, the template appends an extra `<|im_start|>assistant\n` at the end, signaling "now generate a response." We set it to `False` because the assistant's response is already in the conversation — we're not generating, we're reading.

---

## 4. The Embedding Layer — Numbers Become Vectors

### The concept

Token IDs are just integers — they have no mathematical relationship to each other. The embedding layer converts each integer into a dense vector (a list of floats). This is a simple table lookup:

```
Token ID 4521 → look up row 4521 in a (vocab_size × hidden_dim) matrix → get a vector of 4096 floats
```

After embedding, the shape of our data is:

```
Before embedding: (1, num_tokens)              — integers
After embedding:  (1, num_tokens, hidden_dim)   — floats
```

For example, if the conversation is 200 tokens and the model has `hidden_dim=4096`:
```
(1, 200, 4096)
 │   │    └── Each token is now a 4096-dimensional vector
 │   └── 200 tokens in the sequence
 └── Batch size of 1
```

### The code

This happens inside `self.model(input_ids)` — you don't see it explicitly in `activations.py` because it's internal to the HuggingFace model. The model's `forward()` method calls `self.embed_tokens(input_ids)` as its first step.

The extractor just passes token IDs in and hooks fire at layer boundaries after embedding has already happened:

```python
# activations.py — line 102
_ = self.model(input_ids)  # Embedding + all layers happen inside here
```

---

## 5. Transformer Layers — Where Activations Live

### The concept

After embedding, data passes through N transformer layers (N=46 for Gemma 2 27B, N=64 for Qwen 3 32B, N=80 for Llama 3.3 70B). Each layer does two things:

1. **Self-attention:** Each token "looks at" all other tokens to understand context. The vector for the word "bank" gets modified based on whether "river" or "money" appears nearby.

2. **Feed-forward network:** Each token's vector is independently transformed through a nonlinear function, adding computational depth.

The output of each layer has the **exact same shape** as the input:

```
Input to layer k:   (1, num_tokens, hidden_dim)
Output of layer k:  (1, num_tokens, hidden_dim)
```

The shape doesn't change — but the *meaning* of the vectors evolves. Early layers encode surface features (syntax, word identity). Middle layers encode semantic features (meaning, intent, persona). Late layers encode prediction features (what word comes next).

### The code — finding the layers

Different model architectures store their layers in different places. `ProbingModel.get_layers()` tries multiple paths:

```python
# model.py — get_layers(), lines 140-146

layer_paths = [
    ('model.model.layers', lambda m: m.model.layers),           # Llama, Gemma 2, Qwen
    ('model.language_model.layers', lambda m: m.language_model.layers),  # Gemma 3 (vision)
    ('model.transformer.h', lambda m: m.transformer.h),         # GPT-style
    ('model.transformer.layers', lambda m: m.transformer.layers),
    ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),    # GPT-NeoX
]
```

Once found, the layers are a `ModuleList` — essentially a Python list of layer modules that can be indexed: `layers[0]`, `layers[22]`, `layers[63]`, etc.

### What a layer's output looks like

When layer 22 processes data, its output is:

```python
output = (hidden_states, attention_weights, ...)
# hidden_states shape: (batch_size, num_tokens, hidden_dim)
# e.g., (1, 200, 4096)
```

Some models return a tuple (with attention weights, cache, etc.), others return just the tensor. That's why the hook checks:

```python
act_tensor = output[0] if isinstance(output, tuple) else output
```

---

## 6. Intercepting Activations with Forward Hooks

### The concept

A **forward hook** is a callback function you attach to any module in a PyTorch network. PyTorch guarantees: every time data passes through that module (during `module.forward()`), your callback is invoked with the module's input and output. You can read the output, copy it, even modify it.

This is the fundamental mechanism that makes activation extraction possible without modifying the model's code.

### The code — registering and firing hooks

Here's the exact sequence, annotated:

```python
# activations.py — full_conversation(), lines 82-106

# 1. Prepare storage
activations = []   # Hooks will append to this list
handles = []       # Track hooks so we can remove them later

# 2. Create a hook function for each target layer
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        # 'output' is what layer_idx just produced
        # It might be a tuple: (hidden_states, attn_weights, ...)
        act_tensor = output[0] if isinstance(output, tuple) else output

        # act_tensor shape: (batch_size, num_tokens, hidden_dim)
        # [0, :, :] removes the batch dimension → (num_tokens, hidden_dim)
        # .cpu() moves from GPU to CPU to free GPU memory
        activations.append(act_tensor[0, :, :].cpu())
    return hook_fn

# 3. Register hooks on the target layers
model_layers = self.probing_model.get_layers()
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]   # e.g., the TransformerDecoderLayer at index 22
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)

# 4. Run the forward pass — hooks fire automatically as data flows through
try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    # 5. Remove hooks (ALWAYS, even if forward pass crashes)
    for handle in handles:
        handle.remove()
```

### What physically happens during step 4

When `self.model(input_ids)` executes:

```
1. input_ids → embed_tokens() → embedded tensor (1, 200, 4096)
2. embedded tensor → layers[0].forward() → output_0 (1, 200, 4096)
3.                                          ↓ NO HOOK on layer 0 (if not requested)
4. output_0 → layers[1].forward() → output_1
5. ...
6. output_21 → layers[22].forward() → output_22 (1, 200, 4096)
7.                                     ↓ HOOK FIRES!
8.                                     hook_fn receives output_22
9.                                     hook_fn does: activations.append(output_22[0,:,:].cpu())
10.                                    activations is now [tensor of shape (200, 4096)]
11. output_22 → layers[23].forward() → output_23
12. ... continues through remaining layers ...
13. final output → lm_head → logits (discarded as _)
```

The crucial point: **the forward pass runs normally.** The hooks are passive observers — they copy data out but don't change the flow. The model produces the same logits it would without hooks.

### Why `.cpu()` here but not in `batch_conversations()`?

In `full_conversation()`, each layer's activation is immediately moved to CPU:

```python
activations.append(act_tensor[0, :, :].cpu())  # ← .cpu()
```

In `batch_conversations()`, activations stay on GPU:

```python
layer_outputs[layer_idx] = act_tensor  # ← no .cpu()
```

The difference in strategy:

- **`full_conversation()` extracts one conversation at a time.** Moving to CPU immediately frees GPU memory before the next layer's hook fires. Since the result is used in notebooks (small scale), CPU storage is fine.

- **`batch_conversations()` is used by the pipeline.** The next step (`SpanMapper.map_spans()`) needs to slice into these tensors and compute means — doing that on GPU is 10-100x faster than on CPU. So they stay on GPU until after `map_spans()` finishes.

### Why `torch.inference_mode()` instead of `torch.no_grad()`?

Both disable gradient computation (we're not training, so no gradients needed). But `inference_mode` goes further — it also disables autograd's internal bookkeeping (version counters on tensors, storage of computation graphs). This makes it faster and uses less memory. Since we never backpropagate through extracted activations, `inference_mode` is strictly better.

---

## 7. What Happens to Special Tokens (EOS, BOS, Padding)

This is one of the most important practical questions. Let's trace exactly what happens.

### BOS / EOS tokens in single-conversation extraction

When you extract activations for one conversation via `full_conversation()`, the token sequence includes all special tokens:

```
Tokens:   [BOS] [system] [\n] [You] [are] [a] [pirate] [EOS] [BOS] [user] [\n] [How] [do] [you] ... [EOS]
Position:   0      1       2    3     4    5     6        7      8     9     10   11   12   13        N-1
```

The activations tensor has shape `(num_tokens, hidden_dim)` — **every token gets an activation, including BOS, EOS, newline, and role markers.** Nothing is excluded at this stage.

The model processes special tokens just like regular tokens. They have embeddings, they participate in self-attention, and they produce activations. The EOS token's activation at position 7 reflects "I've finished the system prompt" — it has absorbed context from all preceding tokens via self-attention.

### So the EOS activation is included in the raw output?

**Yes.** `full_conversation()` returns activations for ALL positions, special tokens included. It's the downstream consumer's job to decide which positions matter.

In this project, that downstream consumer is `SpanMapper`, which uses span boundaries to select only the content tokens of specific turns (ignoring role markers, BOS, EOS). More on this in [Part 10](#10-from-raw-activations-to-per-turn-means-the-spanmapper-step).

### Padding tokens in batch extraction

When `batch_conversations()` pads sequences to equal length, it adds padding tokens and marks them with an attention mask:

```python
# activations.py — batch_conversations(), lines 302-312

for ids in batch_full_ids:
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]          # Truncate if too long

    # Pad to max length
    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

    input_ids_batch.append(padded_ids)
    attention_mask_batch.append(attention_mask)
```

Visually, for a batch of 3 conversations (lengths 150, 200, 180), padded to 200:

```
Conversation 0: [real real real ... real PAD PAD PAD ... PAD]    150 real + 50 pad
Conversation 1: [real real real ... real real real real ... real] 200 real + 0 pad
Conversation 2: [real real real ... real PAD PAD PAD ... PAD]    180 real + 20 pad

Attention mask:
Conversation 0: [1 1 1 ... 1 0 0 0 ... 0]
Conversation 1: [1 1 1 ... 1 1 1 1 ... 1]
Conversation 2: [1 1 1 ... 1 0 0 0 ... 0]
```

The attention mask is passed to the model:

```python
# activations.py — lines 338-341

_ = self.model(
    input_ids=input_ids_tensor,
    attention_mask=attention_mask_tensor,   # ← Tells the model which tokens are real
)
```

### What does the attention mask actually do?

Inside each transformer layer's self-attention mechanism, the attention mask prevents real tokens from attending to padding tokens. Without it, the model would treat padding as meaningful input and produce corrupted activations.

With the mask:
- Token "pirate" at position 6 can attend to tokens at positions 0-6 (all real)
- Token "pirate" CANNOT attend to padding positions
- The padding positions themselves still produce activations (they go through the computation), but those activations are meaningless garbage

### Are padding activations included in the output?

**Yes.** `batch_conversations()` returns the full `(num_layers, batch_size, max_seq_len, hidden_dim)` tensor, including padding positions. It's `SpanMapper`'s job to only use the real (non-padding) positions by checking `batch_metadata['truncated_lengths']`:

```python
# spans.py — map_spans(), lines 83-86

actual_length = batch_metadata['truncated_lengths'][conv_id]
if start_idx >= actual_length:
    continue   # Skip — this span is in the padding region
```

### What about the pad token itself?

Notice this line in `ProbingModel.__init__()`:

```python
# model.py — lines 49-50

if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
```

Some tokenizers don't have a dedicated pad token. In that case, the EOS token is reused as the pad token. This is safe because the attention mask prevents the model from treating these padding-EOS tokens as real end-of-sequence markers.

### Summary: special token handling

| Token | Included in raw activations? | Included in final per-turn mean? |
|-------|------|------|
| BOS (`<\|im_start\|>`) | Yes | **No** — `SpanMapper` spans only cover content, not markers |
| EOS (`<\|im_end\|>`) | Yes | **No** — spans end before the EOS |
| Role markers (`user`, `assistant`) | Yes | **No** — spans start after the role marker |
| Newline after role marker | Yes | **Depends** — may be inside or outside the span boundary |
| Content tokens | Yes | **Yes** — this is what spans select |
| Padding tokens | Yes (but garbage values) | **No** — `SpanMapper` checks `truncated_lengths` |

---

## 8. Why Mean Over Tokens? Why Not Just Use One Token?

### The concept

After extracting activations, you have a matrix of shape `(num_tokens, hidden_dim)` — one vector per token. But the pipeline needs a single vector per turn to compare roles. The standard approach in this project: **take the mean across all content tokens in the assistant's turn.**

```
Token activations for "Arrr! When some scurvy dog crosses me...":
  Token "Arrr"     → [0.12, -0.34, 0.78, ...]   ← (hidden_dim,)
  Token "!"        → [0.15, -0.31, 0.80, ...]
  Token "When"     → [0.11, -0.29, 0.75, ...]
  Token "some"     → [0.13, -0.33, 0.77, ...]
  ...
  Token "me"       → [0.14, -0.30, 0.79, ...]
  Token "..."      → [0.10, -0.35, 0.76, ...]

Mean activation:   → [0.125, -0.32, 0.775, ...]   ← single (hidden_dim,) vector
```

### Why not just use the last token?

The last token (the EOS or the final content token) is often used in classification tasks because in autoregressive models, it has "seen" the entire sequence via self-attention. But for persona measurement, the mean is preferred because:

1. **Stability.** A single token's activation is noisy — it's heavily influenced by the specific word at that position. The mean smooths out word-specific noise and captures the overall "character" of the response.

2. **The persona signal is distributed.** The model doesn't concentrate its persona information at one token. It's spread across all tokens — each word choice, each phrasing pattern, carries persona signal.

3. **Robustness to response length.** A short response (10 tokens) and a long response (200 tokens) produce vectors in the same space. If you used only the last token, the two vectors would be at very different "positions" in the sequence, making comparison less meaningful.

### Where the mean is computed in code

The mean over tokens doesn't happen in `ActivationExtractor` — it happens in `SpanMapper.map_spans()`:

```python
# spans.py — map_spans(), lines 96-108

# batch_activations[:, conv_id, start_idx:end_idx, :]
# has shape (num_layers, span_length, hidden_size)
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

span_length = span_activations.size(1)
if span_length > 0:
    if span_length == 1:
        mean_activation = span_activations.squeeze(1)       # (num_layers, hidden_size)
    else:
        mean_activation = span_activations.mean(dim=1)       # (num_layers, hidden_size)
    turn_activations.append(mean_activation)
```

And then in `2_activations.py`, the pipeline takes only assistant turns and averages across them too:

```python
# 2_activations.py — lines 114-120

assistant_act = conv_acts[1::2]   # All assistant turns (odd indices)
if assistant_act.shape[0] > 0:
    mean_act = assistant_act.mean(dim=0).cpu()   # (num_layers, hidden_size)
    all_activations.append(mean_act)
```

So there are **two levels of averaging:**
1. `SpanMapper`: mean across tokens within one assistant turn → `(num_layers, hidden_dim)`
2. Pipeline: mean across all assistant turns in the conversation → `(num_layers, hidden_dim)`

For single-turn conversations (the common case in this project), both levels just produce the same result since there's only one assistant turn.

---

## 9. Batched Extraction — Padding, Masking, and Why It's Tricky

### The concept

Processing one conversation at a time is simple but slow. A GPU can process 16 conversations in roughly the same time as 1 (it's a parallel processor). But batching requires all sequences to be the same length — hence padding.

### The problem with padding

Padding creates two issues:

**Issue 1: Wasted computation.** If one conversation is 400 tokens and another is 100 tokens, you pad both to 400. The model processes 400 tokens for the short conversation even though 300 of them are meaningless. Wasted GPU cycles and memory.

**Issue 2: Corrupted attention.** Without an attention mask, the model's self-attention mechanism would let real tokens attend to padding tokens, mixing garbage into the real activations. The attention mask prevents this.

### The code

```python
# activations.py — batch_conversations(), lines 289-316

# Step 1: Find the longest conversation, cap at max_length
actual_max_len = max(len(ids) for ids in batch_full_ids)
max_seq_len = min(max_length, actual_max_len)

# Step 2: Pad each conversation to max_seq_len
for ids in batch_full_ids:
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]   # Truncate if too long

    padded_ids = ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids))
    attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

# Step 3: Stack into 2D tensors
input_ids_tensor = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
attention_mask_tensor = torch.tensor(attention_mask_batch, dtype=torch.long, device=device)

# Step 4: Forward pass with attention mask
_ = self.model(
    input_ids=input_ids_tensor,
    attention_mask=attention_mask_tensor,
)
```

### Why `max_length` exists

The `max_length` parameter (default 4096 in `batch_conversations`, set to 2048 in the pipeline) prevents memory explosions. If one conversation in a batch of 16 is 10,000 tokens long, every conversation in the batch would be padded to 10,000 tokens. That's `16 × 10,000 × 4,096 × 2 bytes ≈ 1.3 GB` per layer, times 64 layers = 83 GB just for activations. Truncation trades off completeness for feasibility.

### Why padding is on the right (not left)

```python
padded_ids = ids + [pad_token_id] * (max_seq_len - len(ids))
# Real tokens first, then padding
```

For activation extraction, right-padding is correct because token positions must stay stable. If you left-padded, the span indices (which say "the assistant's response starts at token 43") would be wrong — padding tokens would shift everything to the right.

Note: `ProbingModel` sets `tokenizer.padding_side = "left"` (line 51 in `model.py`), which is used during *generation* (where left-padding is standard). But `batch_conversations()` does its own manual right-padding, overriding this setting.

---

## 10. From Raw Activations to Per-Turn Means (The SpanMapper Step)

### The concept

`ActivationExtractor.batch_conversations()` returns a raw 4D tensor:

```
(num_layers, batch_size, max_seq_len, hidden_dim)
 e.g., (64, 16, 512, 4096)
```

This contains activations for every token at every layer — including system prompts, user questions, role markers, and padding. We only want the **content tokens of the assistant's response,** averaged into a single vector per layer.

This is what `SpanMapper` does. It takes the raw activations + span metadata and slices out just the parts we care about.

### The span metadata

`ConversationEncoder.build_batch_turn_spans()` produces spans like:

```python
[
    {"conversation_id": 0, "turn": 0, "role": "user",      "start": 15, "end": 42, "n_tokens": 27},
    {"conversation_id": 0, "turn": 1, "role": "assistant",  "start": 45, "end": 187, "n_tokens": 142},
    {"conversation_id": 1, "turn": 0, "role": "user",      "start": 12, "end": 38, "n_tokens": 26},
    {"conversation_id": 1, "turn": 1, "role": "assistant",  "start": 41, "end": 201, "n_tokens": 160},
    ...
]
```

The `start` and `end` are token indices into the conversation's token sequence. They point to the **content** of each turn — not the structural tokens around it. `start` is inclusive, `end` is exclusive.

### The slicing

```python
# spans.py — map_spans(), line 97

span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
# shape: (num_layers, span_length, hidden_size)
# e.g.,  (64, 142, 4096) for a 142-token assistant response
```

This is a tensor slice — no data is copied, it's a view into the same memory. Then the mean is computed:

```python
mean_activation = span_activations.mean(dim=1)  # (num_layers, hidden_size)
```

This collapses the 142 token vectors into 1 vector per layer by averaging.

### The pipeline then takes only assistant turns

```python
# 2_activations.py — lines 114-120

# conv_acts shape: (num_turns, num_layers, hidden_size)
# turns are in order: [user, assistant, user, assistant, ...]
# [1::2] selects indices 1, 3, 5, ... = all assistant turns
assistant_act = conv_acts[1::2]
mean_act = assistant_act.mean(dim=0).cpu()  # (num_layers, hidden_size)
```

So the full reduction chain is:

```
Raw activations:  (64 layers, 512 tokens, 4096 hidden)
                              ↓
SpanMapper mean (tokens 45-187 only):
                  (64 layers, 4096 hidden)     ← one vector per layer for this turn
                              ↓
Pipeline mean across assistant turns:
                  (64 layers, 4096 hidden)     ← same shape, averaged if multi-turn
```

---

## 11. Why Intermediate Layers? Why Not the Final Output?

### The concept

The model's final output is logits — probability distributions over the vocabulary for the next token. These are useful for generation but terrible for persona analysis because:

1. **Logits are about prediction, not representation.** The logit vector says "the next word is probably 'the' (p=0.12) or 'a' (p=0.08)." It doesn't say "I'm currently behaving like a pirate."

2. **Logits are too specific.** Two responses to the same question — both fully pirate-like but using different words — would have very different logits but should have similar persona vectors.

3. **Middle layers encode abstract features.** Research consistently shows that early layers encode surface features (syntax, word identity), middle layers encode semantic/conceptual features (meaning, intent, persona, role), and late layers specialize for next-token prediction. The persona signal peaks at middle layers.

This is why the project uses `target_layer` values near the middle of each model:

```python
# models.py — MODEL_CONFIGS

"google/gemma-2-27b-it":            {"target_layer": 22, "total_layers": 46},   # 48%
"Qwen/Qwen3-32B":                   {"target_layer": 32, "total_layers": 64},   # 50%
"meta-llama/Llama-3.3-70B-Instruct": {"target_layer": 40, "total_layers": 80},  # 50%
```

### Why extract ALL layers instead of just the target layer?

The pipeline extracts all layers by default (`--layers all`), not just the target layer. This costs more memory and time but enables:

1. **The axis is computed at every layer.** `compute_axis()` produces shape `(n_layers, hidden_dim)`. While projection typically uses the target layer, having all layers enables per-layer analysis (e.g., "at which layer does the persona signal emerge?").

2. **`axis_norm_per_layer()`** computes the L2 norm of the axis at each layer, revealing which layers carry the strongest signal.

3. **The notebooks do multi-layer analysis** (PCA across layers, cosine similarity per layer).

---

## 12. Putting It All Together — The Complete Pipeline Trace

Here's one conversation traced through every step, from disk to the final axis.

### Input

File: `data/roles/instructions/pirate.json`
```json
{"instruction": [{"pos": "You are a pirate captain on the high seas..."}], ...}
```

File: `data/extraction_questions.jsonl`
```json
{"question": "How do you handle conflict with others?"}
```

### Step 1 — Generate (produces the conversation text)

```
vLLM generates → pirate.jsonl:
{
  "conversation": [
    {"role": "system", "content": "You are a pirate captain..."},
    {"role": "user", "content": "How do you handle conflict?"},
    {"role": "assistant", "content": "Arrr! When some scurvy dog..."}
  ],
  "prompt_index": 0,
  "question_index": 47,
  "label": "pos"
}
```

### Step 2 — Extract Activations (this is what `activations.py` does)

```python
# Pipeline creates the three objects
pm = ProbingModel("Qwen/Qwen3-32B")                           # Loads model
encoder = ConversationEncoder(pm.tokenizer, pm.model_name)     # For formatting
extractor = ActivationExtractor(pm, encoder)                   # For extraction
span_mapper = SpanMapper(pm.tokenizer)                         # For aggregation

# 1. TOKENIZE: conversation → token IDs
#    apply_chat_template wraps in <|im_start|>system\n...<|im_end|>\n...
#    Result: [151644, 8948, 198, 2610, 525, ...] — say 200 tokens

# 2. PAD: batch of 16 conversations, pad to longest (say 350 tokens)
#    Real tokens get attention_mask=1, padding gets attention_mask=0

# 3. FORWARD PASS: model processes the batch
#    Input:  (16, 350)           — token IDs
#    Layer 0 → Layer 1 → ... → Layer 63
#    Hooks fire at every layer, capturing:
#    Output: (64, 16, 350, 4096) — activations for all layers, all tokens

# 4. SPAN MAPPING: find assistant turn tokens (say positions 45-200)
#    Slice: batch_activations[:, conv_id, 45:200, :]  → (64, 155, 4096)
#    Mean over 155 tokens:                             → (64, 4096)

# 5. SAVE: activations_dict["pos_p0_q47"] = tensor(64, 4096)
torch.save(activations_dict, "pirate.pt")
```

### Steps 3-5 — Score, Filter, Compute Axis

```python
# Step 3: LLM judge gives score=3 (fully playing pirate)
# Step 4: pirate_vector = mean of all score=3 activations → (64, 4096)
# Step 5: axis = mean(default_vectors) - mean([pirate_vector, doctor_vector, ...]) → (64, 4096)
```

### Using the axis

```python
# Later, to check if a model is drifting away from the Assistant:
from assistant_axis import project

projection = project(some_new_activation, axis, layer=32)
# projection > 0 → more Assistant-like
# projection < 0 → drifting toward character behavior
```

That projection value is a single scalar derived from:
- A conversation that was tokenized (with all its special tokens)
- Fed through a 64-layer transformer
- Intercepted at layer 32 via a forward hook
- Averaged over the assistant's content tokens (excluding BOS, EOS, role markers, padding)
- Dot-producted with the normalized axis direction at layer 32
