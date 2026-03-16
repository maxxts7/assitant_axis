# The Activation Calculation — What the Numbers Actually Are

How a transformer layer turns one set of numbers into another, what the hook captures, and every mathematical step from raw text to the final axis projection scalar — with the actual code at every stage.

---

## Table of Contents

- [1. What "Activation" Means Physically](#1-what-activation-means-physically)
- [2. Inside a Transformer Layer — The Full Computation](#2-inside-a-transformer-layer--the-full-computation)
  - [2.1 Layer Normalization](#21-layer-normalization)
  - [2.2 Self-Attention — The Core of Context](#22-self-attention--the-core-of-context)
  - [2.3 Feed-Forward Network — The Knowledge Store](#23-feed-forward-network--the-knowledge-store)
  - [2.4 Residual Connections — Why Layers Add, Not Replace](#24-residual-connections--why-layers-add-not-replace)
- [3. What the Hook Captures — The Exact Tensor](#3-what-the-hook-captures--the-exact-tensor)
- [4. One Token's Journey Through All 64 Layers](#4-one-tokens-journey-through-all-64-layers)
- [5. The Mean Over Tokens — Collapsing a Sequence to One Vector](#5-the-mean-over-tokens--collapsing-a-sequence-to-one-vector)
- [6. The Mean Over Responses — Collapsing 1200 Conversations to One Vector](#6-the-mean-over-responses--collapsing-1200-conversations-to-one-vector)
- [7. The Subtraction — Computing the Axis](#7-the-subtraction--computing-the-axis)
- [8. The Dot Product — Projection Onto the Axis](#8-the-dot-product--projection-onto-the-axis)
- [9. Normalization — Why and When](#9-normalization--why-and-when)
- [10. Cosine Similarity — Comparing Two Directions](#10-cosine-similarity--comparing-two-directions)
- [11. The Complete Numerical Trace — One Conversation End to End](#11-the-complete-numerical-trace--one-conversation-end-to-end)
- [12. Shape Tracking — Every Tensor at Every Step](#12-shape-tracking--every-tensor-at-every-step)

---

## 1. What "Activation" Means Physically

An activation is a list of floating-point numbers. For Qwen 3 32B, it's a list of 4096 numbers. Each number is stored as a `bfloat16` (2 bytes). One activation for one token at one layer is 8,192 bytes (8 KB).

```
One activation vector (Qwen 3 32B at layer 32):
[-0.0234, 0.1562, -0.0039, 0.0781, ..., -0.0156]
 ←─────────────── 4096 numbers ──────────────────→
```

These numbers don't have individual human-readable meanings. Dimension 1,847 doesn't mean "pirate-ness." The information is distributed across all 4,096 dimensions collectively. This is why the axis needs 4,096 dimensions too — it must compare the full representation, not individual features.

The numbers are computed by the transformer layer. Each layer takes in a vector of 4,096 numbers (from the previous layer's output) and outputs a new vector of 4,096 numbers. The input vector carries one meaning; the output vector carries a refined meaning after incorporating attention to other tokens and passing through the feed-forward network.

---

## 2. Inside a Transformer Layer — The Full Computation

A transformer layer is a composition of sub-modules. Here's what happens to a single token's activation vector as it passes through one layer. These are the models used in this project:

| Model | hidden_dim | num_heads | head_dim | intermediate_size | layers |
|-------|-----------|-----------|----------|-------------------|--------|
| Gemma 2 27B | 3,584 | 16 | 256 | 36,864 | 46 |
| Qwen 3 32B | 5,120 | 40 | 128 | 27,648 | 64 |
| Llama 3.3 70B | 8,192 | 64 | 128 | 28,672 | 80 |

Each layer does the same sequence of operations:

```
Input: x — shape (seq_len, hidden_dim)
                    │
         ┌──────────┴──────────┐
         │   Layer Norm (RMS)   │  Normalize the magnitudes
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   Self-Attention     │  Each token looks at all others
         └──────────┬──────────┘
                    │
              x + attention_output      ← RESIDUAL CONNECTION (add, don't replace)
                    │
         ┌──────────┴──────────┐
         │   Layer Norm (RMS)   │  Normalize again
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │   Feed-Forward (MLP) │  Non-linear transformation
         └──────────┬──────────┘
                    │
              x + ffn_output            ← RESIDUAL CONNECTION
                    │
Output: new x — shape (seq_len, hidden_dim)   ← THIS is what the hook captures
```

### 2.1 Layer Normalization

Before each sub-module, the input is normalized using **RMSNorm** (Root Mean Square Normalization):

```
For each token's vector x (4096 dims):

    rms = sqrt(mean(x²))               # The root-mean-square of the vector
    x_normalized = x / (rms + ε)        # Divide by rms to normalize
    output = γ * x_normalized           # Scale by learned weight γ
```

**Why:** Without normalization, activations can grow or shrink as they pass through layers, making training unstable. RMSNorm keeps the magnitude roughly constant while preserving direction.

**Concrete numbers** for one token (Qwen, hidden_dim=5120):

```
Input x:        [-0.0234, 0.1562, -0.0039, ..., -0.0156]   5120 values
mean(x²):       0.00847
rms:            0.0920
x / rms:        [-0.254, 1.698, -0.042, ..., -0.170]        magnitudes now ~1.0
γ * (x / rms):  [-0.189, 1.257, -0.031, ..., -0.126]        scaled by learned weights
```

### 2.2 Self-Attention — The Core of Context

Self-attention lets each token incorporate information from every other token. This is how the model knows that "pirate" in the system prompt should influence the activation of "handle" in the question.

**Step 1: Project into Q, K, V**

For each token's vector (5120 dims for Qwen), three linear projections produce queries, keys, and values:

```
Q = x @ W_Q      # (5120) @ (5120, 5120) → (5120)     "What am I looking for?"
K = x @ W_K      # (5120) @ (5120, 5120) → (5120)     "What do I offer?"
V = x @ W_V      # (5120) @ (5120, 5120) → (5120)     "What info do I carry?"
```

These are split into multiple "heads" (40 heads for Qwen, each with head_dim=128):

```
Q reshaped: (40 heads, 128 dims per head)
K reshaped: (40 heads, 128 dims per head)
V reshaped: (40 heads, 128 dims per head)
```

**Step 2: Compute attention scores**

For each head, each token computes how much attention to pay to every other token:

```
For head h, token i attending to token j:
    score[i,j] = Q_h[i] · K_h[j] / sqrt(128)
```

The division by `sqrt(head_dim)` prevents the dot products from growing too large (which would make softmax saturate).

For a 200-token sequence with 40 heads:
```
Raw scores: (40 heads, 200 tokens, 200 tokens) = 40 × 200 × 200 = 1.6M numbers
```

**Step 3: Apply causal mask and softmax**

In autoregressive (causal) models, token i can only attend to tokens 0...i (not future tokens). This is enforced by setting future positions to -infinity before softmax:

```
score[i,j] = -inf    if j > i    (can't look at future tokens)

weights = softmax(scores, dim=-1)
# Each row sums to 1.0 — it's a probability distribution over which tokens to attend to
```

**This is also where the attention mask for padding operates.** Padding positions get `-inf` scores, so they receive zero weight after softmax.

**Step 4: Weighted sum of values**

```
output_h[i] = Σⱼ weights[i,j] × V_h[j]
# For each token i: weighted average of all tokens' values
```

**Step 5: Concatenate heads and project**

```
all_heads = concat(output_0, output_1, ..., output_39)   # (40 × 128) = 5120 dims
attention_output = all_heads @ W_O                        # (5120) @ (5120, 5120) → (5120)
```

The result is a new 5120-dim vector for each token that incorporates context from all attended tokens.

### 2.3 Feed-Forward Network — The Knowledge Store

After attention, each token's vector passes through a feed-forward network independently (no cross-token interaction):

```
For Qwen (SwiGLU variant):
    gate = x @ W_gate        # (5120) @ (5120, 27648) → (27648)
    up   = x @ W_up          # (5120) @ (5120, 27648) → (27648)
    hidden = SiLU(gate) * up  # Element-wise: sigmoid(gate) * gate * up
    output = hidden @ W_down  # (27648) @ (27648, 5120) → (5120)
```

The intermediate size (27,648 for Qwen) is the FFN's width — it's much larger than the hidden dimension. This "expand then contract" pattern gives the FFN enough capacity to store learned associations (facts, patterns, behaviors).

**SiLU** (Sigmoid Linear Unit) is a smooth activation function: `SiLU(x) = x × sigmoid(x)`. The **gating** mechanism (`SiLU(gate) * up`) lets the network learn to selectively activate different features.

### 2.4 Residual Connections — Why Layers Add, Not Replace

After each sub-module (attention and FFN), the output is **added** to the input, not used to replace it:

```
after_attention = x + attention_output(norm(x))
after_ffn = after_attention + ffn_output(norm(after_attention))
```

**Why this matters for activations:** Because of residual connections, information from early layers flows directly to later layers without being forced through every intermediate computation. The activation at layer 32 contains:
- The original embedding (faintly)
- Contributions from every layer 0-31
- Layer 32's own attention and FFN contribution

This is why activations at middle layers are rich — they accumulate information from many processing stages.

---

## 3. What the Hook Captures — The Exact Tensor

The hook in `activations.py` fires **after** the layer's full computation (attention + FFN + residual connections):

```python
# activations.py — lines 86-91
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu())
    return hook_fn
```

`output` is what the layer's `forward()` method returned. For Llama/Qwen/Gemma layers, this is a tuple where `output[0]` is:

```
output[0] shape: (batch_size, seq_len, hidden_dim)
# e.g., (1, 200, 4096) for a 200-token conversation on Gemma 2 27B

# Each position contains the FULL residual stream at this layer:
# output[0][0, 42, :]  → the 4096-dim activation for token 42 at this layer
```

This tensor has already gone through:
1. RMSNorm → Self-attention → Residual add
2. RMSNorm → Feed-forward → Residual add

It represents the model's complete understanding of the conversation at this layer, at every token position.

---

## 4. One Token's Journey Through All 64 Layers

Let's trace token 42 (say the word "conflict" in the question "How do you handle conflict?") through all 64 layers of Qwen 3 32B.

```
EMBEDDING:
  Token ID 12847 → look up row 12847 in (152064, 5120) embedding table
  Result: h₀ = [0.0312, -0.0156, 0.0078, ..., -0.0234]  (5120 dims)
  This is a "dumb" representation — it only knows the word "conflict"

LAYER 0:
  h₀ → RMSNorm → Self-Attention (attends to "How", "do", "you", "handle")
       → residual add → RMSNorm → FFN → residual add
  h₁ = h₀ + attention₀ + ffn₀
  Now "conflict" knows its syntactic position (after "handle")

LAYER 1-15 (early layers):
  h₁ → h₂ → ... → h₁₅
  Each layer refines: word identity, syntax, basic semantics
  "conflict" now has absorbed the system prompt "You are a pirate captain"

LAYER 16-40 (middle layers):
  h₁₆ → ... → h₃₂ → ... → h₄₀
  Semantic features emerge: "conflict in the context of being a pirate"
  The persona information is strongest here — this is where activation
  extraction matters most

LAYER 41-63 (late layers):
  h₄₁ → ... → h₆₃
  Prediction features: "what words should follow?"
  The model is preparing to generate "Arrr! When some scurvy dog..."

LM HEAD:
  h₆₃ → linear projection → logits (152064 dims — one per vocab word)
  logits tell us the probability of each possible next word
  (Not used by activation extraction — we discard this)
```

The hook at layer 32 captures `h₃₂` — the activation vector at the point where semantic understanding is richest but before the model has committed to specific output words.

---

## 5. The Mean Over Tokens — Collapsing a Sequence to One Vector

The raw extraction gives one vector per token per layer. For a 200-token response at 64 layers:

```
Raw: (64, 200, 4096)
     layers  tokens  dimensions
```

The pipeline needs one vector per conversation (at each layer). The mean over tokens achieves this:

```python
# spans.py — map_spans(), lines 96-107
span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]
# shape: (64, 155, 4096) — only assistant response tokens (positions 45-200)

mean_activation = span_activations.mean(dim=1)
# shape: (64, 4096) — averaged across the 155 tokens
```

**What the mean does mathematically:**

```
For each layer L and each dimension d:
    mean[L, d] = (act[L, token_0, d] + act[L, token_1, d] + ... + act[L, token_154, d]) / 155
```

This is computed independently for each of the 4096 dimensions at each of the 64 layers.

**What information survives the mean:**
- Directions that are consistently present across tokens (like "I'm in pirate mode") are preserved
- Directions that vary randomly per token (like "the current word is 'cutlass' vs 'arrr'") cancel out
- The mean captures the **stable representation** — the persona, not the individual words

**What information is lost:**
- Token order (the mean is commutative — scrambling tokens gives the same mean)
- Token-specific features (which word was generated at each position)
- Positional information (early vs late in the response)

This trade-off is acceptable because persona is a conversation-level property, not a token-level one.

---

## 6. The Mean Over Responses — Collapsing 1200 Conversations to One Vector

After Step 5, each conversation produces one vector of shape `(64, 4096)`. For the pirate role with 800 score-3 responses, we have 800 such vectors. These are averaged into the pirate's **role vector**:

```python
# pipeline/4_vectors.py — the core logic
filtered = [act for key, act in activations.items() if scores[key] == 3]
# 800 tensors, each (64, 4096)

role_vector = torch.stack(filtered).mean(dim=0)
# stack: (800, 64, 4096)
# mean(dim=0): (64, 4096) — averaged across 800 responses
```

**What the mean does:**

```
For each layer L and each dimension d:
    pirate_vector[L, d] = (response_0[L, d] + response_1[L, d] + ... + response_799[L, d]) / 800
```

**What survives:** The consistent pirate signal — dimensions that are systematically different when the model is being a pirate. Noise from individual questions, word choices, and response lengths cancels out.

**Why minimum 50 responses:** Statistical reliability. With fewer samples, the mean is noisy and might not represent the true pirate direction. With 800 samples, the standard error is `σ / sqrt(800)` — very small.

---

## 7. The Subtraction — Computing the Axis

```python
# axis.py — compute_axis(), lines 28-55
def compute_axis(role_activations, default_activations):
    role_mean = role_activations.mean(dim=0)       # (64, 4096)
    default_mean = default_activations.mean(dim=0)  # (64, 4096)
    axis = default_mean - role_mean                  # (64, 4096)
    return axis
```

But in the pipeline, this happens at a higher level — Step 5 loads the per-role vectors (already averaged) and computes:

```python
# pipeline/5_axis.py — the core logic
role_mean = torch.stack(all_role_vectors).mean(dim=0)       # Mean of 275 role vectors
default_mean = torch.stack(default_vectors).mean(dim=0)     # Mean of default vector(s)

axis = default_mean - role_mean                              # (64, 4096)
```

**What the subtraction does:**

```
For each layer L and each dimension d:
    axis[L, d] = default_mean[L, d] - role_mean[L, d]
```

Dimensions where the default and roles have the same average value → `axis[L, d] ≈ 0`. These dimensions are irrelevant to persona.

Dimensions where the default is consistently higher than roles → `axis[L, d] > 0`. These dimensions indicate "assistant-ness."

Dimensions where roles are consistently higher → `axis[L, d] < 0`. These dimensions indicate "character-ness."

**The axis vector is not unit-length.** Its magnitude varies by layer:

```python
# axis.py — axis_norm_per_layer()
norms = axis.float().norm(dim=1).numpy()
# e.g., [0.12, 0.15, ..., 0.85, 0.92, ..., 0.45, 0.31]
#        early layers    middle layers      late layers
# The axis is strongest at middle layers — where persona lives
```

---

## 8. The Dot Product — Projection Onto the Axis

```python
# axis.py — project(), lines 58-88
def project(activations, axis, layer, normalize=True):
    act = activations[layer].float()    # (4096,) — one token/turn's activation
    ax = axis[layer].float()            # (4096,) — the axis at this layer

    if normalize:
        ax = ax / (ax.norm() + 1e-8)    # Make unit length

    return float(act @ ax)               # Dot product → scalar
```

**The dot product `act @ ax`:**

```
projection = Σᵢ act[i] × ax[i]    for i = 0, 1, ..., 4095
           = act[0]×ax[0] + act[1]×ax[1] + ... + act[4095]×ax[4095]
```

This is a sum of 4,096 products. Each dimension contributes: if `act[i]` and `ax[i]` have the same sign, dimension `i` contributes positively. If opposite signs, negatively.

**Geometric interpretation:** The dot product measures how much the activation vector "points in the same direction" as the axis. If the activation is perfectly aligned with the axis, the projection equals the activation's magnitude. If perpendicular, zero. If opposite, negative.

```
                    ax (axis direction, normalized)
                    ↗
                   /
                  /  θ
      act ───────•────────→
                 │
                 │ projection = |act| × cos(θ)
                 │
```

---

## 9. Normalization — Why and When

### Axis normalization in `project()`

```python
if normalize:
    ax = ax / (ax.norm() + 1e-8)
```

`ax.norm()` is the L2 norm (Euclidean length): `sqrt(Σᵢ ax[i]²)`

The `+ 1e-8` prevents division by zero if the axis is all zeros at some layer.

**Why normalize:** Without normalization, `act @ ax` depends on both:
1. How aligned `act` is with the axis (what we care about)
2. How long the axis vector is (an irrelevant scaling factor)

Normalizing to unit length (`norm = 1.0`) removes factor 2, so the projection only measures alignment.

**The batch version:**

```python
# axis.py — project_batch(), lines 91-116
acts = activations[:, layer, :].float()   # (batch, 4096)
ax = axis[layer].float()                   # (4096,)

if normalize:
    ax = ax / (ax.norm() + 1e-8)

return acts @ ax                           # (batch, 4096) @ (4096,) → (batch,)
```

Matrix-vector multiplication: each row of `acts` is dot-producted with `ax`, producing one scalar per sample.

### Normalization in cosine similarity

```python
# axis.py — cosine_similarity_per_layer(), lines 119-143
v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)
similarities = (v1_norm * v2_norm).sum(dim=1)
```

Here, **both** vectors are normalized (not just the axis). The result is cosine similarity — a value between -1 and +1 that measures directional alignment regardless of magnitude.

**`dim=1, keepdim=True`:** The norm is computed along dimension 1 (the hidden_dim axis). `keepdim=True` keeps the result as shape `(n_layers, 1)` instead of `(n_layers,)`, enabling broadcasting in the division.

**Why `v1_norm * v2_norm` then `.sum(dim=1)` instead of `v1_norm @ v2_norm.T`?** Because these are 2D tensors `(n_layers, hidden_dim)`, not batches of vectors. Element-wise multiply then sum along dim=1 computes the dot product at each layer independently. Matrix multiply would give cross-layer products, which is wrong.

---

## 10. Cosine Similarity — Comparing Two Directions

```python
# axis.py — cosine_similarity_per_layer()
# Compares two (n_layers, hidden_dim) tensors layer by layer

# Usage: compare a role vector with the axis
similarities = cosine_similarity_per_layer(pirate_vector, axis)
# Returns: numpy array of length n_layers
# e.g., [-0.82, -0.79, ..., -0.91, -0.88, ..., -0.45, -0.32]
#         ← pirate is anti-aligned with the axis (expected!)
```

**Why cosine and not dot product for comparison?** Different roles produce vectors of different magnitudes. A verbose role (generating long responses) might have larger activation magnitudes than a terse role. Cosine similarity normalizes this away — it only compares directions.

```
cosine(pirate_vector, axis) = -0.91 at layer 32
# Interpretation: the pirate direction is almost exactly opposite to the axis
# This makes sense — the axis points TOWARD assistant, AWAY FROM roles

cosine(doctor_vector, axis) = -0.45 at layer 32
# The doctor is less anti-aligned — it's closer to the assistant than the pirate
# Doctors are more "professional" and "helpful" — assistant-adjacent traits
```

---

## 11. The Complete Numerical Trace — One Conversation End to End

Let's trace exact shapes through every step for one pirate conversation on Qwen 3 32B.

```
INPUT TEXT:
  System: "You are a pirate captain on the high seas..."
  User:   "How do you handle conflict with others?"
  Model:  "Arrr! When some scurvy dog crosses me, I draw me cutlass..."

STEP 1: TOKENIZE
  apply_chat_template → "<|im_start|>system\nYou are a pirate...<|im_end|>\n..."
  tokenizer("...") → token IDs: [151644, 8948, 198, ..., 151645]
  Shape: (1, 247)    — 1 batch, 247 tokens

STEP 2: EMBED
  embed_tokens(input_ids) → (1, 247, 5120)
  Each of the 247 tokens is now a 5120-dim vector

STEP 3: PASS THROUGH 64 LAYERS
  Layer 0:  (1, 247, 5120) → RMSNorm → Attention → Add → RMSNorm → FFN → Add → (1, 247, 5120)
  Layer 1:  (1, 247, 5120) → ... → (1, 247, 5120)
  ...
  Layer 32: (1, 247, 5120) → ... → (1, 247, 5120)   ← HOOK FIRES, captures this
  ...
  Layer 63: (1, 247, 5120) → ... → (1, 247, 5120)   ← HOOK FIRES, captures this

STEP 4: HOOK CAPTURES (for all 64 layers)
  activations.append(act_tensor[0, :, :].cpu())
  Each capture: (247, 5120) — one per layer
  After stacking: (64, 247, 5120)
  Memory: 64 × 247 × 5120 × 2 bytes = 163 MB

STEP 5: SPAN MAPPING (take only assistant tokens)
  ConversationEncoder identifies assistant response: tokens 89-247
  SpanMapper slices: batch_activations[:, 0, 89:247, :]
  Shape: (64, 158, 5120)     — 158 assistant response tokens

STEP 6: MEAN OVER TOKENS
  span_activations.mean(dim=1)
  Shape: (64, 5120)           — one vector per layer
  Memory: 64 × 5120 × 2 = 655 KB (a massive reduction from 163 MB)

STEP 7: SAVED TO DISK
  activations_dict["pos_p2_q47"] = tensor(64, 5120)
  torch.save(activations_dict, "pirate.pt")

--- LATER, IN STEP 4 OF THE PIPELINE ---

STEP 8: FILTER BY SCORE
  scores["pos_p2_q47"] == 3    (fully pirate)
  → This activation is included

STEP 9: MEAN OVER ALL SCORE-3 RESPONSES
  800 tensors of shape (64, 5120) → stack → (800, 64, 5120) → mean(dim=0) → (64, 5120)
  This is the pirate_vector

--- STEP 5 OF THE PIPELINE ---

STEP 10: COMPUTE AXIS
  role_mean: mean of 275 role vectors → (64, 5120)
  default_mean: default vector → (64, 5120)
  axis = default_mean - role_mean → (64, 5120)

--- USING THE AXIS ---

STEP 11: PROJECT A NEW ACTIVATION
  new_activation: (64, 5120) — from some new conversation
  axis[32]: (5120,) — the axis at the target layer
  normalized: axis[32] / axis[32].norm() → (5120,) with norm 1.0

  projection = new_activation[32] @ normalized_axis[32]
             = Σᵢ new_activation[32, i] × normalized_axis[32, i]
             = 37.4     (a scalar — this response is assistant-like)
```

---

## 12. Shape Tracking — Every Tensor at Every Step

A complete reference for every tensor shape transformation in the pipeline.

### During extraction (one conversation)

| Step | Operation | Shape | Size (Qwen 32B) |
|------|-----------|-------|-----------------|
| Tokenize | `tokenizer(text)` | `(1, T)` | `(1, 247)` |
| Embed | `embed_tokens(ids)` | `(1, T, H)` | `(1, 247, 5120)` |
| Layer output | Hook captures | `(1, T, H)` | `(1, 247, 5120)` |
| Remove batch dim | `[0, :, :]` | `(T, H)` | `(247, 5120)` |
| Stack layers | `torch.stack(...)` | `(L, T, H)` | `(64, 247, 5120)` |
| Span slice | `[:, start:end, :]` | `(L, T', H)` | `(64, 158, 5120)` |
| Mean over tokens | `.mean(dim=1)` | `(L, H)` | `(64, 5120)` |

### During extraction (batch of 16)

| Step | Operation | Shape | Size |
|------|-----------|-------|------|
| Pad/truncate | Manual loop | `(B, T_max)` | `(16, 512)` |
| Layer output | Hook captures | `(B, T_max, H)` | `(16, 512, 5120)` |
| Stack layers | `torch.stack(...)` | `(L, B, T_max, H)` | `(64, 16, 512, 5120)` |
| Span slice (one conv) | `[:, conv_id, s:e, :]` | `(L, T', H)` | `(64, 158, 5120)` |
| Mean over tokens | `.mean(dim=1)` | `(L, H)` | `(64, 5120)` |

### During axis computation

| Step | Operation | Shape |
|------|-----------|-------|
| All score-3 activations for one role | `torch.stack(filtered)` | `(N, L, H)` e.g., `(800, 64, 5120)` |
| Role vector | `.mean(dim=0)` | `(L, H)` = `(64, 5120)` |
| All role vectors | `torch.stack(...)` | `(R, L, H)` = `(275, 64, 5120)` |
| Role mean | `.mean(dim=0)` | `(L, H)` = `(64, 5120)` |
| Default mean | Same | `(L, H)` = `(64, 5120)` |
| **Axis** | `default - role` | `(L, H)` = `(64, 5120)` |

### During projection

| Step | Operation | Shape |
|------|-----------|-------|
| Activation at target layer | `activations[layer]` | `(H,)` = `(5120,)` |
| Axis at target layer | `axis[layer]` | `(H,)` = `(5120,)` |
| Normalized axis | `ax / ax.norm()` | `(H,)` = `(5120,)` |
| **Projection** | `act @ normalized_ax` | scalar |

### Legend

| Symbol | Meaning | Typical values |
|--------|---------|---------------|
| `B` | Batch size | 1 or 16 |
| `T` | Sequence length (tokens) | 100-500 |
| `T_max` | Padded sequence length | 512 or 2048 |
| `T'` | Span length (assistant tokens only) | 50-400 |
| `H` | Hidden dimension | 3584, 5120, or 8192 |
| `L` | Number of layers | 46, 64, or 80 |
| `N` | Number of score-3 responses per role | 50-1000 |
| `R` | Number of roles | 275 |
