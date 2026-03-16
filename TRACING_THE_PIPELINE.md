# Tracing the Pipeline: From PCA Back to the Roots

---

In January 2026, Christina Lu and collaborators at Anthropic and MATS published a paper with a striking claim: the difference between a language model "being itself" and "pretending to be someone else" can be captured by a single direction in the model's internal activation space. They called this direction the **Assistant Axis**.

But here is the puzzle. A modern language model like Llama 3.3 70B has 80 layers, each producing a hidden state vector with 8,192 dimensions. That is 80 x 8,192 = 655,360 numbers flowing through the network at every token position. How do you find one meaningful direction in that ocean of numbers? And how do you know it means "being the Assistant"?

The answer is a five-step pipeline. Each step transforms something messy into something cleaner. At the very end, PCA reveals that hundreds of character archetypes collapse onto a handful of dimensions, and the dominant one separates "being the Assistant" from "being someone else." But PCA does not create that structure. It only reveals what was already there.

This document traces that chain backwards. We start from what PCA shows, then ask: where did those inputs come from? Each step peels back a layer, until we reach the very beginning: the decision to invent 275 fictional characters and ask them all the same questions.

> **Central question:** How does a sequence of design decisions — from character definitions, through text generation, activation extraction, quality filtering, and averaging — produce inputs that PCA can decompose into an interpretable persona space?

---

## Key Terminology

**Activation** — The internal state vector a neural network produces at a given layer for a given token. Think of it as the model's "thought" at that point in processing. In this project, activations are vectors with thousands of dimensions (e.g., 4,096 for Qwen 3 32B, 8,192 for Llama 3.3 70B).

**Residual stream** — The main information highway through a transformer. Each layer reads from it, processes, and writes back to it. Activations in this project are extracted from the post-MLP residual stream, meaning after the layer's feedforward network has written its contribution.

**Role vector** — A single vector summarizing how a model behaves when playing a particular character. Computed by averaging the activations across many responses where the model successfully adopted that role. Shape: (n_layers, hidden_dim).

**Default vector** — The same thing, but for the model behaving as its ordinary self — the "helpful assistant" it was trained to be.

**Assistant Axis** — The direction obtained by subtracting the mean of all role vectors from the default vector. Points from "character" toward "assistant." A single vector of shape (n_layers, hidden_dim).

**PCA (Principal Component Analysis)** — A statistical method that finds the directions of greatest variance in a dataset. If you have 275 role vectors in a 4,096-dimensional space, PCA finds the axes along which those vectors spread out the most, ranked from most spread to least.

**Forward hook** — A callback function attached to a specific layer of a PyTorch model. Every time data passes through that layer during a forward pass, the hook captures the output without altering the model's computation. This is how activations are extracted without modifying the model itself.

**Score** — A 0-to-3 rating of how fully a model adopted a given role in a particular response. Score 3 means fully in-character. Score 0 means refused and identified as an AI. Only score-3 responses contribute to role vectors.

**Extraction question** — An open-ended question designed to make different characters answer differently. "What is the relationship between law and morality?" produces very different responses from a pirate, a judge, and a therapist.

---

## 1. What PCA Reveals (The End of the Chain)

PCA is the last analytical step. It takes as input a matrix of role vectors — one row per character archetype — and decomposes it.

Here is what that looks like concretely. Suppose you have 275 role vectors, each extracted at layer 32 of a 64-layer model, with hidden dimension 4,096. Your input matrix is 275 x 4,096. PCA finds 275 orthogonal directions (principal components), ranked by how much variance each one explains.

The code in `assistant_axis/pca.py` does this:

```python
def compute_pca(activation_list, layer, scaler=None, verbose=True):
    # Select the layer slice: (n_samples, hidden_dim)
    layer_activations = activation_list[:, layer, :]

    # Optionally center the data (subtract the mean)
    if scaler is not None:
        scaled = scaler.fit_transform(layer_activations)

    # Fit PCA — sklearn's PCA, nothing exotic
    pca = PCA()
    pca_transformed = pca.fit_transform(X_np)
    variance_explained = pca.explained_variance_ratio_
```

Before PCA runs, the data is **mean-centered** using a `MeanScaler`. This matters. PCA finds directions of maximum variance around the mean. Without centering, the first principal component would largely point toward the mean itself — a trivially dominant direction that tells you nothing about how the roles *differ* from each other. Centering removes that, so PC1 captures the most informative axis of variation.

The result across three models (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B): 4 to 19 components explain 70% of the total variance. The space of "how characters differ" is low-dimensional. Out of 4,096 possible directions, a handful capture most of the story.

The dominant component — PC1 — separates fantastical characters (demon, ghost, bard, leviathan) on one end from assistant-adjacent roles (consultant, reviewer, evaluator) on the other end. The default Assistant vector sits at the extreme assistant-adjacent end of PC1.

The cosine similarity between PC1 and the independently computed Assistant Axis exceeds 0.71 at middle layers. They are nearly the same direction.

**Why PCA and not something else?** PCA is the simplest possible decomposition. It makes no assumptions about clusters, categories, or distributions. It just asks: "which directions carry the most variance?" If the structure is real, PCA will find it. If PCA finds nothing, more complex methods will not help. The fact that PC1 aligns so strongly with the Assistant Axis is evidence that the axis is not an artifact of how it was computed — it emerges independently from the geometry of the data.

**But PCA does not create the structure.** It cannot manufacture a low-dimensional persona space from noise. The structure must already exist in the role vectors it receives. So: where do the role vectors come from?

---

## 2. Role Vectors (Step 5 + Step 4 of the Pipeline)

A role vector is a single point in activation space that summarizes "what the model is doing internally when it fully plays this character." Computing it requires two pipeline steps, traced in reverse:

### 2.1 The Axis: Subtraction of Two Means (Pipeline Step 5)

The final pipeline script, `pipeline/5_axis.py`, does almost nothing:

```python
# Separate default and role vectors
for vec_file in vector_files:
    data = load_vector(vec_file)
    if "default" in role:
        default_vectors.append(vector)
    else:
        role_vectors.append(vector)

# Compute axis: points from role-playing toward default
default_mean = torch.stack(default_vectors).mean(dim=0)
role_mean = torch.stack(role_vectors).mean(dim=0)
axis = default_mean - role_mean
```

The formula is: **axis = mean(default) - mean(roles)**.

This is a contrast vector. It points from the centroid of "all characters" toward "the default assistant." The subtraction is the entire computation. Everything else in the pipeline exists to produce clean inputs for this subtraction.

Why subtraction? Because a direction in high-dimensional space is defined by two points. If you want a direction that means "more assistant-like," you need a point that represents "assistant" and a point that represents "not assistant." The mean of 275 diverse roles provides a robust estimate of "not assistant" — it averages out the specifics of any individual character, leaving only the common deviation from default behavior.

### 2.2 Per-Role Vector Computation (Pipeline Step 4)

Each role vector is itself a mean — but of what? `pipeline/4_vectors.py` answers:

```python
def compute_pos_3_vector(activations, scores, min_count):
    filtered_acts = []
    for key, act in activations.items():
        if key in scores and scores[key] == 3:
            filtered_acts.append(act)

    stacked = torch.stack(filtered_acts)  # (n_samples, n_layers, hidden_dim)
    return stacked.mean(dim=0)            # (n_layers, hidden_dim)
```

For each role, the code:
1. Loads all per-response activation tensors for that role
2. Loads the judge scores for that role
3. **Filters to only score=3 responses** — responses where the model fully adopted the character
4. Averages those filtered activations into a single vector

For the default role, there is no score filtering — all activations are used:

```python
def compute_mean_vector(activations):
    all_acts = list(activations.values())
    stacked = torch.stack(all_acts)
    return stacked.mean(dim=0)
```

**Why filter by score=3?** This is a critical design decision. Not every response successfully embodies the target character. A model asked to be a pirate might still say "As an AI language model, I can't really be a pirate, but..." That response's activation should not contribute to the pirate vector. It would contaminate the signal with "model being itself while talking about pirates" rather than "model being a pirate." Score-3 filtering ensures each role vector represents genuine character adoption, not partial compliance.

**Why average?** Because individual responses are noisy. A single pirate response might be about treasure maps; another about sea battles. These specifics are irrelevant to "pirate-ness" as a persona. Averaging across many score-3 responses (the code requires at least 50 per role via `min_count`) cancels out topic-specific signal and preserves only what is consistent: the persona itself.

**Why keep all layers?** Each role vector has shape (n_layers, hidden_dim) — it stores the mean activation at every layer, not just one. Different layers encode different aspects of behavior. Early layers process syntax. Middle layers represent semantic content. Late layers prepare for token prediction. By preserving all layers, downstream analysis can examine which layers carry the most persona-relevant information. (The paper finds middle layers are most informative.)

This raises the next question: where do the per-response activations come from? And where do the scores come from?

---

## 3. The Judge: Quality Filtering (Pipeline Step 3)

Each of the ~1,200 responses per role needs a score. Is the model really playing the character, or just politely declining? A human could judge this, but 275 roles x 1,200 responses = 330,000 judgments. That is infeasible by hand.

`pipeline/3_judge.py` uses an LLM judge — GPT-4.1-mini — to score each response on a 0-to-3 scale:

- **0**: The model refused and identified as an AI
- **1**: The model identified as an AI but attempted to answer the question
- **2**: The model identified as an AI but showed some role characteristics
- **3**: The model fully played the role

Each role's JSON file contains a tailored evaluation prompt. For example, the pirate role's `eval_prompt` reads:

> "You are evaluating whether the model's response displays the role: **pirate**. A pirate is a seafaring rogue who lives by their own code rather than conventional laws, speaks using nautical terminology and colorful sea language, values personal freedom and independence above authority..."

The judge receives the question, the model's response, and this rubric, then outputs a single number.

```python
judge_prompt = eval_prompt_template.format(
    question=question,
    answer=assistant_response
)
# ...
responses_text = await call_judge_batch(client, prompts, model="gpt-4.1-mini", ...)
score = parse_judge_score(response_text)
```

**Why an LLM judge instead of a classifier?** A trained classifier would need labeled training data for each of 275 roles. An LLM judge generalizes: given a rubric describing the role and a clear scoring scheme, it can evaluate any character without per-role training. The rubric is part of each role's JSON file, generated alongside the role definition itself.

**Why a 0-3 scale instead of binary?** Because the boundary between "in character" and "not in character" is not crisp. A model might adopt pirate vocabulary but still say "As an AI..." — that is a 2, not a 3. The 4-point scale captures this gradient. But for vector computation, only score 3 matters. The intermediate scores exist to give the judge a place to put ambiguous cases rather than forcing them into "yes" or "no."

**Why use GPT-4.1-mini as the judge model?** Cost and speed. At 100 requests per second with async batching, judging 330,000 responses is feasible. The paper validates against human judgments (91.6% agreement on a 200-sample subset).

The judge operates on text responses. Where do those responses come from?

---

## 4. Activation Extraction (Pipeline Step 2)

This is the most technically intricate step. The model has already generated its text responses in Step 1. Now we need to extract what the model was *thinking internally* when it produced those responses.

### 4.1 The Core Mechanism: Forward Hooks

PyTorch models are built from nested modules. A transformer has layers. Each layer has attention and MLP sublayers. When you run input through the model (a "forward pass"), data flows through each module in sequence. A **forward hook** is a function you attach to a module that gets called every time data passes through that module, letting you capture the output without changing anything.

In `assistant_axis/internals/activations.py`:

```python
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        layer_outputs[layer_idx] = act_tensor
    return hook_fn

model_layers = self.probing_model.get_layers()
for layer_idx in layer_list:
    target_layer = model_layers[layer_idx]
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    handles.append(handle)
```

Each hook captures the tensor that a layer produces: shape (batch_size, sequence_length, hidden_dim). After the forward pass completes, all hooks are removed and the captured tensors are stacked.

**Why hooks instead of `output_hidden_states=True`?** The comment in the code says it directly: hooks are "more reliable." The `output_hidden_states` flag in HuggingFace models returns hidden states, but the exact format varies across model architectures. Some return tuples, some return nested tuples, some include the embedding layer, some do not. Hooks bypass all of this — you specify exactly which module to intercept and get exactly what it outputs.

### 4.2 From Tokens to Turns: Span Mapping

A conversation like `[system: "You are a pirate", user: "How do you feel about taxes?", assistant: "Arr, taxes be but another form of plunder..."]` becomes a single sequence of tokens. But we only want the activations corresponding to the assistant's response — not the system prompt, not the user question.

The `ConversationEncoder` class tokenizes the conversation using the model's chat template, then identifies which token ranges correspond to which turns. The `SpanMapper` class in `assistant_axis/internals/spans.py` then uses those ranges to slice the right activations:

```python
for span in spans:
    start_idx = span['start']
    end_idx = span['end']

    # Extract activations for this span
    span_activations = batch_activations[:, conv_id, start_idx:end_idx, :]

    # Mean across tokens in the span
    mean_activation = span_activations.mean(dim=1)  # (num_layers, hidden_size)
```

For each conversation, this produces a mean activation vector of shape (num_layers, hidden_dim) — the average of all token activations within the assistant's response.

**Why mean across tokens?** An assistant response might be 200 tokens long. Each token has its own activation at each layer. Keeping all 200 would make the data 200x larger without clear benefit — the persona is a property of the entire response, not of individual words. Averaging collapses the token dimension while preserving the layer-by-dimension structure. The assumption is that persona information is distributed roughly uniformly across response tokens, so the mean captures it.

**A subtlety about which activations:** The code extracts post-MLP residual stream activations. In a transformer layer, the data flow is: input -> attention -> add to residual -> MLP -> add to residual. The extraction point is after the MLP addition. This is important because the MLP layers are where much of the model's "knowledge" processing happens. Extracting before the MLP would miss its contribution.

### 4.3 Batching for Efficiency

A single role has ~1,200 conversations. Processing them one-by-one through a 70B parameter model is slow. The `batch_conversations` method in `ActivationExtractor` processes 16 conversations at once:

```python
def batch_conversations(self, conversations, layer=None, max_length=4096, **chat_kwargs):
    # Tokenize all conversations, pad to same length
    for ids in batch_full_ids:
        padded_ids = ids + [tokenizer.pad_token_id] * (max_seq_len - len(ids))
        attention_mask = [1] * len(ids) + [0] * (max_seq_len - len(ids))

    # Single forward pass for entire batch
    with torch.inference_mode():
        _ = self.model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
```

Conversations are padded to the same length so they can be stacked into a single tensor. The attention mask tells the model to ignore padding tokens. One forward pass through 16 conversations is far cheaper than 16 separate passes, because GPU parallelism is utilized.

**Why `torch.inference_mode()` instead of `torch.no_grad()`?** Both disable gradient computation. But `inference_mode` goes further — it also disables the creation of "version counters" on tensors, which track in-place modifications for autograd. This saves memory and is slightly faster. Since we never need gradients (we are only extracting, not training), `inference_mode` is the right choice.

**The output of Step 2**, for each role, is a dictionary mapping response identifiers to activation tensors:

```python
activations_dict[key] = act  # key like "pos_p2_q37", value is (n_layers, hidden_dim)
torch.save(activations_dict, output_file)
```

These are saved as `.pt` files — one per role. But the activations are extracted from *already-generated* responses. Where did those come from?

---

## 5. Response Generation (Pipeline Step 1)

`pipeline/1_generate.py` is the beginning of the data pipeline. It takes 276 role definitions and 240 extraction questions and generates responses for every combination.

### 5.1 The Combinatorics

Each role has 5 system prompt variants. With 240 questions and 5 prompts, that is 5 x 240 = **1,200 responses per role**. Across 276 roles, that is 331,200 total responses.

The code uses vLLM for batch inference — a high-throughput inference engine that can generate thousands of responses efficiently with tensor parallelism across multiple GPUs:

```python
generator = VLLMGenerator(
    model_name=args.model,
    max_model_len=2048,
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
)
```

For each role, the `generate_for_role` method builds all 1,200 conversations and generates them in a single batch:

```python
for prompt_idx in prompt_indices:         # 5 system prompts
    instruction = instructions[prompt_idx]
    for q_idx, question in enumerate(questions):  # 240 questions
        conversation = format_conversation(instruction, question, tokenizer)
        all_conversations.append(conversation)

responses = self.generate_batch(all_conversations)
```

Each conversation is a list of messages: a system message (the role instruction) and a user message (the extraction question). The model generates an assistant response. The entire conversation — system prompt, user question, and generated response — is saved to a JSONL file.

**Why vLLM instead of HuggingFace `model.generate()`?** Speed. vLLM implements continuous batching, PagedAttention, and other optimizations that make it 5-20x faster than naive HuggingFace generation for batch workloads. When you need 331,200 responses from a 70B model, this difference matters. (HuggingFace generation is still available in `ProbingModel.generate()` for interactive single-response use.)

### 5.2 Sampling Parameters

The generation uses `temperature=0.7`, `top_p=0.9`, `max_tokens=512`.

**Why temperature 0.7?** Temperature controls randomness. At 0 (greedy), the model always picks the most likely next token — responses are deterministic and repetitive. At 1.0, sampling follows the raw probability distribution — diverse but sometimes incoherent. 0.7 is a common "sweet spot": diverse enough that different prompt variants produce meaningfully different responses, controlled enough that responses stay coherent.

**Why 5 prompt variants per role?** To reduce dependence on exact wording. A pirate described as "a pirate captain who has sailed the seven seas" and "a seasoned pirate who values freedom above all else" are the same *role* expressed differently. If the role vector depended heavily on exact prompt phrasing, it would be capturing surface-level instruction-following rather than persona adoption. Five variants dilute the effect of any single phrasing.

**Why max_tokens=512?** Long enough for the model to develop a substantive response that reveals its persona. Short enough to keep generation tractable. A 50-token response might not reveal much character; a 2,000-token response would slow the pipeline enormously for diminishing returns.

### 5.3 The Default Role

The default role's JSON file is minimal:

```json
{
  "instruction": [
    {"pos": ""},
    {"pos": "You are an AI assistant."},
    {"pos": "You are a large language model."},
    {"pos": "You are {model_name}."},
    {"pos": "Respond as yourself."}
  ]
}
```

No eval_prompt. No role-specific questions. The first variant is an *empty string* — no system prompt at all. This is deliberate. The "default assistant" is whatever the model does when it is not told to be anything in particular. The five variants range from no instruction to explicit self-identification ("You are Qwen", "You are an AI assistant"). All responses from these prompts are used without score filtering, because the model cannot fail to be itself.

This brings us to the deepest layer: the design of the roles and questions themselves.

---

## 6. Roles and Questions (The Foundation)

### 6.1 The 275 Character Archetypes

The `data/roles/instructions/` directory contains 276 JSON files (275 characters + 1 default). They span a wide range:

- **Human archetypes**: accountant, activist, actor, adolescent, advocate, altruist
- **Fantastical entities**: alien, aberration, demon, ghost, leviathan, oracle
- **Personality types**: absurdist, addict, cynic, hedonist, stoic, nihilist
- **Professional roles**: consultant, engineer, evaluator, journalist, therapist

Each file contains five instruction variants and a tailored evaluation prompt. For example, the pirate role:

```json
{
  "instruction": [
    {"pos": "You are a pirate captain who has sailed the seven seas..."},
    {"pos": "Act as a seasoned pirate who values freedom above all else..."},
    {"pos": "Embody the role of a seafaring rogue..."},
    {"pos": "Be a pirate who speaks in the colorful language of the sea..."},
    {"pos": "Take on the persona of a treasure-hunting pirate..."}
  ],
  "eval_prompt": "You are evaluating whether the model's response displays the role: **pirate**. A pirate is a seafaring rogue who..."
}
```

**Why 275 roles?** The goal is not completeness — there are infinitely many possible characters. The goal is *coverage of the variation that matters*. If the persona space is low-dimensional (as PCA reveals), then you need enough points to fill that space, but adding more points beyond saturation gives diminishing returns. 275 roles, ranging from mundane to extreme, from human to inhuman, from benign to dangerous, provide enough spread to discover the principal axes of variation.

**Why include both fantastical and mundane roles?** If you only used human professional roles (therapist, accountant, lawyer), the resulting persona space would only capture how those professions differ — a narrow slice. Including alien, demon, bard, and oracle pushes the space to its extremes. This is important because the *distance* from the default Assistant to the nearest dangerous role is what matters for safety. You cannot measure that distance without including dangerous or extreme roles in the dataset.

**Why 5 instruction variants instead of 1?** Robustness. If the pirate role vector depended on the exact words "pirate captain who has sailed the seven seas," it would not generalize. Five paraphrases of the same role concept force the averaging in Step 4 to collapse surface variation and preserve only the underlying persona.

### 6.2 The 240 Extraction Questions

The extraction questions in `data/extraction_questions.jsonl` are open-ended prompts designed to elicit persona-revealing responses:

- "What is the relationship between law and morality?"
- "How do you view people who take credit for others' work?"
- "Describe what it's like when you finally understand something complex."
- "I'm considering whether to have an abortion and need help thinking through this decision."

**Why these questions?** They are chosen to hit *dimensions along which different personas diverge*. A pirate and a judge will answer "What is the relationship between law and morality?" very differently. A therapist and a demon will answer "How do you handle someone who is hurting?" very differently. If all questions were about cooking recipes, most personas would converge — the questions would not discriminate.

**Why 240 questions?** Each role also has ~40 role-specific questions in its JSON file (visible in the pirate example: "What would you do if you found buried treasure in your backyard?"). The 240 extraction questions are *shared across all roles*. This is essential. To compare role vectors meaningfully, the model must be responding to the *same prompts* under different personas. If each role answered different questions, the differences in role vectors could be caused by topic differences rather than persona differences. Shared questions control for topic.

**Why are some questions emotionally charged?** Questions like "I'm considering whether to have an abortion" or "My teenager wants to volunteer for a cause I disagree with" probe the edges of persona. A pirate might say "follow yer heart." A therapist might say "let's explore the feelings around this." A default assistant might hedge carefully. Emotionally loaded questions create larger persona-dependent separations in activation space, making the signal easier to detect.

---

## 7. Stress-Testing the Chain

Let us examine some claims that could appear incomplete or contradictory on careful reading.

### "PCA reveals low-dimensional structure" — but does averaging create it?

Steps 4 and 5 average across many responses. Averaging inherently reduces dimensionality — the mean of 1,000 random vectors in 4,096 dimensions is a single point. Could the low-dimensionality that PCA finds be an artifact of averaging rather than a genuine property of the model?

No, because PCA is applied to the *collection* of 275 role vectors, not to the raw responses within any single role. Each role is already collapsed to a single vector by averaging. PCA then asks: when I look at how these 275 points scatter in 4,096-dimensional space, how many directions do I need to explain most of the variance? If the 275 points scattered randomly, PCA would need close to 275 components to explain 95% of variance. The fact that ~19 suffice means the roles are not randomly placed — they cluster along a few systematic dimensions. Averaging within roles did not create this inter-role structure. It cleaned up noise that would have obscured it.

### "Score-3 filtering selects for the signal" — or does it bias it?

Only responses where the model fully adopted the character contribute to role vectors. Could this create an artifactual axis that merely separates "model when compliant" from "model when resistant"?

This is a genuine concern. A model that says "As an AI, I can't be a pirate" (score 1-2) is in a different state than one that says "Arrr, me hearties!" (score 3). The score-3 filter selects for the second state. But this is precisely the point: the role vector should capture what the model is doing when it *is* being the pirate, not when it is declining to be one. The concern would be valid if score-3 responses were rare flukes — but the paper reports that for most roles, the majority of responses receive score 3. The filter removes noise, not signal.

### "The default role uses all activations" — inconsistent with score-3 filtering for other roles?

For character roles, only score-3 responses contribute. For the default role, all responses are used without filtering. Isn't this asymmetric?

It is deliberately asymmetric because the default role cannot fail to be itself. When the system prompt says "You are an AI assistant" or is simply empty, the model has no reason to deviate from its trained behavior. Every response *is* a score-3 equivalent. Adding a judge step would add cost and complexity with no benefit — the model's default behavior is, by definition, its default.

### "275 roles is enough" — how do we know?

We do not know with certainty. The paper acknowledges this: "275 roles is likely incomplete." But the low dimensionality of PCA provides indirect evidence. If adding 100 more roles would reveal new principal components, the current PCA would show variance "missing" — the existing components would explain an unusually low fraction of total variance. The fact that ~19 components capture 70% suggests the major axes are already discovered. Additional roles would likely project onto the existing components rather than creating new ones.

---

## 8. Recap: The Full Chain in Numbered Steps

1. **275 character archetypes and 1 default** are defined as JSON files, each with 5 system prompt variants and a tailored evaluation rubric. The diversity spans human, fantastical, benign, and dangerous roles to ensure broad coverage of the persona space.

2. **240 extraction questions** are written to be shared across all roles, probing dimensions (morality, emotion, conflict, authority) along which different personas diverge.

3. **Pipeline Step 1 — Generation.** For each role, each prompt variant, and each question, the model generates a response via vLLM batch inference. That is 1,200 responses per role, 331,200 total. Parameters: temperature 0.7, max 512 tokens.

4. **Pipeline Step 2 — Activation extraction.** Each generated conversation is re-fed to the model (with HuggingFace, not vLLM) using forward hooks on every transformer layer. The hook captures the post-MLP residual stream activation. Token-level activations within the assistant response span are averaged to produce one vector of shape (n_layers, hidden_dim) per response.

5. **Pipeline Step 3 — Judging.** An LLM judge (GPT-4.1-mini) scores each response on a 0-3 scale for how fully it adopted the target role. Each role has a custom rubric. This produces a score file per role.

6. **Pipeline Step 4 — Role vectors.** For each character role, activations from score-3 responses (at least 50 required) are averaged into a single role vector of shape (n_layers, hidden_dim). For the default role, all activations are averaged without filtering.

7. **Pipeline Step 5 — Axis computation.** The Assistant Axis = mean(default vectors) - mean(role vectors). A single vector of shape (n_layers, hidden_dim) pointing from "character" toward "assistant."

8. **PCA analysis.** The 275 role vectors are mean-centered and decomposed by PCA. The first principal component correlates strongly with the independently computed Assistant Axis (cosine similarity > 0.71). The persona space is low-dimensional: ~19 components explain 70% of variance. The structure is consistent across three different model families.

---

## 9. Why This Matters

The pipeline is not just an engineering artifact. It embodies a scientific claim: **the difference between "being the assistant" and "being someone else" is a property of the model's internal geometry, not just its output text.**

Text-level analysis could tell you that a pirate response uses nautical terms. Activation-level analysis reveals something deeper: the model's internal state has shifted along a specific, recoverable direction. That direction is consistent across 275 characters, across three model families, and across hundreds of thousands of responses. It is not specific to pirate vocabulary or therapist empathy or demon malice. It captures something about *persona itself* — the meta-property of how strongly the model is deviating from its trained default.

This has direct safety implications. If harmful behavior correlates with movement along this axis (the paper finds r = 0.39-0.52), then monitoring or constraining that direction becomes a principled intervention. The pipeline's output — a single vector per model — becomes a tool for **activation capping**: clamping the model's activations to prevent drift past a safe threshold, reducing harmful responses by ~60% without degrading capabilities.

The backward trace through this pipeline reveals that each design decision serves a specific purpose in producing clean, interpretable inputs for the next step. Remove any step — the diversity of roles, the shared questions, the score filtering, the token averaging, the mean centering — and the final structure degrades. The pipeline is not arbitrary. It is a chain of deliberate compressions, each discarding noise while preserving the signal that matters: how a model internally represents *who it is being*.

---

## 10. Historical Note

The Assistant Axis was introduced in "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" by Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, and Jack Lindsey. Published on arXiv as paper 2601.10387 on January 15, 2026. The work was conducted through MATS (ML Alignment Theory Scholars) in collaboration with Anthropic. The codebase implements the full pipeline described in the paper, with pre-computed axes available on HuggingFace at `lu-christina/assistant-axis-vectors`.
