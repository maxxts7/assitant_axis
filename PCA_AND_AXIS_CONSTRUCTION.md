# The PCA Calculation and Axis Construction: Why Subtract the Mean of Roles?

How the Assistant Axis is built, why the mean of all role vectors is subtracted from the default, what this achieves mathematically, what the alternatives were, and why this particular choice was made.

---

## Table of Contents

- [1. The Setup: What You Have Before Any Calculation](#1-the-setup-what-you-have-before-any-calculation)
- [2. The Axis Formula: What It Does and What It Means](#2-the-axis-formula-what-it-does-and-what-it-means)
- [3. Why Subtract the Mean of ALL Roles?](#3-why-subtract-the-mean-of-all-roles)
- [4. The PCA Connection: How the Axis Relates to PC1](#4-the-pca-connection-how-the-axis-relates-to-pc1)
- [5. The Mean Subtraction in PCA: A Separate but Related Operation](#5-the-mean-subtraction-in-pca-a-separate-but-related-operation)
- [6. The Alternatives That Were Not Chosen](#6-the-alternatives-that-were-not-chosen)
- [7. The Motivation: Why This Design Works for Safety](#7-the-motivation-why-this-design-works-for-safety)
- [8. Putting It All Together: The Full Mathematical Picture](#8-putting-it-all-together-the-full-mathematical-picture)

---

## 1. The Setup: What You Have Before Any Calculation

Before any axis computation or PCA, the pipeline has already produced:

- **275 role vectors**, each of shape `(n_layers, hidden_dim)`. Each vector is the mean activation of a model when it is *fully* playing a particular character (pirate, demon, doctor, poet, etc.). Only responses that scored 3 on the role adherence scale are included -- these are responses where the model completely adopted the persona.

- **1 default vector**, same shape. This is the mean activation of the model when prompted with neutral system prompts ("You are an AI assistant", "Respond as yourself", or no system prompt at all). No score filtering is applied because the model is always "being itself" by definition.

Each of these 276 vectors lives in a space of dimensionality `hidden_dim` (3,584 for Gemma, 5,120 for Qwen, 8,192 for Llama) at each layer. The question is: what structure exists in this cloud of 276 points?

Here is what these vectors look like conceptually at a single layer, projected into a simplified 2D space:

```
                              ● default
                             .
                            .
                           .
          ● demon         .        ● consultant
                    ● pirate
        ● anarchist       ● poet
                       ● doctor
             ● cult_leader     ● reviewer
                     ● therapist
                 ● alien
```

The default vector sits off to one side. The role vectors cluster on the other side, spread out according to their individual characteristics, but all generally *away* from the default. The axis aims to capture the direction from that cloud of roles toward the default.

---

## 2. The Axis Formula: What It Does and What It Means

The formula is:

```
axis = default_mean - role_mean
```

Where:
- `default_mean` is the default vector (or mean of default variants): shape `(n_layers, hidden_dim)`
- `role_mean` is the mean of all 275 role vectors: shape `(n_layers, hidden_dim)`

In code (`pipeline/5_axis.py`, lines 76-80):

```python
default_mean = default_stacked.mean(dim=0)   # (n_layers, hidden_dim)
role_mean = role_stacked.mean(dim=0)          # (n_layers, hidden_dim)
axis = default_mean - role_mean               # (n_layers, hidden_dim)
```

**What this subtraction produces, dimension by dimension:**

For each of the 4,096+ dimensions at each layer:

| Situation | axis[d] value | Meaning |
|-----------|--------------|---------|
| default[d] >> role_mean[d] | Large positive | This dimension is characteristic of assistant behavior |
| default[d] << role_mean[d] | Large negative | This dimension is characteristic of role-playing behavior |
| default[d] ≈ role_mean[d] | Near zero | This dimension doesn't distinguish assistant from roles |

The axis is a vector where each dimension encodes *how much that dimension matters for being the assistant rather than a character*. Dimensions irrelevant to persona wash out to zero. Dimensions critical to persona get large magnitudes.

---

## 3. Why Subtract the Mean of ALL Roles?

This is the central design decision. There are three things to explain: why subtract (rather than some other operation), why use the mean of roles (rather than a single role), and why use all 275 roles (rather than a selected subset).

### 3.1 Why subtraction?

Subtraction gives you a **direction** in activation space. If you have two points A and B, the vector `B - A` points from A to B. The axis `default - role_mean` points from the average role-playing state to the default assistant state.

This direction is what you need for projection. When you project a new activation onto this direction (via dot product), you get a scalar that tells you where on the role-to-assistant spectrum that activation falls. High projection = closer to assistant. Low projection = closer to role-playing.

No other operation gives you this. Addition would be meaningless. Division operates element-wise and conflates magnitude with direction. Only subtraction produces a direction between two points in high-dimensional space.

### 3.2 Why use the mean of roles rather than a single role?

If you subtracted a single role -- say the pirate vector -- you would get:

```
axis_pirate = default - pirate_vector
```

This direction separates "pirate" from "assistant." It would excel at detecting pirate-ness. But it would be contaminated with pirate-specific features -- dimensions that distinguish pirates from *other characters*, not just from the assistant. When you project a demon's activation onto this axis, the result would be noisy: some pirate-specific dimensions would contribute positively (if the demon happens to share pirate features) and others negatively, even though both are equally "not the assistant."

By averaging all 275 roles, character-specific features cancel out:

```
role_mean = (pirate + demon + doctor + poet + ... + alien) / 275
```

- The dimension for "nautical vocabulary" is high only in the pirate. Averaged with 274 other roles, it gets diluted by a factor of 275. It barely appears in the mean.
- The dimension for "medical knowledge" is high only in the doctor. Same dilution.
- But the dimension for "not being the default assistant" is high in *all* roles. It survives the averaging intact.

What remains in `role_mean` is the **shared non-assistant signal** -- the common direction that all characters have simply by virtue of *not being the assistant*. The subtraction `default - role_mean` then isolates exactly this shared direction.

This is a classic application of the statistical principle: averaging reduces noise (role-specific variance) while preserving signal (the shared component).

### 3.3 Why all 275 roles rather than a subset?

The more roles you average over, the more completely role-specific features cancel. With 5 roles, you still have substantial noise from individual characters. With 275 roles spanning humans, non-humans, professionals, villains, fantastical creatures, and abstract archetypes, the cancellation is thorough.

The paper reports that the persona space is low-dimensional (4-19 components explain 70% of variance, depending on the model). This means the 275 vectors don't scatter uniformly across 4,096 dimensions -- they occupy a small subspace. The more densely you sample this subspace, the more accurately your mean represents its center, and the cleaner the resulting axis direction.

The 275 roles were specifically chosen to be diverse: human characters (pirate, doctor, poet), non-human characters (demon, alien, dragon), abstract roles (oracle, embodiment, leviathan), and assistant-adjacent roles (consultant, reviewer, evaluator). This diversity ensures the mean is genuinely representative and not biased toward any particular character family.

---

## 4. The PCA Connection: How the Axis Relates to PC1

Here is the key empirical finding: when you run PCA on the 275 role vectors (after mean-centering), the first principal component (PC1) aligns closely with the Assistant Axis.

### What PCA finds

PCA on the 275 role vectors at a target layer (e.g., layer 22 for Gemma 2 27B) yields:

```
PCA Results (Gemma 2 27B, layer 22):
  PC1: 48.8% of variance explained
  PC2:  9.8%
  PC3:  7.0%
  PC4:  5.5%
  First 4 PCs: 71.1% cumulative
  Dimensions for 70% variance: 4
  Dimensions for 90% variance: 18
```

PC1 alone captures nearly half the variance. This means that the single biggest source of variation among the 275 role vectors is a single direction. That direction correlates with the Assistant Axis at cosine similarity > 0.60 at all layers and > 0.71 at middle layers (paper, Section 3.1).

### Why this happens

The PCA finding is not a coincidence. It emerges from the structure of the data:

1. **All roles share one major source of variation**: how far they are from the default assistant. The demon is very far. The consultant is close. This variation dominates all others.

2. **Role-specific variations are individually small and uncorrelated**: the pirate's nautical features, the doctor's medical features, and the poet's literary features each account for a tiny fraction of the total variance, and they don't correlate with each other.

3. **PCA finds directions of maximum variance**: since the "distance from assistant" direction has the most variance (all 275 roles vary along it), PCA identifies it as PC1.

The Assistant Axis (computed via `default - role_mean`) and PC1 (computed via eigendecomposition of the covariance matrix) are two different mathematical routes to approximately the same direction. The axis uses explicit knowledge of the default vector. PCA discovers the direction blindly from the variance structure. Their agreement is the validation: the axis direction isn't an arbitrary choice -- it's the dominant structure in the data.

### Why the paper recommends the axis over PC1

Despite their alignment, the paper recommends using the contrast vector (`default - role_mean`) rather than PC1 for practical use:

1. **Reproducibility**: PC1 is determined by the eigendecomposition, which can flip sign arbitrarily (both `+PC1` and `-PC1` are equally valid eigenvectors). You need an additional step to determine which direction is "toward assistant." The contrast vector has a built-in orientation.

2. **Stability**: PC1 can shift if you change the set of roles or add new ones. The contrast vector only requires the default vector and the mean, making it more stable across different role sets.

3. **Guarantee**: PC1 is not *guaranteed* to correspond to the assistant direction in every model. It reflects maximum variance, which usually but not necessarily aligns with the assistant direction. The contrast vector is defined to point toward the assistant by construction.

---

## 5. The Mean Subtraction in PCA: A Separate but Related Operation

There are **two** mean subtractions in this project, and they serve different purposes. They are easy to confuse.

### Mean subtraction #1: In the axis formula

```python
axis = default_mean - role_mean
```

This subtracts the mean of 275 role vectors from the default vector. It produces the axis direction. This is a subtraction of **two specific centroids** to get a direction.

### Mean subtraction #2: In PCA preprocessing (the `MeanScaler`)

```python
# pca.py, MeanScaler
scaler = MeanScaler()
scaled = scaler.fit_transform(role_vectors_at_layer)
# Internally: scaled = role_vectors_at_layer - mean(role_vectors_at_layer)
```

This subtracts the **global mean** (mean of all 275 role vectors) from each individual role vector before running PCA. This is standard PCA preprocessing -- it centers the data cloud at the origin.

### Why PCA requires centering

PCA finds directions of maximum variance. Variance is defined relative to the mean. If you don't subtract the mean first, PC1 will point toward the mean of the data from the origin -- it captures where the data is located, not how it varies. This is uninteresting.

After centering, the data cloud is centered at the origin, and PCA captures the directions along which the centered data *spreads out* the most. These are the genuine directions of variation.

In the PCA notebook (`notebooks/pca.ipynb`, cell 8):

```python
scaler = MeanScaler()
pca_transformed, variance_explained, n_components, pca, scaler = compute_pca(
    role_vectors_at_layer,
    layer=None,
    scaler=scaler
)
```

The `MeanScaler` computes `mean(all 275 role vectors)` and subtracts it from each. This centers the role-vector cloud at the origin. Then `PCA()` finds the axes of maximum spread in the centered cloud.

### How the two relate

The mean subtracted in PCA preprocessing (the global mean of role vectors) is the same `role_mean` used in the axis formula. So the axis formula can be rewritten as:

```
axis = default_mean - role_mean
     = (default_mean - role_mean) + 0
     = (default_mean - role_mean)
```

After PCA centering, the default vector's position in the centered space is:

```
default_centered = default_mean - role_mean = axis
```

The centered default vector IS the axis. This is why PC1 (the direction of maximum variance in the centered data) aligns with the axis -- the default vector sits at an extreme of the distribution along the dominant direction, and its position after centering directly gives the axis.

This correspondence is not accidental. It's a mathematical consequence of the fact that:
1. The dominant variation in the role vectors is "distance from assistant"
2. Centering the role vectors by their own mean places the assistant at one extreme
3. PC1 points along that extreme

---

## 6. The Alternatives That Were Not Chosen

### Alternative 1: Use PC1 directly as the axis

**What it would look like:**
```python
pca = PCA(n_components=1)
pca.fit(mean_centered_role_vectors)
axis = pca.components_[0]  # The first principal component
```

**Pros:** No need for a default vector at all. The axis emerges purely from the structure of role-playing activations.

**Cons:**
- Sign ambiguity: PC1 has no inherent orientation. You'd need to check which end corresponds to the assistant.
- No guarantee PC1 captures the assistant direction in all models. In a model where role variations along some other dimension (e.g., formality) happened to be larger than the assistant direction, PC1 would capture formality instead.
- Less interpretable: the axis no longer has a clear semantic definition ("from roles toward assistant"). It's "the direction of maximum variance in role space."

**Why it was not chosen:** The paper found that PC1 and the contrast vector agree (cosine > 0.71 at middle layers), so there's no accuracy gain. But the contrast vector is more principled, reproducible, and guaranteed to point the right way. The paper states: the contrast vector method is recommended for reproducibility since PC1 is not guaranteed to correspond to an Assistant Axis in every model.

### Alternative 2: Subtract a single role vector

**What it would look like:**
```python
axis = default_mean - pirate_vector  # or demon_vector, or any single role
```

**Pros:** Simpler. No need to collect 275 roles.

**Cons:** As discussed in Section 3.2, this axis would be contaminated by pirate-specific features. It would work well for detecting pirate-ness but poorly as a general persona drift detector. The axis would not be robust to new, unseen character types.

**Why it was not chosen:** The goal is a general-purpose axis that detects drift toward *any* character, not just one specific role. The mean of many roles achieves this by cancelling role-specific noise.

### Alternative 3: Use a weighted mean of roles

**What it would look like:**
```python
# Weight roles by how different they are from the default
distances = [(role_vec - default_mean).norm() for role_vec in role_vectors]
weights = distances / sum(distances)
role_mean = sum(w * v for w, v in zip(weights, role_vectors))
axis = default_mean - role_mean
```

**Pros:** Roles that are more different from the assistant (demon, anarchist) would contribute more to the axis. Roles that are nearly identical to the assistant (consultant, reviewer) would contribute less. This might sharpen the axis.

**Cons:**
- Introduces a hyperparameter (the weighting scheme). Uniform averaging is parameter-free.
- Could bias the axis toward extreme roles, making it less sensitive to subtle drift.
- The empirical finding that PC1 already captures ~48% of variance with uniform weights suggests the unweighted mean is already effective.

**Why it was not chosen:** No evidence that weighting improves performance. The uniform mean is simpler, has no hyperparameters, and the resulting axis already correlates strongly with PC1, confirming that the dominant signal is captured.

### Alternative 4: Subtract the median of roles instead of the mean

**What it would look like:**
```python
role_median = torch.median(torch.stack(role_vectors), dim=0).values
axis = default_mean - role_median
```

**Pros:** The median is more robust to outliers. If a few extreme roles (e.g., the "leviathan" or "hive mind") have unusual activations, the median wouldn't be distorted by them.

**Cons:**
- The mean is the minimum-variance estimator of the center of a symmetric distribution. For cancelling role-specific noise, it's optimal.
- The median in high-dimensional spaces behaves differently from 1D intuition. The coordinate-wise median doesn't necessarily correspond to any meaningful "central" direction in 4,096 dimensions.
- PCA operates on the covariance matrix, which is defined in terms of means. Using the median would break the clean connection between the axis and PC1.

**Why it was not chosen:** The mean has better theoretical properties for this application. The role vectors are not known to have outlier problems -- they're all generated from the same pipeline with the same score filtering.

### Alternative 5: Use a learned linear probe

**What it would look like:**
```python
# Train a logistic regression: "default" vs "role-playing"
X = all_activation_vectors  # (n_samples, hidden_dim)
y = labels  # 1 for default, 0 for role-playing

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X, y)
axis = clf.coef_[0]  # The learned decision boundary normal
```

**Pros:** A supervised approach that directly optimizes for separating default from role-playing activations. Can handle non-uniform distributions and find optimal boundaries.

**Cons:**
- Requires labelled data (already have it, so this isn't a strong con).
- The learned axis depends on the training data distribution, regularization, and other hyperparameters.
- The connection to PCA would be lost -- a probe's direction doesn't necessarily align with the direction of maximum variance.
- Overfitting risk if the training distribution doesn't match deployment.
- The simple subtraction `default - mean(roles)` is equivalent to the optimal Bayes classifier direction when the two classes (default and roles) have equal covariance matrices and differ only in their means (Fisher's Linear Discriminant). Given that both classes are sampled from the same model architecture, this assumption is reasonable.

**Why it was not chosen:** The simple subtraction is analytically motivated, has no hyperparameters, and already aligns with PC1. A learned probe would add complexity without clear benefit.

### Alternative 6: Don't subtract anything -- use the default vector directly

**What it would look like:**
```python
axis = default_mean  # Just use the default vector as-is
```

**Pros:** Even simpler. No need for role vectors at all.

**Cons:** The default vector points from the origin to where the assistant lives in activation space. This direction includes everything about the model's default state -- not just persona, but also general language patterns, architecture-specific biases, and other factors unrelated to persona. The projection of a new activation onto this direction would measure "overall similarity to the default state," not specifically "persona alignment."

The subtraction removes the shared baseline (the part of activation space that all roles and the default have in common) and isolates the persona-specific direction. Without subtraction, you're measuring raw position rather than relative position on the persona spectrum.

**Why it was not chosen:** The axis needs to capture the *difference* between assistant and non-assistant behavior, not the absolute location of the assistant. Subtraction provides this.

---

## 7. The Motivation: Why This Design Works for Safety

### The safety problem this solves

Language models drift away from their default assistant persona during conversations. This drift correlates with harmful outputs:

- **Jailbreaks** (65-88% success rate) push the model deep into character via a single adversarial prompt
- **Therapy conversations** gradually erode professional boundaries over 15-30 turns
- **Philosophical discussions** invite the model to adopt non-assistant perspectives
- **Delusion reinforcement** occurs when the model validates false beliefs instead of maintaining its corrective function

The correlation between low Assistant Axis projection and harmful response rate is r = 0.39-0.52 (p < 0.001) across models. The axis captures something real about the model's safety posture.

### Why the mean-subtraction design serves safety

The mean of all roles represents the **average non-assistant state**. By subtracting it from the default, the axis captures the general direction of "being the assistant" rather than the direction away from any specific harmful persona. This is critical for safety because:

1. **New attack vectors are handled automatically.** The axis doesn't need to know about specific jailbreak characters. Any persona that moves the model away from assistant-like behavior will show up as decreased projection, whether it's a character the axis was trained on or a novel one.

2. **Gradual drift is detectable.** Because the axis measures a continuous spectrum (not a binary "safe/unsafe"), it can detect slow persona erosion in therapy or philosophy conversations where no single turn is clearly dangerous.

3. **Activation capping has a principled threshold.** The threshold `tau` for capping is set at the 25th percentile of the default assistant's projection distribution. This is meaningful because the axis was constructed relative to the mean of roles -- the threshold defines a floor in terms of "how far from the assistant center is too far."

### Why PCA validates this design

The fact that PCA independently discovers the same direction (PC1 ≈ axis) without being told about the default vector is strong evidence that the axis captures genuine structure rather than an artifact of the construction method. It means:

- The dominant organizing principle of the persona space really is "how assistant-like is the model?"
- This isn't imposed by the formula -- it's discovered by the data
- The formula `default - mean(roles)` happens to recover this natural structure because the default is genuinely at the extreme of PC1

The PCA variance explained numbers tell us how much of the persona space the axis accounts for:

| Model | PC1 Variance | Interpretation |
|-------|-------------|----------------|
| Gemma 2 27B | 48.8% | Nearly half of all persona variation is along the assistant axis |
| Qwen 3 32B | ~40% | Similar dominance |
| Llama 3.3 70B | ~40% | Consistent across architectures |

A 40-48% share from a single direction in a 4,096-dimensional space is extremely high. For comparison, in random data, each direction would capture ~0.024% of variance (1/4096). The axis captures 1,600-2,000x more variance than a random direction.

---

## 8. Putting It All Together: The Full Mathematical Picture

Here is the complete chain from raw activations to a projection scalar, with every mean subtraction and its purpose identified.

### Step 1: Collect role vectors

For each role `r` (of 275 roles), collect `N_r` response activations that scored 3:

```
v_r = (1/N_r) * sum(activation_i for i in score_3_responses_of_role_r)
```

Shape: `(n_layers, hidden_dim)` per role.

**Mean over responses:** cancels response-specific noise (individual questions, word choices). What survives is the stable per-role activation pattern.

### Step 2: Compute role mean

```
role_mean = (1/275) * sum(v_r for r in all_roles)
```

Shape: `(n_layers, hidden_dim)`.

**Mean over roles:** cancels role-specific features (pirate nautical dims, doctor medical dims). What survives is the shared "non-assistant" signal.

### Step 3: Compute axis

```
axis = default_mean - role_mean
```

Shape: `(n_layers, hidden_dim)`.

**Subtraction:** isolates the direction from "average non-assistant" to "assistant." Dimensions shared by both wash to zero. Only persona-discriminating dimensions survive.

### Step 4: PCA validation (separate computation)

```
centered_roles = [v_r - role_mean for v_r in all_role_vectors]
covariance = (1/274) * sum(c @ c.T for c in centered_roles)
eigenvalues, eigenvectors = eigen(covariance)
PC1 = eigenvectors[:, 0]  # direction of maximum variance
```

**Mean centering for PCA:** required so that PCA finds directions of *variation*, not direction of the centroid. After centering, the centered default vector equals the axis, and PC1 aligns with it.

**The validation:** cosine_similarity(axis, PC1) > 0.71 at middle layers. The axis formula recovers the dominant structure that PCA discovers blindly.

### Step 5: Projection

```
projection = activation[layer] @ (axis[layer] / ||axis[layer]||)
```

**Normalization:** ensures the projection measures pure directional alignment, independent of the axis vector's magnitude.

**Result:** a single scalar. High = assistant-like. Low = drifting toward role-playing. Below threshold = activation capping intervenes.

### The chain of means, summarized

| Mean | What it averages over | What it cancels | What survives |
|------|----------------------|-----------------|---------------|
| Mean over tokens in a response | 155 token activations | Token-specific features (individual words) | Conversation-level representation |
| Mean over responses for a role | 800 score-3 responses | Question-specific features | Stable role representation |
| Mean over all 275 roles | 275 role vectors | Role-specific features | Shared "non-assistant" direction |
| Subtraction (default - role_mean) | n/a | Shared baseline between assistant and roles | Pure persona-discriminating direction |
| PCA mean centering | Centers the role cloud at origin | Global location of the cloud | Directions of genuine variation |

Each mean removes one layer of noise. The final axis is what remains after all noise sources have been averaged away -- a clean direction that captures the single most important axis of variation in how language models represent their persona.
