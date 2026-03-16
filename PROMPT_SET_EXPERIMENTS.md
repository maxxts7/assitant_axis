# How Prompt Set Composition Shapes the Activation Subspace

## The Core Question

The current assistant axis emerges from contrasting 275 diverse roles against the default assistant. PCA on those 276 role vectors reveals a remarkably low-dimensional structure: ~4 components capture 70% of variance in a 5,120-dimensional space, and PC1 alone accounts for ~49%.

But PCA operates on whatever data you give it. The eigenvectors and eigenvalues are entirely determined by the covariance matrix, which is entirely determined by the input points. Change the points, change the PCA.

> Is the low-dimensional structure a property of the model's internal representation, or an artifact of the particular 275 roles chosen?

Three experiments probe this from different angles. Each holds the model and pipeline constant and varies only the prompt set.

---

## What PCA Is Actually Sensitive To

Before designing experiments, we need to be precise about what changes in the input data will and won't change PCA's output.

**PCA finds the eigenvectors of the covariance matrix C = (1/n) X^T X**, where X is the centered data matrix (rows = role vectors, columns = hidden dimensions).

Changing the prompt set changes X, which changes C, which changes the eigenvectors and eigenvalues. But not all changes are equal:

- **Adding points along an existing direction of high variance** increases that eigenvalue but doesn't change the eigenvector much. The axis rotates only if the new points pull the covariance structure in a genuinely new direction.

- **Adding points in a previously low-variance direction** can promote that direction from a minor PC to a major one. Directions that were "invisible" (low eigenvalue) in the original set can become dominant.

- **Removing points that were driving a dominant PC** can demote it. If PC1 was dominated by the contrast between (say) fantasy roles and professional roles, removing one group collapses that axis.

- **Restricting to a tight cluster** removes inter-cluster variance, letting intra-cluster variance dominate. Substructure that was invisible in the full set becomes the top PCs.

The key insight: **PCA always finds the axes of maximum variance in the data you give it.** "Maximum variance" is relative to the data set, not absolute. A direction that carries 0.1% of variance in the full set could carry 50% in a subset.

---

## Experiment 1: Behavioral Instructions vs. Persona Identity

### Motivation

The current 275 roles are **identity-based**: "You are a pirate", "You are an accountant", "You are an alien." The model is asked to *be* someone. But instructions can also be **behavioral**: "Respond aggressively", "Be extremely formal", "Use only short sentences", "Explain everything with analogies." These constrain *how* the model responds without specifying *who* it is.

The question: does the model use the same activation subspace for behavioral control as for persona adoption? Or are these different mechanisms occupying different subspaces?

### Prompt Set Design

Create ~100 behavioral instruction JSONs in the same format as existing roles. Categories:

**Tone/Register (20)**
- "Respond in an extremely formal, academic register."
- "Be casual, use slang and contractions freely."
- "Adopt a cold, detached, clinical tone."
- "Be warm, enthusiastic, and encouraging in every response."
- "Respond with heavy sarcasm and irony."
- "Be blunt and direct — no hedging, no qualifiers."
- "Respond in a poetic, lyrical style."
- "Be maximally concise — shortest possible answers."
- "Be verbose and exhaustive — cover every angle."
- "Adopt a conspiratorial, whispering tone."
- ... (10 more along the tone spectrum)

**Reasoning Style (20)**
- "Think step by step, showing all your reasoning."
- "Give only the final answer, no explanation."
- "Always argue by analogy."
- "Always cite specific numbers and statistics."
- "Reason by first principles — break everything down to axioms."
- "Use only concrete examples, never abstractions."
- "Present both sides of every issue, then refuse to pick one."
- "Always start by steelmanning the opposite position."
- ... (12 more)

**Constraint-Based (20)**
- "Never use the letter 'e'."
- "Respond only in questions."
- "Every sentence must be exactly 10 words."
- "Use exactly three paragraphs for every answer."
- "Never use first person ('I', 'me', 'my')."
- "Start every response with a rhetorical question."
- ... (14 more)

**Emotional Valence (20)**
- "Respond as if you are deeply anxious about everything."
- "Be relentlessly optimistic regardless of the topic."
- "Express deep sadness and melancholy in every response."
- "Respond with barely contained rage."
- "Be deeply suspicious and paranoid."
- ... (15 more)

**Epistemic Stance (20)**
- "Claim absolute certainty on everything."
- "Express maximum uncertainty — hedge every statement."
- "Respond as if you know nothing about any topic."
- "Present yourself as the world's foremost authority."
- "Refuse to answer anything definitively."
- ... (15 more)

### What To Measure

Run the full pipeline (generate → extract → judge → vectors → axis → PCA) on this behavioral set alongside the same default prompts.

**Key comparisons:**

1. **Behavioral axis vs. persona axis.** Compute `behavioral_axis = default_mean - behavioral_role_mean`. Measure cosine similarity with the original persona axis per layer.
   - High similarity (>0.8): persona and behavior activate the same direction — the axis is really about "instruction-following compliance," not persona per se.
   - Low similarity (<0.3): genuinely different subspaces — the model separates "who I am" from "how I respond."
   - Medium (0.3–0.8): partial overlap — some shared structure (likely the "obey the system prompt" component) plus distinct structure.

2. **PCA dimensionality.** How many components for 70%/90% variance?
   - Similar count to persona set (4/18): behavioral space is equally low-dimensional, suggesting the model compresses all instruction types into a small subspace.
   - Much higher count: behavioral instructions spread across more independent axes — behaviors are more "orthogonal" to each other than personas are.
   - Much lower count: behaviors are even more redundant than personas — perhaps they all reduce to a single "compliance intensity" axis.

3. **Subspace overlap.** Project the behavioral vectors onto the persona PCA basis (and vice versa). If the first 5 persona PCs explain >80% of behavioral variance, the subspaces are nearly the same. If they explain <30%, they're largely independent.

### Prediction

The "comply with system prompt" component is likely shared — both persona and behavioral instructions require the model to deviate from default behavior, so the axis direction (default vs. not-default) should be similar. But the *within-deviation* structure should differ. Personas cluster by character type; behaviors cluster by communication style. PC2 and beyond will probably diverge even if PC1 is shared.

---

## Experiment 2: Homogeneous Cluster (Fantasy Characters)

### Motivation

In the full 275-role set, the dominant source of variation is the broad contrast: professional roles vs. mythical creatures vs. emotional archetypes vs. abstract concepts. PC1 captures this coarse structure. But within a narrow category — say, fantasy characters — there may be finer-grained axes (hero vs. villain, ancient vs. futuristic, human vs. non-human) that are invisible in the full set because their variance is tiny compared to the inter-category variance.

Analogy: if you PCA the heights of all animals (mice to giraffes), PC1 is just "big vs. small." But if you PCA the heights of only humans, PC1 might capture sex differences, PC2 might capture age, etc. — structure that was invisible in the full-animal analysis.

### Prompt Set Design

Select ~30–40 existing roles that form a coherent fantasy/mythical cluster:

**From the current 276 roles:**
- angel, bard, caveman, demon, elder, eldritch, familiar, ghost, golem, guardian, healer, hermit, jester, leviathan, mystic, oracle, pirate, pilgrim, predator, prey, prophet, revenant, rogue, sage, shaman, shapeshifter, spirit, trickster, vampire, vigilante, wanderer, warrior, witch, wraith

That's ~34 roles. Add the default vector(s) as always.

### What To Measure

Run PCA on only these ~35 vectors (at the target layer).

1. **Does PC1 still align with the full-set axis?**
   Compute cosine similarity between the fantasy-only PC1 and the full-set axis.
   - If high: even within fantasy characters, the dominant variation is still "how assistant-like is this role" — the axis is universal.
   - If low: within this cluster, the dominant variation is something else — perhaps moral alignment (angel vs. demon) or embodiment (ghost vs. warrior).

2. **Do new fine-grained axes emerge?**
   Examine the top 3–5 PCs of the fantasy-only set. Check whether they separate interpretable subgroups:
   - Good vs. evil (angel, guardian, healer vs. demon, predator, vampire, wraith)
   - Corporeal vs. incorporeal (warrior, pirate, caveman vs. ghost, spirit, wraith)
   - Wise/mystical vs. physical/active (sage, oracle, mystic vs. warrior, rogue, pirate)

   If the PCs cleanly separate these subgroups, PCA has found structure that was invisible in the full set.

3. **How does dimensionality change?**
   - If it takes ~4 components for 70% variance (same as full set): the fantasy characters are themselves low-dimensional — they vary mostly along a few axes.
   - If it takes ~15 components: the fine-grained variation is spread across many dimensions — fantasy characters are more "different from each other" in activation space than the full set would suggest.
   - If it takes ~1–2 components: fantasy characters are basically the same in activation space, varying only along one axis (probably the generic "how committed to the role" axis).

### A Critical Subtlety

With only ~35 data points in 5,120 dimensions, PCA can find at most 34 nonzero eigenvalues. This is fine — it's the same min(n, p) constraint as the full set (276 << 5,120). But with fewer points, the eigenvalues are noisier. A variance ratio of 15% on one component might not be meaningfully different from 12% on the next. Consider running bootstrap resampling (randomly drop 5 roles, re-run PCA, check stability of eigenvectors) to assess which components are robust.

---

## Experiment 3: Maximally Divergent Roles

### Motivation

The current 275 roles were chosen to be diverse, but not explicitly to *maximize* the spread in activation space. Some roles may be near-duplicates in activation space (e.g., "accountant" and "auditor" probably produce similar activations). Others may be extreme outliers (e.g., "void," "wind," "coral_reef" — abstract/non-human concepts).

If we deliberately select roles that are maximally spread — as far apart as possible in activation space — do more independent PCA axes emerge? Or does the same low-dimensional structure persist regardless?

This is the most important experiment of the three, because its answer directly addresses the core question: **is the low dimensionality a property of the model or the data?**

### Prompt Set Design

This experiment requires an iterative approach, since "maximally divergent" is defined by the activation space itself.

**Phase 1: Compute pairwise distances.** Using existing role vectors (from a completed pipeline run), compute the cosine distance between all pairs of role vectors at the target layer. This gives a 275 × 275 distance matrix.

**Phase 2: Select maximally spread subset.** Use a greedy farthest-point algorithm:
1. Start with the role farthest from the centroid.
2. Add the role farthest from all currently selected roles.
3. Repeat until you have ~50 roles.

This produces a subset that maximally spans the existing activation space.

**Phase 3: Run PCA on the subset.** Compare with the full-set PCA.

**Phase 4 (the real test): Generate NEW maximally divergent roles.** Design ~50 roles specifically intended to be as different as possible from each other and from existing roles. Categories that push boundaries:

- **Sensory/Abstract**: "You are the concept of silence", "You are the color blue", "You are entropy"
- **Collective/Swarm**: "You are a hive mind of a thousand insects", "You are the collective unconscious of a civilization"
- **Contradictory**: "You are a pacifist who loves violence", "You are an omniscient being who knows nothing"
- **Scale extremes**: "You are a subatomic particle", "You are a galaxy cluster"
- **Temporal**: "You are a being that experiences time backwards", "You are a moment frozen in time"
- **Anti-roles**: "You are the opposite of helpful", "You are deliberately incoherent", "You refuse to be any character at all"

These push far beyond the current role set into territory where the model might need genuinely new activation directions to comply.

### What To Measure

1. **Does effective dimensionality increase?** Compare dims-for-70%-variance between:
   - Full set (275 roles): baseline (~4 dims)
   - Maximally spread subset of existing roles (50 roles)
   - New maximally divergent roles (50 roles)

   Three possible outcomes:
   - **Dimensionality is stable** (~4 dims for 70% across all sets): The low-dimensional structure is a hard property of the model. No matter how you vary the prompts, the model's activation space for "role-playing" is compressed into a small subspace. This would be the strongest finding — it means the model has a fixed, low-dimensional "persona manifold."
   - **Dimensionality increases with divergence** (4 → 8 → 15): The model does have more activation dimensions available, but the original prompt set didn't exercise them. The low dimensionality was partly an artifact of role similarity. The model's persona space is higher-dimensional than the original analysis suggested.
   - **Dimensionality decreases with the subset** (4 → 2): The 275-role set had some spurious dimensions driven by a few outlier roles. Removing the "noise" roles reveals an even simpler structure.

2. **PC1 alignment.** Does the axis direction persist? Compute cosine similarity between:
   - Full-set axis and divergent-set axis
   - Full-set PC1 and divergent-set PC1

   If the axis is stable: it represents a fundamental property of the model (default-vs-instructed).
   If it shifts: the axis direction depends on which roles you contrast against.

3. **New directions.** If dimensionality increases, what are the new PCs? Project the new role vectors onto the full-set PCA basis. Any variance that falls outside the full-set top-10 PCs represents genuinely new directions that the original set didn't probe.

### Implementation Note

Phase 4 (new roles) requires creating new role JSONs and running the full pipeline — not just re-running PCA on existing vectors. This is the most expensive experiment but also the most informative.

Phases 1–3 can be done immediately from existing vector files with no generation or extraction.

---

## Cross-Experiment Analysis

The three experiments are designed to triangulate the answer from different angles. Here's what combinations of outcomes would mean:

### Scenario A: The Model Has a Fixed Low-Dimensional Persona Manifold

| Experiment | Outcome |
|---|---|
| Behavioral | PC1 aligns with persona axis. Same dimensionality (~4 for 70%). |
| Fantasy subset | Dimensionality stays ~4. PC1 still tracks the axis. |
| Divergent | Dimensionality stays ~4 even with extreme roles. |

**Interpretation:** The model compresses all "deviate from default" instructions into a fixed, small subspace. The structure is a property of the model's learned representation, not the prompts. The model has a ~4-dimensional "compliance/persona space" that it uses regardless of what you ask it to be. This would be the most interesting finding for mechanistic interpretability — it implies a small, identifiable circuit governs persona adoption.

### Scenario B: The Subspace Is Prompt-Dependent

| Experiment | Outcome |
|---|---|
| Behavioral | Different PC1 direction. Different dimensionality. |
| Fantasy subset | Different PCs emerge. Dimensionality shifts. |
| Divergent | Higher dimensionality with more extreme roles. |

**Interpretation:** The activation subspace PCA discovers is shaped by the prompts. The model has a higher-dimensional space available for persona/behavioral representation, and PCA is a window whose view depends on where you point it. The "4-dimensional" finding from the original set reflects the diversity of *those particular 275 roles*, not a hard model constraint. Different tasks exercise different dimensions.

### Scenario C: Mixed — Stable Axis, Variable Subspace

| Experiment | Outcome |
|---|---|
| Behavioral | PC1 aligns (~axis). But PC2+ diverge. Higher dimensionality. |
| Fantasy subset | PC1 still tracks axis. New fine-grained PCs appear. |
| Divergent | PC1 stable. More dimensions beyond PC1. |

**Interpretation:** The model has a universal "default vs. instructed" axis (PC1) that is robust to prompt choice — this is a real property of the model. But the remaining structure (PC2 and beyond) depends on which roles are included. The axis is fundamental; the rest of the subspace is task-dependent. This is perhaps the most likely outcome, and it would mean: the axis is a genuine discovery about the model, but claims about "the persona space being 4-dimensional" should be qualified as "4-dimensional for this particular set of 275 roles."

---

## Implementation Plan

### What Can Be Done Immediately (No GPU, No Generation)

If you have existing vector `.pt` files from a completed pipeline run:

**Experiment 2 (Fantasy subset):**
```python
# Load existing vectors, select fantasy roles, re-run PCA
fantasy_roles = ["angel", "bard", "caveman", "demon", "elder", ...]
fantasy_vectors = [v for v in all_vectors if v["role"] in fantasy_roles]
stacked = torch.stack([v["vector"] for v in fantasy_vectors])
pca_result = compute_pca(stacked, layer=target_layer, scaler=MeanScaler())
```

**Experiment 3, Phase 1–3 (Maximally spread subset):**
```python
# Compute pairwise distances, select spread subset, re-run PCA
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(all_role_vectors_at_layer, metric="cosine"))
# Greedy farthest-point selection
selected = [np.argmax(np.linalg.norm(centered, axis=1))]  # start with farthest from centroid
for _ in range(49):
    min_dists = distances[selected].min(axis=0)
    next_idx = np.argmax(min_dists)
    selected.append(next_idx)
```

### What Requires Full Pipeline Runs (GPU Required)

- **Experiment 1:** New behavioral role JSONs → generate → extract → judge → vectors → PCA.
- **Experiment 3, Phase 4:** New divergent role JSONs → same full pipeline.

Each full pipeline run for ~50–100 roles on a single GPU takes roughly:
- Generation: ~2–4 hours (5 prompts × 240 questions × 50 roles = 60,000 generations)
- Extraction: ~1–2 hours
- Judging: ~1–2 hours (API-limited)
- Vectors + Axis + PCA: minutes

### Comparative Metrics to Report

For each experiment, compute and compare:

| Metric | Full Set (baseline) | Exp 1 (Behavioral) | Exp 2 (Fantasy) | Exp 3 (Divergent) |
|---|---|---|---|---|
| N roles | 275 | ~100 | ~34 | ~50 |
| PC1 variance % | ~49% | ? | ? | ? |
| Dims for 70% | ~4 | ? | ? | ? |
| Dims for 90% | ~18 | ? | ? | ? |
| cos(PC1, full-set axis) | ~0.71 | ? | ? | ? |
| cos(new axis, full-set axis) | 1.0 | ? | ? | ? |
| Subspace overlap (top-5 PCs) | 100% | ? | ? | ? |
