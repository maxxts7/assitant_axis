# Task 6: Axis Operations (axis.py)

## Overview

The `axis.py` module provides the core mathematical operations for computing, projecting onto, saving/loading, and analysing the **assistant axis** -- a direction in activation space that captures the difference between role-playing behaviour and default assistant behaviour in language models.

**Core formula:**

```
axis = mean(default_activations) - mean(pos_3_activations)
```

The axis points FROM role-playing TOWARD default assistant behaviour. A higher projection value indicates more "assistant-like" activations; a lower value indicates more "role-playing" activations.

**Imports used:** `torch`, `numpy`, `typing.Union`, `typing.Optional`

---

## Sub-Tasks

---

### Sub-Task 6.1: `compute_axis` -- Compute the Assistant Axis

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `role_activations` | `torch.Tensor` | `(n_role, n_layers, hidden_dim)` | -- | Activations from role-playing responses (score=3) |
| `default_activations` | `torch.Tensor` | `(n_default, n_layers, hidden_dim)` | -- | Activations from default/neutral system prompts |

#### Processing

1. **Compute the mean of role activations** across the sample dimension (dim=0), collapsing `n_role` samples into a single centroid.
2. **Compute the mean of default activations** across the sample dimension (dim=0), collapsing `n_default` samples into a single centroid.
3. **Subtract** the role mean from the default mean to obtain the axis direction.

```python
# Compute means
role_mean = role_activations.mean(dim=0)       # (n_layers, hidden_dim)
default_mean = default_activations.mean(dim=0)  # (n_layers, hidden_dim)

# axis points from role toward default
axis = default_mean - role_mean
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | The assistant axis vector at every layer |

---

### Sub-Task 6.2: `project` -- Project Activations onto the Axis (Single Sample)

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `activations` | `torch.Tensor` | `(n_layers, hidden_dim)` or `(hidden_dim,)` | -- | Activations from a single sample |
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | The assistant axis |
| `layer` | `int` | scalar | -- | Layer index to use for projection |
| `normalize` | `bool` | -- | `True` | Whether to normalize the axis to unit length before projection |

#### Processing

1. **Select the layer slice of the activations.** If the input is 2-D (`n_layers, hidden_dim`), index into the specified layer. If 1-D (`hidden_dim,`), use the tensor as-is. Cast to float32.
2. **Select the layer slice of the axis** and cast to float32.
3. **Optionally normalize** the axis vector to unit length (with epsilon 1e-8 for numerical safety).
4. **Compute the dot product** between the activation vector and the (optionally normalized) axis vector.

```python
# Get layer activations
if activations.ndim == 2:
    act = activations[layer].float()
else:
    act = activations.float()

ax = axis[layer].float()

if normalize:
    ax = ax / (ax.norm() + 1e-8)

return float(act @ ax)
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| projection | `float` | scalar | Dot product of the activation vector with the (normalized) axis. Higher = more assistant-like; lower = more role-playing. |

---

### Sub-Task 6.3: `project_batch` -- Project a Batch of Activations onto the Axis

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `activations` | `torch.Tensor` | `(batch, n_layers, hidden_dim)` | -- | Activations from multiple samples |
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | The assistant axis |
| `layer` | `int` | scalar | -- | Layer index to use |
| `normalize` | `bool` | -- | `True` | Whether to normalize the axis |

#### Processing

1. **Slice all samples at the target layer** using `activations[:, layer, :]`, yielding a `(batch, hidden_dim)` matrix. Cast to float32.
2. **Extract the axis vector** at the target layer and cast to float32.
3. **Optionally normalize** the axis vector (epsilon 1e-8).
4. **Matrix-vector multiply** the batch of activations with the axis vector, producing one scalar projection per sample.

```python
# Get layer activations for all samples
acts = activations[:, layer, :].float()  # (batch, hidden_dim)
ax = axis[layer].float()  # (hidden_dim,)

if normalize:
    ax = ax / (ax.norm() + 1e-8)

return acts @ ax  # (batch,)
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| projections | `torch.Tensor` | `(batch,)` | One projection scalar per sample in the batch |

---

### Sub-Task 6.4: `cosine_similarity_per_layer` -- Per-Layer Cosine Similarity

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `v1` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | First vector (e.g., an axis or activation centroid) |
| `v2` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | Second vector |

#### Processing

1. **Cast both inputs to float32.**
2. **L2-normalize each vector independently along the hidden dimension** (dim=1), with epsilon 1e-8 to prevent division by zero.
3. **Element-wise multiply** the two normalized tensors and **sum along dim=1** to obtain the cosine similarity at each layer.
4. **Convert to NumPy.**

```python
v1 = v1.float()
v2 = v2.float()

# Normalize both vectors
v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)

# Compute dot product per layer
similarities = (v1_norm * v2_norm).sum(dim=1)

return similarities.numpy()
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| similarities | `np.ndarray` | `(n_layers,)` | Cosine similarity between `v1` and `v2` at each layer, in range [-1, 1] |

---

### Sub-Task 6.5: `axis_norm_per_layer` -- Per-Layer L2 Norm of the Axis

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | The assistant axis |

#### Processing

1. **Cast to float32.**
2. **Compute the L2 norm along the hidden dimension** (dim=1), yielding one norm value per layer.
3. **Convert to NumPy.**

```python
return axis.float().norm(dim=1).numpy()
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| norms | `np.ndarray` | `(n_layers,)` | L2 norm of the axis vector at each layer |

---

### Sub-Task 6.6: `save_axis` -- Persist an Axis to Disk

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | -- | The axis tensor to save |
| `path` | `str` | -- | -- | Filesystem path for the `.pt` file |
| `metadata` | `Optional[dict]` | -- | `None` | Optional metadata dict (e.g., model name, layer count, date) |

#### Processing

1. **Construct a save dict** with the axis stored under the key `"axis"`.
2. **If metadata is provided**, add it under the key `"metadata"`.
3. **Serialize with `torch.save`.**

```python
save_dict = {"axis": axis}
if metadata:
    save_dict["metadata"] = metadata
torch.save(save_dict, path)
```

#### Output

| Return | Type | Description |
|---|---|---|
| (none) | `None` | The function writes a `.pt` file to `path` and returns nothing |

**Side effect:** Creates or overwrites a `.pt` file at `path` containing a dict with keys `"axis"` (and optionally `"metadata"`).

---

### Sub-Task 6.7: `load_axis` -- Load an Axis from Disk

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `path` | `str` | -- | -- | Path to the `.pt` file to load |

#### Processing

1. **Load the file** with `torch.load`, forcing CPU placement (`map_location="cpu"`) and allowing arbitrary objects (`weights_only=False`).
2. **Handle both storage formats:**
   - If the loaded object is a dict containing an `"axis"` key, return that value.
   - If the loaded object is a dict without an `"axis"` key, raise `ValueError`.
   - If the loaded object is a raw tensor (legacy format), return it directly.

```python
data = torch.load(path, map_location="cpu", weights_only=False)

# Handle both formats: dict with 'axis' key or raw tensor
if isinstance(data, dict):
    if "axis" in data:
        return data["axis"]
    else:
        raise ValueError("Expected 'axis' key in saved dict")
else:
    return data
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| axis | `torch.Tensor` | `(n_layers, hidden_dim)` | The loaded axis tensor |

**Raises:** `ValueError` if the file contains a dict without an `"axis"` key.

---

### Sub-Task 6.8: `aggregate_role_vectors` -- Aggregate Per-Role Vectors into a Single Mean

#### Input

| Parameter | Type | Shape | Default | Description |
|---|---|---|---|---|
| `vectors` | `dict` | `{str: Tensor(n_layers, hidden_dim)}` | -- | Dict mapping role names to their corresponding vectors |
| `exclude_roles` | `Optional[list]` | -- | `None` | List of role names to exclude from the aggregation (e.g., `["default"]`) |

#### Processing

1. **Default `exclude_roles` to an empty list** if `None`.
2. **Filter out excluded roles** by iterating over the dict and keeping only vectors whose role name is not in the exclusion list.
3. **Guard against empty result** -- raise `ValueError` if no vectors remain after filtering.
4. **Stack** the remaining vectors into a single tensor of shape `(n_roles, n_layers, hidden_dim)`.
5. **Compute the mean** across the role dimension (dim=0).

```python
exclude_roles = exclude_roles or []

filtered = [v for name, v in vectors.items() if name not in exclude_roles]

if not filtered:
    raise ValueError("No vectors remaining after exclusions")

stacked = torch.stack(filtered)  # (n_roles, n_layers, hidden_dim)
return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| mean_vector | `torch.Tensor` | `(n_layers, hidden_dim)` | The mean of all non-excluded role vectors |

**Raises:** `ValueError` if all roles are excluded or the input dict is empty.
