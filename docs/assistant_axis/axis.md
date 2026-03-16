# `assistant_axis/axis.py`

## Overview

This file provides the core computation and projection utilities for the **assistant axis** -- a direction in a language model's activation space that captures the difference between role-playing behavior and default assistant behavior. The module includes functions to compute the axis from activation data, project activations onto the axis, measure cosine similarity between vectors across layers, and save/load the axis to disk.

The central idea: given activations collected from a model when it is role-playing versus when it is responding with a default/neutral system prompt, the axis is defined as the vector pointing **from** the role-playing centroid **toward** the default-assistant centroid. Projecting new activations onto this axis yields a scalar that indicates how "assistant-like" or "role-playing-like" a given response is.

---

## Line-by-Line Explanation

### Lines 1--21: Module Docstring

```python
"""
Assistant axis computation and projection utilities.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Formula:
    axis = mean(default_activations) - mean(pos_3_activations)

Where:
    - default_activations: activations from neutral system prompts
    - pos_3_activations: activations from responses fully playing a role (score=3)

The axis points FROM role-playing TOWARD default assistant behavior.

Example:
    from assistant_axis import load_axis, project

    axis = load_axis("outputs/gemma-2-27b/axis.pt")
    projection = project(activation, axis, layer=22)
"""
```

The module-level docstring explains the purpose of the file, the formula used to compute the axis, the meaning of each term, and gives a quick usage example. `pos_3_activations` refers to activations from responses that were scored 3 (fully committed to a role). The axis direction convention means positive projections are more assistant-like, and negative projections are more role-playing-like.

---

### Line 23: Import `torch`

```python
import torch
```

Imports PyTorch, the tensor library used for all numerical operations in this file -- mean computation, dot products, norms, and serialization.

---

### Line 24: Import `numpy`

```python
import numpy as np
```

Imports NumPy. It is used as the return type for a few functions (`cosine_similarity_per_layer` and `axis_norm_per_layer`) that convert their PyTorch tensor results to NumPy arrays for downstream convenience (e.g., plotting with matplotlib).

---

### Line 25: Type-hint imports

```python
from typing import Union, Optional
```

Imports `Union` and `Optional` from the `typing` module for type annotations. `Optional` is used in several function signatures; `Union` is imported but not used directly in the current code (it may be kept for forward-compatibility or was used in an earlier revision).

---

### Lines 28--55: `compute_axis`

```python
def compute_axis(
    role_activations: torch.Tensor,
    default_activations: torch.Tensor,
) -> torch.Tensor:
```

Defines the function signature. It takes two tensors:

- `role_activations` -- shape `(n_role, n_layers, hidden_dim)`: a batch of activation vectors collected while the model was role-playing (score = 3).
- `default_activations` -- shape `(n_default, n_layers, hidden_dim)`: a batch of activation vectors collected under neutral/default system prompts.

It returns a tensor of shape `(n_layers, hidden_dim)` representing the axis at every layer.

```python
    role_mean = role_activations.mean(dim=0)       # (n_layers, hidden_dim)
    default_mean = default_activations.mean(dim=0)  # (n_layers, hidden_dim)
```

Averages each set of activations across the first (batch) dimension. After this, `role_mean` is the centroid of all role-playing activations and `default_mean` is the centroid of all default activations, both retaining the per-layer structure.

```python
    axis = default_mean - role_mean
```

Subtracts the role-playing centroid from the default centroid. This produces a vector that points from the role-playing cluster toward the default-assistant cluster in activation space. Any activation projected onto this direction will yield a positive value if it is closer to the default side and a negative value if it is closer to the role-playing side.

```python
    return axis
```

Returns the axis tensor of shape `(n_layers, hidden_dim)`.

---

### Lines 58--88: `project`

```python
def project(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
```

Projects a **single** activation tensor onto the axis at a given layer. Parameters:

- `activations` -- either `(n_layers, hidden_dim)` (multi-layer) or `(hidden_dim,)` (single layer already extracted).
- `axis` -- shape `(n_layers, hidden_dim)`.
- `layer` -- integer index selecting which layer to use.
- `normalize` -- if `True` (default), the axis is unit-normalized before projection so the result is a cosine-like scalar rather than a raw dot product.

Returns a Python `float`.

```python
    if activations.ndim == 2:
        act = activations[layer].float()
    else:
        act = activations.float()
```

If the activations tensor has two dimensions (layers x hidden_dim), the function indexes into the requested layer. If it is already one-dimensional (just hidden_dim), it uses it directly. `.float()` casts to 32-bit float to avoid precision issues with half-precision or bfloat16 tensors.

```python
    ax = axis[layer].float()
```

Extracts the axis vector for the requested layer and casts to float32.

```python
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
```

If normalization is enabled, divides the axis vector by its L2 norm, turning it into a unit vector. The small epsilon `1e-8` prevents division by zero in the degenerate case where the axis has zero magnitude.

```python
    return float(act @ ax)
```

Computes the dot product between the activation vector and the (possibly normalized) axis vector using the `@` operator (matrix/vector multiplication). The result is converted from a zero-dimensional tensor to a plain Python `float` and returned. A higher value means the activation is more aligned with default-assistant behavior; a lower value means more aligned with role-playing.

---

### Lines 91--116: `project_batch`

```python
def project_batch(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
```

Batch version of `project`. Instead of a single activation, it takes a batch of activations shaped `(batch, n_layers, hidden_dim)` and returns a tensor of projection values shaped `(batch,)`.

```python
    acts = activations[:, layer, :].float()  # (batch, hidden_dim)
```

Slices out the specified layer from every sample in the batch using `[:, layer, :]`, yielding a 2D tensor where each row is one sample's activation at that layer. Cast to float32.

```python
    ax = axis[layer].float()  # (hidden_dim,)
```

Extracts the axis vector for the requested layer, cast to float32.

```python
    if normalize:
        ax = ax / (ax.norm() + 1e-8)
```

Same unit-normalization as in `project`, with the same epsilon guard.

```python
    return acts @ ax  # (batch,)
```

Matrix-vector multiplication: the `(batch, hidden_dim)` matrix multiplied by the `(hidden_dim,)` vector produces a `(batch,)` vector of scalar projections, one per sample.

---

### Lines 119--143: `cosine_similarity_per_layer`

```python
def cosine_similarity_per_layer(
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> np.ndarray:
```

Computes the cosine similarity between two multi-layer vectors at each layer independently. Both inputs have shape `(n_layers, hidden_dim)`. Returns a NumPy array of length `n_layers`.

```python
    v1 = v1.float()
    v2 = v2.float()
```

Casts both tensors to float32 for numerical stability.

```python
    v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
    v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)
```

Normalizes each vector per layer. `v1.norm(dim=1, keepdim=True)` computes the L2 norm along dimension 1 (the `hidden_dim` axis), producing a `(n_layers, 1)` tensor. Dividing by this (with epsilon guard) makes each row a unit vector. The same is done for `v2`.

```python
    similarities = (v1_norm * v2_norm).sum(dim=1)
```

Element-wise multiplication of the two unit-vector tensors followed by a sum along the hidden dimension. This is equivalent to computing the dot product of two unit vectors per layer, which is the cosine similarity. The result has shape `(n_layers,)`.

```python
    return similarities.numpy()
```

Converts the PyTorch tensor to a NumPy array and returns it.

---

### Lines 146--156: `axis_norm_per_layer`

```python
def axis_norm_per_layer(axis: torch.Tensor) -> np.ndarray:
```

Takes an axis tensor of shape `(n_layers, hidden_dim)` and returns the L2 norm of the axis at each layer as a NumPy array.

```python
    return axis.float().norm(dim=1).numpy()
```

Casts to float32, computes the L2 norm along dimension 1 (the hidden dimension) to get one scalar per layer, and converts to NumPy. This is useful for understanding which layers have the strongest axis signal -- a larger norm means the default and role-playing centroids are further apart at that layer.

---

### Lines 159--175: `save_axis`

```python
def save_axis(
    axis: torch.Tensor,
    path: str,
    metadata: Optional[dict] = None,
):
```

Saves an axis tensor to disk as a `.pt` file. Optionally includes a metadata dictionary (e.g., information about the model, layer count, or training details).

```python
    save_dict = {"axis": axis}
```

Wraps the axis tensor in a dictionary under the key `"axis"`. Using a dictionary rather than saving the raw tensor allows attaching metadata and makes the file format self-describing.

```python
    if metadata:
        save_dict["metadata"] = metadata
```

If a metadata dictionary was provided (and is truthy / non-empty), it is added to the save dictionary under the key `"metadata"`.

```python
    torch.save(save_dict, path)
```

Serializes the dictionary to disk using PyTorch's `torch.save`, which uses Python's `pickle` protocol under the hood. The resulting file can be loaded back with `torch.load`.

---

### Lines 178--197: `load_axis`

```python
def load_axis(path: str) -> torch.Tensor:
```

Loads a previously saved axis from a `.pt` file and returns the axis tensor.

```python
    data = torch.load(path, map_location="cpu", weights_only=False)
```

Loads the file from disk. `map_location="cpu"` ensures the tensor is loaded onto the CPU regardless of what device it was saved from (e.g., if it was saved from a GPU). `weights_only=False` allows loading the full pickled object (necessary because the file may contain a dictionary, not just raw tensor weights).

```python
    if isinstance(data, dict):
        if "axis" in data:
            return data["axis"]
        else:
            raise ValueError("Expected 'axis' key in saved dict")
    else:
        return data
```

Handles two possible file formats:

1. **Dictionary format** (produced by `save_axis`): looks for the `"axis"` key and returns its value. If the dictionary exists but lacks the `"axis"` key, raises a `ValueError` with a descriptive message.
2. **Raw tensor format** (legacy/simple files): if the loaded object is not a dictionary, it is assumed to be the axis tensor itself and is returned directly.

---

### Lines 200--222: `aggregate_role_vectors`

```python
def aggregate_role_vectors(
    vectors: dict,
    exclude_roles: Optional[list] = None,
) -> torch.Tensor:
```

Aggregates multiple per-role vectors (e.g., one mean-activation vector per character role) into a single representative vector by averaging. Parameters:

- `vectors` -- a dictionary mapping role names (strings) to tensors of shape `(n_layers, hidden_dim)`.
- `exclude_roles` -- an optional list of role names to exclude from the aggregation (for example, `["default"]` to leave out the default/neutral role).

```python
    exclude_roles = exclude_roles or []
```

If `exclude_roles` was not provided (i.e., is `None`), defaults to an empty list. This avoids `None`-related errors in the membership check below.

```python
    filtered = [v for name, v in vectors.items() if name not in exclude_roles]
```

Iterates over the dictionary items and keeps only the vectors whose role names are **not** in the exclusion list. The result is a plain Python list of tensors.

```python
    if not filtered:
        raise ValueError("No vectors remaining after exclusions")
```

Safety check: if every role was excluded (or the input dictionary was empty), raises a `ValueError` rather than proceeding with an empty list, which would cause a downstream error in `torch.stack`.

```python
    stacked = torch.stack(filtered)  # (n_roles, n_layers, hidden_dim)
```

Stacks the list of 2D tensors into a single 3D tensor along a new first dimension, yielding shape `(n_roles, n_layers, hidden_dim)`.

```python
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

Averages across the role dimension (dim 0), collapsing the `n_roles` axis. The result is a single `(n_layers, hidden_dim)` tensor representing the mean activation across all included roles. This can then be used as the `role_activations` centroid (after appropriate reshaping) or compared against the default centroid.
