# Task 5: Compute Assistant Axis

## Overview

This task computes the **assistant axis** -- a single direction in activation space that separates role-playing behavior from default assistant behavior. It takes the per-role mean vectors produced by Task 4 and computes:

```
axis = mean(default_vectors) - mean(role_vectors)
```

The resulting axis points **FROM** role-playing **TOWARD** default assistant behavior. A higher projection onto this axis means the model is behaving more like a default assistant; a lower (or more negative) projection means the model is behaving more like it is playing a character role.

The pipeline script lives at `pipeline/5_axis.py`. The reusable library functions live at `assistant_axis/axis.py`.

---

## Sub-Tasks

### Sub-Task 5.1: Load and Classify Per-Role Vectors

#### Input

| Parameter | Type | Description | Source |
|---|---|---|---|
| `--vectors_dir` | `str` (CLI arg, required) | Path to the directory containing `.pt` vector files produced by Task 4 | e.g. `outputs/gemma-2-27b/vectors` |

Each `.pt` file in the directory is a dictionary saved by `torch.save` with the following keys:

| Key | Type | Shape / Values | Description |
|---|---|---|---|
| `"vector"` | `torch.Tensor` | `(n_layers, hidden_dim)` | The mean activation vector for this role |
| `"type"` | `str` | `"mean"` or `"pos_3"` | `"mean"` for default roles (unfiltered), `"pos_3"` for roles filtered to score=3 samples |
| `"role"` | `str` | e.g. `"default"`, `"pirate"`, `"shakespeare"` | The role name, matching the file stem |

#### Processing

The script discovers all `.pt` files in the vectors directory, loads each one, and classifies it as either a **default** vector or a **role** vector based on two criteria: (1) whether the role name contains the substring `"default"`, or (2) whether the vector type is `"mean"`.

```python
def load_vector(vector_file: Path) -> dict:
    """Load vector data from .pt file."""
    return torch.load(vector_file, map_location="cpu", weights_only=False)
```

```python
# Load all vectors
vector_files = sorted(vectors_dir.glob("*.pt"))
print(f"Found {len(vector_files)} vector files")

# Separate default and role vectors
default_vectors = []
role_vectors = []

for vec_file in tqdm(vector_files, desc="Loading vectors"):
    data = load_vector(vec_file)
    vector = data["vector"]
    vector_type = data.get("type", "unknown")
    role = data.get("role", vec_file.stem)

    if "default" in role or vector_type == "mean":
        default_vectors.append(vector)
        print(f"  {role}: default/mean vector")
    else:
        role_vectors.append(vector)
```

Classification rules:

- If the role name contains `"default"` (e.g. `"default"`, `"default_assistant"`) **OR** the vector type is `"mean"` --> classified as a **default vector**.
- Otherwise --> classified as a **role vector** (these are the `"pos_3"` filtered vectors from characters/personas).

If the `"type"` key is missing from the saved dictionary, it defaults to `"unknown"`, which means classification falls back to the role-name check alone. If the `"role"` key is missing, it falls back to the file stem (`vec_file.stem`).

#### Output

| Variable | Type | Description |
|---|---|---|
| `default_vectors` | `list[torch.Tensor]` | List of tensors, each of shape `(n_layers, hidden_dim)` |
| `role_vectors` | `list[torch.Tensor]` | List of tensors, each of shape `(n_layers, hidden_dim)` |

The script validates that both lists are non-empty and exits with an error if either is missing:

```python
if not default_vectors:
    print("Error: No default vectors found")
    sys.exit(1)

if not role_vectors:
    print("Error: No role vectors found")
    sys.exit(1)
```

---

### Sub-Task 5.2: Compute the Axis (Default Mean Minus Role Mean)

#### Input

| Variable | Type | Shape | Description |
|---|---|---|---|
| `default_vectors` | `list[torch.Tensor]` | Each element: `(n_layers, hidden_dim)` | Default/neutral vectors from Sub-Task 5.1 |
| `role_vectors` | `list[torch.Tensor]` | Each element: `(n_layers, hidden_dim)` | Role-playing vectors from Sub-Task 5.1 |

#### Processing

**Step 1: Stack vectors into batched tensors.**

Each list of vectors is stacked along a new batch dimension (dim=0):

```python
default_stacked = torch.stack(default_vectors)  # (n_default, n_layers, hidden_dim)
role_stacked = torch.stack(role_vectors)  # (n_roles, n_layers, hidden_dim)
```

**Step 2: Compute the mean across the batch dimension for each group.**

```python
default_mean = default_stacked.mean(dim=0)  # (n_layers, hidden_dim)
role_mean = role_stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

**Step 3: Subtract to obtain the axis.**

```python
axis = default_mean - role_mean
```

This is the element-wise difference between the two mean vectors. The axis has shape `(n_layers, hidden_dim)`, providing a separate direction vector at every layer of the model.

The equivalent library function in `assistant_axis/axis.py` encapsulates this logic:

```python
def compute_axis(
    role_activations: torch.Tensor,
    default_activations: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the assistant axis from role and default activations.

    Formula: axis = mean(default) - mean(role)

    The axis points FROM role-playing TOWARD assistant behavior.

    Args:
        role_activations: Tensor of shape (n_role, n_layers, hidden_dim)
                         Activations from role-playing responses (score=3)
        default_activations: Tensor of shape (n_default, n_layers, hidden_dim)
                            Activations from default/neutral system prompts

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
    # Compute means
    role_mean = role_activations.mean(dim=0)       # (n_layers, hidden_dim)
    default_mean = default_activations.mean(dim=0)  # (n_layers, hidden_dim)

    # axis points from role toward default
    axis = default_mean - role_mean

    return axis
```

The library also provides a helper for aggregating vectors from a dictionary (useful when working with named roles outside the pipeline):

```python
def aggregate_role_vectors(
    vectors: dict,
    exclude_roles: Optional[list] = None,
) -> torch.Tensor:
    """
    Aggregate per-role vectors into a single mean vector.

    Args:
        vectors: Dict mapping role names to vectors (n_layers, hidden_dim)
        exclude_roles: List of role names to exclude (e.g., ["default"])

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
    exclude_roles = exclude_roles or []

    filtered = [v for name, v in vectors.items() if name not in exclude_roles]

    if not filtered:
        raise ValueError("No vectors remaining after exclusions")

    stacked = torch.stack(filtered)  # (n_roles, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

#### Output

| Variable | Type | Shape | Description |
|---|---|---|---|
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | The assistant axis -- one direction vector per layer |

For reference, with Gemma-2-27B: `n_layers` = number of transformer layers in the model, `hidden_dim` = the residual stream dimension (e.g. 4608 for Gemma-2-27B, though this depends on the model).

---

### Sub-Task 5.3: Compute and Report Diagnostic Statistics

#### Input

| Variable | Type | Shape | Description |
|---|---|---|---|
| `axis` | `torch.Tensor` | `(n_layers, hidden_dim)` | The axis computed in Sub-Task 5.2 |

#### Processing

The script computes the L2 norm of the axis at each layer and prints a diagnostic summary:

```python
print(f"\nAxis shape: {axis.shape}")
print(f"Axis norms per layer (first 10):")
norms = axis.norm(dim=1)
for i, norm in enumerate(norms[:10]):
    print(f"  Layer {i}: {norm:.4f}")
print(f"  ...")
print(f"  Mean norm: {norms.mean():.4f}")
print(f"  Max norm: {norms.max():.4f} (layer {norms.argmax().item()})")
```

`axis.norm(dim=1)` computes the L2 norm along the `hidden_dim` dimension, producing one scalar per layer. This tells you how strongly the default and role-playing centroids differ at each layer. Layers with larger norms have a more pronounced separation and are typically more useful for downstream projection/steering.

The library provides a dedicated utility for this:

```python
def axis_norm_per_layer(axis: torch.Tensor) -> np.ndarray:
    """
    Compute the L2 norm of the axis at each layer.

    Args:
        axis: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of norms, one per layer
    """
    return axis.float().norm(dim=1).numpy()
```

#### Output

Printed to stdout (not saved to a file). Example output:

```
Axis shape: torch.Size([46, 4608])
Axis norms per layer (first 10):
  Layer 0: 0.2134
  Layer 1: 0.3891
  Layer 2: 0.4523
  ...
  Mean norm: 1.2345
  Max norm: 3.4567 (layer 22)
```

The layer with the maximum norm is typically the most informative layer for downstream projection tasks.

---

### Sub-Task 5.4: Save the Axis to Disk

#### Input

| Parameter | Type | Description |
|---|---|---|
| `--output` | `str` (CLI arg, required) | Output file path for the axis tensor |
| `axis` | `torch.Tensor` | Shape `(n_layers, hidden_dim)` |

#### Processing

The pipeline script saves the raw tensor directly:

```python
torch.save(axis, output_path)
print(f"\nSaved axis to {output_path}")
```

The output directory is created if it does not exist:

```python
output_path.parent.mkdir(parents=True, exist_ok=True)
```

Note: the pipeline script saves the **raw tensor** (not wrapped in a dict), whereas the library's `save_axis` function saves a **dict** with an `"axis"` key and optional metadata:

```python
def save_axis(
    axis: torch.Tensor,
    path: str,
    metadata: Optional[dict] = None,
):
    """
    Save axis to a .pt file.

    Args:
        axis: Axis tensor of shape (n_layers, hidden_dim)
        path: Path to save to
        metadata: Optional metadata dict to include
    """
    save_dict = {"axis": axis}
    if metadata:
        save_dict["metadata"] = metadata
    torch.save(save_dict, path)
```

The library's `load_axis` function handles **both** formats transparently:

```python
def load_axis(path: str) -> torch.Tensor:
    """
    Load axis from a .pt file.

    Args:
        path: Path to load from

    Returns:
        Axis tensor of shape (n_layers, hidden_dim)
    """
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

| Artifact | Format | Shape | Path |
|---|---|---|---|
| Axis file | PyTorch `.pt` (raw tensor via pipeline, or dict via library) | `(n_layers, hidden_dim)` | e.g. `outputs/gemma-2-27b/axis.pt` |

---

## Downstream Usage: Projection Utilities

Once the axis is computed and saved, the library in `assistant_axis/axis.py` provides projection functions for using it.

### Single-Sample Projection

```python
def project(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> float:
    """
    Project activations onto the axis at a specific layer.

    Args:
        activations: Tensor of shape (n_layers, hidden_dim) or (hidden_dim,)
        axis: Tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use for projection
        normalize: Whether to normalize the axis before projection

    Returns:
        Projection value (scalar). Higher values indicate more "assistant-like",
        lower values indicate more "role-playing".
    """
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

When `normalize=True` (the default), the axis vector at the chosen layer is converted to a unit vector before the dot product. This means the returned scalar is the **signed projection length** (component of `activations` along the axis direction). When `normalize=False`, the returned value is the raw dot product, which scales with the axis norm.

### Batch Projection

```python
def project_batch(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Project a batch of activations onto the axis.

    Args:
        activations: Tensor of shape (batch, n_layers, hidden_dim)
        axis: Tensor of shape (n_layers, hidden_dim)
        layer: Layer index to use
        normalize: Whether to normalize the axis

    Returns:
        Projection values of shape (batch,)
    """
    # Get layer activations for all samples
    acts = activations[:, layer, :].float()  # (batch, hidden_dim)
    ax = axis[layer].float()  # (hidden_dim,)

    if normalize:
        ax = ax / (ax.norm() + 1e-8)

    return acts @ ax  # (batch,)
```

### Cosine Similarity Between Two Vectors (Per Layer)

```python
def cosine_similarity_per_layer(
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> np.ndarray:
    """
    Compute cosine similarity between two vectors at each layer.

    Args:
        v1: Tensor of shape (n_layers, hidden_dim)
        v2: Tensor of shape (n_layers, hidden_dim)

    Returns:
        Array of cosine similarities, one per layer
    """
    v1 = v1.float()
    v2 = v2.float()

    # Normalize both vectors
    v1_norm = v1 / (v1.norm(dim=1, keepdim=True) + 1e-8)
    v2_norm = v2 / (v2.norm(dim=1, keepdim=True) + 1e-8)

    # Compute dot product per layer
    similarities = (v1_norm * v2_norm).sum(dim=1)

    return similarities.numpy()
```

---

## CLI Reference

```
uv run pipeline/5_axis.py \
    --vectors_dir outputs/gemma-2-27b/vectors \
    --output outputs/gemma-2-27b/axis.pt
```

| Argument | Required | Type | Default | Description |
|---|---|---|---|---|
| `--vectors_dir` | Yes | `str` | -- | Directory containing per-role `.pt` vector files from Task 4 |
| `--output` | Yes | `str` | -- | Path for the output axis `.pt` file |

---

## Data Flow Summary

```
inputs/                                      outputs/
  vectors/                                     axis.pt
    default.pt  ──┐                            (n_layers, hidden_dim)
    pirate.pt   ──┤                                 ▲
    poet.pt     ──┤  load + classify                │
    ...         ──┘       │                         │
                          ▼                         │
              ┌───────────────────────┐             │
              │ default_vectors list  │──► mean ─►  │
              │ role_vectors list     │──► mean ─►  subtract ─► axis
              └───────────────────────┘
```
