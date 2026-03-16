# `pipeline/5_axis.py`

## Overview

This is the **final step** of the pipeline. It takes the per-role mean vectors computed in step 4 and combines them into a single **assistant axis** -- a direction in activation space that separates default assistant behavior from role-playing behavior.

The formula is:

```
axis = mean(default_vectors) - mean(role_vectors)
```

The resulting axis vector **points from role-playing toward default assistant behavior**. It can then be used to measure or steer how "assistant-like" a model's hidden states are at each layer.

---

## Line-by-Line Explanation

### Shebang and module docstring

```python
#!/usr/bin/env python3
"""
Compute the assistant axis from per-role vectors.

Formula: axis = mean(default_vectors) - mean(pos_3_vectors across roles)

The axis points FROM role-playing TOWARD default assistant behavior.

Usage:
    uv run scripts/5_axis.py \
        --vectors_dir outputs/gemma-2-27b/vectors \
        --output outputs/gemma-2-27b/axis.pt
"""
```

The shebang allows direct execution on Unix-like systems. The docstring explains the mathematical formula and directional convention: the axis points **from** role-playing **toward** default. Example usage is shown with two arguments.

---

### Imports

```python
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
```

- `argparse` -- parses command-line arguments.
- `sys` -- used to modify `sys.path` and to call `sys.exit()` on fatal errors.
- `Path` (from `pathlib`) -- cross-platform filesystem path handling.
- `torch` -- PyTorch, used for loading vectors, stacking, computing means, norms, and saving the final axis.
- `tqdm` -- displays a progress bar while loading vector files.

---

### Modifying `sys.path`

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

Adds the project root (two directories above this script) to the front of the Python import path, allowing shared project modules to be imported.

---

### `load_vector` function

```python
def load_vector(vector_file: Path) -> dict:
    """Load vector data from .pt file."""
    return torch.load(vector_file, map_location="cpu", weights_only=False)
```

Loads a `.pt` file saved by step 4. Each file contains a dictionary with keys `"vector"`, `"type"`, and `"role"`. The arguments are:

- `map_location="cpu"` -- loads all tensors onto CPU regardless of the device they were saved from.
- `weights_only=False` -- allows loading the full dictionary (not just raw tensors).

---

### `main` function -- argument parsing

```python
def main():
    parser = argparse.ArgumentParser(description="Compute assistant axis from vectors")
    parser.add_argument("--vectors_dir", type=str, required=True, help="Directory with vector .pt files")
    parser.add_argument("--output", type=str, required=True, help="Output axis.pt file path")
    args = parser.parse_args()
```

Defines two required command-line arguments:

| Argument | Purpose |
|---|---|
| `--vectors_dir` | Path to the directory containing per-role vector `.pt` files from step 4. |
| `--output` | Path where the final axis tensor will be saved (e.g., `outputs/model/axis.pt`). |

---

### `main` function -- path setup

```python
    vectors_dir = Path(args.vectors_dir)
    output_path = Path(args.output)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
```

Converts string arguments to `Path` objects. The **parent directory** of the output file is created if it does not already exist. For example, if `--output` is `outputs/model/axis.pt`, then `outputs/model/` is created.

---

### `main` function -- discovering vector files

```python
    # Load all vectors
    vector_files = sorted(vectors_dir.glob("*.pt"))
    print(f"Found {len(vector_files)} vector files")
```

Finds all `.pt` files in the vectors directory, sorted alphabetically for deterministic order. The total count is printed.

---

### `main` function -- separating default and role vectors

```python
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

This is the core classification loop. For each vector file:

1. The data dictionary is loaded and the `"vector"` tensor, `"type"` string, and `"role"` string are extracted. `data.get()` with defaults provides graceful handling if metadata keys are missing.
2. The vector is classified as a **default vector** if either:
   - The role name contains `"default"`, OR
   - The vector type is `"mean"` (the type assigned by step 4 to default roles).
3. Otherwise, it is classified as a **role vector** (a score=3 filtered vector representing role-playing behavior).

Default vectors are printed individually so the user can verify which roles were treated as defaults.

---

### `main` function -- validation

```python
    print(f"\nLoaded {len(default_vectors)} default vectors, {len(role_vectors)} role vectors")

    if not default_vectors:
        print("Error: No default vectors found")
        sys.exit(1)

    if not role_vectors:
        print("Error: No role vectors found")
        sys.exit(1)
```

Prints the count of each category. If either list is empty, the script exits with an error code of 1. Both categories are required to compute the axis -- the axis is the difference between their means, so both must exist.

---

### `main` function -- computing the axis

```python
    # Compute means
    default_stacked = torch.stack(default_vectors)  # (n_default, n_layers, hidden_dim)
    role_stacked = torch.stack(role_vectors)  # (n_roles, n_layers, hidden_dim)

    default_mean = default_stacked.mean(dim=0)  # (n_layers, hidden_dim)
    role_mean = role_stacked.mean(dim=0)  # (n_layers, hidden_dim)

    # Compute axis: points from role-playing toward default
    axis = default_mean - role_mean
```

This is the mathematical core of the script:

1. **Stack**: `torch.stack` converts each list of 2D tensors `(n_layers, hidden_dim)` into a 3D tensor. The default stack has shape `(n_default, n_layers, hidden_dim)` and the role stack has shape `(n_roles, n_layers, hidden_dim)`.
2. **Mean**: `.mean(dim=0)` averages across the first dimension (the different roles/default variants), producing a single `(n_layers, hidden_dim)` tensor for each category.
3. **Subtract**: The axis is `default_mean - role_mean`. Since subtraction points from the subtrahend toward the minuend, this axis points **from role-playing toward default assistant behavior**.

---

### `main` function -- diagnostic output

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

Prints diagnostic statistics about the computed axis:

- **`axis.shape`**: Confirms the expected `(n_layers, hidden_dim)` shape.
- **`axis.norm(dim=1)`**: Computes the L2 norm (Euclidean length) of the axis vector at each layer, resulting in a 1D tensor of length `n_layers`. This tells you how strongly the axis separates default from role-playing at each layer.
- The first 10 per-layer norms are printed individually.
- **Mean norm**: The average separation strength across all layers.
- **Max norm**: The layer where the axis is strongest, along with its layer index (obtained via `norms.argmax().item()`). This identifies which layer has the largest difference between default and role-playing activations.

---

### `main` function -- saving the axis

```python
    # Save axis
    torch.save(axis, output_path)
    print(f"\nSaved axis to {output_path}")
```

Saves the raw axis tensor (shape `(n_layers, hidden_dim)`) to disk as a `.pt` file using `torch.save`. Unlike step 4, no metadata dictionary is wrapped around it -- just the bare tensor is saved. The output path is printed for confirmation.

---

### Script entry point

```python
if __name__ == "__main__":
    main()
```

Standard Python entry point guard: `main()` runs only when the file is executed as a script, not when imported as a module.
