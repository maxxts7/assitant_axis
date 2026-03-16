# `pipeline/4_vectors.py`

## Overview

This script computes **per-role mean activation vectors** from the activations extracted in step 2 and the scores assigned in step 3. It handles two categories of roles differently:

- **Regular roles**: Only activations whose corresponding score is exactly 3 (meaning the model fully adopted the role) are averaged together.
- **Default roles** (roles whose name contains `"default"`): All activations are averaged without any score filtering.

The output for each role is a `.pt` file containing the mean activation vector, metadata about the vector type, and the role name.

---

## Line-by-Line Explanation

### Shebang and module docstring

```python
#!/usr/bin/env python3
"""
Compute per-role vectors from activations and scores.

For regular roles: computes the mean of activations where score=3 (fully playing role)
For default role: computes the mean of ALL activations (no score filtering)

Usage:
    uv run scripts/4_vectors.py \
        --activations_dir outputs/gemma-2-27b/activations \
        --scores_dir outputs/gemma-2-27b/scores \
        --output_dir outputs/gemma-2-27b/vectors \
        --min_count 50
"""
```

The shebang line (`#!/usr/bin/env python3`) allows the script to be executed directly on Unix-like systems by locating `python3` via the environment. The docstring documents the purpose of the module and shows example usage with the `uv run` command, including the four command-line arguments.

---

### Imports

```python
import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
```

- `argparse` -- used to parse command-line arguments.
- `json` -- used to load the score files, which are stored as JSON.
- `sys` -- used to modify `sys.path` so the project root is importable.
- `Path` (from `pathlib`) -- provides object-oriented filesystem path handling.
- `torch` -- PyTorch, used for loading/saving activation tensors and performing tensor math (stacking, mean).
- `tqdm` -- provides a progress bar when iterating over activation files.

---

### Modifying `sys.path`

```python
sys.path.insert(0, str(Path(__file__).parent.parent))
```

This inserts the **project root directory** (two levels up from this script's location) at the front of the Python module search path. This ensures that any shared modules in the project root can be imported, regardless of the working directory when the script is launched.

---

### `load_scores` function

```python
def load_scores(scores_file: Path) -> dict:
    """Load scores from JSON file."""
    with open(scores_file, 'r') as f:
        return json.load(f)
```

Opens a JSON file at `scores_file` and returns its contents as a Python dictionary. The dictionary maps prompt/response keys to integer scores (0--3). The file is opened in text/read mode (`'r'`).

---

### `load_activations` function

```python
def load_activations(activations_file: Path) -> dict:
    """Load activations from .pt file."""
    return torch.load(activations_file, map_location="cpu", weights_only=False)
```

Loads a PyTorch `.pt` file and returns the stored object (expected to be a dictionary mapping keys to tensors). Two keyword arguments are used:

- `map_location="cpu"` -- forces all tensors to be loaded onto the CPU, regardless of which device they were saved from. This avoids GPU memory issues.
- `weights_only=False` -- allows loading arbitrary Python objects (not just raw tensor weights). This is required because the file contains a dictionary, not just a bare tensor.

---

### `compute_pos_3_vector` function

```python
def compute_pos_3_vector(activations: dict, scores: dict, min_count: int) -> torch.Tensor:
    """
    Compute mean vector from activations where score=3.

    Args:
        activations: Dict mapping keys to tensors (n_layers, hidden_dim)
        scores: Dict mapping keys to scores (0-3)
        min_count: Minimum number of score=3 samples required

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
```

Function signature and docstring. This function computes a mean activation vector using only the samples that received a perfect score of 3 (indicating the model was fully playing the assigned role). Each activation tensor has shape `(n_layers, hidden_dim)` -- one hidden-state vector per layer.

```python
    # Filter activations with score=3
    filtered_acts = []
    for key, act in activations.items():
        if key in scores and scores[key] == 3:
            filtered_acts.append(act)
```

Iterates over every key-activation pair. If the key also exists in the scores dictionary and the score is exactly 3, the activation tensor is added to the `filtered_acts` list. Keys not present in `scores` are silently ignored.

```python
    if len(filtered_acts) < min_count:
        raise ValueError(f"Only {len(filtered_acts)} score=3 samples, need {min_count}")
```

A safety check: if the number of qualifying activations is below the `min_count` threshold (default 50), a `ValueError` is raised. This prevents computing a mean from too few samples, which would be noisy and unreliable.

```python
    # Stack and compute mean
    stacked = torch.stack(filtered_acts)  # (n_samples, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

`torch.stack` combines the list of 2D tensors into a single 3D tensor of shape `(n_samples, n_layers, hidden_dim)`. Then `.mean(dim=0)` computes the element-wise mean across the sample dimension, collapsing it and returning a tensor of shape `(n_layers, hidden_dim)`. This is the mean activation vector for this role.

---

### `compute_mean_vector` function

```python
def compute_mean_vector(activations: dict) -> torch.Tensor:
    """
    Compute mean vector from all activations (no filtering).

    Args:
        activations: Dict mapping keys to tensors (n_layers, hidden_dim)

    Returns:
        Mean vector of shape (n_layers, hidden_dim)
    """
    all_acts = list(activations.values())
    stacked = torch.stack(all_acts)  # (n_samples, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

A simpler variant of `compute_pos_3_vector` that does **no score filtering**. It takes all activation tensors from the dictionary, stacks them into a 3D tensor, and returns the mean across the sample dimension. This is used for "default" roles where every response is representative of default assistant behavior, so no quality filter is needed.

---

### `main` function -- argument parsing

```python
def main():
    parser = argparse.ArgumentParser(description="Compute per-role vectors")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with activation .pt files")
    parser.add_argument("--scores_dir", type=str, required=True, help="Directory with score JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for vector .pt files")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum score=3 samples required")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()
```

Sets up the command-line interface with five arguments:

| Argument | Required | Default | Purpose |
|---|---|---|---|
| `--activations_dir` | Yes | -- | Path to the directory containing per-role `.pt` activation files from step 2. |
| `--scores_dir` | Yes | -- | Path to the directory containing per-role `.json` score files from step 3. |
| `--output_dir` | Yes | -- | Path where the output vector `.pt` files will be saved. |
| `--min_count` | No | 50 | Minimum number of score=3 samples required to compute a vector. |
| `--overwrite` | No | False (flag) | If set, existing output files will be overwritten instead of skipped. |

---

### `main` function -- directory setup

```python
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir)
```

Converts the string arguments into `Path` objects. The output directory is created if it does not already exist (`parents=True` creates any necessary parent directories; `exist_ok=True` prevents errors if the directory is already present).

---

### `main` function -- discovering activation files

```python
    # Get all activation files
    activation_files = sorted(activations_dir.glob("*.pt"))
    print(f"Found {len(activation_files)} activation files")
```

Uses `glob("*.pt")` to find all PyTorch files in the activations directory. The results are sorted alphabetically so processing order is deterministic. The count is printed for the user's awareness.

---

### `main` function -- counters

```python
    successful = 0
    skipped = 0
    failed = 0
```

Three counters are initialized to track processing outcomes: how many roles were successfully processed, how many were skipped (because the output already existed), and how many failed (due to missing data or insufficient samples).

---

### `main` function -- main processing loop

```python
    for act_file in tqdm(activation_files, desc="Computing vectors"):
        role = act_file.stem
        output_file = output_dir / f"{role}.pt"
```

Iterates over each activation file with a `tqdm` progress bar. `act_file.stem` extracts the filename without its extension (e.g., `"pirate"` from `"pirate.pt"`), which is used as the role name. The output file path mirrors the role name in the output directory.

```python
        # Skip if exists (unless --overwrite)
        if output_file.exists() and not args.overwrite:
            skipped += 1
            continue
```

If the output file already exists and `--overwrite` was not specified, the role is skipped. This makes the script **resumable** -- you can re-run it after a failure without redoing completed work.

```python
        # Load activations
        activations = load_activations(act_file)

        if not activations:
            print(f"Warning: No activations for {role}")
            failed += 1
            continue
```

Loads the activation dictionary from disk. If the dictionary is empty (no activations were extracted for this role), a warning is printed and the role is counted as failed.

```python
        try:
            if "default" in role:
                # Default roles: use all activations (no score filtering)
                vector = compute_mean_vector(activations)
                vector_type = "mean"
            else:
                # Regular roles: filter by score=3
                scores_file = scores_dir / f"{role}.json"
                if not scores_file.exists():
                    print(f"Warning: No scores file for {role}")
                    failed += 1
                    continue

                scores = load_scores(scores_file)
                vector = compute_pos_3_vector(activations, scores, args.min_count)
                vector_type = "pos_3"
```

The branching logic that distinguishes default roles from regular roles:

- **If `"default"` appears in the role name**: `compute_mean_vector` is called (no filtering). The type is recorded as `"mean"`.
- **Otherwise**: The corresponding scores JSON file is located. If it does not exist, the role fails. Otherwise, `compute_pos_3_vector` is called with the score-filtering logic. The type is recorded as `"pos_3"`.

```python
            # Save vector
            save_data = {
                "vector": vector,
                "type": vector_type,
                "role": role,
            }
            torch.save(save_data, output_file)
            successful += 1
```

The computed vector is bundled into a dictionary with metadata (`type` and `role`) and saved as a `.pt` file using `torch.save`. This metadata is used downstream by step 5 to distinguish default vectors from role vectors.

```python
        except ValueError as e:
            print(f"Warning: {role}: {e}")
            failed += 1
```

Catches `ValueError` exceptions -- the specific exception raised by `compute_pos_3_vector` when there are not enough score=3 samples. The error message is printed and the role is counted as failed.

---

### `main` function -- summary

```python
    print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")
```

After processing all roles, a summary line is printed showing how many roles were processed successfully, skipped, and failed.

---

### Script entry point

```python
if __name__ == "__main__":
    main()
```

Standard Python idiom: when the file is run as a script (rather than imported as a module), the `main()` function is called.
