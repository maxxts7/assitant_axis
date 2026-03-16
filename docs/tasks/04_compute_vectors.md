# Task 4: Compute Per-Role Vectors

## Overview

This task takes per-role activation `.pt` files (from Task 2) and per-role score `.json` files (from Task 3) and reduces each role to a single **mean activation vector**. For regular roles, only activations whose judge score equals 3 ("fully playing the role") are included in the mean. For default roles, all activations are included without score filtering. The resulting vectors are consumed by Task 5, which subtracts the role-vector mean from the default-vector mean to produce the assistant axis.

**Source file:** `pipeline/4_vectors.py`

**CLI invocation:**

```
uv run pipeline/4_vectors.py \
    --activations_dir outputs/gemma-2-27b/activations \
    --scores_dir outputs/gemma-2-27b/scores \
    --output_dir outputs/gemma-2-27b/vectors \
    --min_count 50
```

---

## Sub-Tasks

### Sub-Task 4.1: Parse CLI Arguments and Initialise Directories

#### Input

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `--activations_dir` | `str` | Yes | -- | Directory containing per-role `.pt` activation files |
| `--scores_dir` | `str` | Yes | -- | Directory containing per-role `.json` score files |
| `--output_dir` | `str` | Yes | -- | Output directory for per-role vector `.pt` files |
| `--min_count` | `int` | No | `50` | Minimum number of score=3 samples required to compute a regular-role vector |
| `--overwrite` | flag | No | `False` | If set, overwrite existing output files |

#### Processing

The script parses command-line arguments and creates the output directory if it does not exist.

```python
def main():
    parser = argparse.ArgumentParser(description="Compute per-role vectors")
    parser.add_argument("--activations_dir", type=str, required=True, help="Directory with activation .pt files")
    parser.add_argument("--scores_dir", type=str, required=True, help="Directory with score JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for vector .pt files")
    parser.add_argument("--min_count", type=int, default=50, help="Minimum score=3 samples required")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activations_dir = Path(args.activations_dir)
    scores_dir = Path(args.scores_dir)
```

#### Output

- Three `Path` objects: `activations_dir`, `scores_dir`, `output_dir`.
- `output_dir` is guaranteed to exist on disk.

---

### Sub-Task 4.2: Discover Activation Files

#### Input

- `activations_dir` -- a directory containing `*.pt` files, one per role (e.g. `pirate.pt`, `default_helpful.pt`).

#### Processing

All `.pt` files in the activations directory are collected and sorted alphabetically. Each file stem is treated as the role name.

```python
# Get all activation files
activation_files = sorted(activations_dir.glob("*.pt"))
print(f"Found {len(activation_files)} activation files")
```

#### Output

- `activation_files` -- a sorted `list[Path]` of every `.pt` file in the activations directory.
- The file stem (e.g. `pirate` from `pirate.pt`) becomes the role identifier used throughout the rest of the script.

---

### Sub-Task 4.3: Load Activations for a Role

#### Input

- A single `.pt` file path (e.g. `activations/pirate.pt`).
- File format: a Python `dict` serialised with `torch.save()`. Keys are sample identifiers of the form `"{label}_p{prompt_index}_q{question_index}"` (e.g. `"pirate_p0_q2"`). Values are `torch.Tensor` objects of shape `(n_layers, hidden_dim)` -- the mean assistant-turn activation per layer for that conversation.

#### Processing

```python
def load_activations(activations_file: Path) -> dict:
    """Load activations from .pt file."""
    return torch.load(activations_file, map_location="cpu", weights_only=False)
```

The file is loaded onto CPU regardless of how it was saved. `weights_only=False` allows arbitrary Python objects (the dict) to be unpickled.

#### Output

- `activations` -- `dict[str, torch.Tensor]` where each value has shape `(n_layers, hidden_dim)`.

---

### Sub-Task 4.4: Load Scores for a Role

#### Input

- A single `.json` file path (e.g. `scores/pirate.json`).
- File format: a flat JSON object mapping the same sample keys to integer scores on the 0-3 scale produced by Task 3's judge.

Example:

```json
{
  "pirate_p0_q0": 3,
  "pirate_p0_q1": 2,
  "pirate_p0_q2": 3,
  "pirate_p1_q0": 0
}
```

Score semantics (from Task 3):

| Score | Meaning |
|---|---|
| 0 | Model refused to answer |
| 1 | Model says it cannot be the role, but can help with related tasks |
| 2 | Model identifies as AI/LLM but has some role attributes |
| 3 | Model is fully playing the role |

#### Processing

```python
def load_scores(scores_file: Path) -> dict:
    """Load scores from JSON file."""
    with open(scores_file, 'r') as f:
        return json.load(f)
```

#### Output

- `scores` -- `dict[str, int]` mapping sample keys to integer scores (0-3).

---

### Sub-Task 4.5: Determine Role Type (Default vs. Regular)

#### Input

- `role` -- the role name string (file stem of the activation file).

#### Processing

The script checks whether the string `"default"` appears anywhere in the role name. This determines which computation path is used.

```python
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

#### Output

- A routing decision:
  - **Default roles** (name contains `"default"`) go to `compute_mean_vector` (Sub-Task 4.6).
  - **Regular roles** go to `compute_pos_3_vector` (Sub-Task 4.7), after loading the corresponding scores file.
- `vector_type` is set to `"mean"` or `"pos_3"` respectively.

---

### Sub-Task 4.6: Compute Mean Vector for Default Roles

#### Input

- `activations` -- `dict[str, torch.Tensor]` where each tensor has shape `(n_layers, hidden_dim)`.
- No filtering is applied; every activation in the dict is used.

#### Processing

All activation tensors are collected into a list and stacked into a 3D tensor, then averaged along the sample dimension.

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

Step-by-step:

1. Extract all tensor values from the dict into a list of length `n_samples`.
2. `torch.stack(all_acts)` produces a tensor of shape `(n_samples, n_layers, hidden_dim)`.
3. `.mean(dim=0)` averages across samples, yielding shape `(n_layers, hidden_dim)`.

#### Output

- A single `torch.Tensor` of shape `(n_layers, hidden_dim)` -- the element-wise mean activation across all conversations for this default role.

---

### Sub-Task 4.7: Compute Score-3 Filtered Vector for Regular Roles

#### Input

| Parameter | Type | Shape / Format | Description |
|---|---|---|---|
| `activations` | `dict[str, torch.Tensor]` | Values: `(n_layers, hidden_dim)` | All activations for the role |
| `scores` | `dict[str, int]` | Values: `0-3` | Judge scores for the role |
| `min_count` | `int` | scalar | Minimum number of score=3 samples required |

#### Processing

Only activations whose corresponding score is exactly 3 are kept. If the count of qualifying samples is below `min_count`, a `ValueError` is raised.

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
    # Filter activations with score=3
    filtered_acts = []
    for key, act in activations.items():
        if key in scores and scores[key] == 3:
            filtered_acts.append(act)

    if len(filtered_acts) < min_count:
        raise ValueError(f"Only {len(filtered_acts)} score=3 samples, need {min_count}")

    # Stack and compute mean
    stacked = torch.stack(filtered_acts)  # (n_samples, n_layers, hidden_dim)
    return stacked.mean(dim=0)  # (n_layers, hidden_dim)
```

Step-by-step:

1. Iterate over every `(key, activation_tensor)` pair in the activations dict.
2. For each key, look it up in the scores dict. Include the activation only if the score equals exactly `3`.
3. If the number of qualifying activations is less than `min_count` (default 50), raise `ValueError`. This error is caught in the main loop and the role is marked as failed.
4. Stack the filtered activations into shape `(n_filtered, n_layers, hidden_dim)`.
5. Take the mean across the first dimension to get the final vector of shape `(n_layers, hidden_dim)`.

#### Output

- A single `torch.Tensor` of shape `(n_layers, hidden_dim)` -- the element-wise mean of only those activations where the model was judged to be "fully playing the role".
- Raises `ValueError` if fewer than `min_count` score=3 samples exist.

---

### Sub-Task 4.8: Save the Per-Role Vector

#### Input

| Field | Type | Description |
|---|---|---|
| `vector` | `torch.Tensor` | Shape `(n_layers, hidden_dim)` -- the computed mean vector |
| `vector_type` | `str` | Either `"mean"` (default role) or `"pos_3"` (regular role) |
| `role` | `str` | The role name (file stem) |

#### Processing

The vector and its metadata are bundled into a dict and saved with `torch.save`.

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

The output file path is `{output_dir}/{role}.pt`.

#### Output

- A `.pt` file at `{output_dir}/{role}.pt` containing a serialised Python dict with three keys:

| Key | Type | Value |
|---|---|---|
| `"vector"` | `torch.Tensor` | Shape `(n_layers, hidden_dim)` |
| `"type"` | `str` | `"mean"` or `"pos_3"` |
| `"role"` | `str` | Role name, e.g. `"pirate"` or `"default_helpful"` |

---

### Sub-Task 4.9: Skip Existing Outputs

#### Input

- `output_file` -- the target `.pt` path for the current role.
- `args.overwrite` -- boolean flag.

#### Processing

Before processing each role, the script checks whether the output file already exists. If it does and `--overwrite` is not set, the role is skipped.

```python
if output_file.exists() and not args.overwrite:
    skipped += 1
    continue
```

#### Output

- The role is either processed or added to the `skipped` counter and bypassed entirely.

---

### Sub-Task 4.10: Error Handling and Summary

#### Input

- Accumulated counters: `successful`, `skipped`, `failed`.

#### Processing

Three classes of failure are handled:

1. **Empty activations** -- the loaded `.pt` file contains no entries.

    ```python
    if not activations:
        print(f"Warning: No activations for {role}")
        failed += 1
        continue
    ```

2. **Missing scores file** -- for a regular (non-default) role, the corresponding `.json` scores file does not exist.

    ```python
    if not scores_file.exists():
        print(f"Warning: No scores file for {role}")
        failed += 1
        continue
    ```

3. **Insufficient score=3 samples** -- the `ValueError` raised by `compute_pos_3_vector` when the count of qualifying activations is below `min_count`.

    ```python
    except ValueError as e:
        print(f"Warning: {role}: {e}")
        failed += 1
    ```

After all roles are processed, a summary is printed:

```python
print(f"\nSummary: {successful} successful, {skipped} skipped, {failed} failed")
```

#### Output

- Console summary with counts of successful, skipped, and failed roles.

---

## Data Flow Summary

```
Task 2 output                    Task 3 output
activations/{role}.pt            scores/{role}.json
  dict[str, Tensor]                dict[str, int]
  key -> (n_layers, hidden_dim)    key -> 0|1|2|3
          |                              |
          +----------+-------------------+
                     |
              Task 4 (this script)
                     |
          +----------+----------+
          |                     |
    "default" in role?     Regular role
          |                     |
    compute_mean_vector    compute_pos_3_vector
    (all activations)      (only score=3 activations)
          |                     |
          +----------+----------+
                     |
          vectors/{role}.pt
            dict with keys:
              "vector": Tensor (n_layers, hidden_dim)
              "type":   "mean" | "pos_3"
              "role":   str
                     |
                     v
              Task 5: axis.pt
```
