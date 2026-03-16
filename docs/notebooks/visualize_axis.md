# notebooks/visualize_axis.ipynb

## Overview

This notebook loads a precomputed "Assistant Axis" vector (a direction in a language model's representation space that distinguishes assistant-like behavior from role-playing behavior) and visualizes how various persona trait vectors align with that axis. It does this by computing the cosine similarity between each trait vector and the axis at a chosen layer, then plotting all the similarities on a number-line chart with labeled extremes and a histogram backdrop.

---

## Cell-by-Cell Explanation

### Cell 0 -- Markdown

> **# Visualize Assistant Axis**
>
> This notebook loads a computed Assistant Axis and visualizes its cosine similarity with different trait persona vectors

This is the title cell. It states the purpose of the notebook: load an axis and visualize how trait persona vectors relate to it via cosine similarity.

---

### Cell 1 -- Code: Imports and setup

```python
import sys
sys.path.insert(0, '..')
```

- `import sys` -- imports Python's `sys` module, which provides access to system-specific parameters and functions.
- `sys.path.insert(0, '..')` -- prepends the parent directory (`..`) to the module search path so that any custom packages living in the repository root can be imported from within the `notebooks/` directory.

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from pathlib import Path
```

- `import torch` -- imports PyTorch, used for loading and manipulating tensor data.
- `import torch.nn.functional as F` -- imports the functional API of PyTorch's neural-network module, used here for `cosine_similarity`.
- `import numpy as np` -- imports NumPy for array operations (sorting, histogram computation, normalization).
- `import matplotlib.pyplot as plt` -- imports Matplotlib's pyplot interface for creating plots.
- `from matplotlib.colors import LinearSegmentedColormap` -- imports the class used to build a custom red-to-blue colormap.
- `from huggingface_hub import hf_hub_download, snapshot_download` -- imports two Hugging Face Hub utilities: `hf_hub_download` downloads a single file from a Hub dataset; `snapshot_download` downloads an entire directory/subset of files from a Hub dataset.
- `from huggingface_hub.utils import disable_progress_bars` -- imports a helper that suppresses download progress bars for cleaner output.
- `from pathlib import Path` -- imports `Path` for filesystem path manipulation.

```python
disable_progress_bars()
```

- Calls `disable_progress_bars()` to turn off Hugging Face Hub progress bars so that cell output stays clean.

---

### Cell 2 -- Markdown

> **## Load Assistant Axis**

Section heading indicating that the next code cell will load the precomputed assistant axis.

---

### Cell 3 -- Code: Load the Assistant Axis from Hugging Face

```python
# Precomputed models include: gemma-2-27b, qwen-3-32b, llama-3.3-70b
MODEL_NAME = "gemma-2-27b"
REPO_ID = "lu-christina/assistant-axis-vectors"
TARGET_LAYER = 22
```

- `MODEL_NAME = "gemma-2-27b"` -- sets the model whose axis vectors will be loaded. The comment notes that `qwen-3-32b` and `llama-3.3-70b` are also available.
- `REPO_ID = "lu-christina/assistant-axis-vectors"` -- the Hugging Face Hub dataset repository that hosts the precomputed vectors.
- `TARGET_LAYER = 22` -- selects layer 22 of the model's residual stream for analysis.

```python
# Load axis from HuggingFace
axis_path = hf_hub_download(repo_id=REPO_ID, filename=f"{MODEL_NAME}/assistant_axis.pt", repo_type="dataset")
axis = torch.load(axis_path, map_location="cpu", weights_only=False)
```

- `axis_path = hf_hub_download(...)` -- downloads the file `gemma-2-27b/assistant_axis.pt` from the dataset repo and returns the local cache path.
- `axis = torch.load(axis_path, map_location="cpu", weights_only=False)` -- deserializes the `.pt` file into a PyTorch tensor on CPU. `weights_only=False` allows loading arbitrary Python objects (needed because the file may contain more than raw tensor weights).

```python
print(f"Axis shape: {axis.shape}")
print(f"Target layer: {TARGET_LAYER}")
```

- Prints the shape of the loaded axis tensor. The output shows `torch.Size([46, 4608])`, meaning 46 layers and a hidden dimension of 4608.
- Prints the target layer index (22).

---

### Cell 4 -- Markdown

> **## Load persona vectors (traits) and compute cosine similarity with the Assistant Axis**

Section heading for the next two code cells, which load trait vectors and compute their cosine similarity with the axis.

---

### Cell 5 -- Code: Download and load all trait vectors

```python
# Download all vectors for this model at once
local_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=f"{MODEL_NAME}/trait_vectors/*.pt"
)
```

- `local_dir = snapshot_download(...)` -- downloads every `.pt` file that matches the glob pattern `gemma-2-27b/trait_vectors/*.pt` from the Hub dataset. It returns the path to the local cache directory containing the downloaded snapshot.

```python
# Load trait vectors
trait_vectors = {p.stem: torch.load(p, map_location="cpu", weights_only=False)
                 for p in Path(local_dir, MODEL_NAME, "trait_vectors").glob("*.pt")}
```

- Constructs a dictionary comprehension that iterates over every `.pt` file in the downloaded `trait_vectors/` directory.
- `p.stem` extracts the filename without extension (e.g., `"helpful"` from `helpful.pt`), which becomes the dictionary key.
- `torch.load(p, map_location="cpu", weights_only=False)` loads each trait vector tensor onto CPU.
- The result is a dictionary mapping trait names (strings) to their corresponding tensors.

```python
print(f"Loaded {len(trait_vectors)} trait vectors")
```

- Prints how many trait vectors were loaded. The output shows 240 trait vectors.

---

### Cell 6 -- Code: Compute cosine similarity at the target layer

```python
# Compute cosine similarity at target layer
axis_layer = axis[TARGET_LAYER]
trait_sims = {name: F.cosine_similarity(vec[TARGET_LAYER], axis_layer, dim=0).item() for name, vec in trait_vectors.items()}
```

- `axis_layer = axis[TARGET_LAYER]` -- extracts the axis vector at layer 22, yielding a 1-D tensor of shape `[4608]`.
- The dictionary comprehension iterates over every trait vector:
  - `vec[TARGET_LAYER]` -- extracts the trait's vector at the same layer (also shape `[4608]`).
  - `F.cosine_similarity(vec[TARGET_LAYER], axis_layer, dim=0)` -- computes the cosine similarity between the trait vector and the axis vector along dimension 0, producing a scalar tensor.
  - `.item()` -- converts the scalar tensor to a plain Python float.
- `trait_sims` is now a dictionary mapping each trait name to its cosine similarity score with the assistant axis.

---

### Cell 7 -- Markdown

> **## Plot cosine similarity with persona vectors (traits) to characterize the Axis**

Section heading introducing the visualization code.

---

### Cell 8 -- Code: Define the `plot_similarity_line` function

```python
def plot_similarity_line(cosine_sims, names, figsize=(8, 3), n_extremes=5, show_histogram=True):
    """Plot cosine similarities on a line."""
    projections = cosine_sims
```

- Defines the function `plot_similarity_line` with parameters:
  - `cosine_sims` -- a NumPy array of cosine similarity values.
  - `names` -- a list of trait names corresponding to each similarity value.
  - `figsize` -- the figure size as a `(width, height)` tuple, defaulting to `(8, 3)`.
  - `n_extremes` -- how many extreme traits to label on each end of the axis, defaulting to 5.
  - `show_histogram` -- whether to overlay a histogram on the plot, defaulting to `True`.
- `projections = cosine_sims` -- aliases the input array for use throughout the function.

```python
    sorted_indices = np.argsort(projections)
    low_extreme_indices = list(sorted_indices[:n_extremes])
    high_extreme_indices = list(sorted_indices[-n_extremes:])
```

- `np.argsort(projections)` -- returns the indices that would sort the projections array in ascending order.
- `low_extreme_indices` -- takes the first `n_extremes` indices (the traits with the lowest cosine similarity, i.e., most "role-playing").
- `high_extreme_indices` -- takes the last `n_extremes` indices (the traits with the highest cosine similarity, i.e., most "assistant-like").

```python
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    custom_cmap = LinearSegmentedColormap.from_list('RedBlue1', ['#e63946', '#457b9d'])
```

- Creates a single figure and axes object with the specified size.
- Builds a custom linear colormap that transitions from red (`#e63946`) to steel blue (`#457b9d`). Red represents the role-playing end; blue represents the assistant-like end.

```python
    proj_norm = (projections - projections.min()) / (projections.max() - projections.min())
    colors = custom_cmap(proj_norm)
```

- `proj_norm` -- normalizes all projection values to the `[0, 1]` range using min-max normalization.
- `colors` -- maps each normalized value through the custom colormap to get an RGBA color for each data point.

```python
    y_pos = np.zeros_like(projections)
    ax.scatter(projections, y_pos, c=colors, marker='D', s=40, alpha=0.6, edgecolors='none', zorder=3)
```

- `y_pos` -- creates an array of zeros the same shape as `projections`, so all points sit on the horizontal axis (y=0).
- `ax.scatter(...)` -- plots all trait similarities as diamond-shaped markers (`'D'`) along the x-axis at y=0. Marker size is 40, transparency is 0.6, no edge colors, and drawn at z-order 3 (above the histogram but below labels).

```python
    if show_histogram:
        hist_counts, bin_edges = np.histogram(projections, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        hist_scale = 0.4
        scaled_heights = hist_counts * hist_scale
        bin_norm = (bin_centers - projections.min()) / (projections.max() - projections.min())
        bin_colors = custom_cmap(bin_norm)
        ax.bar(bin_centers, scaled_heights, width=bin_width, alpha=0.3, color=bin_colors, edgecolor='none', zorder=1)
```

- If `show_histogram` is `True`, this block overlays a density histogram behind the scatter points.
- `np.histogram(projections, bins=30, density=True)` -- computes a 30-bin histogram with density normalization (area sums to 1).
- `bin_centers` -- calculates the center x-coordinate of each bin by averaging adjacent edges.
- `bin_width` -- the width of each bin (uniform since bins are evenly spaced).
- `hist_scale = 0.4` -- a scaling factor to keep the histogram bars small relative to the y-axis range.
- `scaled_heights` -- multiplies the density values by the scale factor so bars fit within the plot.
- `bin_norm` / `bin_colors` -- normalizes bin center positions and maps them to the same red-blue colormap.
- `ax.bar(...)` -- draws the histogram bars with 0.3 transparency, no edge color, at z-order 1 (behind everything else).

```python
    y_above = [0.15, 0.25, 0.35]
    y_below = [-0.15, -0.25, -0.35]
```

- Defines vertical offsets for label placement. Labels alternate above and below the axis to reduce overlap. The first label on each side goes to `+/-0.15`, the second to `+/-0.25`, and the third to `+/-0.35`.

```python
    for i, idx in enumerate(low_extreme_indices):
        label = names[idx].replace('_', ' ').title()
        x_pos = projections[idx]
        if i % 2 == 0:
            y_label = y_above[i // 2] if i // 2 < len(y_above) else y_above[-1]
            va = 'bottom'
            line_end = y_label - 0.02
        else:
            y_label = y_below[i // 2] if i // 2 < len(y_below) else y_below[-1]
            va = 'top'
            line_end = y_label + 0.02
        ax.plot([x_pos, x_pos], [0.02 if y_label > 0 else -0.02, line_end], '-', color='gray', alpha=0.4, linewidth=0.8, zorder=1)
        ax.text(x_pos, y_label, label, ha='center', va=va, fontsize=9, zorder=4)
```

- Loops over the `n_extremes` traits with the lowest cosine similarity (most role-playing).
- `label` -- converts the trait name from `snake_case` to `Title Case` for display.
- `x_pos` -- the cosine similarity value used as the x-coordinate.
- Even-indexed labels are placed above the axis; odd-indexed labels are placed below, each at increasing distance to avoid overlap.
- `va` (vertical alignment) is set to `'bottom'` for above-labels and `'top'` for below-labels.
- `line_end` -- the endpoint of the connecting line, offset slightly from the label position.
- `ax.plot(...)` -- draws a thin gray vertical line connecting the data point on the axis to the label.
- `ax.text(...)` -- places the label text at the computed position with center horizontal alignment, at z-order 4 (on top of everything).

```python
    for i, idx in enumerate(reversed(high_extreme_indices)):
        label = names[idx].replace('_', ' ').title()
        x_pos = projections[idx]
        if i % 2 == 0:
            y_label = y_above[i // 2] if i // 2 < len(y_above) else y_above[-1]
            va = 'bottom'
            line_end = y_label - 0.02
        else:
            y_label = y_below[i // 2] if i // 2 < len(y_below) else y_below[-1]
            va = 'top'
            line_end = y_label + 0.02
        ax.plot([x_pos, x_pos], [0.02 if y_label > 0 else -0.02, line_end], '-', color='gray', alpha=0.4, linewidth=0.8, zorder=1)
        ax.text(x_pos, y_label, label, ha='center', va=va, fontsize=9, zorder=4)
```

- Identical logic to the previous loop, but for the `n_extremes` traits with the highest cosine similarity (most assistant-like).
- `reversed(high_extreme_indices)` -- reverses the order so the most extreme trait is labeled first (and placed closest to the axis).

```python
    max_abs = max(abs(projections.min()), abs(projections.max()))
    ax.annotate('Role-playing', xy=(-max_abs, -0.45), xytext=(-max_abs + max_abs * 0.25, -0.45),
                arrowprops=dict(arrowstyle='->', color='#e63946', lw=2),
                fontsize=12, fontweight='bold', color='#e63946', ha='left', va='center')
    ax.annotate('Assistant-like', xy=(max_abs, -0.45), xytext=(max_abs - max_abs * 0.25, -0.45),
                arrowprops=dict(arrowstyle='->', color='#457b9d', lw=2),
                fontsize=12, fontweight='bold', color='#457b9d', ha='right', va='center')
```

- `max_abs` -- finds the maximum absolute cosine similarity value, used to position the endpoint annotations symmetrically.
- The first `ax.annotate(...)` draws a red left-pointing arrow labeled "Role-playing" at the bottom-left of the plot. The arrow starts at an inset position (`-max_abs + max_abs * 0.25`) and points outward to `-max_abs`.
- The second `ax.annotate(...)` draws a blue right-pointing arrow labeled "Assistant-like" at the bottom-right. The arrow starts at an inset position (`max_abs - max_abs * 0.25`) and points outward to `max_abs`.

```python
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=1)
    ax.tick_params(axis='x', length=12, width=1.5, pad=10)
    ax.tick_params(axis='y', length=0, width=0)
    ax.set_yticks([])
    ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.set_ylim(-0.55, 0.5)
    ax.set_xlim(-1, 1)
    ax.grid(False)
    plt.tight_layout()
    return fig
```

- Hides the top, right, and left spines, leaving only the bottom spine.
- Repositions the bottom spine to y=0 so it serves as the number line.
- Draws a thick black horizontal line at y=0 to reinforce the axis.
- Configures x-axis ticks to be tall (length 12) and thick (width 1.5) with padding of 10.
- Hides y-axis ticks entirely since the y-axis carries no data meaning.
- Removes all y-tick labels with `set_yticks([])`.
- Sets explicit x-ticks at `[-0.8, -0.4, 0, 0.4, 0.8]` for clean reference points.
- Sets y-limits to `[-0.55, 0.5]` to accommodate labels and arrows.
- Sets x-limits to `[-1, 1]`, the natural range of cosine similarity.
- Disables the grid.
- Calls `tight_layout()` to reduce whitespace.
- Returns the figure object.

---

### Cell 9 -- Code: Generate and display the plot

```python
# Plot traits
trait_names = list(trait_sims.keys())
trait_cosine_sims = np.array([trait_sims[n] for n in trait_names])
fig = plot_similarity_line(trait_cosine_sims, trait_names, n_extremes=5)
plt.title(f"Trait Vectors vs Assistant Axis ({MODEL_NAME.replace('-', ' ').title()}, Layer {TARGET_LAYER})")
plt.show()
```

- `trait_names = list(trait_sims.keys())` -- extracts all trait names from the similarity dictionary into a list.
- `trait_cosine_sims = np.array([trait_sims[n] for n in trait_names])` -- builds a NumPy array of cosine similarity values in the same order as `trait_names`.
- `fig = plot_similarity_line(trait_cosine_sims, trait_names, n_extremes=5)` -- calls the visualization function defined in Cell 8, labeling the 5 most extreme traits on each end.
- `plt.title(...)` -- adds a title to the plot: "Trait Vectors vs Assistant Axis (Gemma 2 27B, Layer 22)". The model name has hyphens replaced with spaces and is title-cased.
- `plt.show()` -- renders and displays the figure inline in the notebook.

The output is a number-line plot showing all 240 trait vectors distributed by their cosine similarity with the assistant axis, with the most extreme traits labeled on both ends. Traits that align with role-playing behavior appear on the left (red), while traits that align with assistant-like behavior appear on the right (blue).
