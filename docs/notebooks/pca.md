# notebooks/pca.ipynb

## Overview

This notebook performs Principal Component Analysis (PCA) on "role vectors" extracted from a large language model (Gemma 2 27B). It loads precomputed role vectors (either from HuggingFace Hub or locally), runs PCA on them at a chosen transformer layer, plots the variance explained by each principal component, and then visualizes the cosine similarity of each role vector (and a default vector) with the top 3 principal components. The goal is to understand how different role personas cluster and separate in the model's representation space.

---

## Markdown Cell 1

> # PCA Analysis
>
> This notebook runs PCA on role vectors and examines the top PCs.

This is the title cell. It states the purpose of the notebook: running PCA on role vectors and examining the top principal components.

---

## Code Cell 1 -- Imports

```python
import sys
sys.path.insert(0, '..')
```

- `import sys` -- imports Python's `sys` module, which provides access to system-specific parameters and functions.
- `sys.path.insert(0, '..')` -- inserts the parent directory (`..`) at the beginning of Python's module search path. This allows importing modules from the repository root (one directory up from the `notebooks/` folder).

```python
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
```

- `import torch` -- imports PyTorch, used here for loading `.pt` tensor files and tensor operations.
- `import numpy as np` -- imports NumPy as `np`, used for numerical array operations.
- `from pathlib import Path` -- imports `Path` from the `pathlib` module for object-oriented filesystem path manipulation.
- `from huggingface_hub import snapshot_download` -- imports the `snapshot_download` function, which downloads an entire repository (or a filtered subset) from the HuggingFace Hub to a local cache directory.
- `from huggingface_hub.utils import disable_progress_bars` -- imports a utility to suppress download progress bars for cleaner output.

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

- `import matplotlib.pyplot as plt` -- imports the pyplot interface from matplotlib for creating plots and figures.
- `from matplotlib.colors import LinearSegmentedColormap` -- imports `LinearSegmentedColormap`, which allows creating custom color gradients by interpolating between specified colors.

```python
from assistant_axis import compute_pca, MeanScaler
```

- `from assistant_axis import compute_pca, MeanScaler` -- imports two items from the project's own `assistant_axis` package:
  - `compute_pca`: a function that runs PCA on a set of vectors, returning transformed data, variance explained, and the fitted PCA object.
  - `MeanScaler`: a scaler class that centers data by subtracting the mean (used before PCA).

```python
disable_progress_bars()
```

- `disable_progress_bars()` -- disables HuggingFace Hub's download progress bars so that output is not cluttered during `snapshot_download`.

---

## Markdown Cell 2

> ## Configuration

A section header introducing the configuration variables.

---

## Code Cell 2 -- Configuration

```python
# Model configuration
MODEL_NAME = "gemma-2-27b"
TARGET_LAYER = 22
```

- `MODEL_NAME = "gemma-2-27b"` -- sets the model identifier to `"gemma-2-27b"`. This is used to locate the correct subdirectory of precomputed vectors.
- `TARGET_LAYER = 22` -- specifies which transformer layer's representations to analyze. Layer 22 is selected for the PCA.

```python
# Local paths if you computed the vectors yourself
LOCAL_ROLE_VECTORS_DIR = Path(f"../outputs/{MODEL_NAME}/role_vectors")
LOCAL_DEFAULT_VECTOR_PATH = Path(f"../outputs/{MODEL_NAME}/default_vector.pt")
```

- `LOCAL_ROLE_VECTORS_DIR = Path(f"../outputs/{MODEL_NAME}/role_vectors")` -- constructs a `Path` to the local directory where role vector `.pt` files would be stored if the user computed them locally (e.g., `../outputs/gemma-2-27b/role_vectors`).
- `LOCAL_DEFAULT_VECTOR_PATH = Path(f"../outputs/{MODEL_NAME}/default_vector.pt")` -- constructs a `Path` to the local default vector file (e.g., `../outputs/gemma-2-27b/default_vector.pt`).

```python
# HuggingFace configuration for pre-computed vectors
# Models supported: gemma-2-27b, qwen-3-32b, llama-3.3-70b
REPO_ID = "lu-christina/assistant-axis-vectors"
```

- `REPO_ID = "lu-christina/assistant-axis-vectors"` -- the HuggingFace dataset repository ID where precomputed vectors are hosted. The comment notes that vectors are available for three models: gemma-2-27b, qwen-3-32b, and llama-3.3-70b.

---

## Markdown Cell 3

> ## Load Data
>
> Run **one** of the two sections below.

A section header instructing the user to run only one of the two following code cells (HuggingFace download or local loading).

---

## Code Cell 3 -- Load Data from HuggingFace

```python
# HuggingFace
print(f"Loading from HuggingFace: {REPO_ID}")
```

- Prints a message indicating that data is being loaded from HuggingFace, displaying the repository ID.

```python
# Download all vectors for this model
local_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[f"{MODEL_NAME}/role_vectors/*.pt", f"{MODEL_NAME}/default_vector.pt"]
)
```

- `snapshot_download(...)` -- downloads files from the HuggingFace Hub repository to a local cache directory and returns the path to that local directory.
  - `repo_id=REPO_ID` -- specifies which repository to download from (`lu-christina/assistant-axis-vectors`).
  - `repo_type="dataset"` -- indicates that the repository is a dataset (not a model).
  - `allow_patterns=[...]` -- filters the download to only include files matching these glob patterns: all `.pt` files under `{MODEL_NAME}/role_vectors/` and the single `{MODEL_NAME}/default_vector.pt` file. This avoids downloading vectors for other models.
- `local_dir` -- receives the local cache path where the downloaded files are stored.

```python
# Load role vectors
role_vectors = {p.stem: torch.load(p, map_location="cpu", weights_only=False)
                for p in Path(local_dir, MODEL_NAME, "role_vectors").glob("*.pt")}
print(f"Loaded {len(role_vectors)} role vectors")
```

- `Path(local_dir, MODEL_NAME, "role_vectors").glob("*.pt")` -- constructs the path to the downloaded role vectors directory and globs for all `.pt` files.
- `{p.stem: torch.load(p, map_location="cpu", weights_only=False) for p in ...}` -- a dictionary comprehension that iterates over each `.pt` file:
  - `p.stem` -- the filename without extension, used as the dictionary key (i.e., the role name, like `"poet"` or `"teacher"`).
  - `torch.load(p, map_location="cpu", weights_only=False)` -- loads the PyTorch tensor from disk onto the CPU. `weights_only=False` allows loading arbitrary Python objects (not just raw tensors).
- `role_vectors` -- a dictionary mapping role name strings to their corresponding tensors.
- The print statement shows how many role vectors were loaded (275 in the output).

```python
# Load default vector
default_vector = torch.load(Path(local_dir, MODEL_NAME, "default_vector.pt"), map_location="cpu", weights_only=False)
print(f"Default vector shape: {default_vector.shape}")
```

- `torch.load(Path(local_dir, MODEL_NAME, "default_vector.pt"), ...)` -- loads the default vector tensor from the downloaded file. The default vector represents the model's behavior when no specific role is assigned.
- `default_vector.shape` -- prints the shape of the tensor. The output shows `torch.Size([46, 4608])`, meaning the default vector has values across 46 layers, each of dimension 4608.

---

## Code Cell 4 -- Load Data Locally (Commented Out)

```python
# FOR LOCAL

# print(f"Loading from local: {LOCAL_ROLE_VECTORS_DIR}")

# # Load role vectors
# role_vectors = {p.stem: torch.load(p, map_location="cpu", weights_only=False)
#                 for p in LOCAL_ROLE_VECTORS_DIR.glob("*.pt")}
# print(f"Loaded {len(role_vectors)} role vectors")

# # Load default vector
# default_vector = torch.load(LOCAL_DEFAULT_VECTOR_PATH, map_location="cpu", weights_only=False)
# print(f"Default vector shape: {default_vector.shape}")
```

This entire cell is commented out. It is the alternative to the HuggingFace cell above, for users who have computed vectors locally. The logic is identical to Cell 3 but uses `LOCAL_ROLE_VECTORS_DIR` and `LOCAL_DEFAULT_VECTOR_PATH` instead of the HuggingFace-downloaded paths.

---

## Markdown Cell 4

> ## Run PCA

A section header introducing the PCA computation.

---

## Code Cell 5 -- Run PCA

```python
# Stack role vectors at target layer
role_vectors_at_layer = torch.stack([v[TARGET_LAYER] for v in role_vectors.values()]).float()
role_labels = list(role_vectors.keys())
```

- `[v[TARGET_LAYER] for v in role_vectors.values()]` -- for each role vector tensor (shape `[46, 4608]`), extracts the slice at the target layer index (22). This produces a list of 1-D tensors each of shape `[4608]`.
- `torch.stack([...])` -- stacks the list of 1-D tensors into a single 2-D tensor of shape `(n_roles, 4608)`.
- `.float()` -- converts the tensor to 32-bit floating point (in case it was stored in a different precision).
- `role_vectors_at_layer` -- the resulting tensor of shape `(275, 4608)`, with one row per role.
- `role_labels = list(role_vectors.keys())` -- creates a list of role name strings in the same order as the rows of `role_vectors_at_layer`.

```python
# Run PCA with mean centering
scaler = MeanScaler()
pca_transformed, variance_explained, n_components, pca, scaler = compute_pca(
    role_vectors_at_layer,
    layer=None,
    scaler=scaler
)
print(f"Fitted PCA with {len(variance_explained)} components")
```

- `scaler = MeanScaler()` -- creates a `MeanScaler` instance, which will center the data by subtracting the mean before PCA.
- `compute_pca(role_vectors_at_layer, layer=None, scaler=scaler)` -- runs PCA on the role vectors. Returns five values:
  - `pca_transformed` -- the data projected into the principal component space.
  - `variance_explained` -- an array of the fraction of variance explained by each component.
  - `n_components` -- the number of components (275, equal to the number of role vectors).
  - `pca` -- the fitted scikit-learn PCA object, which holds the principal component directions in `pca.components_`.
  - `scaler` -- the fitted scaler (now containing the computed mean).
  - `layer=None` -- indicates that the input is already a 2-D matrix (no layer indexing needed inside the function).
- The output shows that PCA was fitted with 275 components. It also prints cumulative variance for the first 5 components and elbow/threshold statistics: the elbow is at component 2, 70% variance is reached at 4 dimensions, 80% at 8, 90% at 18, and 95% at 36.

---

## Markdown Cell 5

> ## Plot variance explained

A section header introducing the variance explained plot.

---

## Code Cell 6 -- Plot Variance Explained

```python
def plot_variance_explained(ax, variance_explained, title, max_components=60, show_ylabel=True):
    """Plot variance explained (histogram + cumulative line)."""
```

- Defines a function `plot_variance_explained` that takes:
  - `ax` -- a matplotlib axes object to plot on.
  - `variance_explained` -- an array of per-component variance ratios.
  - `title` -- the plot title string.
  - `max_components` -- how many components to show on the x-axis (default 60).
  - `show_ylabel` -- whether to display the y-axis label.

```python
    n_show = min(len(variance_explained), max_components)
    var_exp = variance_explained[:n_show]
    cumulative = np.cumsum(var_exp)
    components = np.arange(1, n_show + 1)
```

- `n_show` -- the smaller of the total number of components and `max_components`, so at most 60 components are shown.
- `var_exp` -- slices `variance_explained` to the first `n_show` entries.
- `cumulative = np.cumsum(var_exp)` -- computes the cumulative sum of variance explained, so `cumulative[i]` is the total variance explained by the first `i+1` components.
- `components` -- an integer array `[1, 2, ..., n_show]` for x-axis tick positions.

```python
    bar_color = '#6a9bc3'
    line_color = '#1a5276'
```

- Defines color hex codes: a medium blue for the bars and a dark blue for the cumulative line.

```python
    ax.bar(components, var_exp * 100, width=0.8, color=bar_color, alpha=0.6,
           edgecolor='none', label='Individual')
    ax.plot(components, cumulative * 100, color=line_color, linewidth=1, label='Cumulative')
```

- `ax.bar(...)` -- draws a bar chart where each bar's height is the individual variance explained (as a percentage) for that component. Bars are semi-transparent (`alpha=0.6`) with no edge color.
- `ax.plot(...)` -- draws a line chart of the cumulative variance explained (as a percentage) across components.

```python
    # Threshold lines at 70%, 80%, 90%
    for thresh in [70, 80, 90]:
        idx = np.argmax(cumulative >= thresh / 100.0)
        if cumulative[idx] >= thresh / 100.0:
            n_dims = idx + 1
            ax.axhline(y=thresh, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.text(max_components - 1, thresh + 1.5, f'{thresh}% ({n_dims}d)',
                    fontsize=7, color='gray', ha='right', va='bottom')
```

- Loops over three variance thresholds: 70%, 80%, and 90%.
- `np.argmax(cumulative >= thresh / 100.0)` -- finds the index of the first component where cumulative variance reaches or exceeds the threshold.
- `if cumulative[idx] >= thresh / 100.0` -- guards against the case where no component reaches the threshold.
- `n_dims = idx + 1` -- converts 0-based index to 1-based component count.
- `ax.axhline(...)` -- draws a horizontal dashed gray line at the threshold percentage.
- `ax.text(...)` -- places a label near the right edge of the plot showing the threshold and how many dimensions are needed to reach it (e.g., `"70% (4d)"`).

```python
    ax.set_xlim(0, max_components + 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Principal Component')
    if show_ylabel:
        ax.set_ylabel('Variance Explained (%)')
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc='center right', fontsize=8)
```

- `ax.set_xlim(0, max_components + 1)` -- sets x-axis range from 0 to 61.
- `ax.set_ylim(0, 105)` -- sets y-axis range from 0 to 105 (a bit above 100% for visual padding).
- `ax.set_xlabel('Principal Component')` -- labels the x-axis.
- `ax.set_ylabel(...)` -- conditionally labels the y-axis.
- `ax.set_title(title, fontsize=11)` -- sets the plot title.
- `ax.grid(False)` -- disables the grid.
- `ax.spines['top'].set_visible(False)` / `ax.spines['right'].set_visible(False)` -- hides the top and right borders of the plot for a cleaner look.
- `ax.set_xticks(...)` / `ax.set_yticks(...)` -- sets specific tick marks on both axes.
- `ax.legend(...)` -- adds a legend in the center-right area.

```python
fig, ax = plt.subplots(figsize=(6, 3.5))
plot_variance_explained(ax, variance_explained, f"Variance Explained by PCA on Role Vectors ({MODEL_NAME.replace('-', ' ').title()})", max_components=60)
plt.tight_layout()
plt.show()
```

- `fig, ax = plt.subplots(figsize=(6, 3.5))` -- creates a new figure and axes with a width of 6 inches and height of 3.5 inches.
- `plot_variance_explained(...)` -- calls the function defined above, passing in the variance explained data and a title string. The title includes the model name formatted with spaces and title case (e.g., `"Gemma 2 27B"`).
- `plt.tight_layout()` -- adjusts subplot spacing to prevent labels from being clipped.
- `plt.show()` -- renders and displays the figure.

---

## Markdown Cell 6

> ## Cosine Similarity with Top 3 PCs

A section header introducing the cosine similarity visualization.

---

## Code Cell 7 -- Define `plot_pc_lines` Function

```python
def plot_pc_lines(role_cosine_sims, role_labels, default_cosine_sims=None,
                       figsize=(8, 7), n_extremes=5, show_histogram=True):
    """
    Plot top 3 PCs' cosine similarities in a single figure with 3 subplots.

    Args:
        role_cosine_sims: (n_roles, 3) array of cosine similarities
        role_labels: List of role label names
        default_cosine_sims: Optional (3,) array of default vector cosine sims
        figsize: Figure size tuple
        n_extremes: Number of extremes to label on each end
        show_histogram: Whether to show histogram overlay
    """
```

- Defines a function `plot_pc_lines` that creates a figure with 3 vertically stacked subplots, one per principal component. Each subplot displays role vectors as points along a 1-D axis based on their cosine similarity with that PC direction.
  - `role_cosine_sims` -- an `(n_roles, 3)` array where each row is a role and each column is the cosine similarity with PC1, PC2, or PC3.
  - `role_labels` -- list of role name strings.
  - `default_cosine_sims` -- optional `(3,)` array of the default vector's cosine similarities with the 3 PCs.
  - `figsize` -- the figure dimensions.
  - `n_extremes` -- how many extreme roles (highest and lowest) to label on each subplot.
  - `show_histogram` -- whether to overlay a histogram showing the distribution of cosine similarities.

```python
    custom_cmap = LinearSegmentedColormap.from_list('PurpleTeal', ['#9b59b6', '#1abc9c'])
```

- Creates a custom colormap that interpolates from purple (`#9b59b6`) to teal (`#1abc9c`). This will be used to color points by their cosine similarity value.

```python
    fig, axes = plt.subplots(3, 1, figsize=figsize)
```

- Creates a figure with 3 rows and 1 column of subplots, sized according to `figsize`.

```python
    for pc_idx, ax in enumerate(axes):
        projections = role_cosine_sims[:, pc_idx]
```

- Iterates over the 3 axes (one per PC). `pc_idx` is 0, 1, or 2.
- `projections` -- extracts the column of cosine similarities for this PC from the `role_cosine_sims` array.

```python
        # Color based on projection value
        c_norm = (projections + 1) / 2  # maps [-1, 1] to [0, 1]
        colors = custom_cmap(c_norm)
```

- `c_norm = (projections + 1) / 2` -- normalizes cosine similarity values from the range `[-1, 1]` to `[0, 1]` for colormap lookup.
- `colors = custom_cmap(c_norm)` -- maps the normalized values to RGBA colors using the purple-to-teal colormap. Low cosine similarity yields purple, high yields teal.

```python
        # Find extreme indices
        sorted_indices = np.argsort(projections)
        low_indices = sorted_indices[:n_extremes].tolist()
        high_indices = sorted_indices[-n_extremes:][::-1].tolist()
```

- `np.argsort(projections)` -- returns indices that would sort the projections in ascending order.
- `low_indices` -- the first `n_extremes` indices (the roles with the lowest cosine similarity).
- `high_indices` -- the last `n_extremes` indices reversed (the roles with the highest cosine similarity, in descending order).

```python
        # Plot all points
        y_pos = np.zeros_like(projections)
        ax.scatter(projections, y_pos, c=colors, marker='o', s=40, alpha=0.6, edgecolors='none', zorder=3)
```

- `y_pos = np.zeros_like(projections)` -- all points are placed at y=0 (on a horizontal number line).
- `ax.scatter(...)` -- plots all role vectors as colored circles along the x-axis. `s=40` sets the marker size, `alpha=0.6` makes them semi-transparent, and `zorder=3` puts them above background elements.

```python
        # Histogram overlay
        if show_histogram:
            hist_counts, bin_edges = np.histogram(projections, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            scaled_heights = hist_counts * 0.4
            bin_norm = (bin_centers + 1) / 2
            bin_colors = custom_cmap(np.clip(bin_norm, 0, 1))
            ax.bar(bin_centers, scaled_heights, width=bin_width, alpha=0.3, color=bin_colors, edgecolor='none', zorder=1)
```

- If `show_histogram` is `True`, a histogram of the cosine similarities is overlaid.
- `np.histogram(projections, bins=30, density=True)` -- computes a 30-bin histogram with density normalization (area sums to 1).
- `bin_centers` -- the midpoint of each bin edge pair, used as x-positions for the bars.
- `bin_width` -- the width of each histogram bin.
- `scaled_heights = hist_counts * 0.4` -- scales down the histogram heights by 0.4 so the bars don't dominate the plot.
- `bin_norm` / `bin_colors` -- maps bin center values to colors using the same purple-to-teal colormap.
- `ax.bar(...)` -- draws the histogram bars with low opacity (`alpha=0.3`) behind the scatter points (`zorder=1`).

```python
        # Add default vector marker
        if default_cosine_sims is not None:
            val = default_cosine_sims[pc_idx]
            ax.axvline(x=val, color='blue', linestyle='--', linewidth=1, alpha=0.9, zorder=2)
            ax.scatter([val], [0], c='blue', marker='*', s=300, alpha=1.0, zorder=5)
            ax.text(val, 0.55, 'Default Response', ha='center', va='bottom', fontsize=10, color='blue', alpha=0.9)
```

- If a default vector's cosine similarities are provided, this block marks the default vector's position on each PC.
- `val = default_cosine_sims[pc_idx]` -- the default vector's cosine similarity with this PC.
- `ax.axvline(...)` -- draws a vertical dashed blue line at the default vector's position.
- `ax.scatter([val], [0], c='blue', marker='*', s=300, ...)` -- plots a large blue star marker at the default vector's position on the number line.
- `ax.text(...)` -- labels it with the text "Default Response" above the star.

```python
        # Label positions
        y_above = [0.25, 0.35, 0.45]
        y_below = [-0.25, -0.35, -0.45]
```

- Defines y-coordinate positions for labels placed above and below the number line. Labels alternate above and below to reduce overlap.

```python
        # Add labels for low extremes
        for i, idx in enumerate(low_indices):
            label = role_labels[idx].replace('_', ' ').title()
            x_pos = projections[idx]
            if i % 2 == 0:
                y_label = y_above[min(i // 2, len(y_above) - 1)]
                va = 'bottom'
            else:
                y_label = y_below[min(i // 2, len(y_below) - 1)]
                va = 'top'
            line_end = y_label - 0.02 if y_label > 0 else y_label + 0.02
            ax.plot([x_pos, x_pos], [0.02 if y_label > 0 else -0.02, line_end], '-', color='gray', alpha=0.4, linewidth=0.8, zorder=1)
            ax.text(x_pos, y_label, label, ha='center', va=va, fontsize=9, zorder=4)
```

- Loops over the `n_extremes` roles with the lowest cosine similarity for this PC.
- `label = role_labels[idx].replace('_', ' ').title()` -- converts the role name from snake_case to Title Case for display (e.g., `"data_scientist"` becomes `"Data Scientist"`).
- `x_pos = projections[idx]` -- the x-coordinate (cosine similarity) for this role.
- The `if i % 2 == 0` / `else` block alternates label placement above and below the axis. Even-indexed labels go above, odd-indexed below. The y-position steps outward for successive labels to avoid overlap.
- `line_end` -- the endpoint of a thin connecting line between the point on the axis and the label.
- `ax.plot(...)` -- draws a short gray vertical line connecting the axis to the label position.
- `ax.text(...)` -- places the role name label at the computed position.

```python
        # Add labels for high extremes
        for i, idx in enumerate(high_indices):
            label = role_labels[idx].replace('_', ' ').title()
            x_pos = projections[idx]
            if i % 2 == 0:
                y_label = y_above[min(i // 2, len(y_above) - 1)]
                va = 'bottom'
            else:
                y_label = y_below[min(i // 2, len(y_below) - 1)]
                va = 'top'
            line_end = y_label - 0.02 if y_label > 0 else y_label + 0.02
            ax.plot([x_pos, x_pos], [0.02 if y_label > 0 else -0.02, line_end], '-', color='gray', alpha=0.4, linewidth=0.8, zorder=1)
            ax.text(x_pos, y_label, label, ha='center', va=va, fontsize=9, zorder=4)
```

- Identical logic to the low-extremes loop, but applied to the `n_extremes` roles with the highest cosine similarity. Labels are placed on the right side of the number line.

```python
        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, zorder=1)
        ax.tick_params(axis='x', length=12, width=1.5, pad=10)
        ax.tick_params(axis='y', length=0, width=0)
        ax.set_yticks([])
        ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
        ax.set_ylim(-0.55, 0.6)
        ax.set_xlim(-1, 1)
        ax.grid(False)
        ax.set_title(f'PC{pc_idx + 1}', fontsize=12, fontweight='bold', loc='left')
```

- `ax.spines['top']`, `'right'`, `'left'` are hidden, leaving only the bottom spine.
- `ax.spines['bottom'].set_position('zero')` -- moves the bottom spine (x-axis) to y=0.
- `ax.axhline(y=0, ...)` -- draws a thick black horizontal line at y=0 as the number line.
- `ax.tick_params(axis='x', length=12, width=1.5, pad=10)` -- makes x-axis ticks tall and thick with extra padding.
- `ax.tick_params(axis='y', length=0, width=0)` -- hides y-axis ticks since the y-axis is not meaningful.
- `ax.set_yticks([])` -- removes y-axis tick labels.
- `ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])` -- places x-axis ticks at specific cosine similarity values.
- `ax.set_ylim(-0.55, 0.6)` -- sets the y-axis range to accommodate labels above and below.
- `ax.set_xlim(-1, 1)` -- sets the x-axis range to the full cosine similarity range.
- `ax.grid(False)` -- disables the grid.
- `ax.set_title(f'PC{pc_idx + 1}', ...)` -- titles each subplot "PC1", "PC2", or "PC3", bold and left-aligned.

```python
    plt.tight_layout()
    return fig
```

- `plt.tight_layout()` -- adjusts spacing between the 3 subplots.
- Returns the figure object.

---

## Code Cell 8 -- Compute Cosine Similarities and Plot

```python
# Get top 3 PC directions (normalized)
pc_directions = pca.components_[:3]
pc_directions = pc_directions / np.linalg.norm(pc_directions, axis=1, keepdims=True)
```

- `pca.components_[:3]` -- extracts the first 3 principal component vectors from the fitted PCA object. Each row is a direction in the 4608-dimensional space.
- `pc_directions / np.linalg.norm(pc_directions, axis=1, keepdims=True)` -- normalizes each PC direction to unit length. `np.linalg.norm(..., axis=1, keepdims=True)` computes the L2 norm of each row and keeps the result as a column vector for broadcasting. (PCA components from scikit-learn are typically already unit-normalized, but this ensures it.)

```python
# Scale and normalize role vectors
role_vectors_scaled = scaler.transform(role_vectors_at_layer.numpy())
role_vectors_norm = role_vectors_scaled / np.linalg.norm(role_vectors_scaled, axis=1, keepdims=True)
```

- `scaler.transform(role_vectors_at_layer.numpy())` -- converts the PyTorch tensor to a NumPy array and applies the fitted `MeanScaler`'s transform (subtracts the mean that was computed during `compute_pca`). This centers the role vectors.
- `role_vectors_norm` -- normalizes each mean-centered role vector to unit length. This is necessary so that the dot product with the PC direction equals the cosine similarity.

```python
# Compute cosine similarities with PC directions: (n_roles, 3)
role_cosine_sims = role_vectors_norm @ pc_directions.T
```

- `role_vectors_norm @ pc_directions.T` -- matrix multiplication of `(275, 4608)` by `(4608, 3)`, producing an `(275, 3)` array. Since both the role vectors and PC directions are unit-normalized, this dot product equals the cosine similarity between each role vector and each of the 3 PC directions.

```python
# Scale and normalize default vector
default_at_layer = default_vector[TARGET_LAYER].float().numpy().reshape(1, -1)
default_scaled = scaler.transform(default_at_layer)
default_norm = default_scaled / np.linalg.norm(default_scaled)
```

- `default_vector[TARGET_LAYER]` -- extracts the default vector at layer 22.
- `.float().numpy()` -- converts to float32 and then to a NumPy array.
- `.reshape(1, -1)` -- reshapes from `(4608,)` to `(1, 4608)` so it is a 2-D array compatible with the scaler's `transform` method.
- `scaler.transform(default_at_layer)` -- mean-centers the default vector using the same mean as the role vectors.
- `default_norm = default_scaled / np.linalg.norm(default_scaled)` -- normalizes the mean-centered default vector to unit length.

```python
# Compute default vector cosine similarities with PCs
default_cosine_sims = (default_norm @ pc_directions.T)[0]
```

- `default_norm @ pc_directions.T` -- computes the cosine similarity between the default vector and each of the 3 PC directions. The result is shape `(1, 3)`.
- `[0]` -- indexes into the first (only) row, yielding a 1-D array of 3 values.

```python
# Plot
fig = plot_pc_lines(
    role_cosine_sims,
    role_labels,
    default_cosine_sims=default_cosine_sims,
    figsize=(8, 7),
    n_extremes=5,
    show_histogram=True
)
plt.show()
```

- Calls `plot_pc_lines` with the computed cosine similarities for all roles and the default vector. The figure is 8x7 inches, labels the 5 most extreme roles on each end of each PC axis, and includes histogram overlays.
- `plt.show()` -- renders and displays the figure with the 3 PC subplots.
