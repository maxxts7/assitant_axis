# `assistant_axis/pca.py`

## Overview

This file provides PCA (Principal Component Analysis) utilities for analyzing model activations. It contains:

- A helper function to convert tensors/arrays to NumPy arrays (`_to_numpy`)
- Two scaler classes for preprocessing data before PCA (`MeanScaler` and `L2MeanScaler`)
- A function to compute PCA on activation data (`compute_pca`)
- A function to plot explained variance using Plotly (`plot_variance_explained`)

These tools are designed for inspecting the dimensionality and structure of neural network activation spaces.

---

## Line-by-line explanation

### Module docstring (lines 1-13)

```python
"""
PCA utilities for analyzing model activations.

This module provides PCA computation and visualization tools for
analyzing the dimensionality and structure of activation spaces.

Example:
    from assistant_axis import compute_pca, plot_variance_explained

    activations = torch.load("activations.pt")
    pca_result, variance, n_components, pca, scaler = compute_pca(activations, layer=22)
    fig = plot_variance_explained(variance)
"""
```

The module-level docstring describes the purpose of the file and provides a short usage example. It shows how to load activations from disk, run PCA on a specific layer, and then plot the explained variance.

---

### Imports (lines 15-18)

```python
import numpy as np
from sklearn.decomposition import PCA
import torch
import plotly.graph_objects as go
```

- `numpy` is used for numerical array operations throughout the module.
- `PCA` from scikit-learn is the actual PCA implementation used.
- `torch` is imported so the code can accept PyTorch tensors as input and convert them to NumPy arrays.
- `plotly.graph_objects` is used to create interactive variance-explained charts.

---

### `_to_numpy` helper (lines 21-27)

```python
def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")
```

- **Line 21-22**: Defines a private helper function with a docstring.
- **Lines 23-24**: If `x` is already a NumPy array, return it unchanged.
- **Lines 25-26**: If `x` is a PyTorch tensor, detach it from the computation graph (so gradients are not tracked), move it to CPU memory (in case it was on a GPU), and convert it to a NumPy array.
- **Line 27**: If `x` is neither type, raise a `TypeError` with a descriptive message.

---

### `MeanScaler` class (lines 30-76)

#### Class definition and docstring (lines 30-31)

```python
class MeanScaler:
    """Scaler that centers data by subtracting the mean."""
```

Defines a scaler that mean-centers data. This is a common preprocessing step before PCA -- subtracting the mean ensures PCA captures variance rather than the location of the data.

#### `__init__` (lines 33-39)

```python
    def __init__(self, mean=None):
        """
        Args:
            mean: Optional precomputed mean as numpy array or torch tensor.
                  If None, will be computed during fit().
        """
        self.mean = mean
```

The constructor accepts an optional precomputed mean. If no mean is given (`None`), it will be computed later when `fit()` is called. This allows reusing a previously computed mean on new data.

#### `_ensure_mean_numpy` (lines 41-47)

```python
    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)
```

- **Line 42-43**: If no mean has been set, do nothing.
- **Lines 44-45**: If the mean is a PyTorch tensor, convert it to a NumPy array in place (detach, move to CPU, convert).
- **Lines 46-47**: If the mean is neither a tensor nor already a NumPy array, use the `_to_numpy` helper to convert it (which will raise `TypeError` for unsupported types).

#### `fit` (lines 49-57)

```python
    def fit(self, X):
        """Compute mean from X if not provided."""
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self
```

- **Line 51**: Converts the input `X` to a NumPy array.
- **Lines 52-54**: If no mean was provided at construction, compute it. `axes` is a tuple of all axes except the last one. For a 2D array of shape `(n_samples, hidden_dims)`, this computes the mean over `axis=0`, producing a 1D array of shape `(hidden_dims,)`. `keepdims=False` ensures the result does not retain the reduced dimensions.
- **Lines 55-56**: If a mean was provided, ensure it is in NumPy format.
- **Line 57**: Returns `self` to allow method chaining (e.g., `scaler.fit(X).transform(X)`).

#### `transform` (lines 59-65)

```python
    def transform(self, X):
        """Subtract stored mean."""
        if self.mean is None:
            raise RuntimeError("MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        return X_np - self.mean
```

- **Lines 61-62**: Guards against using `transform` before fitting. Raises a `RuntimeError` if the mean has not been computed or provided.
- **Line 63**: Ensures the stored mean is a NumPy array.
- **Line 64**: Converts the input to NumPy.
- **Line 65**: Subtracts the mean from every sample via broadcasting and returns the centered data.

#### `fit_transform` (lines 67-68)

```python
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

A convenience method that calls `fit` and then `transform` in sequence. This is the standard scikit-learn-style API.

#### `state_dict` (lines 70-72)

```python
    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean}
```

Returns the scaler's internal state as a dictionary. This follows the PyTorch convention of `state_dict()` for serialization. The mean is ensured to be NumPy before returning.

#### `load_state_dict` (lines 74-75)

```python
    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
```

Restores the scaler's state from a dictionary previously produced by `state_dict()`. If the stored mean is `None`, it remains `None`; otherwise it is converted to NumPy.

---

### `L2MeanScaler` class (lines 78-127)

#### Class definition and docstring (lines 78-79)

```python
class L2MeanScaler:
    """Scaler that centers data and L2-normalizes."""
```

Similar to `MeanScaler`, but after subtracting the mean it also L2-normalizes each sample. This makes every sample a unit vector after centering, which can be useful when you care about direction rather than magnitude.

#### `__init__` (lines 81-88)

```python
    def __init__(self, mean=None, eps: float = 1e-12):
        """
        Args:
            mean: Optional precomputed mean.
            eps: Small value to avoid division by zero.
        """
        self.mean = mean
        self.eps = eps
```

Like `MeanScaler.__init__`, but adds an `eps` parameter. This small epsilon value prevents division by zero during L2 normalization when a centered vector has zero (or near-zero) norm.

#### `_ensure_mean_numpy` (lines 90-96)

```python
    def _ensure_mean_numpy(self):
        if self.mean is None:
            return
        if isinstance(self.mean, torch.Tensor):
            self.mean = self.mean.detach().cpu().numpy()
        elif not isinstance(self.mean, np.ndarray):
            self.mean = _to_numpy(self.mean)
```

Identical to `MeanScaler._ensure_mean_numpy`. Ensures the stored mean is a NumPy array.

#### `fit` (lines 98-106)

```python
    def fit(self, X):
        """Compute mean from X if not provided."""
        X_np = _to_numpy(X)
        if self.mean is None:
            axes = tuple(range(X_np.ndim - 1))
            self.mean = X_np.mean(axis=axes, keepdims=False)
        else:
            self._ensure_mean_numpy()
        return self
```

Identical to `MeanScaler.fit`. Computes the mean over all axes except the last (feature) dimension if no precomputed mean was supplied.

#### `transform` (lines 108-116)

```python
    def transform(self, X):
        """Subtract stored mean and L2-normalize."""
        if self.mean is None:
            raise RuntimeError("L2MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
        self._ensure_mean_numpy()
        X_np = _to_numpy(X)
        X_centered = X_np - self.mean
        norms = np.linalg.norm(X_centered, ord=2, axis=-1, keepdims=True)
        return X_centered / np.maximum(norms, self.eps)
```

- **Lines 110-111**: Guards against using an unfitted scaler.
- **Line 112**: Ensures the mean is in NumPy format.
- **Line 113**: Converts input to NumPy.
- **Line 114**: Centers the data by subtracting the mean.
- **Line 115**: Computes the L2 (Euclidean) norm of each sample along the last axis. `keepdims=True` keeps the result broadcastable against `X_centered`.
- **Line 116**: Divides each centered sample by its norm. `np.maximum(norms, self.eps)` clamps the norm to at least `eps` to prevent division by zero.

#### `fit_transform` (lines 118-119)

```python
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

Convenience method combining `fit` and `transform`.

#### `state_dict` (lines 121-123)

```python
    def state_dict(self):
        self._ensure_mean_numpy()
        return {"mean": self.mean, "eps": self.eps}
```

Returns the scaler's state, including both the mean and the epsilon value.

#### `load_state_dict` (lines 125-127)

```python
    def load_state_dict(self, state):
        self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
        self.eps = float(state.get("eps", 1e-12))
```

Restores state from a dictionary. Uses `dict.get` with a default of `1e-12` for `eps` to maintain backward compatibility with state dicts that may not include it.

---

### `compute_pca` function (lines 130-213)

#### Signature and docstring (lines 130-143)

```python
def compute_pca(activation_list, layer: int | None, scaler=None, verbose: bool = True):
    """
    Compute PCA on activations.

    Args:
        activation_list: torch.Tensor or np.ndarray of shape (n_samples, n_layers, hidden_dims)
                        or (n_samples, hidden_dims)
        layer: Layer index for 3D input, None for 2D
        scaler: Optional scaler with fit_transform() or fit()/transform() methods
        verbose: Whether to print analysis results

    Returns:
        Tuple of (pca_transformed, variance_explained, n_components, pca, fitted_scaler)
    """
```

The main PCA computation function. It accepts activation data (either 2D or 3D), an optional layer index, an optional scaler, and a verbosity flag. It returns a 5-tuple containing the transformed data, variance ratios, component count, the fitted PCA object, and the fitted scaler.

#### Layer selection for torch.Tensor (lines 145-153)

```python
    if isinstance(activation_list, torch.Tensor):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
```

- **Line 145**: Checks if the input is a PyTorch tensor.
- **Lines 146-149**: For 3D tensors (shape `(n_samples, n_layers, hidden_dims)`), requires a `layer` index and selects that layer's activations across all samples, resulting in a 2D slice of shape `(n_samples, hidden_dims)`.
- **Lines 150-151**: For 2D tensors (shape `(n_samples, hidden_dims)`), uses the input directly.
- **Lines 152-153**: Rejects any other dimensionality.

#### Layer selection for np.ndarray (lines 154-162)

```python
    elif isinstance(activation_list, np.ndarray):
        if activation_list.ndim == 3:
            if layer is None:
                raise ValueError("For 3D activation_list, provide a layer index.")
            layer_activations = activation_list[:, layer, :]
        elif activation_list.ndim == 2:
            layer_activations = activation_list
        else:
            raise ValueError("activation_list must be 2D or 3D")
```

Same logic as the PyTorch branch above, but for NumPy arrays.

#### Type rejection (lines 163-164)

```python
    else:
        raise TypeError("activation_list must be torch.Tensor or np.ndarray")
```

If `activation_list` is neither a tensor nor an ndarray, raise a `TypeError`.

#### Scaling (lines 166-181)

```python
    if scaler is None:
        scaled = layer_activations
        fitted_scaler = None
    else:
        if hasattr(scaler, "fit_transform"):
            scaled = scaler.fit_transform(layer_activations)
            fitted_scaler = scaler
        elif hasattr(scaler, "transform") and hasattr(scaler, "fit"):
            fitted_scaler = scaler.fit(layer_activations)
            scaled = fitted_scaler.transform(layer_activations)
        elif callable(scaler):
            scaled = scaler(layer_activations)
            fitted_scaler = None
        else:
            raise TypeError("scaler must be None, callable, or have fit/transform or fit_transform")
```

- **Lines 167-169**: If no scaler is provided, use the raw activations.
- **Lines 171-173**: If the scaler has a `fit_transform` method (like `MeanScaler` or scikit-learn scalers), use it. The scaler object is retained as `fitted_scaler` so it can be returned and reused.
- **Lines 174-176**: If the scaler has separate `fit` and `transform` methods but not `fit_transform`, call them individually.
- **Lines 177-179**: If the scaler is a plain callable (e.g., a lambda or function), call it directly on the data. No fitted scaler is retained in this case.
- **Lines 180-181**: Reject anything else.

#### PCA computation (lines 183-189)

```python
    X_np = _to_numpy(scaled)
    pca = PCA()
    pca_transformed = pca.fit_transform(X_np)

    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)
```

- **Line 183**: Converts the (possibly scaled) data to a NumPy array.
- **Line 184**: Creates a scikit-learn `PCA` object with no component limit (uses all components).
- **Line 185**: Fits the PCA model and transforms the data in one call. `pca_transformed` has the same shape as the input but in the PCA coordinate system.
- **Line 187**: Extracts the ratio of variance explained by each principal component.
- **Line 188**: Computes the cumulative sum of explained variance ratios.
- **Line 189**: The total number of components equals the number of variance ratios.

#### Verbose output (lines 191-211)

```python
    if verbose:
        print(f"PCA fitted with {n_components} components")
        print(f"Cumulative variance for first 5 components: {cumulative_variance[:5]}")

        def find_elbow_point(variance_explained):
            first_diff = np.diff(variance_explained)
            second_diff = np.diff(first_diff)
            return np.argmax(np.abs(second_diff)) + 1

        elbow_point = find_elbow_point(variance_explained)
        dims_70 = np.argmax(cumulative_variance >= 0.70) + 1
        dims_80 = np.argmax(cumulative_variance >= 0.80) + 1
        dims_90 = np.argmax(cumulative_variance >= 0.90) + 1
        dims_95 = np.argmax(cumulative_variance >= 0.95) + 1

        print("\nPCA Analysis Results:")
        print(f"Elbow point at component: {elbow_point + 1}")
        print(f"Dimensions for 70% variance: {dims_70}")
        print(f"Dimensions for 80% variance: {dims_80}")
        print(f"Dimensions for 90% variance: {dims_90}")
        print(f"Dimensions for 95% variance: {dims_95}")
```

- **Lines 192-193**: Prints the total number of components and the cumulative variance of the first five.
- **Lines 195-198**: Defines a local function `find_elbow_point` that finds the "elbow" in the variance curve. It computes the first derivative (`np.diff`), then the second derivative, and returns the index where the absolute second derivative is largest. This indicates where the rate of change in explained variance shifts most dramatically.
- **Line 200**: Calls `find_elbow_point` on the variance ratios.
- **Lines 201-204**: Finds how many dimensions are needed to reach 70%, 80%, 90%, and 95% of total variance. `np.argmax(cumulative_variance >= threshold)` returns the first index where the condition is true; adding 1 converts from zero-based index to a component count.
- **Lines 206-211**: Prints all the analysis results.

#### Return (line 213)

```python
    return pca_transformed, variance_explained, n_components, pca, fitted_scaler
```

Returns a 5-tuple:
1. `pca_transformed` -- the data projected into PCA space
2. `variance_explained` -- per-component variance ratios
3. `n_components` -- total number of components
4. `pca` -- the fitted scikit-learn `PCA` object (can be reused to transform new data)
5. `fitted_scaler` -- the fitted scaler object (or `None`)

---

### `plot_variance_explained` function (lines 216-306)

#### Signature and docstring (lines 216-235)

```python
def plot_variance_explained(
    variance_explained_or_dict,
    title="PCA Variance Explained",
    subtitle="",
    show_thresholds=True,
    max_components=None
):
    """
    Plot PCA variance explained (individual + cumulative).

    Args:
        variance_explained_or_dict: Array of variance ratios or dict with "variance_explained" key
        title: Plot title
        subtitle: Plot subtitle
        show_thresholds: Whether to show threshold lines (70%, 80%, 90%, 95%)
        max_components: Maximum number of components to show

    Returns:
        Plotly Figure object
    """
```

This function creates an interactive Plotly chart showing individual and cumulative explained variance per principal component.

#### Input handling (lines 236-246)

```python
    if isinstance(variance_explained_or_dict, dict):
        variance_explained = variance_explained_or_dict["variance_explained"]
    else:
        variance_explained = variance_explained_or_dict

    if isinstance(variance_explained, torch.Tensor):
        variance_explained = variance_explained.detach().cpu().numpy()

    variance_explained = np.asarray(variance_explained, dtype=float)
    cumulative_variance = np.cumsum(variance_explained)
    n_components = len(variance_explained)
```

- **Lines 236-239**: Accepts either a raw array/tensor or a dict containing a `"variance_explained"` key (for flexibility with different calling conventions).
- **Lines 241-242**: If the data is a PyTorch tensor, convert it to NumPy.
- **Line 244**: Ensures the data is a float NumPy array regardless of input type.
- **Line 245**: Computes cumulative variance.
- **Line 246**: Gets the total component count.

#### Component limiting (lines 248-251)

```python
    if max_components is not None:
        n_components = min(n_components, max_components)
        variance_explained = variance_explained[:n_components]
        cumulative_variance = cumulative_variance[:n_components]
```

If `max_components` is specified, truncates the data to show only that many components. This is useful when there are hundreds of components but you only want to visualize the first few.

#### X-axis values (line 253)

```python
    component_numbers = np.arange(1, n_components + 1)
```

Creates an array `[1, 2, 3, ..., n_components]` for labeling the x-axis (1-indexed component numbers).

#### Figure creation (line 255)

```python
    fig = go.Figure()
```

Creates an empty Plotly figure object.

#### Bar chart for individual variance (lines 257-264)

```python
    fig.add_trace(
        go.Bar(
            x=component_numbers,
            y=variance_explained * 100,
            name="Individual Variance",
            opacity=0.6
        )
    )
```

Adds a bar chart showing the variance explained by each individual component as a percentage. The bars are semi-transparent (`opacity=0.6`) so the cumulative line is visible behind them.

#### Line chart for cumulative variance (lines 266-273)

```python
    fig.add_trace(
        go.Scatter(
            x=component_numbers,
            y=cumulative_variance * 100,
            mode="lines+markers",
            name="Cumulative Variance"
        )
    )
```

Adds a line-and-marker plot showing the running total of explained variance. This makes it easy to see how many components are needed to reach a given threshold.

#### Y-axis range calculation (lines 275-276)

```python
    max_y = float(np.max([np.max(variance_explained), np.max(cumulative_variance)]) * 100)
    nice_top = np.ceil(max(max_y, 100) / 5) * 5
```

- **Line 275**: Finds the maximum y-value across both traces (as a percentage).
- **Line 276**: Rounds up to the nearest multiple of 5, with a minimum of 100%. This ensures the y-axis always includes the 100% mark and has a clean upper bound.

#### Threshold lines (lines 278-291)

```python
    if show_thresholds and n_components > 0:
        thresholds = [70, 80, 90, 95]
        for thr in thresholds:
            idx = np.argmax(cumulative_variance >= thr / 100.0)
            if cumulative_variance[idx] >= thr / 100.0:
                n_dims = idx + 1
                fig.add_hline(y=thr, line_dash="dash", line_width=1, opacity=0.5)
                fig.add_annotation(
                    x=0.995, xref="paper", xanchor="right",
                    y=thr, yref="y", yshift=-10,
                    text=f"{thr}% ({n_dims} dims)",
                    showarrow=False, align="right",
                    font=dict(size=10, color="gray")
                )
```

- **Line 278**: Only draws thresholds if enabled and there is data.
- **Line 279**: The four standard variance thresholds to mark.
- **Line 281**: Finds the first component index where cumulative variance meets or exceeds the threshold.
- **Line 282**: Verifies the threshold is actually reached (it might not be if total variance is less, though in standard PCA it always sums to 1.0).
- **Line 283**: Converts zero-based index to a 1-based component count.
- **Line 284**: Adds a horizontal dashed line at the threshold percentage.
- **Lines 285-291**: Adds a text annotation at the right edge of the plot, shifted slightly below the threshold line, showing the threshold percentage and how many dimensions are needed to reach it.

#### Layout configuration (lines 293-304)

```python
    fig.update_layout(
        title={"text": title, "subtitle": {"text": subtitle}},
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        hovermode="x unified",
        width=800,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=120)
    )
    fig.update_yaxes(range=[0, nice_top])
```

- **Line 294**: Sets the chart title and subtitle.
- **Line 295-296**: Labels the axes.
- **Line 297**: Uses "x unified" hover mode so hovering shows values for all traces at that x position.
- **Lines 298-299**: Sets the figure dimensions to 800x600 pixels.
- **Lines 300-301**: Places the legend horizontally above the chart.
- **Line 302**: Adds top margin to make room for the legend.
- **Line 304**: Fixes the y-axis to start at 0 and end at the computed `nice_top` value.

#### Return (line 306)

```python
    return fig
```

Returns the fully constructed Plotly `Figure` object. The caller can then display it with `fig.show()`, save it with `fig.write_html()`, or further customize it.
