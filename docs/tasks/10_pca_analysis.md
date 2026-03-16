# Task 10: PCA Analysis (pca.py)

## Overview

The `pca.py` module provides PCA (Principal Component Analysis) computation and visualization tools for analyzing the dimensionality and structure of model activation spaces. It contains:

- **`_to_numpy`** -- a private helper for converting tensors/arrays to NumPy.
- **`MeanScaler`** -- a scaler that mean-centers data.
- **`L2MeanScaler`** -- a scaler that mean-centers and then L2-normalizes data.
- **`compute_pca`** -- runs full PCA on activations (with optional layer selection and scaling).
- **`plot_variance_explained`** -- builds a Plotly figure showing individual and cumulative variance.

---

## Sub-Tasks

---

### Sub-Task 10.1: NumPy Conversion (`_to_numpy`)

#### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `np.ndarray`, `torch.Tensor`, or other | The object to convert. |

#### Processing

Checks the type of `x` and converts accordingly. Raises `TypeError` for unsupported types.

```python
def _to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x)}")
```

Steps:
1. If `x` is already an `np.ndarray`, return it unchanged.
2. If `x` is a `torch.Tensor`, detach from the computation graph, move to CPU, and convert to NumPy via `.numpy()`.
3. Otherwise, raise `TypeError`.

#### Output

| Return | Type | Description |
|--------|------|-------------|
| result | `np.ndarray` | NumPy array with the same data and shape as the input. |

---

### Sub-Task 10.2: Mean Centering (`MeanScaler`)

A scikit-learn-style scaler that centers data by subtracting the mean along all axes except the last (feature) dimension.

#### Sub-Task 10.2.1: Construction (`MeanScaler.__init__`)

##### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean` | `np.ndarray`, `torch.Tensor`, or `None` | `None` | Optional precomputed mean vector. If `None`, will be computed during `fit()`. |

##### Processing

```python
def __init__(self, mean=None):
    self.mean = mean
```

Stores the provided mean (or `None`) on the instance.

##### Output

A `MeanScaler` instance with `self.mean` set.

---

#### Sub-Task 10.2.2: Internal Mean Conversion (`MeanScaler._ensure_mean_numpy`)

##### Input

No explicit parameters -- operates on `self.mean`.

##### Processing

```python
def _ensure_mean_numpy(self):
    if self.mean is None:
        return
    if isinstance(self.mean, torch.Tensor):
        self.mean = self.mean.detach().cpu().numpy()
    elif not isinstance(self.mean, np.ndarray):
        self.mean = _to_numpy(self.mean)
```

Steps:
1. If `self.mean` is `None`, return immediately (nothing to convert).
2. If `self.mean` is a `torch.Tensor`, detach, move to CPU, and convert to NumPy in place.
3. If `self.mean` is neither `None` nor `np.ndarray`, delegate to `_to_numpy()`.

##### Output

Side effect: `self.mean` is guaranteed to be `np.ndarray` or `None` after this call.

---

#### Sub-Task 10.2.3: Fitting (`MeanScaler.fit`)

##### Input

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | `np.ndarray` or `torch.Tensor` | Arbitrary N-D, typically `(n_samples, hidden_dims)` or `(n_samples, n_layers, hidden_dims)` | Data to compute the mean from. |

##### Processing

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

Steps:
1. Convert `X` to NumPy via `_to_numpy`.
2. If no mean was provided at construction time (`self.mean is None`):
   - Build a tuple of all axes **except the last** (the feature dimension). For a 2D array of shape `(n_samples, hidden_dims)`, `axes = (0,)`. For a 3D array `axes = (0, 1)`.
   - Compute the mean over those axes with `keepdims=False`, producing a 1D vector of shape `(hidden_dims,)`.
3. If a mean was already provided, ensure it is converted to NumPy.
4. Return `self` (for chaining).

##### Output

| Return | Type | Description |
|--------|------|-------------|
| `self` | `MeanScaler` | The fitted scaler instance (with `self.mean` now set as `np.ndarray`). |

---

#### Sub-Task 10.2.4: Transform (`MeanScaler.transform`)

##### Input

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | `np.ndarray` or `torch.Tensor` | Same feature dimension as fitted data | Data to center. |

##### Processing

```python
def transform(self, X):
    """Subtract stored mean."""
    if self.mean is None:
        raise RuntimeError("MeanScaler not fitted: call .fit(X) or pass mean to ctor.")
    self._ensure_mean_numpy()
    X_np = _to_numpy(X)
    return X_np - self.mean
```

Steps:
1. Guard: if `self.mean` is `None`, raise `RuntimeError`.
2. Ensure mean is NumPy.
3. Convert `X` to NumPy.
4. Subtract `self.mean` via broadcasting (mean has shape `(hidden_dims,)`, subtracted from the last axis of `X_np`).

##### Output

| Return | Type | Shape | Description |
|--------|------|-------|-------------|
| result | `np.ndarray` | Same as input `X` | Mean-centered data. |

---

#### Sub-Task 10.2.5: Fit + Transform (`MeanScaler.fit_transform`)

##### Input

Same as `fit` / `transform`.

##### Processing

```python
def fit_transform(self, X):
    return self.fit(X).transform(X)
```

Calls `fit(X)` then `transform(X)` in sequence.

##### Output

Same as `transform`.

---

#### Sub-Task 10.2.6: Serialization (`MeanScaler.state_dict` / `load_state_dict`)

##### `state_dict()`

```python
def state_dict(self):
    self._ensure_mean_numpy()
    return {"mean": self.mean}
```

Returns a dictionary `{"mean": np.ndarray | None}` suitable for saving with `torch.save` or `np.savez`.

##### `load_state_dict(state)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | Must contain key `"mean"` (value: array-like or `None`). |

```python
def load_state_dict(self, state):
    self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
```

Restores the scaler's mean from a previously saved state dictionary.

---

### Sub-Task 10.3: L2-Normalized Mean Centering (`L2MeanScaler`)

A scaler that mean-centers data and then L2-normalizes each sample (row). Shares the same API shape as `MeanScaler` but adds normalization after centering.

#### Sub-Task 10.3.1: Construction (`L2MeanScaler.__init__`)

##### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean` | `np.ndarray`, `torch.Tensor`, or `None` | `None` | Optional precomputed mean vector. |
| `eps` | `float` | `1e-12` | Small epsilon to avoid division by zero during L2 normalization. |

##### Processing

```python
def __init__(self, mean=None, eps: float = 1e-12):
    self.mean = mean
    self.eps = eps
```

Stores the mean and epsilon on the instance.

##### Output

An `L2MeanScaler` instance.

---

#### Sub-Task 10.3.2: Internal Mean Conversion (`L2MeanScaler._ensure_mean_numpy`)

Identical logic to `MeanScaler._ensure_mean_numpy`.

```python
def _ensure_mean_numpy(self):
    if self.mean is None:
        return
    if isinstance(self.mean, torch.Tensor):
        self.mean = self.mean.detach().cpu().numpy()
    elif not isinstance(self.mean, np.ndarray):
        self.mean = _to_numpy(self.mean)
```

---

#### Sub-Task 10.3.3: Fitting (`L2MeanScaler.fit`)

##### Input

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | `np.ndarray` or `torch.Tensor` | N-D array | Data to compute the mean from. |

##### Processing

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

Same algorithm as `MeanScaler.fit`: computes the mean over all axes except the last if no precomputed mean was provided.

##### Output

| Return | Type | Description |
|--------|------|-------------|
| `self` | `L2MeanScaler` | The fitted scaler. |

---

#### Sub-Task 10.3.4: Transform (`L2MeanScaler.transform`)

##### Input

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `X` | `np.ndarray` or `torch.Tensor` | Same feature dimension as fitted data | Data to center and normalize. |

##### Processing

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

Steps:
1. Guard: raise `RuntimeError` if not fitted.
2. Ensure mean is NumPy.
3. Convert `X` to NumPy.
4. **Center**: subtract `self.mean` from `X_np` (broadcasting along the last axis).
5. **Compute L2 norms**: `np.linalg.norm` with `ord=2` along the last axis (`axis=-1`), keeping dimensions for broadcasting. Result shape: `(..., 1)`.
6. **Normalize**: divide centered data by `max(norm, eps)` to avoid division by zero. `np.maximum(norms, self.eps)` ensures no norm is smaller than `eps`.

##### Output

| Return | Type | Shape | Description |
|--------|------|-------|-------------|
| result | `np.ndarray` | Same as input `X` | Mean-centered, L2-normalized data. Each row (last-axis slice) has unit L2 norm (unless the original norm was below `eps`). |

---

#### Sub-Task 10.3.5: Fit + Transform (`L2MeanScaler.fit_transform`)

```python
def fit_transform(self, X):
    return self.fit(X).transform(X)
```

Calls `fit(X)` then `transform(X)` in sequence.

---

#### Sub-Task 10.3.6: Serialization (`L2MeanScaler.state_dict` / `load_state_dict`)

##### `state_dict()`

```python
def state_dict(self):
    self._ensure_mean_numpy()
    return {"mean": self.mean, "eps": self.eps}
```

Returns `{"mean": np.ndarray | None, "eps": float}`.

##### `load_state_dict(state)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | Must contain `"mean"` (array-like or `None`). Optionally contains `"eps"`. |

```python
def load_state_dict(self, state):
    self.mean = _to_numpy(state["mean"]) if state["mean"] is not None else None
    self.eps = float(state.get("eps", 1e-12))
```

Restores the scaler from a saved state. If `"eps"` is missing from the state dict, defaults to `1e-12`.

---

### Sub-Task 10.4: PCA Computation (`compute_pca`)

The main entry point for running PCA on activation tensors. Handles layer extraction, optional scaling, full PCA fitting, and diagnostic printing.

#### Input

| Parameter | Type | Shape / Values | Default | Description |
|-----------|------|----------------|---------|-------------|
| `activation_list` | `torch.Tensor` or `np.ndarray` | `(n_samples, n_layers, hidden_dims)` (3D) or `(n_samples, hidden_dims)` (2D) | *required* | The activation data. |
| `layer` | `int` or `None` | Integer index or `None` | *required* | Layer index to select when input is 3D. Must be `None` for 2D input. |
| `scaler` | `MeanScaler`, `L2MeanScaler`, callable, or `None` | -- | `None` | Optional preprocessing scaler. |
| `verbose` | `bool` | `True` / `False` | `True` | Whether to print PCA analysis results. |

#### Processing

The function proceeds in four stages:

**Stage 1: Layer Selection**

```python
# Select layer
if isinstance(activation_list, torch.Tensor):
    if activation_list.ndim == 3:
        if layer is None:
            raise ValueError("For 3D activation_list, provide a layer index.")
        layer_activations = activation_list[:, layer, :]
    elif activation_list.ndim == 2:
        layer_activations = activation_list
    else:
        raise ValueError("activation_list must be 2D or 3D")
elif isinstance(activation_list, np.ndarray):
    if activation_list.ndim == 3:
        if layer is None:
            raise ValueError("For 3D activation_list, provide a layer index.")
        layer_activations = activation_list[:, layer, :]
    elif activation_list.ndim == 2:
        layer_activations = activation_list
    else:
        raise ValueError("activation_list must be 2D or 3D")
else:
    raise TypeError("activation_list must be torch.Tensor or np.ndarray")
```

Steps:
1. Validate that `activation_list` is a `torch.Tensor` or `np.ndarray`.
2. If 3D: require `layer` to be an integer, then slice `activation_list[:, layer, :]` to get a 2D matrix of shape `(n_samples, hidden_dims)`.
3. If 2D: use the data directly.
4. If neither 2D nor 3D, or an unsupported type: raise an error.

**Stage 2: Scaling**

```python
# Scale if requested
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

Steps:
1. If `scaler is None`: pass data through unmodified, set `fitted_scaler = None`.
2. If scaler has `fit_transform`: call it directly (covers `MeanScaler`, `L2MeanScaler`, sklearn scalers).
3. Else if scaler has both `fit` and `transform`: call `fit()` then `transform()` separately.
4. Else if scaler is callable (a plain function): call it on the data, set `fitted_scaler = None`.
5. Otherwise: raise `TypeError`.

**Stage 3: PCA Fitting**

```python
X_np = _to_numpy(scaled)
pca = PCA()
pca_transformed = pca.fit_transform(X_np)

variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)
n_components = len(variance_explained)
```

Steps:
1. Convert scaled data to NumPy.
2. Create a `sklearn.decomposition.PCA()` instance with **no component limit** (fits all `min(n_samples, hidden_dims)` components).
3. Fit and transform the data, producing `pca_transformed` of shape `(n_samples, n_components)`.
4. Extract `explained_variance_ratio_` -- a 1D array of shape `(n_components,)` summing to ~1.0.
5. Compute cumulative variance via `np.cumsum`.

**Stage 4: Verbose Diagnostics**

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

Steps (only when `verbose=True`):
1. Print total component count and cumulative variance for the first 5 components.
2. **Elbow point detection** via the `find_elbow_point` inner function:
   - Compute `first_diff = np.diff(variance_explained)` -- the first discrete derivative.
   - Compute `second_diff = np.diff(first_diff)` -- the second discrete derivative.
   - The elbow is at the index of the maximum absolute second derivative, plus 1 (to account for the index shift introduced by `np.diff`).
3. **Threshold dimensions**: for each of 70%, 80%, 90%, 95%, find the smallest number of components whose cumulative variance reaches that threshold using `np.argmax(cumulative_variance >= threshold) + 1`.
4. Print all results.

#### Output

| Return | Type | Shape | Description |
|--------|------|-------|-------------|
| `pca_transformed` | `np.ndarray` | `(n_samples, n_components)` | Data projected onto principal components. |
| `variance_explained` | `np.ndarray` | `(n_components,)` | Fraction of variance explained by each component. |
| `n_components` | `int` | scalar | Total number of components (equals `min(n_samples, hidden_dims)`). |
| `pca` | `sklearn.decomposition.PCA` | -- | The fitted PCA object (contains `components_`, `mean_`, etc.). |
| `fitted_scaler` | `MeanScaler`, `L2MeanScaler`, or `None` | -- | The fitted scaler instance if one was used, otherwise `None`. |

Returned as a 5-tuple: `(pca_transformed, variance_explained, n_components, pca, fitted_scaler)`.

---

### Sub-Task 10.5: Variance Visualization (`plot_variance_explained`)

Builds an interactive Plotly figure with a bar chart (individual variance per component) and a line chart (cumulative variance), plus optional threshold annotations.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variance_explained_or_dict` | `np.ndarray`, `torch.Tensor`, or `dict` | *required* | Either an array of variance ratios, or a dict with key `"variance_explained"`. |
| `title` | `str` | `"PCA Variance Explained"` | Main plot title. |
| `subtitle` | `str` | `""` | Subtitle displayed below the title. |
| `show_thresholds` | `bool` | `True` | Whether to draw horizontal dashed lines at 70%, 80%, 90%, 95%. |
| `max_components` | `int` or `None` | `None` | If set, truncates display to the first N components. |

#### Processing

**Stage 1: Input Normalization**

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

Steps:
1. If a dict is passed, extract the `"variance_explained"` value.
2. If the value is a `torch.Tensor`, convert to NumPy.
3. Ensure the result is a float NumPy array via `np.asarray`.
4. Compute cumulative variance.

**Stage 2: Component Truncation**

```python
if max_components is not None:
    n_components = min(n_components, max_components)
    variance_explained = variance_explained[:n_components]
    cumulative_variance = cumulative_variance[:n_components]

component_numbers = np.arange(1, n_components + 1)
```

If `max_components` is specified, slice both arrays to that length. Generate 1-based component indices.

**Stage 3: Bar + Line Traces**

```python
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=component_numbers,
        y=variance_explained * 100,
        name="Individual Variance",
        opacity=0.6
    )
)

fig.add_trace(
    go.Scatter(
        x=component_numbers,
        y=cumulative_variance * 100,
        mode="lines+markers",
        name="Cumulative Variance"
    )
)
```

Steps:
1. Create a Plotly `Figure`.
2. Add a **bar trace** for individual variance per component (y-axis in percentage).
3. Add a **scatter trace** (lines + markers) for cumulative variance (y-axis in percentage).

**Stage 4: Threshold Annotations**

```python
max_y = float(np.max([np.max(variance_explained), np.max(cumulative_variance)]) * 100)
nice_top = np.ceil(max(max_y, 100) / 5) * 5

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

Steps:
1. Compute a "nice" y-axis ceiling rounded up to the nearest multiple of 5 (at least 100).
2. For each threshold (70, 80, 90, 95):
   - Find the first component index where cumulative variance reaches that threshold.
   - Verify the threshold is actually reached (guards against data that never reaches 95%, etc.).
   - Draw a horizontal dashed line at that y-value.
   - Add a right-aligned text annotation showing the threshold and the number of dimensions required, shifted 10px below the line.

**Stage 5: Layout Configuration**

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

Sets titles, axis labels, figure size (800x600), horizontal legend above the plot, unified hover mode, and y-axis range from 0 to the computed ceiling.

#### Output

| Return | Type | Description |
|--------|------|-------------|
| `fig` | `plotly.graph_objects.Figure` | Interactive Plotly figure ready for `fig.show()` or `fig.write_html()`. |
