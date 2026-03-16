# `assistant_axis/tests/test_axis.py`

## Overview

This file contains the test suite for the axis computation utilities defined in `assistant_axis.axis`. It uses **pytest** as the testing framework and **PyTorch** for tensor operations. The tests cover six core functions: `compute_axis`, `project`, `project_batch`, `save_axis`, `load_axis`, `cosine_similarity_per_layer`, and `axis_norm_per_layer`. Each function has its own test class with multiple test methods that verify correctness, output shape, edge cases, and type guarantees.

---

## Line-by-Line Explanation

### Lines 1-3: Module Docstring

```python
"""
Tests for axis computation utilities.
"""
```

A module-level docstring that briefly describes the purpose of the file: it contains tests for the axis computation utilities.

---

### Lines 5-7: Imports

```python
import tempfile
import pytest
import torch
```

- `tempfile` -- from the Python standard library. Used later to create temporary files for testing the save/load functionality without polluting the filesystem.
- `pytest` -- the test framework. Although not invoked explicitly in the code, importing it makes pytest fixtures, markers, and other features available if needed. pytest also uses this import to identify the file as a test module.
- `torch` -- PyTorch, the deep learning library. Used to create tensors, perform mathematical operations, and validate tensor properties throughout all test methods.

---

### Lines 9-17: Importing Functions Under Test

```python
from assistant_axis.axis import (
    compute_axis,
    project,
    project_batch,
    save_axis,
    load_axis,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
)
```

Imports seven functions from the `assistant_axis.axis` module. These are the functions being tested:

- `compute_axis` -- computes a direction vector (axis) from two sets of activations.
- `project` -- projects a single activation tensor onto an axis at a given layer.
- `project_batch` -- projects a batch of activations onto an axis at a given layer.
- `save_axis` -- serializes an axis tensor to disk.
- `load_axis` -- deserializes an axis tensor from disk.
- `cosine_similarity_per_layer` -- computes cosine similarity between two tensors for each layer.
- `axis_norm_per_layer` -- computes the L2 norm of an axis tensor for each layer.

---

### Lines 20-21: `TestComputeAxis` Class

```python
class TestComputeAxis:
    """Tests for compute_axis function."""
```

Defines a test class grouping all tests for the `compute_axis` function. pytest discovers classes prefixed with `Test` and runs any methods within them that start with `test_`.

---

### Lines 23-33: `test_basic_computation`

```python
    def test_basic_computation(self):
        """Axis should be default - role activations."""
        n_layers, hidden_dim = 4, 8
        role_acts = torch.randn(10, n_layers, hidden_dim)
        default_acts = torch.randn(10, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        expected = default_acts.mean(dim=0) - role_acts.mean(dim=0)
        assert axis.shape == (n_layers, hidden_dim)
        assert torch.allclose(axis, expected)
```

- **Line 23-24**: Method definition and docstring. The axis should equal the mean of the default activations minus the mean of the role activations.
- **Line 25**: Sets up dimensions: 4 layers, hidden dimension of 8.
- **Lines 26-27**: Creates two random tensors of shape `(10, 4, 8)` -- 10 samples, 4 layers, 8 hidden dimensions. `role_acts` represents activations when the model plays a specific role; `default_acts` represents activations under default behavior.
- **Line 29**: Calls `compute_axis` with both activation sets and stores the result.
- **Line 31**: Manually computes the expected result: take the mean across the sample dimension (dim=0) for each set, then subtract the role mean from the default mean. This gives a tensor of shape `(4, 8)`.
- **Line 32**: Asserts the output shape is `(n_layers, hidden_dim)`, i.e., `(4, 8)`.
- **Line 33**: Asserts the computed axis is numerically close to the expected value using `torch.allclose`, which allows for small floating-point differences.

---

### Lines 35-43: `test_output_shape`

```python
    def test_output_shape(self):
        """Output shape should be (n_layers, hidden_dim)."""
        n_layers, hidden_dim = 32, 4096
        role_acts = torch.randn(50, n_layers, hidden_dim)
        default_acts = torch.randn(20, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)
```

- **Lines 37-38**: Uses larger, more realistic dimensions (32 layers, 4096 hidden dim) and different sample counts (50 vs 20) to verify that the function handles realistic model sizes.
- **Line 40**: Calls `compute_axis`.
- **Line 42-43**: Asserts the output shape collapses the sample dimension, yielding `(32, 4096)`.

---

### Lines 45-53: `test_different_sample_counts`

```python
    def test_different_sample_counts(self):
        """Should work with different numbers of samples."""
        n_layers, hidden_dim = 4, 16
        role_acts = torch.randn(100, n_layers, hidden_dim)
        default_acts = torch.randn(5, n_layers, hidden_dim)

        axis = compute_axis(role_acts, default_acts)

        assert axis.shape == (n_layers, hidden_dim)
```

- Tests that `compute_axis` works when the two input tensors have very different numbers of samples (100 vs 5). Since each set is independently averaged along the sample dimension, mismatched sample counts should not cause errors.
- Asserts the shape is still correct.

---

### Lines 56-57: `TestProject` Class

```python
class TestProject:
    """Tests for project function."""
```

Groups tests for the `project` function, which computes the scalar projection of an activation vector onto a normalized axis vector at a specific layer.

---

### Lines 59-74: `test_basic_projection`

```python
    def test_basic_projection(self):
        """Projection should be dot product with normalized axis."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        result = project(activations, axis, layer)

        # Manual calculation
        act = activations[layer].float()
        ax = axis[layer].float()
        ax_normalized = ax / ax.norm()
        expected = float(act @ ax_normalized)

        assert abs(result - expected) < 1e-5
```

- **Lines 61-63**: Creates random activations and axis tensors of shape `(4, 8)`, and selects layer index 2.
- **Line 65**: Calls `project` to get the scalar projection of the activation at layer 2 onto the axis at layer 2.
- **Lines 68-72**: Manually replicates the expected computation:
  - Extracts the activation vector at the selected layer and casts to float32.
  - Extracts the axis vector at the selected layer and casts to float32.
  - Normalizes the axis vector by dividing by its L2 norm.
  - Computes the dot product (`@` operator) between the activation and the normalized axis, then converts to a Python float.
- **Line 74**: Asserts the result matches the expected value within a tolerance of `1e-5`.

---

### Lines 76-86: `test_unnormalized`

```python
    def test_unnormalized(self):
        """With normalize=False, should be raw dot product."""
        n_layers, hidden_dim = 4, 8
        activations = torch.randn(n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 1

        result = project(activations, axis, layer, normalize=False)

        expected = float(activations[layer].float() @ axis[layer].float())
        assert abs(result - expected) < 1e-5
```

- Tests the `normalize=False` option of the `project` function.
- **Line 83**: Passes `normalize=False` to skip axis normalization.
- **Line 85**: The expected value is simply the raw dot product between the activation and axis vectors at layer 1, without normalizing the axis first.
- **Line 86**: Asserts the result is close to the raw dot product.

---

### Lines 88-95: `test_returns_float`

```python
    def test_returns_float(self):
        """Should return a Python float."""
        activations = torch.randn(4, 8)
        axis = torch.randn(4, 8)

        result = project(activations, axis, layer=0)

        assert isinstance(result, float)
```

- Verifies that `project` returns a native Python `float`, not a PyTorch scalar tensor. This is important for downstream code that may expect a plain float (e.g., for JSON serialization or logging).

---

### Lines 98-99: `TestProjectBatch` Class

```python
class TestProjectBatch:
    """Tests for project_batch function."""
```

Groups tests for `project_batch`, which applies the projection operation across a batch of activation samples.

---

### Lines 101-115: `test_batch_projection`

```python
    def test_batch_projection(self):
        """Should project multiple activations."""
        n_samples, n_layers, hidden_dim = 10, 4, 8
        activations = torch.randn(n_samples, n_layers, hidden_dim)
        axis = torch.randn(n_layers, hidden_dim)
        layer = 2

        results = project_batch(activations, axis, layer)

        assert len(results) == n_samples

        # Verify each result matches individual projection
        for i, result in enumerate(results):
            expected = project(activations[i], axis, layer)
            assert abs(result - expected) < 1e-5
```

- **Lines 103-106**: Creates a batch of 10 activation samples, each with 4 layers and hidden dimension 8, plus a single axis tensor and a target layer.
- **Line 108**: Calls `project_batch`, which should return one scalar projection per sample.
- **Line 110**: Asserts the number of results equals the number of samples (10).
- **Lines 113-115**: Iterates over every result and compares it against calling `project` individually on each sample. This confirms `project_batch` is equivalent to applying `project` in a loop -- it is a correctness check, not a performance test.

---

### Lines 118-119: `TestSaveLoadAxis` Class

```python
class TestSaveLoadAxis:
    """Tests for save_axis and load_axis functions."""
```

Groups tests for the serialization/deserialization pair `save_axis` and `load_axis`.

---

### Lines 121-129: `test_save_and_load`

```python
    def test_save_and_load(self):
        """Saved axis should load correctly."""
        axis = torch.randn(32, 4096)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert torch.allclose(axis, loaded)
```

- **Line 123**: Creates a random axis tensor of shape `(32, 4096)`.
- **Line 125**: Opens a temporary file with a `.pt` extension (a conventional PyTorch file extension). `delete=False` prevents the file from being deleted when the context manager exits, so `load_axis` can read it.
- **Line 126**: Saves the axis tensor to the temporary file.
- **Line 127**: Loads the axis tensor back from the same file.
- **Line 129**: Asserts the loaded tensor is numerically identical to the original using `torch.allclose`.

---

### Lines 131-139: `test_preserves_dtype`

```python
    def test_preserves_dtype(self):
        """Should preserve tensor dtype."""
        axis = torch.randn(4, 8, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_axis(axis, f.name)
            loaded = load_axis(f.name)

        assert loaded.dtype == axis.dtype
```

- Verifies that the data type (dtype) of the tensor is preserved through the save/load round trip.
- **Line 133**: Explicitly creates a tensor with `torch.float32` dtype.
- **Line 139**: Asserts the loaded tensor has the same dtype as the original.

---

### Lines 142-143: `TestCosineSimilarityPerLayer` Class

```python
class TestCosineSimilarityPerLayer:
    """Tests for cosine_similarity_per_layer function."""
```

Groups tests for `cosine_similarity_per_layer`, which computes the cosine similarity between two multi-layer tensors on a per-layer basis.

---

### Lines 145-153: `test_identical_vectors`

```python
    def test_identical_vectors(self):
        """Identical vectors should have similarity 1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, v)

        assert len(similarities) == 4
        for sim in similarities:
            assert abs(sim - 1.0) < 1e-5
```

- **Line 147**: Creates a random tensor with 4 layers.
- **Line 149**: Computes cosine similarity of the tensor with itself. By definition, the cosine of the angle between a vector and itself is 1.
- **Line 151**: Asserts the function returns one similarity value per layer.
- **Lines 152-153**: Asserts every layer's similarity is approximately 1.0.

---

### Lines 155-162: `test_opposite_vectors`

```python
    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1."""
        v = torch.randn(4, 8)

        similarities = cosine_similarity_per_layer(v, -v)

        for sim in similarities:
            assert abs(sim + 1.0) < 1e-5
```

- **Line 159**: Computes cosine similarity between a vector and its negation (`-v`). Opposite vectors have a cosine similarity of -1.
- **Lines 161-162**: Asserts every layer's similarity is approximately -1.0. The check `abs(sim + 1.0) < 1e-5` is equivalent to asserting `sim` is close to `-1.0`.

---

### Lines 164-174: `test_orthogonal_vectors`

```python
    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0."""
        # Create orthogonal vectors
        v1 = torch.zeros(1, 4)
        v1[0, 0] = 1.0
        v2 = torch.zeros(1, 4)
        v2[0, 1] = 1.0

        similarities = cosine_similarity_per_layer(v1, v2)

        assert abs(similarities[0]) < 1e-5
```

- **Lines 167-170**: Manually constructs two orthogonal vectors. `v1` is `[1, 0, 0, 0]` and `v2` is `[0, 1, 0, 0]`. These are standard basis vectors that are perpendicular to each other. Both have shape `(1, 4)` -- 1 layer, 4-dimensional hidden space.
- **Line 172**: Computes their cosine similarity.
- **Line 174**: Asserts the similarity at the single layer (index 0) is approximately 0, which is the cosine of a 90-degree angle.

---

### Lines 177-178: `TestAxisNormPerLayer` Class

```python
class TestAxisNormPerLayer:
    """Tests for axis_norm_per_layer function."""
```

Groups tests for `axis_norm_per_layer`, which computes the L2 (Euclidean) norm of each layer's axis vector.

---

### Lines 180-186: `test_returns_correct_length`

```python
    def test_returns_correct_length(self):
        """Should return norm for each layer."""
        axis = torch.randn(32, 4096)

        norms = axis_norm_per_layer(axis)

        assert len(norms) == 32
```

- Creates an axis tensor with 32 layers and calls `axis_norm_per_layer`.
- Asserts the returned list/sequence has 32 elements -- one norm value per layer.

---

### Lines 188-195: `test_values_are_positive`

```python
    def test_values_are_positive(self):
        """Norms should be positive."""
        axis = torch.randn(4, 8)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm >= 0
```

- Verifies that all computed norms are non-negative. By mathematical definition, L2 norms are always greater than or equal to zero.
- Uses `>= 0` rather than `> 0` because a zero vector would have norm 0 (tested separately below).

---

### Lines 197-204: `test_zero_vector`

```python
    def test_zero_vector(self):
        """Zero vector should have norm 0."""
        axis = torch.zeros(2, 4)

        norms = axis_norm_per_layer(axis)

        for norm in norms:
            assert norm == 0
```

- **Line 199**: Creates an all-zeros tensor with 2 layers and hidden dimension 4.
- **Lines 201-204**: Computes the norms and asserts every layer's norm is exactly 0. This is an edge-case test -- a zero vector has zero magnitude.
