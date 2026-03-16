# `assistant_axis/steering.py`

## Overview

This file implements **activation steering** for transformer models. Activation steering is a technique for intervening on the internal hidden-state activations of a neural network during inference, in order to amplify, suppress, or redirect specific learned features (e.g., sycophancy, refusal, a personality trait). The module provides a context manager (`ActivationSteering`) that temporarily attaches PyTorch forward hooks to selected transformer layers. When the model runs a forward pass, those hooks intercept the activations and modify them according to one of four intervention strategies: **addition**, **ablation**, **mean ablation**, or **capping**. Several convenience factory functions are also provided for common use-cases.

---

## Line-by-line explanation

### Lines 1-16 -- Module docstring

```python
"""
Activation steering utilities for transformer models.

This module provides a context manager for intervening on model activations
during inference, supporting addition, ablation, and mean ablation operations.

Example:
    from assistant_axis import load_model, load_axis, ActivationSteering

    model, tokenizer = load_model("google/gemma-2-27b-it")
    axis = load_axis("outputs/gemma-2-27b/axis.pt")

    with ActivationSteering(model, steering_vectors=[axis[22]],
                           coefficients=[1.0], layer_indices=[22]):
        output = model.generate(...)
"""
```

The module-level docstring describes the purpose of the file and gives a minimal usage example. The example shows loading a model and a precomputed "axis" (a steering vector file), then using `ActivationSteering` as a context manager so that `model.generate(...)` runs with the intervention active.

---

### Lines 18-19 -- Imports

```python
import torch
from typing import Sequence, Union, Iterable, List
```

- `torch` -- the PyTorch library, used for tensor operations, model hooks, and device/dtype management.
- `Sequence, Union, Iterable, List` -- standard Python typing constructs used in type annotations throughout the file.

---

### Lines 22-32 -- Class definition and class docstring

```python
class ActivationSteering:
    """
    Context manager for activation steering supporting:
    - Multiple feature directions simultaneously
    - Both addition and ablation interventions
    - Multiple layers
    - Per-direction coefficients

    For ablation: projects out the direction, then adds back with coefficient
    For addition: standard activation steering (add coeff * direction)
    """
```

Declares the `ActivationSteering` class. The docstring summarises the four capabilities: multiple feature directions, addition/ablation/mean-ablation/capping interventions, multi-layer support, and per-direction coefficients.

---

### Lines 34-41 -- `_POSSIBLE_LAYER_ATTRS`

```python
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",          # GPT-2/Neo, Bloom, etc.
        "encoder.layer",          # BERT/RoBERTa
        "model.layers",           # Llama/Mistral/Gemma 2/Qwen
        "language_model.layers",  # Gemma 3 (vision-language models)
        "gpt_neox.layers",        # GPT-NeoX
        "block",                  # Flan-T5
    )
```

A class-level constant that lists the dotted attribute paths where different transformer architectures store their list of layers. The method `_locate_layer_list` iterates through these paths and returns the first one that resolves to an indexable sequence on the model object. This makes the class architecture-agnostic without requiring the user to specify where layers live.

---

### Lines 43-55 -- `__init__` signature

```python
    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]],
        *,
        coefficients: Union[float, List[float]] = 1.0,
        layer_indices: Union[int, List[int]] = -1,
        intervention_type: str = "addition",
        positions: str = "all",
        mean_activations: Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]], None] = None,
        cap_thresholds: Union[float, List[float], None] = None,
        debug: bool = False,
    ):
```

The constructor accepts:
- `model` -- the transformer model whose activations will be steered.
- `steering_vectors` -- a single tensor, a list of tensors, or a list of plain numeric sequences representing the direction(s) to steer along.
- `*` -- forces all remaining arguments to be keyword-only.
- `coefficients` -- scalar or list of scalars controlling the strength of each steering direction. Defaults to `1.0`.
- `layer_indices` -- which layer(s) to attach hooks to. Defaults to `-1` (the last layer).
- `intervention_type` -- one of `"addition"`, `"ablation"`, `"mean_ablation"`, or `"capping"`.
- `positions` -- `"all"` applies the intervention to every token position; `"last"` applies it only to the final token.
- `mean_activations` -- replacement activation vectors required when using `"mean_ablation"`.
- `cap_thresholds` -- threshold values required when using `"capping"`.
- `debug` -- when `True`, prints diagnostic information.

---

### Lines 56-72 -- `__init__` docstring

```python
        """
        Args:
            model: The transformer model to steer
            steering_vectors: Either a single vector or list of vectors to use for steering
            coefficients: Either a single coefficient or list of coefficients (one per vector)
            layer_indices: Either a single layer index or list of layer indices to intervene at
            intervention_type: "addition" (standard steering), "ablation" (project out then add back),
                             "mean_ablation", or "capping"
            positions: "all" (steer all positions) or "last" (steer only last position)
            mean_activations: For mean_ablation only - replacement activations to add after projection
            cap_thresholds: For capping only - threshold values to cap projected activations at
            debug: Whether to print debugging information

        Note: For 1:1 mapping, steering_vectors, coefficients, and layer_indices must all have same length.
              steering_vectors[i] will be applied at layer_indices[i] with coefficients[i].
              If layer_indices has fewer elements than vectors, it will be broadcast to match.
        """
```

Documents every parameter and the broadcasting rule: if only one `layer_indices` value is given but multiple steering vectors are supplied, that single layer index is replicated for every vector.

---

### Lines 73-77 -- Storing basic attributes and initialising handle list

```python
        self.model = model
        self.intervention_type = intervention_type.lower()
        self.positions = positions.lower()
        self.debug = debug
        self._handles = []
```

- Stores a reference to the model.
- Normalises `intervention_type` and `positions` to lowercase so comparisons are case-insensitive.
- Stores the debug flag.
- Initialises `_handles` as an empty list. This will later hold the PyTorch hook handles so they can be removed cleanly.

---

### Lines 79-83 -- Validation of `intervention_type` and `positions`

```python
        if self.intervention_type not in {"addition", "ablation", "mean_ablation", "capping"}:
            raise ValueError("intervention_type must be 'addition', 'ablation', 'mean_ablation', or 'capping'")

        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")
```

Validates that `intervention_type` and `positions` are one of the allowed values and raises a descriptive `ValueError` otherwise.

---

### Lines 85-89 -- Validation specific to `mean_ablation`

```python
        if self.intervention_type == "mean_ablation":
            if self.positions != "all":
                raise ValueError("mean_ablation only supports positions='all'")
            if mean_activations is None:
                raise ValueError("mean_activations is required for mean_ablation")
```

Mean ablation has additional constraints: it only works when `positions="all"` (it does not support last-position-only intervention), and the caller must supply the `mean_activations` argument.

---

### Lines 91-94 -- Normalising vectors, coefficients, layers, and mean activations

```python
        self.steering_vectors = self._normalize_vectors(steering_vectors)
        self.coefficients = self._normalize_coefficients(coefficients)
        self.layer_indices = self._normalize_layers(layer_indices)
        self.mean_activations = self._normalize_mean_activations(mean_activations) if mean_activations is not None else None
```

Each input is passed through a normalisation helper that ensures it becomes a list of the appropriate type (tensors for vectors, floats for coefficients, ints for layer indices). This lets the rest of the code always work with lists, regardless of whether the user supplied a single value or a list.

---

### Lines 95-107 -- Handling `cap_thresholds`

```python
        self.cap_thresholds = None

        if self.intervention_type == "capping":
            if cap_thresholds is None:
                raise ValueError("cap_thresholds is required when intervention_type='capping'")
            self.cap_thresholds = (
                [float(cap_thresholds)] if isinstance(cap_thresholds, (int, float))
                else [float(t) for t in cap_thresholds]
            )
            if len(self.cap_thresholds) != len(self.steering_vectors):
                raise ValueError(
                    f"Number of cap_thresholds ({len(self.cap_thresholds)}) must match number of vectors ({len(self.steering_vectors)})"
                )
```

- Defaults `cap_thresholds` to `None`.
- When the intervention type is `"capping"`, the thresholds are required. The value is normalised to a list of floats, and its length must match the number of steering vectors.

---

### Lines 109-110 -- Coefficient count validation

```python
        if self.intervention_type != "mean_ablation" and len(self.coefficients) != len(self.steering_vectors):
            raise ValueError(f"Number of coefficients ({len(self.coefficients)}) must match number of vectors ({len(self.steering_vectors)})")
```

For all intervention types except `mean_ablation`, the number of coefficients must equal the number of steering vectors. (Mean ablation does not use coefficients in the same way, so it is exempt.)

---

### Lines 112-113 -- Mean activations count validation

```python
        if self.mean_activations is not None and len(self.mean_activations) != len(self.steering_vectors):
            raise ValueError(f"Number of mean_activations ({len(self.mean_activations)}) must match number of vectors ({len(self.steering_vectors)})")
```

If `mean_activations` was provided, its length must match the number of steering vectors.

---

### Lines 115-118 -- Layer index broadcasting

```python
        if len(self.layer_indices) == 1 and len(self.steering_vectors) > 1:
            self.layer_indices = self.layer_indices * len(self.steering_vectors)
        elif len(self.layer_indices) != len(self.steering_vectors):
            raise ValueError(f"Number of layer_indices ({len(self.layer_indices)}) must match number of vectors ({len(self.steering_vectors)}) or be 1 (for broadcasting)")
```

Implements the broadcasting rule described in the docstring. If only one layer index is given, it is repeated to match the number of vectors. Otherwise the lists must be the same length.

---

### Lines 120-126 -- Building the `vectors_by_layer` lookup

```python
        self.vectors_by_layer = {}
        for i, (vector, coeff, layer_idx) in enumerate(zip(self.steering_vectors, self.coefficients, self.layer_indices)):
            if layer_idx not in self.vectors_by_layer:
                self.vectors_by_layer[layer_idx] = []
            mean_act = self.mean_activations[i] if self.mean_activations is not None else None
            tau = self.cap_thresholds[i] if self.cap_thresholds is not None else None
            self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act, tau))
```

Groups the steering vectors by the layer they should be applied at. `vectors_by_layer` is a dict mapping each layer index to a list of tuples `(vector, coefficient, original_index, mean_activation_or_None, cap_threshold_or_None)`. This allows the hook for a given layer to efficiently find only the interventions relevant to that layer.

---

### Lines 128-132 -- Debug output

```python
        if self.debug:
            print(f"[ActivationSteering] Initialized with:")
            print(f"  - {len(self.steering_vectors)} steering vectors")
            print(f"  - {len(set(self.layer_indices))} unique layers: {sorted(set(self.layer_indices))}")
            print(f"  - Intervention: {self.intervention_type}")
```

When debug mode is enabled, prints a summary of the initialisation: how many vectors, which unique layers, and which intervention type.

---

### Lines 134-159 -- `_normalize_vectors`

```python
    def _normalize_vectors(self, steering_vectors):
        """Convert steering vectors to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(steering_vectors):
            if steering_vectors.ndim == 1:
                vectors = [steering_vectors]
            elif steering_vectors.ndim == 2:
                vectors = [steering_vectors[i] for i in range(steering_vectors.shape[0])]
            else:
                raise ValueError("steering_vectors tensor must be 1D or 2D")
        else:
            vectors = steering_vectors

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Steering vector {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Vector {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result
```

- `p = next(self.model.parameters())` -- grabs the first model parameter to discover the device (CPU/GPU) and dtype (float16, bfloat16, etc.) the model is using.
- If the input is a single tensor, it is split into a list: a 1-D tensor becomes a one-element list; a 2-D tensor is split along the first dimension (each row becomes a separate vector).
- If the input is already a list or other iterable, it is used directly.
- Each element is converted to a tensor via `torch.as_tensor` with the model's dtype and device. The function validates that every vector is 1-D and, if the model exposes a `hidden_size` config attribute, that the vector length matches.

---

### Lines 161-166 -- `_normalize_coefficients`

```python
    def _normalize_coefficients(self, coefficients):
        """Convert coefficients to a list of floats."""
        if isinstance(coefficients, (int, float)):
            return [float(coefficients)]
        else:
            return [float(c) for c in coefficients]
```

If the caller passed a single number, wraps it in a list. Otherwise converts every element to a Python float.

---

### Lines 168-173 -- `_normalize_layers`

```python
    def _normalize_layers(self, layer_indices):
        """Convert layer indices to a list of ints."""
        if isinstance(layer_indices, int):
            return [layer_indices]
        else:
            return list(layer_indices)
```

Same pattern as coefficients: a single int becomes a one-element list, otherwise the iterable is converted to a plain list.

---

### Lines 175-200 -- `_normalize_mean_activations`

```python
    def _normalize_mean_activations(self, mean_activations):
        """Convert mean activations to a list of tensors on the correct device/dtype."""
        p = next(self.model.parameters())

        if torch.is_tensor(mean_activations):
            if mean_activations.ndim == 1:
                vectors = [mean_activations]
            elif mean_activations.ndim == 2:
                vectors = [mean_activations[i] for i in range(mean_activations.shape[0])]
            else:
                raise ValueError("mean_activations tensor must be 1D or 2D")
        else:
            vectors = mean_activations

        result = []
        hidden_size = getattr(self.model.config, "hidden_size", None)

        for i, vec in enumerate(vectors):
            tensor_vec = torch.as_tensor(vec, dtype=p.dtype, device=p.device)
            if tensor_vec.ndim != 1:
                raise ValueError(f"Mean activation {i} must be 1-D, got shape {tensor_vec.shape}")
            if hidden_size and tensor_vec.numel() != hidden_size:
                raise ValueError(f"Mean activation {i} length {tensor_vec.numel()} != model hidden_size {hidden_size}")
            result.append(tensor_vec)

        return result
```

Mirrors `_normalize_vectors` exactly, but for the mean-activation replacement vectors. The same device/dtype alignment and shape validation are applied.

---

### Lines 202-218 -- `_locate_layer_list`

```python
    def _locate_layer_list(self):
        """Find the layer list in the model."""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur, path

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )
```

Iterates over `_POSSIBLE_LAYER_ATTRS`. For each dotted path (e.g., `"model.layers"`), it walks the model's attributes one level at a time. The `for...else` construct means the `else` block runs only if the inner loop completed without hitting a `break` (i.e., every attribute in the path existed). If the final object is indexable (`__getitem__`), it is returned along with the path string. If none of the known paths resolve, a `ValueError` is raised telling the user to add the correct path.

---

### Lines 220-230 -- `_get_layer_module`

```python
    def _get_layer_module(self, layer_idx):
        """Get the module for a specific layer index."""
        layer_list, path = self._locate_layer_list()

        if not (-len(layer_list) <= layer_idx < len(layer_list)):
            raise IndexError(f"layer_idx {layer_idx} out of range for {len(layer_list)} layers")

        if self.debug:
            print(f"[ActivationSteering] Located layer {path}[{layer_idx}]")

        return layer_list[layer_idx]
```

Uses `_locate_layer_list` to find the layer container, validates that `layer_idx` is in bounds (supporting negative indices), and returns the specific layer module. Debug output shows the resolved path.

---

### Lines 232-236 -- `_create_hook_fn`

```python
    def _create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, ins, out):
            return self._apply_layer_interventions(out, layer_idx)
        return hook_fn
```

Returns a closure that captures the `layer_idx`. When PyTorch calls this hook during a forward pass, it receives the module, its inputs (`ins`), and its outputs (`out`). The hook delegates to `_apply_layer_interventions` which performs the actual modification and returns the altered output.

---

### Lines 238-276 -- `_apply_layer_interventions`

```python
    def _apply_layer_interventions(self, activations, layer_idx):
        """Apply only the interventions assigned to this specific layer."""
        if layer_idx not in self.vectors_by_layer:
            return activations

        if torch.is_tensor(activations):
            tensor_out = activations
            was_tuple = False
        elif isinstance(activations, (tuple, list)):
            if not torch.is_tensor(activations[0]):
                return activations
            tensor_out = activations[0]
            was_tuple = True
        else:
            return activations

        modified_out = tensor_out

        for vector, coeff, vector_idx, mean_act, tau in self.vectors_by_layer[layer_idx]:
            if self.intervention_type == "addition":
                modified_out = self._apply_addition(modified_out, vector, coeff)
            elif self.intervention_type == "ablation":
                modified_out = self._apply_ablation(modified_out, vector, coeff)
            elif self.intervention_type == "mean_ablation":
                modified_out = self._apply_mean_ablation(modified_out, vector, mean_act)
            elif self.intervention_type == "capping":
                modified_out = self._apply_cap(modified_out, vector, tau)

            if self.debug:
                v = vector / (vector.norm() + 1e-8)
                pre = torch.einsum('bld,d->bl', tensor_out, v)
                post = torch.einsum('bld,d->bl', modified_out, v)
                print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx}: "
                    f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")

        if was_tuple:
            return (modified_out, *activations[1:])
        else:
            return modified_out
```

- **Early return** -- If this layer has no registered vectors, the activations pass through unchanged.
- **Unpack output format** -- Different transformer implementations return either a plain tensor or a tuple (where the first element is the hidden state and subsequent elements are caches/attention weights). This code detects the format and extracts the hidden-state tensor.
- **Apply each intervention** -- Loops over every `(vector, coeff, vector_idx, mean_act, tau)` tuple assigned to this layer and dispatches to the appropriate intervention method.
- **Debug diagnostics** -- When debug mode is on, computes the mean projection of the activations onto the (normalised) steering vector before and after the intervention, printing both so the user can verify the effect.
- **Re-pack output** -- If the original output was a tuple, the modified tensor is placed back in position 0 and the remaining elements (caches, etc.) are preserved.

---

### Lines 278-288 -- `_apply_addition`

```python
    def _apply_addition(self, activations, vector, coeff):
        """Apply standard activation addition: x + coeff * vector"""
        vector = vector.to(activations.device)
        steer = coeff * vector

        if self.positions == "all":
            return activations + steer
        else:
            result = activations.clone()
            result[:, -1, :] += steer
            return result
```

Standard activation addition. The steering vector is scaled by the coefficient and added to the activations. If `positions="all"`, the vector is broadcast across all token positions (the `[batch, seq_len, hidden]` tensor adds the `[hidden]` vector via broadcasting). If `positions="last"`, only the final token position (`[:, -1, :]`) is modified. The `.clone()` avoids in-place mutation of the original tensor when operating on a single position.

---

### Lines 290-305 -- `_apply_ablation`

```python
    def _apply_ablation(self, activations, vector, coeff):
        """Apply ablation: project out direction, then add back with coefficient."""
        vector = vector.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            projections = torch.einsum('bld,d->bl', activations, vector_norm)
            projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
            return projected_out + coeff * vector
        else:
            result = activations.clone()
            last_pos = result[:, -1, :]
            projection = torch.einsum('bd,d->b', last_pos, vector_norm)
            projected_out = last_pos - torch.einsum('b,d->bd', projection, vector_norm)
            result[:, -1, :] = projected_out + coeff * vector
            return result
```

Ablation works in two steps:
1. **Project out** -- Compute the scalar projection of each activation onto the normalised direction vector (`einsum('bld,d->bl', ...)`), then subtract that component from the activations. This removes the feature direction entirely.
2. **Add back** -- Optionally add back `coeff * vector`. With `coeff=0.0` this is a pure ablation; with `coeff=1.0` the feature is restored at its original magnitude (effectively a no-op). The `1e-8` epsilon prevents division by zero when normalising. The same all-positions vs. last-position logic applies.

---

### Lines 307-315 -- `_apply_mean_ablation`

```python
    def _apply_mean_ablation(self, activations, vector, mean_activation):
        """Apply mean ablation: project out direction, then add mean activation."""
        vector = vector.to(activations.device)
        mean_activation = mean_activation.to(activations.device)
        vector_norm = vector / (vector.norm() + 1e-8)

        projections = torch.einsum('bld,d->bl', activations, vector_norm)
        projected_out = activations - torch.einsum('bl,d->bld', projections, vector_norm)
        return projected_out + mean_activation
```

Similar to ablation, but instead of adding back a scaled version of the same direction, it adds a pre-computed mean activation vector. This replaces the feature's contribution with the "average" contribution observed across a reference dataset. Only supports `positions="all"`.

---

### Lines 317-332 -- `_apply_cap`

```python
    def _apply_cap(self, activations, vector, tau):
        """Apply capping: cap projection onto vector at threshold tau."""
        vector = vector.to(activations.device)
        v = vector / (vector.norm() + 1e-8)

        if self.positions == "all":
            proj = torch.einsum('bld,d->bl', activations, v)
            excess = (proj - tau).clamp(min=0.0)
            return activations - torch.einsum('bl,d->bld', excess, v)
        else:
            result = activations.clone()
            last = result[:, -1, :]
            proj = torch.einsum('bd,d->b', last, v)
            excess = (proj - tau).clamp(min=0.0)
            result[:, -1, :] = last - torch.einsum('b,d->bd', excess, v)
            return result
```

Capping limits how large the projection of the activations onto the steering direction can be. It computes the scalar projection, then calculates the excess above the threshold `tau`. The `clamp(min=0.0)` ensures that only projections exceeding `tau` are reduced -- projections below `tau` are left unchanged. The excess is then subtracted from the activations along the direction vector.

---

### Lines 334-345 -- `__enter__`

```python
    def __enter__(self):
        """Register hooks on all unique layers."""
        for layer_idx in self.vectors_by_layer.keys():
            layer_module = self._get_layer_module(layer_idx)
            hook_fn = self._create_hook_fn(layer_idx)
            handle = layer_module.register_forward_hook(hook_fn)
            self._handles.append(handle)

        if self.debug:
            print(f"[ActivationSteering] Registered {len(self._handles)} hooks")

        return self
```

Called when entering the `with` block. For each unique layer that has at least one steering vector, it:
1. Resolves the layer module via `_get_layer_module`.
2. Creates a hook closure via `_create_hook_fn`.
3. Registers the hook using PyTorch's `register_forward_hook`, which returns a `RemovableHook` handle.
4. Stores the handle so it can be removed later.

Returns `self` so the user can optionally bind the context manager to a variable (e.g., `with ActivationSteering(...) as steerer:`).

---

### Lines 347-349 -- `__exit__`

```python
    def __exit__(self, *exc):
        """Remove all hooks."""
        self.remove()
```

Called when leaving the `with` block (whether normally or via an exception). Delegates to `self.remove()`.

---

### Lines 351-359 -- `remove`

```python
    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            if handle:
                handle.remove()
        self._handles = []

        if self.debug:
            print("[ActivationSteering] Removed all hooks")
```

Iterates over all stored hook handles and calls `.remove()` on each, which unregisters the hook from PyTorch. The handles list is then cleared. This can also be called manually outside a context manager if the user registered hooks via `__enter__` directly.

---

### Lines 362-385 -- `create_feature_ablation_steerer`

```python
def create_feature_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    ablation_coefficients: Union[float, List[float]] = 0.0,
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for feature ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        layer_indices: Layer(s) to intervene at
        ablation_coefficients: Coefficient(s) for ablation. 0.0 = pure ablation, 1.0 = no change
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=ablation_coefficients,
        layer_indices=layer_indices,
        intervention_type="ablation",
        **kwargs
    )
```

A convenience factory function that creates an `ActivationSteering` instance pre-configured for ablation. The default coefficient is `0.0`, meaning a pure ablation (completely remove the direction). A coefficient of `1.0` would leave the activations unchanged. `**kwargs` forwards any additional arguments (e.g., `positions`, `debug`).

---

### Lines 388-413 -- `create_multi_feature_steerer`

```python
def create_multi_feature_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    coefficients: List[float],
    layer_indices: Union[int, List[int]],
    intervention_type: str = "addition",
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for multiple features.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors
        coefficients: List of coefficients (one per feature)
        layer_indices: Layer(s) to intervene at
        intervention_type: "addition" or "ablation"
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=coefficients,
        layer_indices=layer_indices,
        intervention_type=intervention_type,
        **kwargs
    )
```

Another convenience factory for steering with multiple feature directions at once. The caller provides a list of direction vectors, a matching list of coefficients, and the layer indices. The intervention type defaults to `"addition"` but can be overridden.

---

### Lines 416-441 -- `create_mean_ablation_steerer`

```python
def create_mean_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    mean_activations: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    **kwargs
) -> ActivationSteering:
    """
    Create a steerer for mean ablation.

    Args:
        model: The model to steer
        feature_directions: List of feature direction vectors to ablate
        mean_activations: List of mean activation vectors to replace with
        layer_indices: Layer(s) to intervene at
    """
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        layer_indices=layer_indices,
        intervention_type="mean_ablation",
        mean_activations=mean_activations,
        coefficients=[0.0] * len(feature_directions),
        positions="all",
        **kwargs
    )
```

Factory for mean ablation. Sets `intervention_type="mean_ablation"`, forces `positions="all"` (the only mode supported), and sets all coefficients to `0.0` (they are not used by mean ablation, but the constructor expects them for length validation).

---

### Lines 444-454 -- `load_capping_config`

```python
def load_capping_config(config_path: str) -> dict:
    """
    Load a capping config file.

    Args:
        config_path: Path to the .pt config file

    Returns:
        Dict with 'vectors' and 'experiments' keys
    """
    return torch.load(config_path, map_location='cpu', weights_only=False)
```

Loads a capping configuration from a `.pt` file using `torch.load`. The configuration is expected to be a dictionary containing:
- `'vectors'` -- a mapping from vector names to their data (layer index and tensor).
- `'experiments'` -- a list of experiment definitions, each containing a list of interventions.

`map_location='cpu'` ensures the file loads on CPU regardless of what device it was saved from. `weights_only=False` allows loading arbitrary Python objects (not just tensor weights).

---

### Lines 457-537 -- `build_capping_steerer`

```python
def build_capping_steerer(
    model: torch.nn.Module,
    capping_config: dict,
    experiment_id: Union[str, int],
    **kwargs
) -> ActivationSteering:
```

Accepts the model, a capping config dictionary (as loaded by `load_capping_config`), and an experiment identifier (either a string ID like `"layers_46:54-p0.25"` or an integer index).

```python
    # Find the experiment
    experiment = None
    if isinstance(experiment_id, int):
        experiment = capping_config['experiments'][experiment_id]
    else:
        for exp in capping_config['experiments']:
            if exp['id'] == experiment_id:
                experiment = exp
                break

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_id}' not found in config")
```

Locates the experiment in the config. If the ID is an integer, it indexes directly into the experiments list. If it is a string, it searches linearly for a matching `'id'` field. Raises a `ValueError` if the experiment is not found.

```python
    # Collect capping interventions
    vectors = []
    cap_thresholds = []
    layer_indices = []

    for intervention in experiment['interventions']:
        if 'cap' not in intervention:
            continue

        vector_name = intervention['vector']
        cap_value = float(intervention['cap'])

        vec_data = capping_config['vectors'][vector_name]
        layer_idx = vec_data['layer']
        vector = vec_data['vector'].to(dtype=torch.float32)

        vectors.append(vector)
        cap_thresholds.append(cap_value)
        layer_indices.append(layer_idx)
```

Iterates over the experiment's list of interventions. Only those containing a `'cap'` key are processed (interventions without `'cap'` are skipped). For each capping intervention:
- Looks up the vector by name in the config's `'vectors'` dictionary.
- Extracts the layer index and the direction tensor (cast to float32).
- Appends the vector, its cap threshold, and the layer index to the respective lists.

```python
    if not vectors:
        raise ValueError(f"No capping interventions found in experiment '{experiment_id}'")

    vectors_tensor = torch.stack(vectors)
```

Validates that at least one capping intervention was found. Stacks all vectors into a single 2-D tensor (which `_normalize_vectors` will later split back into individual 1-D tensors).

```python
    return ActivationSteering(
        model=model,
        steering_vectors=vectors_tensor,
        layer_indices=layer_indices,
        intervention_type="capping",
        cap_thresholds=cap_thresholds,
        coefficients=[0.0] * len(vectors),
        positions="all",
        **kwargs
    )
```

Constructs and returns an `ActivationSteering` instance configured for capping. Coefficients are set to `0.0` (they are not used by the capping intervention but are required for constructor validation). Positions default to `"all"`.
