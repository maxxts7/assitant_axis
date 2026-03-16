# Task 7: Activation Steering (steering.py)

## Overview

`steering.py` provides a context-manager-based system for intervening on transformer model activations during inference. The central class, `ActivationSteering`, registers PyTorch forward hooks on specified transformer layers that modify the hidden-state tensor as it passes through the model. Four intervention types are supported:

| Intervention | Effect |
|---|---|
| `addition` | Adds a scaled steering vector to the activations. |
| `ablation` | Projects out a direction, then adds the direction back scaled by a coefficient (0 = pure removal). |
| `mean_ablation` | Projects out a direction, then adds a pre-computed mean activation in its place. |
| `capping` | Clips the component of the activation along a direction so it never exceeds a threshold. |

Five helper functions wrap common configurations: `create_feature_ablation_steerer`, `create_multi_feature_steerer`, `create_mean_ablation_steerer`, `load_capping_config`, and `build_capping_steerer`.

---

## Sub-Tasks

### Sub-Task 7.1: ActivationSteering Constructor (`__init__`)

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | *(required)* | The transformer model whose activations will be steered. |
| `steering_vectors` | `Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]]` | *(required)* | One or more direction vectors. A 1-D tensor is treated as a single vector; a 2-D tensor is split along dim 0. |
| `coefficients` | `Union[float, List[float]]` | `1.0` | Scaling factor(s), one per vector. |
| `layer_indices` | `Union[int, List[int]]` | `-1` | Layer index/indices where hooks are registered. If length 1 with multiple vectors, it is broadcast. |
| `intervention_type` | `str` | `"addition"` | One of `"addition"`, `"ablation"`, `"mean_ablation"`, `"capping"`. |
| `positions` | `str` | `"all"` | `"all"` steers every sequence position; `"last"` steers only the final position. |
| `mean_activations` | `Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]], None]` | `None` | Required for `mean_ablation` -- replacement activations to add after projection. |
| `cap_thresholds` | `Union[float, List[float], None]` | `None` | Required for `capping` -- threshold values per vector. |
| `debug` | `bool` | `False` | Print diagnostic information. |

#### Processing

1. **Store basic fields and validate `intervention_type` / `positions`.**

```python
self.model = model
self.intervention_type = intervention_type.lower()
self.positions = positions.lower()
self.debug = debug
self._handles = []

if self.intervention_type not in {"addition", "ablation", "mean_ablation", "capping"}:
    raise ValueError("intervention_type must be 'addition', 'ablation', 'mean_ablation', or 'capping'")

if self.positions not in {"all", "last"}:
    raise ValueError("positions must be 'all' or 'last'")
```

2. **Validate `mean_ablation`-specific constraints** (requires `positions="all"` and `mean_activations`).

```python
if self.intervention_type == "mean_ablation":
    if self.positions != "all":
        raise ValueError("mean_ablation only supports positions='all'")
    if mean_activations is None:
        raise ValueError("mean_activations is required for mean_ablation")
```

3. **Normalize all collection inputs** via private helpers (see Sub-Tasks 7.2--7.5).

```python
self.steering_vectors = self._normalize_vectors(steering_vectors)
self.coefficients = self._normalize_coefficients(coefficients)
self.layer_indices = self._normalize_layers(layer_indices)
self.mean_activations = self._normalize_mean_activations(mean_activations) if mean_activations is not None else None
self.cap_thresholds = None
```

4. **Normalize and validate `cap_thresholds`** if intervention is `capping`.

```python
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

5. **Cross-validate lengths** of coefficients, mean_activations, and layer_indices against steering_vectors. Broadcast `layer_indices` if it has length 1.

```python
if self.intervention_type != "mean_ablation" and len(self.coefficients) != len(self.steering_vectors):
    raise ValueError(f"Number of coefficients ({len(self.coefficients)}) must match number of vectors ({len(self.steering_vectors)})")

if self.mean_activations is not None and len(self.mean_activations) != len(self.steering_vectors):
    raise ValueError(f"Number of mean_activations ({len(self.mean_activations)}) must match number of vectors ({len(self.steering_vectors)})")

if len(self.layer_indices) == 1 and len(self.steering_vectors) > 1:
    self.layer_indices = self.layer_indices * len(self.steering_vectors)
elif len(self.layer_indices) != len(self.steering_vectors):
    raise ValueError(f"Number of layer_indices ({len(self.layer_indices)}) must match number of vectors ({len(self.steering_vectors)}) or be 1 (for broadcasting)")
```

6. **Build `vectors_by_layer` lookup** -- a dict mapping each layer index to a list of `(vector, coeff, vector_idx, mean_act, tau)` tuples.

```python
self.vectors_by_layer = {}
for i, (vector, coeff, layer_idx) in enumerate(zip(self.steering_vectors, self.coefficients, self.layer_indices)):
    if layer_idx not in self.vectors_by_layer:
        self.vectors_by_layer[layer_idx] = []
    mean_act = self.mean_activations[i] if self.mean_activations is not None else None
    tau = self.cap_thresholds[i] if self.cap_thresholds is not None else None
    self.vectors_by_layer[layer_idx].append((vector, coeff, i, mean_act, tau))
```

#### Output

A fully initialised `ActivationSteering` instance with the following key attributes:

| Attribute | Type | Description |
|---|---|---|
| `steering_vectors` | `List[torch.Tensor]` | Normalised list of 1-D tensors on model device/dtype. |
| `coefficients` | `List[float]` | One float per vector. |
| `layer_indices` | `List[int]` | One layer index per vector (broadcast if needed). |
| `mean_activations` | `List[torch.Tensor] or None` | One tensor per vector (mean_ablation only). |
| `cap_thresholds` | `List[float] or None` | One threshold per vector (capping only). |
| `vectors_by_layer` | `Dict[int, List[Tuple]]` | Lookup table grouping interventions by layer. |

---

### Sub-Task 7.2: Normalize Steering Vectors (`_normalize_vectors`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `steering_vectors` | `Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]]` | Raw input from the caller. |

(Also reads `self.model.parameters()` for device/dtype and `self.model.config.hidden_size` for shape validation.)

#### Processing

1. **Infer target device and dtype** from the first model parameter.

```python
p = next(self.model.parameters())
```

2. **Unpack tensor inputs**: a 1-D tensor becomes a single-element list; a 2-D tensor is split along dim 0; anything else raises `ValueError`.

```python
if torch.is_tensor(steering_vectors):
    if steering_vectors.ndim == 1:
        vectors = [steering_vectors]
    elif steering_vectors.ndim == 2:
        vectors = [steering_vectors[i] for i in range(steering_vectors.shape[0])]
    else:
        raise ValueError("steering_vectors tensor must be 1D or 2D")
else:
    vectors = steering_vectors
```

3. **Convert each element** to a tensor on the correct device/dtype, validate it is 1-D, and validate its length matches `hidden_size`.

```python
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

#### Output

`List[torch.Tensor]` -- each element is a 1-D tensor of shape `(hidden_size,)` on the model's device and dtype.

---

### Sub-Task 7.3: Normalize Coefficients (`_normalize_coefficients`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `coefficients` | `Union[float, int, List[float]]` | Raw coefficient(s). |

#### Processing

Wraps a scalar in a single-element list; converts every element to `float`.

```python
def _normalize_coefficients(self, coefficients):
    """Convert coefficients to a list of floats."""
    if isinstance(coefficients, (int, float)):
        return [float(coefficients)]
    else:
        return [float(c) for c in coefficients]
```

#### Output

`List[float]` -- one or more float values.

---

### Sub-Task 7.4: Normalize Layer Indices (`_normalize_layers`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `layer_indices` | `Union[int, List[int]]` | Raw layer index or indices. |

#### Processing

Wraps a scalar `int` in a single-element list; otherwise converts to a plain `list`.

```python
def _normalize_layers(self, layer_indices):
    """Convert layer indices to a list of ints."""
    if isinstance(layer_indices, int):
        return [layer_indices]
    else:
        return list(layer_indices)
```

#### Output

`List[int]` -- one or more integer layer indices.

---

### Sub-Task 7.5: Normalize Mean Activations (`_normalize_mean_activations`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `mean_activations` | `Union[torch.Tensor, List[torch.Tensor], List[Sequence[float]]]` | Raw mean activation input. |

(Also reads `self.model.parameters()` for device/dtype and `self.model.config.hidden_size` for validation.)

#### Processing

Identical logic to `_normalize_vectors` (Sub-Task 7.2): infers device/dtype from model parameters, unpacks 1-D/2-D tensors or lists, converts each to a 1-D tensor, and validates against `hidden_size`.

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

#### Output

`List[torch.Tensor]` -- each element is a 1-D tensor of shape `(hidden_size,)` on the model's device and dtype.

---

### Sub-Task 7.6: Locate Layer List (`_locate_layer_list`)

#### Input

None (reads `self.model` and the class-level `_POSSIBLE_LAYER_ATTRS`).

The candidate attribute paths are:

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

#### Processing

Iterates through each dotted path, resolving attributes via `getattr`. If every component in the path resolves and the final object supports `__getitem__`, it is returned.

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

#### Output

`Tuple[ModuleList, str]` -- the indexable layer container and the dotted attribute path string that resolved to it (e.g., `"model.layers"`).

---

### Sub-Task 7.7: Get Layer Module (`_get_layer_module`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `layer_idx` | `int` | Index into the layer list (supports negative indexing). |

#### Processing

Calls `_locate_layer_list` (Sub-Task 7.6), validates the index is in bounds, and returns the module at that position.

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

#### Output

`torch.nn.Module` -- the specific transformer layer module.

---

### Sub-Task 7.8: Create Hook Function (`_create_hook_fn`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `layer_idx` | `int` | The layer this hook is associated with. |

#### Processing

Returns a closure that, when called by PyTorch's forward-hook mechanism, delegates to `_apply_layer_interventions`.

```python
def _create_hook_fn(self, layer_idx):
    """Create a hook function for a specific layer."""
    def hook_fn(module, ins, out):
        return self._apply_layer_interventions(out, layer_idx)
    return hook_fn
```

The hook signature `(module, ins, out)` matches PyTorch's `register_forward_hook` contract. Returning a value from the hook replaces the module's output.

#### Output

`Callable[[Module, Tuple, Any], Any]` -- a hook function ready for `register_forward_hook`.

---

### Sub-Task 7.9: Apply Layer Interventions (`_apply_layer_interventions`)

#### Input

| Parameter | Type | Description |
|---|---|---|
| `activations` | `torch.Tensor` or `Tuple[torch.Tensor, ...]` | Raw output from the transformer layer. Shape of the tensor component is `(batch, seq_len, hidden_size)`. |
| `layer_idx` | `int` | Which layer produced these activations. |

#### Processing

1. **Early exit** if no vectors are assigned to this layer.

```python
if layer_idx not in self.vectors_by_layer:
    return activations
```

2. **Unwrap tuple outputs**. Many transformer layers return `(hidden_states, attention_weights, ...)`. The code extracts the first tensor and remembers whether it was a tuple.

```python
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
```

3. **Loop over all interventions for this layer** and apply the appropriate method.

```python
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
```

4. **Debug logging** projects the original and modified activations onto the direction and prints per-vector diagnostics.

```python
    if self.debug:
        v = vector / (vector.norm() + 1e-8)
        pre = torch.einsum('bld,d->bl', tensor_out, v)
        post = torch.einsum('bld,d->bl', modified_out, v)
        print(f"[ActivationSteering] Layer {layer_idx}, vec {vector_idx}: "
            f"pre mean={pre.mean():.3f} | post mean={post.mean():.3f}")
```

5. **Re-wrap** into a tuple if the original output was a tuple.

```python
if was_tuple:
    return (modified_out, *activations[1:])
else:
    return modified_out
```

#### Output

Same type as the input `activations` -- either a `torch.Tensor` of shape `(batch, seq_len, hidden_size)` or a tuple whose first element has been modified.

---

### Sub-Task 7.10: Addition Intervention (`_apply_addition`)

#### Input

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `activations` | `torch.Tensor` | `(batch, seq_len, hidden_size)` | Current hidden states. |
| `vector` | `torch.Tensor` | `(hidden_size,)` | Steering direction. |
| `coeff` | `float` | scalar | Scaling factor. |

#### Processing

Computes `x + coeff * vector`. When `positions="last"`, only position `[:, -1, :]` is modified.

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

**Mathematical formulation:**

- `positions="all"`: For every position `t`, `h_t' = h_t + alpha * v`
- `positions="last"`: Only `h_T' = h_T + alpha * v`; all other positions unchanged.

#### Output

`torch.Tensor` of shape `(batch, seq_len, hidden_size)` -- the steered activations.

---

### Sub-Task 7.11: Ablation Intervention (`_apply_ablation`)

#### Input

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `activations` | `torch.Tensor` | `(batch, seq_len, hidden_size)` | Current hidden states. |
| `vector` | `torch.Tensor` | `(hidden_size,)` | Direction to ablate. |
| `coeff` | `float` | scalar | How much of the direction to add back (0 = full ablation, 1 = no change). |

#### Processing

Projects out the direction using the dot product, then adds the direction back scaled by `coeff`.

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

**Mathematical formulation:**

Let `v_hat = v / ||v||`. For each position:

1. Compute scalar projection: `p = h . v_hat`
2. Remove component: `h_perp = h - p * v_hat`
3. Add back scaled: `h' = h_perp + alpha * v`

Note: when `coeff=0`, this is pure ablation (the component is fully removed). When `coeff=1.0` and `vector` is unit-norm, the result is the original activation (no change).

#### Output

`torch.Tensor` of shape `(batch, seq_len, hidden_size)` -- activations with the specified direction ablated and optionally partially restored.

---

### Sub-Task 7.12: Mean Ablation Intervention (`_apply_mean_ablation`)

#### Input

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `activations` | `torch.Tensor` | `(batch, seq_len, hidden_size)` | Current hidden states. |
| `vector` | `torch.Tensor` | `(hidden_size,)` | Direction to ablate. |
| `mean_activation` | `torch.Tensor` | `(hidden_size,)` | Mean activation vector to add after ablation. |

Note: This method only supports `positions="all"` (enforced in `__init__`).

#### Processing

Projects out the direction and replaces it with the pre-computed mean activation vector.

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

**Mathematical formulation:**

Let `v_hat = v / ||v||`. For each position:

1. Compute scalar projection: `p = h . v_hat`
2. Remove component: `h_perp = h - p * v_hat`
3. Replace with mean: `h' = h_perp + mean_act`

This replaces sample-specific variance along the direction with a fixed "average" value.

#### Output

`torch.Tensor` of shape `(batch, seq_len, hidden_size)`.

---

### Sub-Task 7.13: Capping Intervention (`_apply_cap`)

#### Input

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `activations` | `torch.Tensor` | `(batch, seq_len, hidden_size)` | Current hidden states. |
| `vector` | `torch.Tensor` | `(hidden_size,)` | Direction along which to cap. |
| `tau` | `float` | scalar | Threshold value. Projections above this are clamped down. |

#### Processing

Computes the projection onto the direction, calculates the excess above `tau`, and subtracts it.

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

**Mathematical formulation:**

Let `v_hat = v / ||v||`. For each position:

1. Compute scalar projection: `p = h . v_hat`
2. Compute excess: `e = max(0, p - tau)`
3. Subtract excess: `h' = h - e * v_hat`

If `p <= tau`, the activation is unchanged. If `p > tau`, the component along `v_hat` is reduced to exactly `tau`. Negative projections are never modified.

#### Output

`torch.Tensor` of shape `(batch, seq_len, hidden_size)` -- activations with capped projections.

---

### Sub-Task 7.14: Enter Context (`__enter__`)

#### Input

None beyond `self` (reads `self.vectors_by_layer`).

#### Processing

Iterates over every unique layer index in `vectors_by_layer`, retrieves the corresponding `torch.nn.Module`, creates a hook function for that layer, and registers it as a PyTorch forward hook.

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

#### Output

Returns `self` (the `ActivationSteering` instance), enabling `with ... as steerer:` usage. Side effect: hooks are now registered on the model and will fire on every forward pass.

---

### Sub-Task 7.15: Exit Context (`__exit__`) and `remove()`

#### Input

| Parameter | Type | Description |
|---|---|---|
| `*exc` | exception info | Standard context-manager exception triple (type, value, traceback). Ignored. |

#### Processing

Delegates to `remove()`, which iterates over stored hook handles and calls `.remove()` on each, then clears the list.

```python
def __exit__(self, *exc):
    """Remove all hooks."""
    self.remove()

def remove(self):
    """Remove all registered hooks."""
    for handle in self._handles:
        if handle:
            handle.remove()
    self._handles = []

    if self.debug:
        print("[ActivationSteering] Removed all hooks")
```

#### Output

`None`. Side effect: all forward hooks are removed from the model; subsequent forward passes proceed without intervention. Exceptions are not suppressed (returns `None` implicitly from `__exit__`).

---

### Sub-Task 7.16: Helper -- `create_feature_ablation_steerer`

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | *(required)* | The model to steer. |
| `feature_directions` | `List[torch.Tensor]` | *(required)* | Feature direction vectors to ablate. |
| `layer_indices` | `Union[int, List[int]]` | *(required)* | Layer(s) to intervene at. |
| `ablation_coefficients` | `Union[float, List[float]]` | `0.0` | How much to add back after ablation (0 = pure ablation). |
| `**kwargs` | | | Forwarded to `ActivationSteering`. |

#### Processing

Wraps `ActivationSteering` with `intervention_type="ablation"`.

```python
def create_feature_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    ablation_coefficients: Union[float, List[float]] = 0.0,
    **kwargs
) -> ActivationSteering:
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=ablation_coefficients,
        layer_indices=layer_indices,
        intervention_type="ablation",
        **kwargs
    )
```

#### Output

`ActivationSteering` instance configured for ablation.

---

### Sub-Task 7.17: Helper -- `create_multi_feature_steerer`

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | *(required)* | The model to steer. |
| `feature_directions` | `List[torch.Tensor]` | *(required)* | Feature direction vectors. |
| `coefficients` | `List[float]` | *(required)* | One coefficient per feature. |
| `layer_indices` | `Union[int, List[int]]` | *(required)* | Layer(s) to intervene at. |
| `intervention_type` | `str` | `"addition"` | `"addition"` or `"ablation"`. |
| `**kwargs` | | | Forwarded to `ActivationSteering`. |

#### Processing

Wraps `ActivationSteering` with user-specified `intervention_type`.

```python
def create_multi_feature_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    coefficients: List[float],
    layer_indices: Union[int, List[int]],
    intervention_type: str = "addition",
    **kwargs
) -> ActivationSteering:
    return ActivationSteering(
        model=model,
        steering_vectors=feature_directions,
        coefficients=coefficients,
        layer_indices=layer_indices,
        intervention_type=intervention_type,
        **kwargs
    )
```

#### Output

`ActivationSteering` instance configured for multi-feature steering.

---

### Sub-Task 7.18: Helper -- `create_mean_ablation_steerer`

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | *(required)* | The model to steer. |
| `feature_directions` | `List[torch.Tensor]` | *(required)* | Feature direction vectors to ablate. |
| `mean_activations` | `List[torch.Tensor]` | *(required)* | Replacement mean activation vectors. |
| `layer_indices` | `Union[int, List[int]]` | *(required)* | Layer(s) to intervene at. |
| `**kwargs` | | | Forwarded to `ActivationSteering`. |

#### Processing

Wraps `ActivationSteering` with `intervention_type="mean_ablation"`, `positions="all"`, and dummy coefficients of `0.0` (one per direction).

```python
def create_mean_ablation_steerer(
    model: torch.nn.Module,
    feature_directions: List[torch.Tensor],
    mean_activations: List[torch.Tensor],
    layer_indices: Union[int, List[int]],
    **kwargs
) -> ActivationSteering:
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

#### Output

`ActivationSteering` instance configured for mean ablation.

---

### Sub-Task 7.19: Helper -- `load_capping_config`

#### Input

| Parameter | Type | Description |
|---|---|---|
| `config_path` | `str` | Path to a `.pt` file containing a serialised capping configuration dictionary. |

#### Processing

Uses `torch.load` with `map_location='cpu'` and `weights_only=False`.

```python
def load_capping_config(config_path: str) -> dict:
    return torch.load(config_path, map_location='cpu', weights_only=False)
```

#### Output

`dict` with the following expected structure:

| Key | Type | Description |
|---|---|---|
| `"vectors"` | `Dict[str, Dict]` | Mapping from vector name to `{"layer": int, "vector": torch.Tensor}`. |
| `"experiments"` | `List[Dict]` | Each experiment has `"id"` (str) and `"interventions"` (list of dicts with `"vector"`, `"cap"` keys). |

---

### Sub-Task 7.20: Helper -- `build_capping_steerer`

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | *(required)* | The model to steer. |
| `capping_config` | `dict` | *(required)* | Config dict as returned by `load_capping_config`. |
| `experiment_id` | `Union[str, int]` | *(required)* | Either the string ID (e.g., `"layers_46:54-p0.25"`) or an integer index. |
| `**kwargs` | | | Forwarded to `ActivationSteering`. |

#### Processing

1. **Look up the experiment** by integer index or string ID.

```python
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

2. **Collect capping interventions** from the experiment, filtering for entries that have a `"cap"` key. For each, resolve the named vector from `capping_config['vectors']`.

```python
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

if not vectors:
    raise ValueError(f"No capping interventions found in experiment '{experiment_id}'")
```

3. **Stack vectors into a 2-D tensor** and construct the `ActivationSteering` instance.

```python
vectors_tensor = torch.stack(vectors)

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

#### Output

`ActivationSteering` instance configured for capping, with one vector/threshold/layer per intervention entry in the selected experiment.
