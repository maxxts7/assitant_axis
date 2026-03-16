# `assistant_axis/__init__.py`

## Overview

This is the package initialisation file for the `assistant_axis` library. It serves two purposes:

1. **Documentation** -- the module-level docstring describes what the package does and shows a quick-start example.
2. **Public API surface** -- it imports every symbol the library wants to expose and lists them in `__all__`, so that users can write `from assistant_axis import <name>` without knowing which internal module defines it.

No logic runs here; the file is purely declarative.

---

## Line-by-line explanation

### Lines 1-20 -- Module docstring

```python
"""
Assistant Axis - Tools for computing and steering with the assistant axis.

The assistant axis is a direction in activation space that captures the difference
between role-playing and default assistant behavior in language models.

Example:
    from assistant_axis import get_config, load_axis, ActivationSteering
    from assistant_axis.internals import ProbingModel

    # Load model and axis
    pm = ProbingModel("google/gemma-2-27b-it")
    axis = load_axis("outputs/axis.pt")
    config = get_config("google/gemma-2-27b-it")

    # Steer model outputs
    with ActivationSteering(pm.model, steering_vectors=[axis[config["target_layer"]]],
                           coefficients=[1.0], layer_indices=[config["target_layer"]]):
        output = pm.model.generate(...)
"""
```

This triple-quoted string is the module's docstring. It is not assigned to a variable; Python automatically stores it as `assistant_axis.__doc__`.

- **Line 2** gives a one-line summary of the package.
- **Lines 4-5** explain the concept behind the "assistant axis": it is a direction in a language model's activation space that separates role-playing behaviour from default assistant behaviour.
- **Lines 7-19** provide a minimal usage example:
  - Import the public helpers (`get_config`, `load_axis`, `ActivationSteering`) and the internal `ProbingModel` class.
  - Instantiate a `ProbingModel` around a Gemma-2 27B model.
  - Load a pre-computed axis from disk and retrieve the model-specific configuration.
  - Use `ActivationSteering` as a context manager to temporarily inject the steering vector into a specific layer, then generate text under that intervention.

---

### Line 22 -- Imports from `.models`

```python
from .models import get_config, MODEL_CONFIGS
```

A relative import from the `assistant_axis.models` sub-module. Two names are brought into the package namespace:

- `get_config` -- a function that returns the configuration dictionary (e.g. target layer, tokenizer name) for a given model identifier.
- `MODEL_CONFIGS` -- a mapping (likely a `dict`) that holds every supported model's configuration.

---

### Lines 23-32 -- Imports from `.axis`

```python
from .axis import (
    compute_axis,
    load_axis,
    save_axis,
    project,
    project_batch,
    cosine_similarity_per_layer,
    axis_norm_per_layer,
    aggregate_role_vectors,
)
```

A relative import from the `assistant_axis.axis` sub-module. Eight names are imported:

| Name | Purpose |
|------|---------|
| `compute_axis` | Computes the assistant axis direction from activation data. |
| `load_axis` | Loads a previously saved axis from a file (e.g. a `.pt` tensor file). |
| `save_axis` | Serialises an axis to disk. |
| `project` | Projects a single activation vector onto the axis. |
| `project_batch` | Batch version of `project`, operating on multiple vectors at once. |
| `cosine_similarity_per_layer` | Computes cosine similarity between an activation and the axis for every layer. |
| `axis_norm_per_layer` | Computes the norm (magnitude) of the axis at each layer. |
| `aggregate_role_vectors` | Aggregates activation vectors collected under different roles into a summary representation. |

---

### Lines 33-38 -- Imports from `.generation`

```python
from .generation import (
    generate_response,
    format_conversation,
    VLLMGenerator,
    RoleResponseGenerator,
)
```

A relative import from the `assistant_axis.generation` sub-module. Four names are imported:

| Name | Purpose |
|------|---------|
| `generate_response` | A convenience function to produce a single model response. |
| `format_conversation` | Converts a conversation (list of messages) into the string format the model expects. |
| `VLLMGenerator` | A class that wraps vLLM for high-throughput batch generation. |
| `RoleResponseGenerator` | A class that generates responses under specific role-play prompts, used for collecting activation data. |

---

### Lines 39-46 -- Imports from `.steering`

```python
from .steering import (
    ActivationSteering,
    create_feature_ablation_steerer,
    create_multi_feature_steerer,
    create_mean_ablation_steerer,
    load_capping_config,
    build_capping_steerer,
)
```

A relative import from the `assistant_axis.steering` sub-module. Six names are imported:

| Name | Purpose |
|------|---------|
| `ActivationSteering` | Context manager that hooks into a model's forward pass and adds (or removes) a steering vector at specified layers. |
| `create_feature_ablation_steerer` | Factory that builds a steerer which ablates (zeroes out) a single feature direction. |
| `create_multi_feature_steerer` | Factory that builds a steerer operating on multiple feature directions simultaneously. |
| `create_mean_ablation_steerer` | Factory that builds a steerer which ablates the mean activation component. |
| `load_capping_config` | Loads a JSON or similar configuration file that specifies capping parameters for steering. |
| `build_capping_steerer` | Constructs a steerer that caps (clamps) the projection onto the axis rather than fully ablating it. |

---

### Lines 47-52 -- Imports from `.pca`

```python
from .pca import (
    compute_pca,
    plot_variance_explained,
    MeanScaler,
    L2MeanScaler,
)
```

A relative import from the `assistant_axis.pca` sub-module. Four names are imported:

| Name | Purpose |
|------|---------|
| `compute_pca` | Runs PCA (Principal Component Analysis) on a matrix of activations. |
| `plot_variance_explained` | Generates a plot showing how much variance each principal component explains. |
| `MeanScaler` | A transformer that centres data by subtracting the mean. |
| `L2MeanScaler` | A transformer that centres data by subtracting the mean and then L2-normalises each sample. |

---

### Lines 54-84 -- `__all__`

```python
__all__ = [
    # Models
    "get_config",
    "MODEL_CONFIGS",
    # Axis
    "compute_axis",
    "load_axis",
    "save_axis",
    "project",
    "project_batch",
    "cosine_similarity_per_layer",
    "axis_norm_per_layer",
    "aggregate_role_vectors",
    # Generation
    "generate_response",
    "format_conversation",
    "VLLMGenerator",
    "RoleResponseGenerator",
    # Steering
    "ActivationSteering",
    "create_feature_ablation_steerer",
    "create_multi_feature_steerer",
    "create_mean_ablation_steerer",
    "load_capping_config",
    "build_capping_steerer",
    # PCA
    "compute_pca",
    "plot_variance_explained",
    "MeanScaler",
    "L2MeanScaler",
]
```

`__all__` is a module-level list of strings that defines the **public API** of the package. It controls two things:

1. **`from assistant_axis import *`** -- only the names listed here will be imported by a wildcard import.
2. **Documentation tools** (e.g. Sphinx, pdoc) use `__all__` to decide which symbols to document.

The list contains every name imported in lines 22-52 and is organised into five commented sections mirroring the sub-modules they come from: Models, Axis, Generation, Steering, and PCA. The total count is 20 public names.

---

### Line 85 -- End of file

The file ends with no trailing code. Python considers the package fully initialised at this point; all 20 public symbols are available directly under `assistant_axis`.
