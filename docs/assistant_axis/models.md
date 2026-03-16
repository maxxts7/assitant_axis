# `assistant_axis/models.py`

## Overview

This file provides model configuration utilities for the `assistant_axis` project. It serves two purposes:

1. It stores a hard-coded registry (`MODEL_CONFIGS`) of known language models and their project-specific settings, such as which transformer layer to target for axis computation.
2. It exposes a `get_config()` function that looks up a model's configuration from the registry, or dynamically infers a reasonable configuration by querying the model's architecture through HuggingFace Transformers.

The module deliberately does **not** handle model loading; it points users to `ProbingModel` from `assistant_axis.internals` for that.

---

## Line-by-Line Explanation

### Lines 1--16: Module Docstring

```python
"""
Model configuration utilities.

This module provides configuration lookup for known models, including
project-specific values like target layers for axis computation.

For model loading, use ProbingModel from assistant_axis.internals instead.

Example:
    from assistant_axis import get_config
    from assistant_axis.internals import ProbingModel

    pm = ProbingModel("google/gemma-2-27b-it")
    config = get_config("google/gemma-2-27b-it")
    target_layer = config["target_layer"]
"""
```

The module-level docstring explains the purpose of this file: providing configuration lookup for known models. It clarifies that model *loading* lives elsewhere (`ProbingModel`), and gives a usage example showing how to obtain a model's config dictionary and extract the `target_layer` value from it.

---

### Lines 17--18: Blank Lines

```python

```

Two blank lines separate the module docstring from the first top-level definition, following PEP 8 style conventions.

---

### Lines 19--23: `MODEL_CONFIGS` Dictionary Opening and Comment

```python
MODEL_CONFIGS = {
    # Format: model_name -> {target_layer, total_layers, short_name, capping_config, capping_experiment}
    # target_layer is the recommended layer for axis computation (typically ~middle)
    # capping_config is the HuggingFace path to the capping config file
    # capping_experiment is the recommended experiment ID for activation capping
```

This begins the `MODEL_CONFIGS` dictionary, a module-level constant that maps HuggingFace model identifier strings to configuration dictionaries. The comments document the schema: each entry may contain `target_layer`, `total_layers`, `short_name`, `capping_config`, and `capping_experiment`. Not every entry uses all fields.

---

### Lines 24--28: Gemma 2 27B Entry

```python
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
```

The first registered model is Google's Gemma 2 27B instruction-tuned variant. It has 46 transformer layers and layer 22 (roughly the middle) is designated as the target for axis computation. The `short_name` `"Gemma"` provides a human-friendly label. This entry does not include `capping_config` or `capping_experiment`, meaning activation capping is not configured for this model.

---

### Lines 29--35: Qwen 3 32B Entry

```python
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
```

The second registered model is Alibaba's Qwen 3 32B. It has 64 layers with layer 32 (the midpoint) as the target. Unlike Gemma, this entry includes two additional fields:
- `capping_config`: a relative path (`"qwen-3-32b/capping_config.pt"`) pointing to a serialized PyTorch file that stores the capping configuration.
- `capping_experiment`: the string `"layers_46:54-p0.25"` identifies a specific experiment setup -- it targets layers 46 through 54 with a parameter value of 0.25 for activation capping.

---

### Lines 36--43: Llama 3.3 70B Entry and Dictionary Closing

```python
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
        "capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
    },
}
```

The third and final registered model is Meta's Llama 3.3 70B Instruct. It is the largest model in the registry at 80 layers, with layer 40 as the target. Like Qwen, it has a `capping_config` path and a `capping_experiment` string (targeting layers 56 through 72 at p=0.25). The closing `}` ends the `MODEL_CONFIGS` dictionary.

---

### Lines 44--45: Blank Lines

```python

```

Two blank lines separate the constant from the function definition, per PEP 8.

---

### Lines 46--56: `get_config` Function Signature and Docstring

```python
def get_config(model_name: str) -> dict:
    """
    Get configuration for a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dict with target_layer, total_layers, and short_name.
        If model is not in known configs, infers values from model architecture.
    """
```

This defines `get_config`, a function that accepts a single argument `model_name` (a string, expected to be a HuggingFace model identifier like `"google/gemma-2-27b-it"`) and returns a `dict`. The docstring explains that the returned dictionary contains at least `target_layer`, `total_layers`, and `short_name`, and that for unknown models the function will attempt to infer these values automatically.

---

### Lines 57--58: Known-Model Lookup

```python
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
```

The function first checks whether `model_name` exists as a key in the `MODEL_CONFIGS` dictionary. If it does, it returns a **copy** of the corresponding config dict. Using `.copy()` is important: it prevents callers from accidentally mutating the shared module-level `MODEL_CONFIGS` data, which would affect all subsequent calls.

---

### Lines 60--64: Inferring Config from HuggingFace -- Setup

```python
    # Try to infer config from model
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        total_layers = config.num_hidden_layers
```

If the model is not in `MODEL_CONFIGS`, the function falls back to dynamic inference. It imports `AutoConfig` from the `transformers` library (HuggingFace Transformers) *inside* the function body -- this is a lazy import, so the `transformers` dependency is only required when the fallback path is actually used. `AutoConfig.from_pretrained(model_name)` downloads (or loads from cache) the model's configuration metadata from HuggingFace Hub. The `num_hidden_layers` attribute is then extracted to get the total number of transformer layers.

---

### Line 65: Computing the Default Target Layer

```python
        target_layer = total_layers // 2  # Default to middle layer
```

The target layer is set to the integer floor-division of `total_layers` by 2, placing it at approximately the middle of the network. This is a reasonable default since the known models in `MODEL_CONFIGS` all use a layer near the midpoint.

---

### Lines 67--78: Inferring the Short Name

```python
        # Infer short name from model name
        model_lower = model_name.lower()
        if "gemma" in model_lower:
            short_name = "Gemma"
        elif "qwen" in model_lower:
            short_name = "Qwen"
        elif "llama" in model_lower:
            short_name = "Llama"
        elif "mistral" in model_lower:
            short_name = "Mistral"
        else:
            short_name = model_name.split("/")[-1].split("-")[0]
```

The model name string is lowered for case-insensitive comparison. A chain of `if`/`elif` checks whether the name contains a known model family substring (`"gemma"`, `"qwen"`, `"llama"`, `"mistral"`) and assigns the corresponding canonical short name. Note that `"mistral"` is handled here even though no Mistral model appears in `MODEL_CONFIGS` -- this anticipates future use.

If none of the known families match, the fallback on line 78 performs string manipulation:
1. `model_name.split("/")[-1]` takes everything after the last `/`, stripping the organization prefix (e.g., `"meta-llama/Llama-3.3-70B-Instruct"` becomes `"Llama-3.3-70B-Instruct"`).
2. `.split("-")[0]` takes everything before the first hyphen (e.g., `"Llama"`).

This is a best-effort heuristic that works well for HuggingFace naming conventions.

---

### Lines 80--84: Returning the Inferred Config

```python
        return {
            "target_layer": target_layer,
            "total_layers": total_layers,
            "short_name": short_name,
        }
```

A new dictionary is constructed and returned with the three inferred values. Note that unlike entries in `MODEL_CONFIGS`, the inferred config never includes `capping_config` or `capping_experiment` -- those are only available for models that have been manually registered.

---

### Lines 85--86: Error Handling

```python
    except Exception as e:
        raise ValueError(f"Could not infer config for model {model_name}: {e}")
```

If anything inside the `try` block fails -- for example, if the model name is invalid, the HuggingFace Hub is unreachable, or the downloaded config lacks `num_hidden_layers` -- the broad `except Exception` catches it and re-raises a `ValueError` with a descriptive message that includes both the model name and the original error. This converts arbitrary exceptions (network errors, attribute errors, etc.) into a single, clear exception type for callers to handle.
