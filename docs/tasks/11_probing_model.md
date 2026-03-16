# Task 11: Probing Model (internals/model.py)

## Overview

The `ProbingModel` class wraps a HuggingFace causal language model and its tokenizer into a single object that provides helper methods for text generation and activation extraction. It is the central abstraction passed through the pipeline instead of raw `(model, tokenizer)` tuples. The companion function `get_config()` in `models.py` supplies per-model metadata (target layer, total layers, short name, capping config) used by downstream tasks.

**Source files:**
- `assistant_axis/internals/model.py` -- `ProbingModel` class
- `assistant_axis/models.py` -- `MODEL_CONFIGS` dict and `get_config()` function

---

## Sub-Tasks

### Sub-Task 11.1: ProbingModel Constructor (`__init__`)

Loads a HuggingFace model and tokenizer from a model identifier, configuring device placement and precision.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | HuggingFace model identifier for the base model (e.g. `"google/gemma-2-27b-it"`) |
| `device` | `Optional[str]` | `None` | Device specification: `None` / `"auto"` for all GPUs, `"cuda:X"` for single GPU, or a `dict` for a custom device map |
| `max_memory_per_gpu` | `Optional[Dict[int, str]]` | `None` | Maps GPU IDs to max memory strings (e.g. `{0: "40GiB", 1: "40GiB"}`) |
| `chat_model_name` | `Optional[str]` | `None` | Separate HuggingFace identifier for the tokenizer (when it differs from the base model) |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Data type for model weights |

#### Processing

1. **Store metadata** -- `model_name`, `chat_model_name`, and `dtype` are saved as instance attributes.

2. **Load tokenizer** -- The tokenizer is loaded from `chat_model_name` if provided, otherwise from `model_name`. Padding is configured:

```python
tokenizer_source = chat_model_name if chat_model_name else model_name
self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

# Set padding token if not set
if self.tokenizer.pad_token is None:
    self.tokenizer.pad_token = self.tokenizer.eos_token
self.tokenizer.padding_side = "left"
```

3. **Build model loading kwargs** -- A `model_kwargs` dict is constructed based on the `device` and `max_memory_per_gpu` parameters. The logic follows a priority chain:

```python
model_kwargs = {
    "dtype": dtype,
}

if max_memory_per_gpu is not None:
    # Use custom memory limits (for multi-worker setups)
    model_kwargs["device_map"] = "auto"
    model_kwargs["max_memory"] = max_memory_per_gpu
elif device is None or device == "auto":
    # Use all available GPUs automatically
    model_kwargs["device_map"] = "auto"
elif isinstance(device, dict):
    # Custom device map provided
    model_kwargs["device_map"] = device
elif isinstance(device, str) and device.startswith("cuda:"):
    # Single GPU specified - try to use it, but allow sharding if needed
    model_kwargs["device_map"] = "auto"
    gpu_id = int(device.split(":")[-1])
    # Limit to just this GPU
    model_kwargs["max_memory"] = {gpu_id: "139GiB"}
    # Set other GPUs to 0 to prevent usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if i != gpu_id and i not in model_kwargs["max_memory"]:
                model_kwargs["max_memory"][i] = "0GiB"
else:
    # Fallback to auto
    model_kwargs["device_map"] = "auto"
```

4. **Load model** -- The model is loaded via `AutoModelForCausalLM` and set to evaluation mode:

```python
self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
self.model.eval()
```

5. **Initialize caches** -- Layer and model-type caches are set to `None` for lazy loading:

```python
self._layers: Optional[nn.ModuleList] = None
self._model_type: Optional[str] = None
```

#### Output

A fully initialized `ProbingModel` instance with the following attributes:

| Attribute | Type | Description |
|---|---|---|
| `self.model` | `AutoModelForCausalLM` | The loaded HuggingFace model in eval mode |
| `self.tokenizer` | `AutoTokenizer` | The loaded tokenizer with left-padding configured |
| `self.model_name` | `str` | The base model identifier |
| `self.chat_model_name` | `Optional[str]` | The tokenizer model identifier (if different) |
| `self.dtype` | `torch.dtype` | The weight precision |
| `self._layers` | `None` | Layer cache (populated lazily by `get_layers()`) |
| `self._model_type` | `None` | Model type cache (populated lazily by `detect_type()`) |

---

### Sub-Task 11.2: Alternative Constructor (`from_existing`)

Creates a `ProbingModel` from an already-loaded model and tokenizer without going through the full `__init__` loading process. Useful for backward compatibility.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nn.Module` | *(required)* | An already-loaded HuggingFace model |
| `tokenizer` | `AutoTokenizer` | *(required)* | An already-loaded tokenizer |
| `model_name` | `Optional[str]` | `None` | Model name; auto-detected from the model's `name_or_path` attribute if not provided |

This is a `@classmethod`.

#### Processing

1. **Create empty instance** -- Bypasses `__init__` entirely using `cls.__new__(cls)`.

2. **Populate attributes manually**:

```python
instance = cls.__new__(cls)
instance.model = model
instance.tokenizer = tokenizer
instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
instance.chat_model_name = None
instance.dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.bfloat16
instance._layers = None
instance._model_type = None
return instance
```

#### Output

Returns a `ProbingModel` instance wrapping the provided model and tokenizer. The `dtype` is inferred from the model's first parameter if possible, otherwise defaults to `torch.bfloat16`.

---

### Sub-Task 11.3: Get Transformer Layers (`get_layers`)

Locates and returns the transformer layer `ModuleList` from the model, handling different HuggingFace model architectures. Uses lazy caching.

#### Input

No parameters. Operates on `self.model`.

#### Processing

1. **Return cache if available**:

```python
if self._layers is not None:
    return self._layers
```

2. **Try known architecture paths** -- Iterates over a list of `(path_name, accessor_lambda)` tuples covering major model families:

```python
layer_paths = [
    ('model.model.layers', lambda m: m.model.layers),           # Standard (Llama, Gemma 2, Qwen)
    ('model.language_model.layers', lambda m: m.language_model.layers),  # Vision-language (Gemma 3, LLaVA)
    ('model.transformer.h', lambda m: m.transformer.h),         # GPT-style
    ('model.transformer.layers', lambda m: m.transformer.layers),  # Transformer variants
    ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),     # GPT-NeoX
]

for path_name, path_func in layer_paths:
    try:
        layers = path_func(self.model)
        if layers is not None and hasattr(layers, '__len__') and len(layers) > 0:
            self._layers = layers
            return self._layers
    except AttributeError:
        continue
```

3. **Raise informative error** -- If no path matches, constructs a detailed `AttributeError` with model-specific guidance:

```python
model_class = type(self.model).__name__
model_name = getattr(self.model, 'name_or_path', 'Unknown')

error_msg = f"Could not find transformer layers for model '{model_name}' (class: {model_class}). "

if 'gemma' in model_name.lower() and '3' in model_name:
    error_msg += "For Gemma 3 vision models, try loading with Gemma3ForConditionalGeneration instead."
elif 'llava' in model_name.lower():
    error_msg += "For LLaVA models, layers should be at model.language_model.layers."
else:
    tried_paths = [path_name for path_name, _ in layer_paths]
    error_msg += f"Tried paths: {tried_paths}"

raise AttributeError(error_msg)
```

#### Output

| Return | Type | Description |
|---|---|---|
| `self._layers` | `nn.ModuleList` | The transformer layer list; indexable by layer number, supports `len()` |

Raises `AttributeError` if no layers are found.

---

### Sub-Task 11.4: Detect Model Type (`detect_type`)

Determines the model family from the model name string. Uses lazy caching.

#### Input

No parameters. Operates on `self.model_name`.

#### Processing

1. **Return cache if available**:

```python
if self._model_type is not None:
    return self._model_type
```

2. **Match model name against known families** (case-insensitive):

```python
model_name_lower = self.model_name.lower()

if 'qwen' in model_name_lower:
    self._model_type = 'qwen'
elif 'llama' in model_name_lower or 'meta-llama' in model_name_lower:
    self._model_type = 'llama'
elif 'gemma' in model_name_lower:
    self._model_type = 'gemma'
else:
    self._model_type = 'unknown'

return self._model_type
```

#### Output

| Return | Type | Possible Values |
|---|---|---|
| Model type | `str` | `'qwen'`, `'llama'`, `'gemma'`, or `'unknown'` |

Three convenience boolean properties delegate to `detect_type()`:

```python
@property
def is_qwen(self) -> bool:
    return self.detect_type() == 'qwen'

@property
def is_gemma(self) -> bool:
    return self.detect_type() == 'gemma'

@property
def is_llama(self) -> bool:
    return self.detect_type() == 'llama'
```

---

### Sub-Task 11.5: Check System Prompt Support (`supports_system_prompt`)

Checks whether the model's chat template supports a system prompt.

#### Input

No parameters. Operates on `self.model_name`.

#### Processing

Only Gemma 2 models lack system prompt support. All other models (Gemma 3, Llama, Qwen, etc.) support them:

```python
def supports_system_prompt(self) -> bool:
    return 'gemma-2' not in self.model_name.lower()
```

#### Output

| Return | Type | Description |
|---|---|---|
| Supports system prompt | `bool` | `False` if `model_name` contains `"gemma-2"` (case-insensitive), `True` otherwise |

---

### Sub-Task 11.6: Text Generation (`generate`)

Generates text from a prompt using the loaded model. Supports both chat-formatted and raw prompts, including a "swapped role" mode.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | *(required)* | Input text prompt |
| `max_new_tokens` | `int` | `300` | Maximum number of tokens to generate |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `do_sample` | `bool` | `True` | Whether to use sampling (vs greedy decoding) |
| `chat_format` | `bool` | `True` | Whether to apply the chat template |
| `swap` | `bool` | `False` | Whether to use swapped-role formatting |
| `**chat_kwargs` | `Any` | -- | Additional keyword arguments forwarded to `apply_chat_template` |

#### Processing

1. **Format the prompt** -- Three branches depending on `chat_format` and `swap`:

```python
if chat_format:
    if swap:
        # Swapped format: user says the prompt, then we continue
        messages = [{"role": "user", "content": "Hello."}, {"role": "model", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        # Swap 'model' back to 'user' in the template
        parts = formatted_prompt.rsplit('model', 1)
        if len(parts) == 2:
            formatted_prompt = 'user'.join(parts)
    else:
        # Standard format
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
else:
    formatted_prompt = prompt
```

2. **Tokenize and move to device**:

```python
inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
```

3. **Generate with `torch.no_grad()`**:

```python
with torch.no_grad():
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=self.tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
```

4. **Decode only the new tokens** (strips the input prefix):

```python
generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
return generated_text.strip()
```

#### Output

| Return | Type | Description |
|---|---|---|
| Generated text | `str` | The decoded new tokens only (prompt excluded), stripped of leading/trailing whitespace. Special tokens are preserved. |

---

### Sub-Task 11.7: Sample Next Token (`sample_next_token`)

Performs a single forward pass and samples one token from the model's output logits. Used for step-by-step autoregressive generation with activation capture.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `torch.Tensor` | *(required)* | Current input token IDs, shape `(1, seq_len)` |
| `suppress_eos` | `bool` | `True` | Whether to suppress the EOS token by setting its logit to `-inf` |

#### Processing

1. **Forward pass** -- Run the model under `torch.no_grad()` and extract last-position logits:

```python
with torch.no_grad():
    outputs = self.model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last token logits
```

2. **Suppress EOS** -- If enabled, set the EOS token logit to negative infinity:

```python
if suppress_eos:
    eos_token_id = self.tokenizer.eos_token_id
    if eos_token_id is not None:
        logits[eos_token_id] = -float('inf')
```

3. **Sample** -- Convert logits to probabilities via softmax and draw one sample:

```python
probs = torch.softmax(logits, dim=-1)
next_token_id = torch.multinomial(probs, 1).item()
```

4. **Concatenate** -- Append the new token to the input sequence:

```python
updated_input_ids = torch.cat([
    input_ids,
    torch.tensor([[next_token_id]], device=input_ids.device)
], dim=1)

return next_token_id, updated_input_ids
```

#### Output

| Return | Type | Shape | Description |
|---|---|---|---|
| `next_token_id` | `int` | scalar | The sampled token ID |
| `updated_input_ids` | `torch.Tensor` | `(1, seq_len + 1)` | The input sequence with the new token appended |

---

### Sub-Task 11.8: Capture Hidden State (`capture_hidden_state`)

Captures the hidden-state activation vector at a specific layer and token position using a PyTorch forward hook. This is the core activation extraction primitive.

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `torch.Tensor` | *(required)* | Input token IDs, shape `(1, seq_len)` |
| `layer` | `int` | *(required)* | Layer index to capture from (0-based) |
| `position` | `int` | `-1` | Token position to capture (`-1` for the last token) |

#### Processing

1. **Define capture hook** -- A closure that intercepts the layer's output and clones the hidden state at the target position:

```python
captured_state = None

def capture_hook(module, input, output):
    nonlocal captured_state
    # Handle tuple outputs (some models return (hidden_states, ...))
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output

    # Capture the hidden state at specified position
    captured_state = hidden_states[0, position, :].clone().cpu()
```

2. **Register the hook** on the target layer:

```python
layer_module = self.get_layers()[layer]
hook_handle = layer_module.register_forward_hook(capture_hook)
```

3. **Run forward pass** under `torch.inference_mode()`, then remove the hook:

```python
try:
    with torch.inference_mode():
        _ = self.model(input_ids)
finally:
    hook_handle.remove()
```

4. **Validate** -- Raise if capture failed:

```python
if captured_state is None:
    raise ValueError(f"Failed to capture hidden state at layer {layer}, position {position}")

return captured_state
```

#### Output

| Return | Type | Shape | Device | Description |
|---|---|---|---|---|
| `captured_state` | `torch.Tensor` | `(hidden_size,)` | CPU | The hidden-state vector at the specified layer and position, cloned and moved to CPU |

Raises `ValueError` if the hook did not fire.

---

### Sub-Task 11.9: Hidden Size Property (`hidden_size`)

Returns the model's hidden dimension size.

#### Input

No parameters. Operates on `self.model.config`.

#### Processing

```python
@property
def hidden_size(self) -> int:
    return self.model.config.hidden_size
```

#### Output

| Return | Type | Description |
|---|---|---|
| Hidden size | `int` | The hidden dimension of the model (e.g., 2304 for Gemma-2-2B, 3584 for Qwen3-32B, 8192 for Llama-3.3-70B) |

---

### Sub-Task 11.10: Device Property (`device`)

Returns the device of the model's first parameter. For multi-GPU models with `device_map="auto"`, this returns the device of whichever parameter comes first in iteration order.

#### Input

No parameters.

#### Processing

```python
@property
def device(self) -> torch.device:
    return next(self.model.parameters()).device
```

#### Output

| Return | Type | Description |
|---|---|---|
| Device | `torch.device` | The device of the first model parameter (e.g. `device(type='cuda', index=0)`) |

---

### Sub-Task 11.11: Cleanup (`close`)

Frees GPU memory by deleting the model and tokenizer, clearing CUDA caches, and running garbage collection.

#### Input

No parameters.

#### Processing

1. **Delete model and tokenizer**:

```python
if self.model is not None:
    del self.model
    self.model = None

if self.tokenizer is not None:
    del self.tokenizer
    self.tokenizer = None

self._layers = None
```

2. **Clear GPU memory** (only if CUDA is available):

```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

3. **Force garbage collection**:

```python
import gc
gc.collect()
```

#### Output

No return value. After calling `close()`:
- `self.model` is `None`
- `self.tokenizer` is `None`
- `self._layers` is `None`
- GPU memory has been freed and caches emptied

---

### Sub-Task 11.12: Model Configuration Lookup (`get_config` from `models.py`)

Looks up or infers per-model metadata used by downstream tasks (axis computation target layer, capping config, etc.).

#### Input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *(required)* | HuggingFace model identifier (e.g. `"Qwen/Qwen3-32B"`) |

#### Processing

1. **Check known configs** -- If the model name matches a key in `MODEL_CONFIGS`, return a copy:

```python
MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "target_layer": 22,
        "total_layers": 46,
        "short_name": "Gemma",
    },
    "Qwen/Qwen3-32B": {
        "target_layer": 32,
        "total_layers": 64,
        "short_name": "Qwen",
        "capping_config": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "target_layer": 40,
        "total_layers": 80,
        "short_name": "Llama",
        "capping_config": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
    },
}

if model_name in MODEL_CONFIGS:
    return MODEL_CONFIGS[model_name].copy()
```

2. **Infer from model architecture** -- For unknown models, download the HuggingFace config and compute defaults:

```python
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    total_layers = config.num_hidden_layers
    target_layer = total_layers // 2  # Default to middle layer

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

    return {
        "target_layer": target_layer,
        "total_layers": total_layers,
        "short_name": short_name,
    }
except Exception as e:
    raise ValueError(f"Could not infer config for model {model_name}: {e}")
```

#### Output

| Return | Type | Description |
|---|---|---|
| Config dict | `dict` | Always contains `target_layer` (`int`), `total_layers` (`int`), `short_name` (`str`). Known models may also include `capping_config` (`str`) and `capping_experiment` (`str`). |

Raises `ValueError` if the model is not in `MODEL_CONFIGS` and its HuggingFace config cannot be downloaded/parsed.
