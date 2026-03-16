# `assistant_axis/internals/model.py`

## Overview

This file defines the `ProbingModel` class, which wraps a HuggingFace causal language model and its tokenizer into a single convenient object. It provides utilities for loading models onto GPUs (with flexible device placement), generating text, sampling individual tokens, extracting hidden-state activations from specific transformer layers, and cleaning up GPU memory. The class is designed to be the central object passed around the codebase instead of raw `(model, tokenizer)` tuples.

---

## Line-by-Line Explanation

### Module Docstring

```python
"""ProbingModel - Wraps HuggingFace model with utilities for activation extraction."""
```

A one-line module docstring summarising the purpose of the file: it provides a wrapper class focused on activation extraction from HuggingFace models.

---

### Imports

```python
from __future__ import annotations
```

Enables PEP 604-style postponed evaluation of annotations so that type hints like `ProbingModel` can be used inside the class body before the class is fully defined. All annotations in the module become strings that are resolved lazily.

```python
from typing import Dict, Optional
```

Imports `Dict` and `Optional` from the `typing` module. `Optional[X]` is shorthand for `X | None`, and `Dict[K, V]` represents a dictionary type. These are used in the constructor's type hints.

```python
import torch
```

Imports PyTorch, the deep-learning framework used for tensor operations, GPU management, and model inference throughout the class.

```python
import torch.nn as nn
```

Imports PyTorch's neural-network module. Used here for the `nn.ModuleList` type annotation (the container type that holds transformer layers) and the `nn.Module` type used in `from_existing`.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

Imports two HuggingFace `transformers` auto-classes:
- `AutoTokenizer`: automatically selects and loads the correct tokenizer for a given model name.
- `AutoModelForCausalLM`: automatically selects and loads the correct causal (autoregressive) language model architecture for a given model name.

---

### Class Definition and Docstring

```python
class ProbingModel:
    """
    Wraps a HuggingFace model and tokenizer with helper methods for generation
    and activation extraction.

    This is the central object you pass around instead of (model, tokenizer) tuples.
    """
```

Defines the `ProbingModel` class. The docstring explains its role: it bundles a model and tokenizer together and exposes convenience methods for text generation and activation (hidden state) extraction.

---

### `__init__` Method

```python
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_memory_per_gpu: Optional[Dict[int, str]] = None,
        chat_model_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
```

The constructor accepts five parameters:
- `model_name`: the HuggingFace Hub identifier for the model to load (e.g. `"meta-llama/Llama-3-8B"`).
- `device`: controls where the model is placed. Can be `None` (auto), a CUDA device string like `"cuda:0"`, a custom device-map dictionary, or `"auto"`.
- `max_memory_per_gpu`: an optional dictionary mapping GPU indices to maximum memory strings (e.g. `{0: "40GiB"}`), useful when sharing GPUs across workers.
- `chat_model_name`: an optional alternative model name to load the tokenizer from, useful when the base model and chat-tuned model use different tokenizers.
- `dtype`: the data type for model weights, defaulting to `torch.bfloat16` (a 16-bit floating point format that saves memory while preserving a wide dynamic range).

```python
        """
        Initialize and load a HuggingFace model and tokenizer.

        Args:
            model_name: HuggingFace model identifier for the base model
            device: Device specification - can be:
                - None: use all available GPUs with device_map="auto"
                - "cuda:X": use single GPU (will auto-shard if model is too large)
                - dict: custom device_map
            max_memory_per_gpu: Optional dict mapping GPU ids to max memory (e.g. {0: "40GiB", 1: "40GiB"})
            chat_model_name: Optional HuggingFace model identifier for tokenizer (if different from base model)
            dtype: Data type for model weights (default: torch.bfloat16)
        """
```

The constructor's docstring documents each argument and explains the different `device` options.

```python
        self.model_name = model_name
        self.chat_model_name = chat_model_name
        self.dtype = dtype
```

Stores the model name, optional chat model name, and data type as instance attributes for later reference.

---

#### Tokenizer Loading

```python
        # Load tokenizer from chat_model_name if provided, otherwise from model_name
        tokenizer_source = chat_model_name if chat_model_name else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
```

Determines which model identifier to use for the tokenizer. If `chat_model_name` is provided, the tokenizer is loaded from that model (because chat-tuned models sometimes have a different tokenizer with special chat tokens). Otherwise, the base `model_name` is used. `AutoTokenizer.from_pretrained` downloads (or loads from cache) the correct tokenizer.

```python
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
```

Many models (especially decoder-only LLMs) do not define a padding token. If none is set, the end-of-sequence token is reused as the pad token. `padding_side = "left"` ensures that when batching, padding is added to the left side of sequences. This is important for causal (left-to-right) language models because the final token positions then align across sequences in a batch, making generation straightforward.

---

#### Model Loading -- Building Keyword Arguments

```python
        # Build model loading kwargs
        model_kwargs = {
            "dtype": dtype,
        }
```

Begins constructing a dictionary of keyword arguments that will be passed to `AutoModelForCausalLM.from_pretrained`. The data type is always included.

```python
        if max_memory_per_gpu is not None:
            # Use custom memory limits (for multi-worker setups)
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory_per_gpu
```

If the caller provided explicit per-GPU memory limits, the model is loaded with `device_map="auto"` (HuggingFace's Accelerate library automatically distributes model layers across devices) and the memory constraints are passed through. This is useful when multiple processes share the same GPUs and each must stay within a memory budget.

```python
        elif device is None or device == "auto":
            # Use all available GPUs automatically
            model_kwargs["device_map"] = "auto"
```

If no device is specified (or explicitly set to `"auto"`), the model is loaded with automatic device mapping, spreading layers across all available GPUs.

```python
        elif isinstance(device, dict):
            # Custom device map provided
            model_kwargs["device_map"] = device
```

If `device` is a dictionary, it is treated as a custom device map (e.g. `{0: "cuda:0", 1: "cuda:1", "lm_head": "cuda:0"}`), giving the caller full control over which model component goes where.

```python
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
```

When a specific CUDA device string like `"cuda:2"` is given, the method still uses `device_map="auto"` (to benefit from automatic layer placement) but constrains memory so only the requested GPU is used. It parses the GPU index from the string, assigns it a generous 139 GiB budget, and sets every other GPU's budget to `"0GiB"` so that Accelerate will not place any layers on them.

```python
        else:
            # Fallback to auto
            model_kwargs["device_map"] = "auto"
```

Any unrecognised `device` value falls back to automatic device mapping.

---

#### Model Loading -- Instantiation

```python
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
```

Loads the model from the HuggingFace Hub (or local cache) using the constructed keyword arguments. `model.eval()` switches the model to evaluation mode, which disables dropout and other training-only behaviours.

```python
        # Cache for layers (lazy loaded)
        self._layers: Optional[nn.ModuleList] = None
        self._model_type: Optional[str] = None
```

Initialises two private caches to `None`. `_layers` will store the result of `get_layers()` after its first call so that repeated access does not re-traverse the model's attribute tree. `_model_type` will cache the result of `detect_type()`.

---

### `from_existing` Class Method

```python
    @classmethod
    def from_existing(cls, model: nn.Module, tokenizer: AutoTokenizer, model_name: Optional[str] = None) -> ProbingModel:
```

A class method that serves as an alternative constructor. It creates a `ProbingModel` from an already-loaded model and tokenizer, avoiding the overhead of loading from disk or downloading.

```python
        """
        Create a ProbingModel from an already-loaded model and tokenizer.

        This is useful for backwards compatibility or when you already have a model loaded.

        Args:
            model: Already-loaded HuggingFace model
            tokenizer: Already-loaded tokenizer
            model_name: Optional model name (will try to detect from model if not provided)

        Returns:
            ProbingModel wrapping the provided model and tokenizer
        """
```

Docstring explaining the purpose and parameters.

```python
        # Create an "empty" instance without going through __init__
        instance = cls.__new__(cls)
```

Uses `__new__` to allocate the object without calling `__init__`. This avoids the automatic model loading that `__init__` performs.

```python
        instance.model = model
        instance.tokenizer = tokenizer
```

Assigns the already-loaded model and tokenizer to the new instance.

```python
        instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
```

Sets the model name. If the caller did not provide one, it tries to read the `name_or_path` attribute that HuggingFace models typically set during loading. Falls back to `'Unknown'`.

```python
        instance.chat_model_name = None
```

Sets `chat_model_name` to `None` since we are wrapping an existing model and do not have a separate chat tokenizer source.

```python
        instance.dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.bfloat16
```

Infers the data type from the model's first parameter. If the model does not have a `parameters()` method (unusual edge case), defaults to `torch.bfloat16`.

```python
        instance._layers = None
        instance._model_type = None
        return instance
```

Initialises the layer and model-type caches to `None` (they will be populated lazily) and returns the new instance.

---

### `hidden_size` Property

```python
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.model.config.hidden_size
```

A read-only property that returns the model's hidden dimension size (e.g. 4096 for many Llama models) from the model's configuration object. This is the dimensionality of each hidden-state vector.

---

### `device` Property

```python
    @property
    def device(self) -> torch.device:
        """Get the device of the first model parameter."""
        return next(self.model.parameters()).device
```

A read-only property that returns the device (CPU or a specific CUDA GPU) where the first model parameter resides. This is used as the default device for tokenized inputs.

---

### `get_layers` Method

```python
    def get_layers(self) -> nn.ModuleList:
        """
        Get the transformer layers from the model, handling different architectures.

        Returns:
            The layers object (usually a ModuleList) that can be indexed and has len()

        Raises:
            AttributeError: If no layers can be found with helpful error message
        """
```

Returns the transformer layer list from the model. Different model architectures store their layers at different attribute paths, so this method tries several known paths.

```python
        if self._layers is not None:
            return self._layers
```

Returns the cached layers immediately if they were already found in a previous call.

```python
        # Try common paths for transformer layers
        layer_paths = [
            ('model.model.layers', lambda m: m.model.layers),  # Standard language models (Llama, Gemma 2, Qwen, etc.)
            ('model.language_model.layers', lambda m: m.language_model.layers),  # Vision-language models (Gemma 3, LLaVA, etc.)
            ('model.transformer.h', lambda m: m.transformer.h),  # GPT-style models
            ('model.transformer.layers', lambda m: m.transformer.layers),  # Some transformer variants
            ('model.gpt_neox.layers', lambda m: m.gpt_neox.layers),  # GPT-NeoX models
        ]
```

Defines a list of `(name, accessor_function)` tuples. Each tuple represents one known attribute path where transformer layers might live. The names are for error messages; the lambdas perform the actual attribute access. The five paths cover the major HuggingFace model families.

```python
        for path_name, path_func in layer_paths:
            try:
                layers = path_func(self.model)
                if layers is not None and hasattr(layers, '__len__') and len(layers) > 0:
                    self._layers = layers
                    return self._layers
            except AttributeError:
                continue
```

Iterates through each path. For each one, it attempts to access the layers using the lambda. If the access succeeds and the result is a non-empty sequence (has `__len__` and length > 0), the layers are cached and returned. If the attribute path does not exist, an `AttributeError` is caught and the loop continues to the next path.

```python
        # If we get here, no layers were found
        model_class = type(self.model).__name__
        model_name = getattr(self.model, 'name_or_path', 'Unknown')
```

If none of the known paths worked, collects diagnostic information: the model's Python class name and its HuggingFace identifier.

```python
        # Provide specific guidance for known cases
        error_msg = f"Could not find transformer layers for model '{model_name}' (class: {model_class}). "
```

Begins constructing a descriptive error message.

```python
        if 'gemma' in model_name.lower() and '3' in model_name:
            error_msg += "For Gemma 3 vision models, try loading with Gemma3ForConditionalGeneration instead."
        elif 'llava' in model_name.lower():
            error_msg += "For LLaVA models, layers should be at model.language_model.layers."
        else:
            # Show what paths were tried
            tried_paths = [path_name for path_name, _ in layer_paths]
            error_msg += f"Tried paths: {tried_paths}"
```

Adds model-specific guidance to the error message. For Gemma 3 vision models, it suggests using a different model class. For LLaVA models, it points to the expected path. For all other models, it lists every path that was attempted, so the user can debug the issue.

```python
        raise AttributeError(error_msg)
```

Raises an `AttributeError` with the constructed message.

---

### `detect_type` Method

```python
    def detect_type(self) -> str:
        """
        Detect the model family (qwen, llama, gemma, etc).

        Returns:
            Model type as a string: 'qwen', 'llama', 'gemma', or 'unknown'
        """
```

Detects which model family this model belongs to based on its name. Returns a simple lowercase string identifier.

```python
        if self._model_type is not None:
            return self._model_type
```

Returns the cached result if the model type has already been detected.

```python
        model_name_lower = self.model_name.lower()
```

Converts the model name to lowercase for case-insensitive matching.

```python
        if 'qwen' in model_name_lower:
            self._model_type = 'qwen'
        elif 'llama' in model_name_lower or 'meta-llama' in model_name_lower:
            self._model_type = 'llama'
        elif 'gemma' in model_name_lower:
            self._model_type = 'gemma'
        else:
            self._model_type = 'unknown'
```

Checks the model name for known substrings and sets the type accordingly. The `'meta-llama'` check is technically redundant (since it contains `'llama'`), but makes the intent explicit. Any unrecognised model gets the type `'unknown'`.

```python
        return self._model_type
```

Returns the detected (and now cached) model type.

---

### `is_qwen`, `is_gemma`, `is_llama` Properties

```python
    @property
    def is_qwen(self) -> bool:
        """Check if this is a Qwen model."""
        return self.detect_type() == 'qwen'

    @property
    def is_gemma(self) -> bool:
        """Check if this is a Gemma model."""
        return self.detect_type() == 'gemma'

    @property
    def is_llama(self) -> bool:
        """Check if this is a Llama model."""
        return self.detect_type() == 'llama'
```

Three boolean convenience properties. Each calls `detect_type()` and compares the result to the relevant family string. They provide a clean API for conditional logic elsewhere in the codebase (e.g. `if probing_model.is_qwen: ...`).

---

### `supports_system_prompt` Method

```python
    def supports_system_prompt(self) -> bool:
        """
        Check if this model supports system prompts in its chat template.

        Returns:
            True if the model supports system prompts, False otherwise.

        Note:
            Only Gemma 2 doesn't support system prompts. All other models
            (including Gemma 3, Llama, Qwen, etc.) support them.
        """
        return 'gemma-2' not in self.model_name.lower()
```

Returns `True` unless the model name contains `"gemma-2"`. Gemma 2 is the only supported model family that lacks system-prompt support in its chat template. This method lets callers decide whether to include a system message when formatting chat prompts.

---

### `generate` Method

```python
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        do_sample: bool = True,
        chat_format: bool = True,
        swap: bool = False,
        **chat_kwargs,
    ) -> str:
```

Generates text from a prompt. Parameters:
- `prompt`: the input text.
- `max_new_tokens`: maximum number of tokens to generate (default 300).
- `temperature`: controls randomness; higher values produce more diverse outputs.
- `do_sample`: if `True`, uses sampling; if `False`, uses greedy decoding.
- `chat_format`: whether to wrap the prompt in the model's chat template.
- `swap`: enables a special "swapped role" formatting mode.
- `**chat_kwargs`: any additional keyword arguments forwarded to `apply_chat_template`.

```python
        # Format as chat if requested
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
```

When `swap=True`, a two-message conversation is constructed: a dummy user greeting followed by the actual prompt attributed to the `"model"` role. After applying the chat template, the last occurrence of the string `'model'` in the formatted output is replaced with `'user'`. This creates a prompt where it appears the user said both messages, which can be useful for certain probing or jailbreak-evaluation scenarios.

```python
            else:
                # Standard format
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
```

The standard path: wraps the prompt as a single user message and applies the chat template. `tokenize=False` returns a string (not token IDs), and `add_generation_prompt=True` appends the model's expected reply prefix.

```python
        else:
            formatted_prompt = prompt
```

If `chat_format` is `False`, the raw prompt is used as-is with no template applied.

```python
        # Tokenize and move to the device of the first model parameter
        # This handles multi-GPU models correctly
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
```

Tokenizes the formatted prompt into PyTorch tensors and moves them to the same device as the model's first parameter. For multi-GPU models using `device_map="auto"`, the first parameter's device is where the model expects its input.

```python
        # Generate
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

Runs the model's `generate` method inside a `torch.no_grad()` context to disable gradient computation (saving memory and time). The generation parameters are passed through, and a `repetition_penalty` of 1.1 is applied to discourage the model from repeating itself. `pad_token_id` is set to the EOS token ID to avoid warnings.

```python
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        return generated_text.strip()
```

Extracts only the newly generated tokens by slicing off the input portion (`outputs[0][inputs.input_ids.shape[1]:]`). Decodes them back to a string with `skip_special_tokens=False` (preserving special tokens like EOS so the caller can see them if needed). Returns the result after stripping leading/trailing whitespace.

---

### `sample_next_token` Method

```python
    def sample_next_token(
        self,
        input_ids: torch.Tensor,
        suppress_eos: bool = True,
    ) -> tuple[int, torch.Tensor]:
```

Samples a single next token from the model's logits. Returns a tuple of the sampled token ID and the updated input IDs tensor (with the new token appended). The `suppress_eos` flag controls whether the end-of-sequence token can be sampled.

```python
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]  # Last token logits
```

Runs a forward pass through the model with gradient computation disabled. Extracts the logits for the last token position of the first (and typically only) sequence in the batch. These logits are a vector of scores across the entire vocabulary.

```python
            # Suppress EOS token if requested
            if suppress_eos:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None:
                    logits[eos_token_id] = -float('inf')
```

If `suppress_eos` is `True`, sets the logit for the EOS token to negative infinity. After softmax, this gives it a probability of zero, preventing it from being sampled. This is useful when you want to force the model to keep generating.

```python
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
```

Converts logits to probabilities using softmax, then samples one token ID from the probability distribution using `torch.multinomial`. `.item()` extracts the Python integer from the single-element tensor.

```python
            # Update input_ids
            updated_input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token_id]], device=input_ids.device)
            ], dim=1)

            return next_token_id, updated_input_ids
```

Creates the updated input IDs by concatenating the original `input_ids` with the newly sampled token (wrapped in a tensor of matching shape and device). Returns both the token ID and the extended sequence so the caller can continue generation.

---

### `capture_hidden_state` Method

```python
    def capture_hidden_state(
        self,
        input_ids: torch.Tensor,
        layer: int,
        position: int = -1,
    ) -> torch.Tensor:
```

Captures the hidden-state activation at a specific layer and token position during a forward pass. This is the core "probing" capability of the class. Parameters:
- `input_ids`: the tokenized input.
- `layer`: which transformer layer to capture from (0-indexed).
- `position`: which token position's hidden state to extract (`-1` means the last token).

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

Defines a forward hook function that will be called when the target layer executes. The `nonlocal` keyword allows the inner function to write to `captured_state` in the enclosing scope. The hook handles two output formats: some layers return a tuple (where the first element is the hidden states), while others return the hidden states directly. It extracts the hidden state at the specified position from the first batch element (`[0, position, :]`), clones it (to avoid it being overwritten by later computation), and moves it to CPU.

```python
        # Register hook on target layer
        layer_module = self.get_layers()[layer]
        hook_handle = layer_module.register_forward_hook(capture_hook)
```

Uses `get_layers()` to find the transformer layers, indexes into the specified layer, and registers the hook. `register_forward_hook` returns a handle that can be used to remove the hook later.

```python
        try:
            with torch.inference_mode():
                _ = self.model(input_ids)
        finally:
            hook_handle.remove()
```

Runs a full forward pass through the model inside `torch.inference_mode()` (which is more efficient than `torch.no_grad()` as it also disables version counting and view tracking). The output is discarded (`_ =`) because the hook has already captured the data we need. The `finally` block ensures the hook is always removed, even if an error occurs during the forward pass. This prevents hooks from accumulating.

```python
        if captured_state is None:
            raise ValueError(f"Failed to capture hidden state at layer {layer}, position {position}")

        return captured_state
```

If the hook was never triggered (which would indicate something went wrong), raises a `ValueError`. Otherwise, returns the captured hidden-state tensor.

---

### `close` Method

```python
    def close(self):
        """Clean up model resources and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._layers = None
```

Deletes the model and tokenizer objects and sets their attributes to `None`. Also clears the cached layers reference. The `del` statements remove the Python references, making the objects eligible for garbage collection.

```python
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
```

If a CUDA GPU is available, `empty_cache()` releases all unused cached memory back to the GPU so that other processes or allocations can use it. `synchronize()` blocks until all CUDA operations have completed, ensuring the memory is fully freed before proceeding.

```python
        # Force garbage collection
        import gc
        gc.collect()
```

Imports Python's garbage collector and forces an immediate collection cycle. This ensures that any circular references or lingering objects (particularly large tensors) are cleaned up promptly rather than waiting for automatic garbage collection. The `import gc` is done locally here rather than at module level because it is only needed during cleanup.
