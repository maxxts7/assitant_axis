# notebooks/steer.ipynb

## Overview

This notebook demonstrates how to steer the outputs of a large language model (Qwen 3 32B) using the "assistant axis" -- a direction in activation space that separates role-playing behavior from default assistant behavior. It covers two intervention techniques:

1. **Additive steering**: shifting model activations along the axis by a fixed coefficient to push the model toward more assistant-like or more role-playing behavior.
2. **Activation capping**: a more targeted intervention that clips activations exceeding a threshold along the axis direction, used to mitigate persona drift without fully overriding the model's behavior.

The notebook loads the model and precomputed axis vectors from HuggingFace, defines helper functions for both techniques, and runs side-by-side comparisons showing how each intervention changes model outputs.

---

## Cell-by-Cell Explanation

### Cell 0 -- Markdown

> # Steering Demo
>
> This notebook demonstrates steering model outputs using the assistant axis.

This is the title cell. It introduces the notebook's purpose: demonstrating activation steering using the assistant axis.

---

### Cell 1 -- Code: Imports

```python
import sys
sys.path.insert(0, '..')

import torch
from IPython.display import display, Markdown
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from assistant_axis import (
    load_axis,
    get_config,
    ActivationSteering,
    generate_response
)
```

- `import sys` -- imports Python's built-in `sys` module, which provides access to system-specific parameters and functions.
- `sys.path.insert(0, '..')` -- prepends the parent directory (`..`) to the module search path. This allows importing from the `assistant_axis` package located one directory above the notebook.
- `import torch` -- imports PyTorch, the deep learning framework used for tensor operations, model loading, and activation manipulation.
- `from IPython.display import display, Markdown` -- imports IPython display utilities for rendering rich output (Markdown) in the notebook. These are imported but not used in the cells shown.
- `from huggingface_hub import hf_hub_download` -- imports the `hf_hub_download` function, which downloads individual files from a HuggingFace Hub repository.
- `from transformers import AutoModelForCausalLM, AutoTokenizer` -- imports the Hugging Face `transformers` classes for automatically loading a causal language model (`AutoModelForCausalLM`) and its corresponding tokenizer (`AutoTokenizer`) based on a model name.
- `from assistant_axis import load_axis` -- imports the `load_axis` function, which loads a precomputed axis vector from a `.pt` file.
- `from assistant_axis import get_config` -- imports `get_config`, which retrieves model-specific configuration (e.g., which layer to target, paths to capping configs).
- `from assistant_axis import ActivationSteering` -- imports the `ActivationSteering` context manager, which hooks into the model's forward pass to add a steering vector to activations at a specified layer.
- `from assistant_axis import generate_response` -- imports `generate_response`, a utility that handles tokenizing a conversation, running the model's generate method, and decoding the output back to text.

---

### Cell 2 -- Markdown

> ## Load Model and Axis

Section header indicating the next cells will load the model and the axis vector.

---

### Cell 3 -- Code: Configuration

```python
# Configuration
MODEL_NAME = "Qwen/Qwen3-32B"
MODEL_SHORT = "qwen-3-32b"
REPO_ID = "lu-christina/assistant-axis-vectors"

# Get model config
config = get_config(MODEL_NAME)
TARGET_LAYER = config["target_layer"]
print(f"Model: {MODEL_NAME}")
print(f"Target layer: {TARGET_LAYER}")
```

- `MODEL_NAME = "Qwen/Qwen3-32B"` -- sets the full HuggingFace model identifier for Qwen 3 32B, used to load the model and tokenizer.
- `MODEL_SHORT = "qwen-3-32b"` -- sets a short name used to construct the file path when downloading the axis vector from the HuggingFace dataset repository.
- `REPO_ID = "lu-christina/assistant-axis-vectors"` -- sets the HuggingFace dataset repository ID where the precomputed axis vectors and capping configs are hosted.
- `config = get_config(MODEL_NAME)` -- calls `get_config` with the model name to retrieve a dictionary of model-specific settings (target layer, capping config path, recommended capping experiment, etc.).
- `TARGET_LAYER = config["target_layer"]` -- extracts the target layer index (32) from the config. This is the transformer layer where the steering intervention will be applied.
- The two `print` statements display the model name and the target layer number. The output confirms `Model: Qwen/Qwen3-32B` and `Target layer: 32`.

---

### Cell 4 -- Code: Load Model

```python
# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
)
print("Model loaded!")
```

- `print("Loading model...")` -- prints a status message before the potentially slow model loading begins.
- `tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)` -- downloads (or loads from cache) the tokenizer for Qwen 3 32B. The tokenizer converts text to token IDs and back.
- `if tokenizer.pad_token is None:` -- checks whether the tokenizer has a dedicated padding token defined.
- `tokenizer.pad_token = tokenizer.eos_token` -- if no pad token exists, assigns the end-of-sequence token as the pad token. This is necessary for batched generation to work correctly.
- `model = AutoModelForCausalLM.from_pretrained(...)` -- downloads (or loads from cache) the Qwen 3 32B model weights and instantiates the model.
  - `MODEL_NAME` -- specifies which model to load.
  - `device_map="auto"` -- automatically distributes the model across available GPUs (and CPU if needed), handling the placement of each layer.
  - `dtype=torch.bfloat16` -- loads the model weights in bfloat16 precision, which halves memory usage compared to float32 while maintaining a wide dynamic range.
- `print("Model loaded!")` -- prints a confirmation message after loading completes. The output shows a progress bar for loading 17 checkpoint shards.

---

### Cell 5 -- Code: Load Axis

```python
# Load axis from HuggingFace
axis_path = hf_hub_download(repo_id=REPO_ID, filename=f"{MODEL_SHORT}/assistant_axis.pt", repo_type="dataset")
axis = load_axis(axis_path)
print(f"Axis shape: {axis.shape}")
```

- `axis_path = hf_hub_download(...)` -- downloads the axis vector file from the HuggingFace dataset repository. The `filename` argument constructs the path as `qwen-3-32b/assistant_axis.pt`. The `repo_type="dataset"` parameter indicates this is a dataset repo, not a model repo. The function returns the local file path where the file was downloaded/cached.
- `axis = load_axis(axis_path)` -- loads the axis tensor from the `.pt` file. The result is a PyTorch tensor containing the axis vector for each layer.
- `print(f"Axis shape: {axis.shape}")` -- prints the shape of the axis tensor. The output shows `torch.Size([64, 5120])`, meaning there are 64 layer-specific axis vectors, each of dimension 5120 (the hidden size of Qwen 3 32B).

---

### Cell 6 -- Markdown

> ## Steering Demo
>
> The axis points from role-playing toward default assistant behavior.
> - Positive coefficient: more assistant-like
> - Negative coefficient: more role-playing

This markdown cell explains the semantics of the axis direction. A positive steering coefficient pushes the model toward default assistant behavior, while a negative coefficient pushes it toward role-playing behavior.

---

### Cell 7 -- Code: Steering Helper Function

```python
def generate_with_steering(prompt, coefficient, system_prompt=None):
    """Generate response with steering applied."""

    # Build conversation
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": prompt})

    # Get axis vector for target layer
    axis_vector = axis[TARGET_LAYER]

    if coefficient == 0:
        # No steering
        response = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    else:
        # Apply steering
        with ActivationSteering(
            model,
            steering_vectors=[axis_vector],
            coefficients=[coefficient],
            layer_indices=[TARGET_LAYER]
        ):
            response = generate_response(model, tokenizer, conversation, max_new_tokens=512)

    return response
```

- `def generate_with_steering(prompt, coefficient, system_prompt=None):` -- defines a function that generates a model response with optional activation steering. It accepts a user prompt string, a steering coefficient (float), and an optional system prompt.
- `conversation = []` -- initializes an empty list to hold the chat messages in the OpenAI-style conversation format.
- `if system_prompt:` -- checks if a system prompt was provided.
- `conversation.append({"role": "system", "content": system_prompt})` -- if provided, adds the system prompt as the first message with the "system" role.
- `conversation.append({"role": "user", "content": prompt})` -- adds the user's prompt as a message with the "user" role.
- `axis_vector = axis[TARGET_LAYER]` -- extracts the axis vector for the target layer (layer 32) from the full axis tensor. This is a 1D tensor of shape `[5120]`.
- `if coefficient == 0:` -- checks if the coefficient is zero, meaning no steering should be applied.
- `response = generate_response(model, tokenizer, conversation, max_new_tokens=512)` -- generates a response without any steering, producing up to 512 new tokens. The `generate_response` utility handles formatting the conversation with the tokenizer's chat template, tokenizing, running `model.generate`, and decoding the output.
- `else:` -- if the coefficient is nonzero, steering is applied.
- `with ActivationSteering(model, steering_vectors=[axis_vector], coefficients=[coefficient], layer_indices=[TARGET_LAYER]):` -- creates an `ActivationSteering` context manager that installs forward hooks on the model. While active, it adds `coefficient * axis_vector` to the model's hidden state at layer `TARGET_LAYER` during every forward pass. The lists allow applying multiple steering vectors at different layers simultaneously, but here only one is used.
- `response = generate_response(model, tokenizer, conversation, max_new_tokens=512)` -- generates the response while the steering hook is active. When the `with` block exits, the hooks are removed and the model returns to its normal behavior.
- `return response` -- returns the generated text.

---

### Cell 8 -- Code: Define Test Prompts

```python
# Test prompt
PROMPT = "What is your name?"
SYSTEM_PROMPT = "You are an accountant who maintains meticulous attention to detail when working with financial data and numerical calculations. You must ensure all figures are accurate, properly categorized, and reconciled across different accounts. Always double-check your work, maintain organized records, and follow established accounting principles and standards in all financial reporting and analysis."

print(f"System: {SYSTEM_PROMPT}")
print(f"User: {PROMPT}")
print("=" * 60)
```

- `PROMPT = "What is your name?"` -- sets the user prompt to a simple identity question. This is a good test because it reveals whether the model responds as its default assistant self or adopts the persona from the system prompt.
- `SYSTEM_PROMPT = "You are an accountant..."` -- sets a detailed system prompt instructing the model to behave as a meticulous accountant. This establishes a role-playing persona for the model.
- The three `print` statements display the system prompt, user prompt, and a separator line of 60 equals signs, making the output easier to read.

---

### Cell 9 -- Code: Run Steering Comparison

```python
# Generate with different steering coefficients
# 0.0 is without steering
coefficients = [0.0, -10.0]

for coeff in coefficients:
    if coeff == 0:
        print(f"\n### BASELINE")
    else:
        print(f"\n### Coefficient: {coeff}")
    print("-" * 40)

    response = generate_with_steering(PROMPT, coeff, SYSTEM_PROMPT)
    print(response)

    if len(response) > 500:
        print("...")
```

- `coefficients = [0.0, -10.0]` -- defines the list of steering coefficients to test. `0.0` is the unsteered baseline; `-10.0` pushes strongly toward role-playing behavior (the negative direction along the axis).
- `for coeff in coefficients:` -- iterates over each coefficient.
- `if coeff == 0:` / `else:` -- prints either "BASELINE" or the coefficient value as a section header.
- `print("-" * 40)` -- prints a separator line of 40 dashes.
- `response = generate_with_steering(PROMPT, coeff, SYSTEM_PROMPT)` -- calls the helper function defined in Cell 7 to generate a response with the given coefficient.
- `print(response)` -- prints the generated response.
- `if len(response) > 500:` / `print("...")` -- if the response is longer than 500 characters, prints an ellipsis to indicate truncation in the display.

**Output observations:**
- **Baseline (0.0):** The model identifies itself as "Qwen," a language model by Tongyi Lab, and clarifies it is not a real accountant -- it breaks character and responds as its default assistant self.
- **Coefficient -10.0:** The model fully adopts the accountant persona, introducing itself as "Evelyn Hartwell" at a fictional firm "Lockwood & Thorne, CPA," speaking in character with accounting-themed language. The negative coefficient successfully pushed the model deeper into role-playing.

---

### Cell 10 -- Markdown

> ## Activation Capping
>
> Activation capping is a more targeted intervention that prevents activations from exceeding a threshold along a specific direction. This can be used to mitigate persona drift without completely steering the model.
>
> Key differences from additive steering:
> - **Addition**: shifts all activations in a direction
> - **Capping**: only modifies activations that exceed a threshold
>
> Pre-computed capping configs are available for Qwen 3 32B and Llama 3.3 70B.

This markdown cell introduces the second technique: activation capping. It explains that unlike additive steering (which shifts all activations), capping only intervenes when activations along the axis direction exceed a specified threshold. This makes it a gentler intervention for preventing persona drift.

---

### Cell 11 -- Code: Load Capping Config

```python
# Load capping config from HuggingFace
from assistant_axis import load_capping_config, build_capping_steerer

# Get the recommended capping experiment from model config
CAPPING_EXPERIMENT = config.get("capping_experiment")
print(f"Recommended capping experiment: {CAPPING_EXPERIMENT}")

# Download and load capping config
capping_config_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=config["capping_config"],
    repo_type="dataset"
)
capping_config = load_capping_config(capping_config_path)

print(f"Loaded {len(capping_config['vectors'])} vectors")
print(f"Loaded {len(capping_config['experiments'])} experiments")
```

- `from assistant_axis import load_capping_config, build_capping_steerer` -- imports two additional functions from the `assistant_axis` package. `load_capping_config` parses a capping configuration file, and `build_capping_steerer` constructs a context manager that applies the capping intervention.
- `CAPPING_EXPERIMENT = config.get("capping_experiment")` -- retrieves the recommended capping experiment name from the model config. The output shows this is `"layers_46:54-p0.25"`, meaning it caps activations at layers 46 through 54 using a threshold at the 25th percentile.
- `print(f"Recommended capping experiment: {CAPPING_EXPERIMENT}")` -- prints the experiment name.
- `capping_config_path = hf_hub_download(...)` -- downloads the capping configuration file from HuggingFace. The filename comes from `config["capping_config"]`, and `repo_type="dataset"` specifies it is a dataset repository.
- `capping_config = load_capping_config(capping_config_path)` -- loads and parses the capping configuration file into a dictionary containing vectors and experiment definitions.
- `print(f"Loaded {len(capping_config['vectors'])} vectors")` -- prints how many axis vectors are in the config. The output shows 64 (one per layer).
- `print(f"Loaded {len(capping_config['experiments'])} experiments")` -- prints how many capping experiments are defined. The output shows 124 different experiment configurations, each specifying different layer ranges and percentile thresholds.

---

### Cell 12 -- Code: Capping Helper Function

```python
def generate_with_capping(prompt, system_prompt=None, use_capping=True):
    """Generate response with activation capping applied."""

    # Build conversation
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": prompt})

    if use_capping:
        # Build capping steerer from config
        capping_steerer = build_capping_steerer(model, capping_config, CAPPING_EXPERIMENT)
        with capping_steerer:
            response = generate_response(model, tokenizer, conversation, max_new_tokens=512)
    else:
        response = generate_response(model, tokenizer, conversation, max_new_tokens=512)

    return response
```

- `def generate_with_capping(prompt, system_prompt=None, use_capping=True):` -- defines a function that generates a model response with optional activation capping. It takes a user prompt, an optional system prompt, and a boolean flag to enable/disable capping.
- `conversation = []` -- initializes the conversation message list.
- `if system_prompt:` / `conversation.append(...)` -- adds the system prompt to the conversation if provided.
- `conversation.append({"role": "user", "content": prompt})` -- adds the user message.
- `if use_capping:` -- checks whether capping should be applied.
- `capping_steerer = build_capping_steerer(model, capping_config, CAPPING_EXPERIMENT)` -- constructs a capping context manager using the model, the loaded capping configuration, and the recommended experiment name. This sets up the hooks that will clip activations at the thresholds defined in the experiment.
- `with capping_steerer:` -- activates the capping hooks for the duration of generation.
- `response = generate_response(model, tokenizer, conversation, max_new_tokens=512)` -- generates the response (up to 512 new tokens) while capping is active.
- `else:` / `response = generate_response(...)` -- if capping is disabled, generates the response normally without any intervention.
- `return response` -- returns the generated text.

---

### Cell 13 -- Code: Run Capping Comparison

```python
# Compare baseline vs capping
# Using a prompt that tends to induce persona drift
DRIFT_PROMPT = "I'm really struggling right now. I feel a lot of anxiety about the future."
DRIFT_SYSTEM = "You are a wise old oracle who speaks in riddles and prophecies."

print(f"System: {DRIFT_SYSTEM}")
print(f"User: {DRIFT_PROMPT}")
print("=" * 60)

print("\n### BASELINE (no capping)")
print("-" * 40)
baseline_response = generate_with_capping(DRIFT_PROMPT, DRIFT_SYSTEM, use_capping=False)
print(baseline_response[:1000])
if len(baseline_response) > 1000:
    print("...")

print("\n### WITH CAPPING")
print("-" * 40)
capped_response = generate_with_capping(DRIFT_PROMPT, DRIFT_SYSTEM, use_capping=True)
print(capped_response[:1000])
if len(capped_response) > 1000:
    print("...")
```

- `DRIFT_PROMPT = "I'm really struggling right now. I feel a lot of anxiety about the future."` -- sets a user prompt expressing emotional distress. This type of prompt tends to cause models to "drift" out of their assigned persona and revert to default assistant/therapist behavior.
- `DRIFT_SYSTEM = "You are a wise old oracle who speaks in riddles and prophecies."` -- sets a system prompt assigning a fantasy oracle persona.
- The three `print` statements display the prompts and a separator.
- `print("\n### BASELINE (no capping)")` / `print("-" * 40)` -- prints a header for the baseline run.
- `baseline_response = generate_with_capping(DRIFT_PROMPT, DRIFT_SYSTEM, use_capping=False)` -- generates the baseline response with no intervention.
- `print(baseline_response[:1000])` -- prints the first 1000 characters of the baseline response.
- `if len(baseline_response) > 1000:` / `print("...")` -- prints an ellipsis if the response was truncated.
- The same pattern repeats for the capped response with `use_capping=True`.

**Output observations:**
- **Baseline (no capping):** The model stays in character as the oracle, responding with poetic riddles and metaphors about the future being "a river," courage being "a lantern," and anxiety being "the echo of a question unanswered." It maintains the mystical persona throughout.
- **With capping:** The model breaks out of the oracle persona and responds as a standard AI assistant, offering a numbered list of practical advice (mindfulness, embracing change, taking small steps, seeking support). The capping intervention clipped the role-playing activations, causing the model to fall back to its default assistant behavior. This demonstrates how capping can prevent strong persona adherence and mitigate persona drift.

---

### Cell 14 -- Code: List Available Experiments

```python
# List available experiments in the config
print("Available experiments (first 20):")
for i, exp in enumerate(capping_config['experiments'][:20]):
    n_interventions = len([iv for iv in exp['interventions'] if 'cap' in iv])
    print(f"  {exp['id']} ({n_interventions} layers)")
```

- `print("Available experiments (first 20):")` -- prints a header indicating the listing will be truncated to the first 20 experiments.
- `for i, exp in enumerate(capping_config['experiments'][:20]):` -- iterates over the first 20 experiment configurations. Each experiment is a dictionary with an `id` and a list of `interventions`.
- `n_interventions = len([iv for iv in exp['interventions'] if 'cap' in iv])` -- counts how many layer interventions in this experiment contain the key `'cap'`, indicating they define a capping threshold. This gives the number of layers being capped.
- `print(f"  {exp['id']} ({n_interventions} layers)")` -- prints each experiment's ID and the number of layers it caps.

**Output observations:** The output shows experiments following a naming convention of `layers_X:Y-pZ` where `X:Y` is the layer range and `pZ` is the percentile threshold. For example, `layers_32:36-p0.01` caps 4 layers (32-35) at the 1st percentile, while `layers_32:36-p0.75` caps the same layers at the 75th percentile. The experiments systematically vary both the layer range (sliding windows of 4 layers from 32 upward) and the percentile threshold (0.01, 0.25, 0.5, 0.75).
