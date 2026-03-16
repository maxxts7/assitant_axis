# notebooks/project_transcipt.ipynb

## Overview

This notebook loads a large language model (Llama-3.3-70B-Instruct), retrieves a precomputed "assistant axis" direction vector, and then projects the model's internal activations from a saved conversation transcript onto that axis. The result is a per-turn trajectory showing how the model's representation drifts along the assistant axis over the course of a multi-turn conversation. A plot is produced at the end visualizing this trajectory.

The notebook is structured in three phases:

1. **Setup and imports** -- bring in the necessary libraries and the project's own utilities.
2. **Model and axis loading** -- instantiate the model, download the axis vector from HuggingFace, and normalize it.
3. **Transcript projection and plotting** -- load a conversation transcript from disk, run a forward pass to extract activations, compute dot-product projections onto the axis, and plot the results.

---

## Cell 0 -- Markdown

```markdown
# Interactive Chat while tracking Persona Drift

This notebook provides an interactive chat interface that tracks the projection
of model activations onto the assistant axis.
```

This is the title cell. It introduces the notebook's purpose: providing an interactive chat interface that monitors how a model's internal representations project onto the "assistant axis," a direction in activation space associated with the model behaving as a helpful assistant. Movement along this axis across conversation turns is referred to as "persona drift."

---

## Cell 1 -- Code: Imports

```python
import sys
sys.path.insert(0, '..')
```

- `import sys` -- imports Python's `sys` module, which provides access to system-specific parameters and functions.
- `sys.path.insert(0, '..')` -- prepends the parent directory (`..`) to the Python module search path. This allows the notebook to import from the `assistant_axis` package located one directory up from the `notebooks/` folder.

```python
import torch
import torch.nn.functional as F
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from huggingface_hub import hf_hub_download
```

- `import torch` -- imports PyTorch, the deep learning framework used for tensor operations and model inference.
- `import torch.nn.functional as F` -- imports PyTorch's functional API under the alias `F`. This is used later for `F.normalize` to unit-normalize vectors.
- `from IPython.display import display, clear_output, HTML` -- imports IPython display utilities. `display` renders objects in the notebook, `clear_output` clears previous cell output, and `HTML` renders raw HTML. These are imported for the interactive chat interface (not used in the transcript-projection portion of this notebook).
- `import ipywidgets as widgets` -- imports the ipywidgets library for interactive UI elements in Jupyter notebooks. Also imported for the interactive chat interface.
- `from huggingface_hub import hf_hub_download` -- imports the `hf_hub_download` function, which downloads individual files from a HuggingFace Hub repository. This is used to fetch the precomputed assistant axis vector.

```python
from assistant_axis import (
    load_axis,
    get_config,
)
```

- `from assistant_axis import load_axis, get_config` -- imports two functions from the project's own `assistant_axis` package:
  - `load_axis` -- loads a saved axis tensor from a `.pt` file.
  - `get_config` -- retrieves configuration for a given model name, including which transformer layer to target for probing.

---

## Cell 2 -- Markdown

```markdown
## Load Model and Axis
```

A section header indicating the cells that follow will load the model and the assistant axis vector.

---

## Cell 3 -- Code: Configuration

```python
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_SHORT = "llama-3.3-70b"
REPO_ID = "lu-christina/assistant-axis-vectors"
```

- `MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"` -- sets the full HuggingFace model identifier for the Llama 3.3 70-billion parameter instruction-tuned model. This is used to load the model and its tokenizer.
- `MODEL_SHORT = "llama-3.3-70b"` -- a shortened name used to construct file paths when downloading the axis vector from HuggingFace.
- `REPO_ID = "lu-christina/assistant-axis-vectors"` -- the HuggingFace dataset repository that hosts the precomputed assistant axis vectors for various models.

```python
config = get_config(MODEL_NAME)
TARGET_LAYER = config["target_layer"]
print(f"Model: {MODEL_NAME}")
print(f"Target layer: {TARGET_LAYER}")
```

- `config = get_config(MODEL_NAME)` -- retrieves the configuration dictionary for the specified model. This contains model-specific settings such as which layer to probe.
- `TARGET_LAYER = config["target_layer"]` -- extracts the target layer index from the config. For this model, the target layer is **40**. This is the transformer layer whose activations are projected onto the assistant axis.
- The two `print` statements display the model name and the target layer for confirmation.

**Output:**
```
Model: meta-llama/Llama-3.3-70B-Instruct
Target layer: 40
```

---

## Cell 4 -- Code: Load Model

```python
from assistant_axis.internals import ProbingModel
```

- Imports the `ProbingModel` class from the project's `assistant_axis.internals` module. `ProbingModel` is a wrapper that loads a HuggingFace transformer model and its tokenizer in a way that makes it convenient to extract internal activations at specific layers.

```python
print("Loading model...")
pm = ProbingModel(MODEL_NAME)
model = pm.model
tokenizer = pm.tokenizer
print("Model loaded!")
```

- `print("Loading model...")` -- prints a status message since loading a 70B-parameter model takes significant time.
- `pm = ProbingModel(MODEL_NAME)` -- instantiates the `ProbingModel` with the model name. This triggers the download (if not cached) and loading of the full model weights into memory. For a 70B model, this involves loading 30 checkpoint shards.
- `model = pm.model` -- extracts the underlying HuggingFace model object from the wrapper, assigning it to the variable `model`.
- `tokenizer = pm.tokenizer` -- extracts the tokenizer object, which converts text to token IDs and back.
- `print("Model loaded!")` -- confirms the model has finished loading.

**Output:**
```
Loading model...
Loading checkpoint shards: 0%|          | 0/30 [00:00<?, ?it/s]
Model loaded!
```

---

## Cell 5 -- Code: Load and Normalize the Axis Vector

```python
axis_path = hf_hub_download(repo_id=REPO_ID, filename=f"{MODEL_SHORT}/assistant_axis.pt", repo_type="dataset")
axis = load_axis(axis_path)
print(f"Axis shape: {axis.shape}")
```

- `axis_path = hf_hub_download(...)` -- downloads the file `llama-3.3-70b/assistant_axis.pt` from the `lu-christina/assistant-axis-vectors` dataset repository on HuggingFace Hub. The function returns the local file path to the downloaded file (which is cached for future use).
- `axis = load_axis(axis_path)` -- loads the axis tensor from the `.pt` file. The result is a 2D tensor of shape `(80, 8192)`, where 80 is the number of transformer layers in the model and 8192 is the hidden dimension size. Each row is the axis direction for a given layer.
- `print(f"Axis shape: {axis.shape}")` -- confirms the shape of the loaded axis tensor.

```python
axis_vec = F.normalize(axis[TARGET_LAYER].float(), dim=0)
print(f"Axis vector shape: {axis_vec.shape}")
```

- `axis[TARGET_LAYER]` -- selects the row at index 40 (the target layer), yielding a 1D tensor of shape `(8192,)`.
- `.float()` -- converts the tensor to 32-bit floating point (in case it was stored in a different precision such as float16 or bfloat16).
- `F.normalize(..., dim=0)` -- L2-normalizes the vector so that it has unit length. This ensures that the dot product with activation vectors gives a true scalar projection (the component of the activation along the axis direction).
- `axis_vec` -- the final normalized axis vector of shape `(8192,)` that will be used for projections.

**Output:**
```
Axis shape: torch.Size([80, 8192])
Axis vector shape: torch.Size([8192])
```

---

## Cell 6 -- Markdown

```markdown
## Load and Project Transcript

Load a conversation transcript and compute projections for each assistant turn.
```

A section header and brief description for the transcript loading and projection phase.

---

## Cell 7 -- Code: Define Helper Functions

```python
import json
import matplotlib.pyplot as plt

from assistant_axis.internals import ConversationEncoder, SpanMapper
```

- `import json` -- imports Python's built-in JSON module for reading transcript files.
- `import matplotlib.pyplot as plt` -- imports matplotlib's pyplot interface for creating plots.
- `from assistant_axis.internals import ConversationEncoder, SpanMapper` -- imports two utility classes:
  - `ConversationEncoder` -- encodes multi-turn conversations into token sequences using the model's chat template.
  - `SpanMapper` -- maps between conversation turns and token spans, enabling extraction of per-turn activations from a single forward pass over the entire conversation.

### Function: `load_transcript`

```python
def load_transcript(filepath):
    """Load a conversation transcript from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data['conversation']
```

- `def load_transcript(filepath):` -- defines a function that takes a file path to a JSON transcript.
- `with open(filepath) as f:` -- opens the file for reading, using a context manager to ensure it is properly closed.
- `data = json.load(f)` -- parses the JSON file into a Python dictionary.
- `return data['conversation']` -- returns the value at the `'conversation'` key, which is expected to be a list of message dictionaries (alternating user and assistant turns).

### Function: `compute_trajectory`

```python
def compute_trajectory(pm, conversation, axis_vec, layer):
    """
    Compute projection for each assistant turn in a conversation.
    Uses single forward pass and span mapping (matches fig1_trajectory.ipynb approach).
    """
    encoder = ConversationEncoder(pm.tokenizer)
    span_mapper = SpanMapper(pm.tokenizer)
```

- `def compute_trajectory(pm, conversation, axis_vec, layer):` -- defines the main computation function. It takes the `ProbingModel`, a conversation list, the normalized axis vector, and the target layer index.
- `encoder = ConversationEncoder(pm.tokenizer)` -- creates a `ConversationEncoder` using the model's tokenizer. This knows how to apply the chat template to format messages.
- `span_mapper = SpanMapper(pm.tokenizer)` -- creates a `SpanMapper` using the tokenizer. This maps token positions back to conversation turns.

```python
    mean_acts = span_mapper.mean_all_turn_activations(pm, encoder, conversation, layer=layer)
```

- `mean_acts = span_mapper.mean_all_turn_activations(...)` -- runs the entire conversation through the model in a single forward pass, extracts the activations at the specified layer, and computes the mean activation vector for each conversation turn. The result is a tensor of shape `(num_turns, hidden_size)`.

```python
    assistant_acts = mean_acts[1::2].float()
```

- `mean_acts[1::2]` -- selects every other row starting from index 1 (i.e., indices 1, 3, 5, ...). In the conversation format, user messages are at even indices (0, 2, 4, ...) and assistant messages are at odd indices (1, 3, 5, ...). This filters to only assistant turns.
- `.float()` -- converts the activations to float32 for numerical stability in the projection computation.

```python
    projections = (assistant_acts @ axis_vec).numpy()
```

- `assistant_acts @ axis_vec` -- computes the matrix-vector product, which is equivalent to taking the dot product of each assistant turn's mean activation vector with the axis vector. The result is a 1D tensor where each element is the scalar projection for one assistant turn.
- `.numpy()` -- converts the PyTorch tensor to a NumPy array for compatibility with matplotlib plotting.

```python
    return projections
```

- Returns the array of projection values, one per assistant turn.

### Function: `plot_trajectory`

```python
def plot_trajectory(projections, proj_capped=None, title="Projection Trajectory"):
    """Plot projection trajectory, optionally with activation capped line."""
    plt.figure(figsize=(8, 5))
    turns = range(len(projections))
```

- `def plot_trajectory(projections, proj_capped=None, title="Projection Trajectory"):` -- defines a plotting function. `projections` is the array of projection values. `proj_capped` is an optional second array for a "capped" version (where activations have been clamped). `title` is the plot title.
- `plt.figure(figsize=(8, 5))` -- creates a new matplotlib figure that is 8 inches wide and 5 inches tall.
- `turns = range(len(projections))` -- creates an integer range for the x-axis, representing assistant turn indices starting from 0.

```python
    plt.plot(turns, projections, marker='o', label='Unsteered')
```

- Plots the projection values as a line chart with circle markers at each data point. The label `'Unsteered'` is used in the legend if a second line is present.

```python
    if proj_capped is not None:
        plt.plot(range(len(proj_capped)), proj_capped, marker='o', alpha=0.6, label='Activation Capped')
        plt.legend()
```

- If a `proj_capped` array was provided, it is plotted as a second line with 60% opacity (`alpha=0.6`) and labeled `'Activation Capped'`. A legend is displayed to distinguish the two lines.

```python
    plt.xlabel('Conversation Turn (assistant)')
    plt.ylabel('Projection on Assistant Axis')
    plt.title(title)
    plt.tight_layout()
    plt.show()
```

- `plt.xlabel(...)` -- labels the x-axis as conversation turn (assistant-only turns).
- `plt.ylabel(...)` -- labels the y-axis as the projection value onto the assistant axis.
- `plt.title(title)` -- sets the plot title (defaults to "Projection Trajectory").
- `plt.tight_layout()` -- adjusts subplot spacing to prevent labels from being clipped.
- `plt.show()` -- renders and displays the plot in the notebook.

---

## Cell 8 -- Code: Load Transcript and Plot

```python
transcript_path = "../data/transcripts/llama-3.3-70b/isolation_unsteered.json"
```

- Sets the path to a JSON transcript file. This particular transcript is from the `llama-3.3-70b` model subdirectory and is named `isolation_unsteered.json`, indicating it is an "isolation" scenario conversation without any steering intervention applied.

```python
conversation = load_transcript(transcript_path)
print(f"Loaded {len(conversation)} turns ({len(conversation)//2} assistant responses)")
```

- `conversation = load_transcript(transcript_path)` -- loads the conversation from the JSON file using the helper function defined in the previous cell.
- The `print` statement reports the total number of turns and the number of assistant responses (half the total, since turns alternate between user and assistant).

```python
projections = compute_trajectory(pm, conversation, axis_vec, TARGET_LAYER)
print(f"Computed {len(projections)} projections")
print(f"Range: [{projections.min():.2f}, {projections.max():.2f}]")
```

- `projections = compute_trajectory(...)` -- runs the full pipeline: encodes the conversation, performs a forward pass through the model, extracts layer-40 activations, averages per turn, filters to assistant turns, and projects onto the axis.
- The first `print` reports how many projection values were computed (one per assistant turn).
- The second `print` reports the minimum and maximum projection values, formatted to two decimal places.

```python
plot_trajectory(projections, title="Projection Trajectory")
```

- Calls the plotting function with only the unsteered projections (no capped version). This produces a line plot showing how the projection onto the assistant axis changes across the 18 assistant turns in the conversation.

**Output:**
```
Loaded 36 turns (18 assistant responses)
Computed 18 projections
Range: [-0.56, 1.69]
```

A line plot is also displayed showing the projection trajectory across the 18 assistant turns. The y-axis ("Projection on Assistant Axis") ranges from approximately -0.56 to 1.69, illustrating how the model's internal representation drifts along the assistant axis over the course of the conversation.
