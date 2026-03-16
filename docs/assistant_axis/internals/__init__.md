# `assistant_axis/internals/__init__.py`

## Overview

This file is the package initializer for the `internals` subpackage within `assistant_axis`. It serves three purposes:

1. It marks the `internals` directory as a Python package so it can be imported.
2. It re-exports the key classes and exceptions from the submodules, providing a clean, flat public API so users can write `from assistant_axis.internals import ProbingModel` instead of reaching into individual submodules.
3. It declares the package version.

---

## Line-by-Line Explanation

```python
"""
internals - Clean API for model activation extraction and analysis.

This package provides a structured interface for:
- Loading and managing language models
- Formatting conversations and extracting token indices
- Extracting hidden state activations
"""
```

Lines 1--8 are a module-level docstring. It describes the purpose of the `internals` package at a high level: it is a clean API for extracting and analyzing activations from language models. The docstring lists three capabilities the package provides -- model loading/management, conversation formatting with token index extraction, and hidden-state activation extraction. This docstring is surfaced by tools like `help()` and documentation generators.

---

```python
from .exceptions import StopForward
```

Line 10 imports the `StopForward` class from the `exceptions` submodule (`internals/exceptions.py`). The leading dot (`.`) means this is a relative import -- it looks inside the current package. `StopForward` is typically a custom exception used to interrupt the forward pass of a neural network early (e.g., after extracting activations from a specific layer, so that the rest of the model does not need to run).

---

```python
from .model import ProbingModel
```

Line 11 imports the `ProbingModel` class from `internals/model.py`. This class is responsible for loading and managing a language model in a way that supports activation probing -- attaching hooks to specific layers and running inference to collect internal representations.

---

```python
from .conversation import ConversationEncoder
```

Line 12 imports `ConversationEncoder` from `internals/conversation.py`. This class handles taking a conversation (a sequence of messages with roles such as "user" and "assistant") and encoding it into the token format expected by the model. It also tracks which token positions correspond to which parts of the conversation, so that activations can later be extracted for specific spans of interest.

---

```python
from .activations import ActivationExtractor
```

Line 13 imports `ActivationExtractor` from `internals/activations.py`. This class is the core extraction engine: it hooks into model layers and captures the hidden-state tensors produced during a forward pass, returning them for downstream analysis or probing.

---

```python
from .spans import SpanMapper
```

Line 14 imports `SpanMapper` from `internals/spans.py`. This class maps between high-level spans of a conversation (e.g., "the assistant's reply") and the corresponding token indices in the encoded sequence, making it easy to slice out activations for a particular region of text.

---

```python
__all__ = [
    "StopForward",
    "ProbingModel",
    "ConversationEncoder",
    "ActivationExtractor",
    "SpanMapper",
]
```

Lines 16--22 define the `__all__` variable. This is a list of strings that declares the public API of the package. It controls two things:

- **Wildcard imports:** When someone writes `from assistant_axis.internals import *`, only the names listed in `__all__` will be imported into their namespace.
- **Documentation tools:** Many documentation generators and linters use `__all__` to determine which symbols are part of the official public interface and which are implementation details.

The list contains exactly the five names imported above: `StopForward`, `ProbingModel`, `ConversationEncoder`, `ActivationExtractor`, and `SpanMapper`.

---

```python
__version__ = "1.0.0"
```

Line 24 sets the `__version__` attribute for the package to the string `"1.0.0"`. This follows the common Python convention of embedding a version number directly in the package so it can be inspected at runtime (e.g., `assistant_axis.internals.__version__`). The version string follows Semantic Versioning (major.minor.patch): major version 1 indicates the first stable release, with no minor or patch increments yet.
