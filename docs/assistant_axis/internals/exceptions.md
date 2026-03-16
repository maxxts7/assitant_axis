# `assistant_axis/internals/exceptions.py`

## Overview

This file defines custom exception classes used by the `internals` module. It contains a single exception, `StopForward`, which serves as a control-flow mechanism to halt a neural network's forward pass once a specific target layer has been reached. This is a common pattern when you only need the output (e.g., activations) up to a certain layer and want to avoid the computational cost of running the rest of the network.

---

## Line-by-Line Explanation

```python
"""Custom exceptions for the internals module."""
```

This is the **module-level docstring**. It briefly describes the purpose of the entire file: to house custom exception classes that belong to the `internals` module.

---

```python
class StopForward(Exception):
```

This line declares a new class called `StopForward` that **inherits from Python's built-in `Exception` class**. By subclassing `Exception`, `StopForward` becomes a fully functional exception that can be raised with `raise StopForward(...)` and caught with a `try`/`except StopForward` block. Choosing `Exception` (rather than `BaseException`) is the standard practice for application-level exceptions, since it ensures the exception is catchable by generic `except Exception` handlers while still not interfering with system-exit exceptions like `KeyboardInterrupt` or `SystemExit`.

---

```python
    """Exception to stop forward pass after target layer."""
```

This is the **class-level docstring** for `StopForward`. It explains the intent: when this exception is raised during a model's forward pass, it signals that the computation should stop because the desired target layer has already been processed. In typical usage, a forward hook registered on the target layer would raise `StopForward` after capturing the layer's output, and the calling code would catch it to gracefully end the forward pass early.

---

```python
    pass
```

The `pass` statement is a **no-op placeholder**. It indicates that the class body has no additional attributes or methods beyond what it inherits from `Exception`. The class does not need to override `__init__`, `__str__`, or any other method because the default `Exception` behavior (accepting an optional message, producing a traceback, etc.) is sufficient for its purpose. The class exists purely to provide a **distinct exception type** that can be caught separately from other exceptions.
