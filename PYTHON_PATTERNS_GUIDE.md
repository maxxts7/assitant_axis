# Python Patterns and Language Features in This Codebase

Every non-obvious Python pattern used in the `assistant_axis` library and pipeline scripts, explained with the actual code and the rationale for why it's used here.

---

## Table of Contents

- [1. Closures and Factory Functions](#1-closures-and-factory-functions)
- [2. Context Managers — `__enter__` and `__exit__`](#2-context-managers--__enter__-and-__exit__)
- [3. `try` / `finally` — Guaranteeing Cleanup](#3-try--finally--guaranteeing-cleanup)
- [4. `@property` — Computed Attributes](#4-property--computed-attributes)
- [5. `@classmethod` and `cls.__new__(cls)` — Alternative Constructors](#5-classmethod-and-cls__new__cls--alternative-constructors)
- [6. `@staticmethod` — Utility Methods Without `self`](#6-staticmethod--utility-methods-without-self)
- [7. `from __future__ import annotations` — Postponed Evaluation](#7-from-__future__-import-annotations--postponed-evaluation)
- [8. `TYPE_CHECKING` — Imports That Only Exist for IDEs](#8-type_checking--imports-that-only-exist-for-ides)
- [9. `__all__` — Controlling What Gets Exported](#9-__all__--controlling-what-gets-exported)
- [10. `**kwargs` — Forwarding Unknown Arguments](#10-kwargs--forwarding-unknown-arguments)
- [11. `isinstance()` — Runtime Type Branching](#11-isinstance--runtime-type-branching)
- [12. `nonlocal` — Modifying Enclosing Scope Variables](#12-nonlocal--modifying-enclosing-scope-variables)
- [13. `async` / `await` — Concurrent I/O](#13-async--await--concurrent-io)
- [14. `asyncio.Lock` — Thread-Safe Async State](#14-asynciolock--thread-safe-async-state)
- [15. `asyncio.gather()` — Running Many Coroutines in Parallel](#15-asynciogather--running-many-coroutines-in-parallel)
- [16. Tuple Unpacking and Star Expressions](#16-tuple-unpacking-and-star-expressions)
- [17. Lambda Functions and Inline Callables](#17-lambda-functions-and-inline-callables)
- [18. `hasattr()` / `getattr()` — Duck Typing and Safe Access](#18-hasattr--getattr--duck-typing-and-safe-access)
- [19. `torch.einsum()` — Einstein Summation Notation](#19-torcheinsum--einstein-summation-notation)
- [20. `.clone()` vs Direct Mutation — Defensive Copies](#20-clone-vs-direct-mutation--defensive-copies)
- [21. Multiprocessing — `mp.Process`, `start`, `join`](#21-multiprocessing--mpprocess-start-join)
- [22. `os.environ` — Environment Variables as Configuration](#22-osenviron--environment-variables-as-configuration)
- [23. List Comprehensions and Dict Comprehensions](#23-list-comprehensions-and-dict-comprehensions)

---

## 1. Closures and Factory Functions

A **closure** is a function that remembers variables from the scope where it was defined, even after that scope has ended. A **factory function** creates and returns closures.

This is the most important Python pattern in this codebase — it appears 8 times across 3 files and is the mechanism that makes forward hooks work.

### The problem closures solve

```python
# THIS IS BUGGY — DO NOT USE
for layer_idx in [0, 1, 2]:
    def hook_fn(module, input, output):
        print(f"Layer {layer_idx}")    # Captures the VARIABLE, not the VALUE
    register_hook(hook_fn)

# When hooks fire later:
# ALL three print "Layer 2" because layer_idx is 2 after the loop ends
```

In Python, the inner function captures a *reference* to the variable `layer_idx`, not a *copy* of its value. After the loop finishes, `layer_idx == 2` for all three closures.

### The factory function fix

```python
# activations.py — lines 86-91
def create_hook_fn(layer_idx):          # Factory function — creates a NEW scope
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu())
    return hook_fn                       # Returns the closure

for layer_idx in layer_list:
    handle = target_layer.register_forward_hook(create_hook_fn(layer_idx))
    #                                          ^^^^^^^^^^^^^^^^^^^^^^^^
    # Each CALL to create_hook_fn creates a new scope.
    # The inner hook_fn captures layer_idx from THAT scope.
```

Each call to `create_hook_fn(22)` creates a new function object with its own binding of `layer_idx=22`. The loop variable can change freely — each closure has its own copy.

### Three variations in activations.py

```python
# Variation 1 (full_conversation): Appends to a list, captures nothing extra
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations.append(act_tensor[0, :, :].cpu())       # Appends to outer list
    return hook_fn

# Variation 2 (at_newline): Stores in a dict by layer, captures newline_pos
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        activations[layer_idx] = act_tensor[0, newline_pos, :].cpu()  # Dict key + position
    return hook_fn

# Variation 3 (batch_conversations): Keeps tensor on GPU, no indexing
def create_hook_fn(layer_idx):
    def hook_fn(module, input, output):
        act_tensor = output[0] if isinstance(output, tuple) else output
        layer_outputs[layer_idx] = act_tensor                # Full tensor, stays on GPU
    return hook_fn
```

Each captures different outer variables (`activations` list, `activations` dict, `newline_pos`, `layer_outputs` dict). The closure "remembers" these even after `create_hook_fn` returns.

### The steering.py variation — a method factory

```python
# steering.py — lines 232-236
def _create_hook_fn(self, layer_idx):
    def hook_fn(module, ins, out):
        return self._apply_layer_interventions(out, layer_idx)  # Returns modified output
    return hook_fn
```

This hook **returns** a value (unlike extraction hooks which just capture data). When a forward hook returns something, PyTorch replaces the layer's output with the returned value. This is how `ActivationSteering` modifies the model's behavior.

---

## 2. Context Managers — `__enter__` and `__exit__`

A **context manager** is an object used with the `with` statement. Python calls `__enter__` when entering the block and `__exit__` when leaving — even if an exception occurs.

### How `ActivationSteering` uses it

```python
# steering.py — lines 334-359

def __enter__(self):
    """Register hooks on all unique layers."""
    for layer_idx in self.vectors_by_layer.keys():
        layer_module = self._get_layer_module(layer_idx)
        hook_fn = self._create_hook_fn(layer_idx)
        handle = layer_module.register_forward_hook(hook_fn)
        self._handles.append(handle)
    return self                          # The 'as' target in 'with ... as steerer:'

def __exit__(self, *exc):
    """Remove all hooks."""
    self.remove()                        # Always called, even on exception

def remove(self):
    for handle in self._handles:
        if handle:
            handle.remove()
    self._handles = []
```

**Usage:**
```python
with ActivationSteering(model, steering_vectors=[axis[22]], ...):
    output = model.generate(...)       # Hooks are active here
# Hooks are automatically removed here, even if generate() crashes
```

### What `__exit__` receives

```python
def __exit__(self, *exc):
    # exc is a tuple of (exc_type, exc_value, traceback)
    # If no exception: (None, None, None)
    # If exception: (ValueError, ValueError("bad"), <traceback object>)
    self.remove()
    # Returning None (or False) lets the exception propagate
    # Returning True would suppress the exception (not done here)
```

The `*exc` collects all three arguments into a tuple. This code ignores them — it always cleans up and lets any exception propagate.

### Other context managers in this codebase

```python
# torch.inference_mode() — disables autograd
with torch.inference_mode():
    _ = self.model(input_ids)

# File I/O — closes file on exit
with open(role_file, 'r') as f:
    return json.load(f)

# JSONL reader — closes file on exit
with jsonlines.open(responses_file, 'r') as reader:
    for entry in reader:
        responses.append(entry)

# asyncio.Lock — releases lock on exit
async with self.lock:
    # Critical section — only one coroutine can be here at a time
    self.tokens -= 1
```

### `with` vs `try/finally`

Both guarantee cleanup. `with` is cleaner when the cleanup logic is encapsulated in an object. `try/finally` is used when the cleanup is ad-hoc:

```python
# Context manager style (steering.py):
with ActivationSteering(model, ...):
    model.generate(...)

# try/finally style (activations.py):
handles = []
for layer_idx in layer_list:
    handle = layer.register_forward_hook(hook_fn)
    handles.append(handle)
try:
    _ = self.model(input_ids)
finally:
    for handle in handles:
        handle.remove()
```

Why doesn't `ActivationExtractor` use a context manager? Because the hooks are registered and removed within a single method call. There's no user-facing "start" and "end" — it's all internal. `ActivationSteering` wraps hooks around *user code* (the generation call inside the `with` block), so a context manager is natural.

---

## 3. `try` / `finally` — Guaranteeing Cleanup

`finally` runs *no matter what* — even if the `try` block raises an exception, even if it returns early, even if it calls `sys.exit()`.

```python
# activations.py — lines 100-106
try:
    with torch.inference_mode():
        _ = self.model(input_ids)     # Might throw OutOfMemoryError
finally:
    for handle in handles:
        handle.remove()               # Always runs
```

**Why it matters here:** If the forward pass crashes (e.g., GPU out of memory), the hooks are still attached to the model's layers. Without cleanup:
- Every future forward pass through the model would trigger the hooks
- The `activations` list would keep growing with garbage data
- Memory would leak
- `ActivationSteering` hooks would keep modifying the model even after the error

### `try/except` vs `try/finally`

```python
# try/except — catches and handles errors
try:
    result = risky_operation()
except ValueError as e:
    print(f"Error: {e}")        # Only runs if ValueError is raised

# try/finally — cleanup regardless of errors
try:
    result = risky_operation()
finally:
    cleanup()                   # Always runs, error or not

# Both together
try:
    result = risky_operation()
except ValueError as e:
    handle_error(e)             # Handle the error
finally:
    cleanup()                   # Clean up regardless
```

The pipeline scripts use `try/except` for individual role processing (skip bad roles, continue with others):

```python
# 2_activations.py — lines 222-231
for role_file in tqdm(role_files):
    try:
        success = process_role(pm, role_file, ...)
    except Exception as e:
        failed_count += 1
        worker_logger.error(f"Exception processing {role_file.stem}: {e}")
        # Loop continues — other roles are still processed
```

---

## 4. `@property` — Computed Attributes

`@property` makes a method behave like an attribute. You access it with `obj.name` instead of `obj.name()`.

```python
# model.py — lines 116-124
@property
def hidden_size(self) -> int:
    return self.model.config.hidden_size

@property
def device(self) -> torch.device:
    return next(self.model.parameters()).device

# Usage:
pm = ProbingModel("Qwen/Qwen3-32B")
print(pm.hidden_size)    # 4096 — no parentheses
print(pm.device)         # cuda:0 — no parentheses
```

**Why use `@property` instead of storing the value in `__init__`?**

For `device`: On a multi-GPU model, the "device" isn't fixed — different parameters are on different GPUs. `next(self.model.parameters()).device` always returns the device of the first parameter, which is the correct device to send inputs to.

For `hidden_size`: It reads from the model's config object, which always exists after loading. It's a minor style choice — storing it in `__init__` would also work.

### Properties that cache (model type detection)

```python
# model.py — lines 198-211
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

These call `detect_type()`, which caches its result in `self._model_type`:

```python
def detect_type(self) -> str:
    if self._model_type is not None:
        return self._model_type         # Return cached value
    # ... detection logic ...
    self._model_type = 'qwen'           # Cache for next time
    return self._model_type
```

This is a manual cache pattern. The first call does the work; subsequent calls return instantly.

---

## 5. `@classmethod` and `cls.__new__(cls)` — Alternative Constructors

### `@classmethod` — a constructor that takes the class, not an instance

```python
# model.py — lines 90-114
@classmethod
def from_existing(cls, model, tokenizer, model_name=None):
    instance = cls.__new__(cls)          # Create instance WITHOUT calling __init__
    instance.model = model
    instance.tokenizer = tokenizer
    instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
    instance.chat_model_name = None
    instance.dtype = next(model.parameters()).dtype
    instance._layers = None
    instance._model_type = None
    return instance
```

**What `cls` is:** The class itself (e.g., `ProbingModel`). Unlike `self` (which is an instance), `cls` lets you create new instances.

**Why not just call `ProbingModel(...)`?** Because `__init__` downloads and loads the model from HuggingFace. If you already have a loaded model, you don't want to re-download it. `from_existing()` creates a `ProbingModel` wrapper around a model that's already in memory.

### `cls.__new__(cls)` — creating an instance without `__init__`

Normal object creation in Python:

```python
pm = ProbingModel("Qwen/Qwen3-32B")
# Step 1: Python calls ProbingModel.__new__(ProbingModel) to create the object
# Step 2: Python calls ProbingModel.__init__(pm, "Qwen/Qwen3-32B") to initialize it
```

`cls.__new__(cls)` does only step 1. The object exists (it has memory allocated, its class is set) but no attributes are set. `from_existing()` then manually sets every attribute that `__init__` would have set.

**The danger:** If you forget to set an attribute that `__init__` would set, accessing it later causes `AttributeError`. That's why `from_existing()` carefully mirrors every attribute from `__init__`: `model_name`, `chat_model_name`, `dtype`, `_layers`, `_model_type`.

---

## 6. `@staticmethod` — Utility Methods Without `self`

A static method doesn't receive `self` (instance) or `cls` (class). It's just a regular function namespaced under the class.

```python
# conversation.py — lines 840-865
@staticmethod
def _longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

@staticmethod
def _strip_trailing_special(ids: List[int], special_ids: set) -> List[int]:
    i = len(ids)
    while i > 0 and ids[i-1] in special_ids:
        i -= 1
    return ids[:i]

@staticmethod
def _find_subsequence(hay: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(hay):
        return -1
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return i
    return -1
```

**Why `@staticmethod` instead of a standalone function?** Organization. These are helper functions used only by `ConversationEncoder`. Putting them as static methods keeps them grouped with their class without polluting the module namespace. They don't need `self` because they operate only on their arguments.

---

## 7. `from __future__ import annotations` — Postponed Evaluation

```python
# activations.py — line 3
from __future__ import annotations
```

**What it does:** Makes all type annotations in the file into strings that are evaluated lazily, not at import time. Without it:

```python
class Foo:
    def method(self) -> Bar:    # ERROR: Bar is not defined yet
        pass

class Bar:
    pass
```

With `from __future__ import annotations`:

```python
from __future__ import annotations

class Foo:
    def method(self) -> Bar:    # OK: "Bar" is stored as the string "Bar", not evaluated
        pass

class Bar:
    pass
```

**Why this codebase uses it:** The `ActivationExtractor` class references `ProbingModel` and `ConversationEncoder` in its type hints. But those classes are defined in different files, and importing them at the top of the file would create circular imports. With postponed annotations, the type hints are just strings at runtime — no import needed.

---

## 8. `TYPE_CHECKING` — Imports That Only Exist for IDEs

```python
# activations.py — lines 5-10
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ProbingModel
    from .conversation import ConversationEncoder
```

**What `TYPE_CHECKING` is:** A constant that is `False` at runtime but `True` when a type checker (mypy, pyright, IDE autocomplete) analyzes the code.

**The problem it solves:** Circular imports. If `activations.py` imports `model.py` and `model.py` imports `activations.py`, Python crashes at import time. But type checkers need the imports to understand the types.

**The solution:** Import only during type checking (which doesn't execute code, just analyzes it). At runtime, the imports never execute. The type annotations work because `from __future__ import annotations` makes them strings.

```python
# At runtime:
#   TYPE_CHECKING = False
#   The "if" block is skipped
#   ProbingModel is never imported
#   But that's fine because annotations are strings, not references

# During type checking:
#   TYPE_CHECKING = True
#   ProbingModel is imported
#   Type checker can validate: "is probing_model really a ProbingModel?"
```

### Where the circular dependency would occur

```
activations.py wants to reference ProbingModel (for type hints)
    └→ would need to import from model.py
        └→ model.py doesn't import activations.py, but...

spans.py wants to reference ConversationEncoder (for type hints)
    └→ would need to import from conversation.py
        └→ conversation.py doesn't import spans.py directly, but
           if conversation.py were to use SpanMapper, you'd have a cycle
```

The `TYPE_CHECKING` pattern preemptively breaks any potential cycle.

---

## 9. `__all__` — Controlling What Gets Exported

```python
# __init__.py — lines 54-84
__all__ = [
    "get_config",
    "MODEL_CONFIGS",
    "compute_axis",
    "load_axis",
    ...
]
```

**What `__all__` does:** When someone writes `from assistant_axis import *`, only the names in `__all__` are imported. Without `__all__`, `import *` imports everything — including internal helpers, imported modules, and other junk.

**What it doesn't do:** `__all__` doesn't prevent direct imports. `from assistant_axis.judge import RateLimiter` works even though `RateLimiter` is not in the top-level `__all__`. It only controls the `import *` behavior.

```python
# internals/__init__.py — lines 16-22
__all__ = [
    "StopForward",
    "ProbingModel",
    "ConversationEncoder",
    "ActivationExtractor",
    "SpanMapper",
]
```

This defines the public API of the internals package. Users import from here:
```python
from assistant_axis.internals import ProbingModel, ActivationExtractor
```

---

## 10. `**kwargs` — Forwarding Unknown Arguments

`**kwargs` collects extra keyword arguments into a dictionary. This codebase uses it heavily for passing model-specific parameters through multiple layers of function calls.

### The forwarding chain

```python
# Layer 1 — User calls:
extractor.full_conversation(conversation, enable_thinking=False)
#                                         ^^^^^^^^^^^^^^^^^^^^^^
#                                         This becomes chat_kwargs = {"enable_thinking": False}

# Layer 2 — full_conversation passes it through:
def full_conversation(self, conversation, layer=None, chat_format=True, **chat_kwargs):
    formatted_prompt = self.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, **chat_kwargs
        #                                                         ^^^^^^^^^^^^
        # Unpacks {"enable_thinking": False} as enable_thinking=False
    )

# Layer 3 — apply_chat_template receives it as a regular keyword argument:
def apply_chat_template(self, conversation, tokenize=False, ..., enable_thinking=False):
    # Qwen's template checks this parameter
```

**Why not just have `enable_thinking` as an explicit parameter?** Because different models need different template options. Qwen needs `enable_thinking`. Gemma might need something else. `**kwargs` lets the code pass through any model-specific option without changing the function signature when a new model is added.

### The `**` unpacking operator

```python
d = {"a": 1, "b": 2}

# ** unpacks the dict into keyword arguments:
func(**d)    # equivalent to func(a=1, b=2)

# Works in reverse too — collects keyword arguments into a dict:
def func(**kwargs):
    print(kwargs)    # {"a": 1, "b": 2}
```

---

## 11. `isinstance()` — Runtime Type Branching

`isinstance(obj, Type)` checks if `obj` is an instance of `Type` (or a subclass of it). This codebase uses it extensively for handling multiple input formats.

### The three-way layer parameter pattern

```python
# activations.py — lines 57-66
if isinstance(layer, int):           # User passed layer=22
    single_layer_mode = True
    layer_list = [layer]
elif isinstance(layer, list):        # User passed layer=[20, 22, 24]
    single_layer_mode = False
    layer_list = layer
else:                                # User passed layer=None
    single_layer_mode = False
    layer_list = list(range(len(self.probing_model.get_layers())))
```

This pattern appears in every extraction method. It lets one parameter accept three different types.

### Tuple-checking for model outputs

```python
# activations.py — line 89
act_tensor = output[0] if isinstance(output, tuple) else output
```

### Multi-type checking

```python
# steering.py — line 246
elif isinstance(activations, (tuple, list)):    # Either tuple OR list
```

Passing a tuple of types checks against any of them.

---

## 12. `nonlocal` — Modifying Enclosing Scope Variables

Python has three scopes: local (inside the function), enclosing (the function that contains this function), and global (module level). By default, a nested function can READ enclosing variables but not REASSIGN them.

```python
# model.py — lines 345-356
def capture_hidden_state(self, input_ids, layer, position=-1):
    captured_state = None                  # Enclosing scope variable

    def capture_hook(module, input, output):
        nonlocal captured_state            # "I want to REASSIGN, not create a new local"
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        captured_state = hidden_states[0, position, :].clone().cpu()
        #              ^ Without nonlocal, this creates a NEW local variable
        #                and the outer captured_state stays None

    layer_module = self.get_layers()[layer]
    hook_handle = layer_module.register_forward_hook(capture_hook)
    try:
        with torch.inference_mode():
            _ = self.model(input_ids)
    finally:
        hook_handle.remove()

    return captured_state                  # Now contains the captured data
```

**Why `nonlocal` is needed here but not in `activations.py`:**

In `activations.py`, hooks do `activations.append(...)` or `layer_outputs[key] = ...`. These are **mutations** of existing objects (appending to a list, adding to a dict), not reassignments. Python allows mutations of enclosing variables without `nonlocal`.

In `model.py`, `captured_state = hidden_states[...]` is a **reassignment** — replacing the `None` with a tensor. Without `nonlocal`, Python treats it as creating a new local variable inside `capture_hook`, and the outer `captured_state` stays `None`.

```python
# Mutation (no nonlocal needed):
activations.append(x)         # Mutates the list — OK
activations[key] = x          # Mutates the dict — OK

# Reassignment (needs nonlocal):
captured_state = x            # Without nonlocal → creates LOCAL variable
                               # With nonlocal → reassigns ENCLOSING variable
```

---

## 13. `async` / `await` — Concurrent I/O

The `judge.py` module uses async Python to make hundreds of API calls concurrently without blocking.

### The problem

Scoring 1200 responses at 1 API call per response, each taking ~200ms:
- Sequential: 1200 × 200ms = **4 minutes**
- Concurrent (50 at a time): 1200 ÷ 50 × 200ms = **4.8 seconds**

### How `async`/`await` works

```python
# judge.py — lines 96-120
async def call_judge_single(client, prompt, model, max_tokens, rate_limiter):
    await rate_limiter.acquire()          # Wait for rate limit token

    try:
        response = await client.chat.completions.create(  # Send API request
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None
```

**`async def`** declares a coroutine — a function that can be paused and resumed.

**`await`** pauses the current coroutine until the awaited operation completes. While paused, the event loop runs other coroutines. When the API response arrives, this coroutine resumes.

**The key insight:** `await` does NOT block the thread. While waiting for one API response, Python can send 49 other API requests. This is concurrency without threads.

### Synchronous wrapper

```python
# judge.py — lines 214-243
def score_responses_sync(...):
    return asyncio.run(score_responses(...))
```

`asyncio.run()` starts the event loop, runs the async function to completion, and returns the result. This lets synchronous code use async functions without restructuring.

---

## 14. `asyncio.Lock` — Thread-Safe Async State

```python
# judge.py — lines 39-65
class RateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()           # Async-aware lock

    async def acquire(self):
        async with self.lock:                # Only one coroutine can enter at a time
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
```

**Why a lock?** Multiple coroutines call `acquire()` concurrently. Without the lock, two coroutines could both read `self.tokens == 1`, both decrement it, and both proceed — violating the rate limit.

**`async with self.lock`** is the async equivalent of a mutex. Only one coroutine can be inside the `async with` block at a time. Others wait (without blocking the thread — they yield to the event loop).

**Token bucket algorithm:** The rate limiter replenishes tokens over time (`self.rate` tokens per second). Each API call consumes one token. If no tokens are available, the coroutine sleeps until one regenerates.

---

## 15. `asyncio.gather()` — Running Many Coroutines in Parallel

```python
# judge.py — lines 123-154
async def call_judge_batch(client, prompts, model, max_tokens, rate_limiter, batch_size=50):
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        tasks = [
            call_judge_single(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in batch_results:
            if isinstance(result, Exception):
                processed.append(None)
            else:
                processed.append(result)
        results.extend(processed)

    return results
```

**`asyncio.gather(*tasks)`** runs all coroutines concurrently and returns their results in order.

**`return_exceptions=True`** means that if one coroutine raises an exception, it's returned as a value in the results list instead of crashing the whole gather. The code then checks `isinstance(result, Exception)` to handle failures gracefully.

**`*tasks`** unpacks the list: `gather(*[a, b, c])` becomes `gather(a, b, c)`.

---

## 16. Tuple Unpacking and Star Expressions

### Basic tuple unpacking

```python
# 2_activations.py — line 169
for i, (act, meta) in enumerate(zip(activations_list, metadata)):
    # i = the index
    # act = activations_list[i]
    # meta = metadata[i]
```

`zip()` pairs up elements from two lists. `enumerate()` adds an index. The nested `(act, meta)` unpacks each pair.

### Star unpacking in tuple reconstruction

```python
# steering.py — line 274
if was_tuple:
    return (modified_out, *activations[1:])
```

The original layer output was `(hidden_states, attention_weights, cache)`. After modifying `hidden_states`, this reconstructs the tuple with the modified first element and the original remaining elements.

`*activations[1:]` unpacks `activations[1:]` into individual elements:

```python
activations = (tensor_A, tensor_B, tensor_C)
(modified, *activations[1:])
# → (modified, tensor_B, tensor_C)
```

### Star in function parameters

```python
# steering.py — line 347
def __exit__(self, *exc):
    # *exc captures all arguments into a tuple
    # __exit__ receives (exc_type, exc_value, traceback) — all captured into exc
    self.remove()
```

---

## 17. Lambda Functions and Inline Callables

A `lambda` is a one-line anonymous function.

```python
# model.py — lines 141-145 — Lambdas in a data structure
layer_paths = [
    ('model.model.layers',          lambda m: m.model.layers),
    ('model.language_model.layers', lambda m: m.language_model.layers),
    ('model.transformer.h',         lambda m: m.transformer.h),
    ('model.transformer.layers',    lambda m: m.transformer.layers),
    ('model.gpt_neox.layers',       lambda m: m.gpt_neox.layers),
]
```

Each lambda takes a model `m` and tries to access a specific attribute path. The list is iterated with try/except to find which path works for the current architecture.

```python
# spans.py — line 65 — Lambda as sort key
spans_by_conversation[conv_id].sort(key=lambda x: x['turn'])
```

Sorts span dicts by their `'turn'` value. `key=` tells `sort()` what value to compare.

```python
# 2_activations.py — line 343 — Lambda for argument parsing
parser.add_argument("--thinking",
    type=lambda x: x.lower() in ['true', '1', 'yes'],
    default=False)
```

`argparse` calls `type(value)` to convert the string argument. This lambda converts strings like `"true"`, `"1"`, `"yes"` to `True` and anything else to `False`.

---

## 18. `hasattr()` / `getattr()` — Duck Typing and Safe Access

### `getattr(obj, name, default)` — access an attribute by name with a fallback

```python
# model.py — line 109
instance.model_name = model_name or getattr(model, 'name_or_path', 'Unknown')
```

If `model_name` is `None`, try `model.name_or_path`. If that attribute doesn't exist, use `'Unknown'`.

```python
# conversation.py — line 28
self.model_name = (model_name or getattr(tokenizer, "name_or_path", "")).lower()
```

Same pattern — try to get the model name from the tokenizer, defaulting to an empty string.

### `hasattr(obj, name)` — check before access

```python
# pca.py — lines 171-178
if hasattr(scaler, "fit_transform"):
    scaled = scaler.fit_transform(layer_activations)
elif hasattr(scaler, "transform") and hasattr(scaler, "fit"):
    fitted_scaler = scaler.fit(layer_activations)
    scaled = fitted_scaler.transform(layer_activations)
elif callable(scaler):
    scaled = scaler(layer_activations)
```

This is **duck typing** — "if it has `fit_transform`, use it like a sklearn-style transformer; if it has `fit` and `transform` separately, use those; if it's just callable, call it." The code doesn't check the class — it checks the capabilities.

```python
# steering.py — line 212
if hasattr(cur, "__getitem__"):
    return cur, path
```

Checks if the object supports indexing (`obj[0]`). `__getitem__` is the dunder method for `[]` access.

---

## 19. `torch.einsum()` — Einstein Summation Notation

`einsum` is a compact notation for tensor operations. It appears 8 times in `steering.py`.

```python
# steering.py — line 296
projections = torch.einsum('bld,d->bl', activations, vector_norm)
```

**Reading the notation:** `'bld,d->bl'`
- Input 1: `activations` has dimensions `b`atch, `l`ength, `d`epth (hidden dim)
- Input 2: `vector_norm` has dimension `d`epth
- Output: dimensions `b`atch, `l`ength
- `d` appears in both inputs but not in the output → **summed over** (dot product)

This computes the dot product of each token's activation with the vector, for every token in every batch element. It's equivalent to:

```python
# Without einsum (harder to read):
projections = (activations * vector_norm.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
```

### The four einsum patterns in this code

```python
# Dot product: (batch, seq, hidden) · (hidden,) → (batch, seq)
torch.einsum('bld,d->bl', activations, vector)

# Outer product: (batch, seq) × (hidden,) → (batch, seq, hidden)
torch.einsum('bl,d->bld', scalars, vector)

# Dot product on last position: (batch, hidden) · (hidden,) → (batch,)
torch.einsum('bd,d->b', last_position, vector)

# Outer product without seq: (batch,) × (hidden,) → (batch, hidden)
torch.einsum('b,d->bd', scalars, vector)
```

---

## 20. `.clone()` vs Direct Mutation — Defensive Copies

```python
# steering.py — lines 286-288
result = activations.clone()     # Create a copy
result[:, -1, :] += steer       # Modify the copy
return result                    # Return the copy, original unchanged
```

**Why `.clone()`?** Without it:

```python
# WRONG — modifies the original tensor
activations[:, -1, :] += steer   # Original tensor is modified in-place
return activations                # Caller's tensor is now corrupted
```

In-place modification of the `output` tensor in a forward hook would corrupt the model's computation graph. `.clone()` creates an independent copy so the original flows through the model undisturbed.

**When `.clone()` is NOT used:**

```python
# steering.py — line 284
return activations + steer    # Creates a NEW tensor — no need for clone
```

`+` already creates a new tensor. In-place `+=` modifies the existing tensor. The rule: if you're going to modify a tensor that someone else might be using, clone first.

---

## 21. Multiprocessing — `mp.Process`, `start`, `join`

```python
# 2_activations.py — lines 314-327
processes = []
for worker_id in range(num_workers):
    if role_chunks[worker_id]:
        p = mp.Process(
            target=process_roles_on_worker,           # Function to run
            args=(worker_id, gpu_chunks[worker_id],   # Positional arguments
                  role_chunks[worker_id], args)
        )
        p.start()              # Launch the process (non-blocking)
        processes.append(p)

for p in processes:
    p.join()                   # Wait for process to finish (blocking)
```

**`mp.Process(target=fn, args=(...))` —** Creates a process object. Does NOT start it yet.

**`p.start()` —** Launches the process. The function `fn` begins executing in a separate OS process. `start()` returns immediately — the parent doesn't wait.

**`p.join()` —** Blocks the parent until the child process finishes. Without `join()`, the parent might exit while workers are still running (orphaned processes).

**Why `start()` is called in the first loop and `join()` in the second:** All workers must start before any is waited on. If you did `start(); join()` for each worker sequentially, only one worker would run at a time — no parallelism.

```python
# Wrong — sequential execution:
for w in range(4):
    p = mp.Process(target=fn)
    p.start()    # Start worker
    p.join()     # WAIT for it to finish before starting the next one

# Correct — parallel execution:
processes = []
for w in range(4):
    p = mp.Process(target=fn)
    p.start()    # Start ALL workers
    processes.append(p)
for p in processes:
    p.join()     # THEN wait for all of them
```

---

## 22. `os.environ` — Environment Variables as Configuration

```python
# 2_activations.py — lines 189-190
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
```

`os.environ` is a dict-like object that reads and writes environment variables. Changes are visible to the current process and any child processes it spawns.

```python
# 2_activations.py — lines 245-252
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    gpu_ids = [int(x.strip()) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
else:
    gpu_ids = list(range(torch.cuda.device_count()))
```

This reads `CUDA_VISIBLE_DEVICES` if it was set externally (e.g., by the user or a job scheduler like SLURM). If not set, it falls back to querying PyTorch for the GPU count.

**Why `if x.strip()`?** Handles edge cases like trailing commas: `"0,1,"` would produce `['0', '1', '']`. The `if x.strip()` filters out the empty string.

---

## 23. List Comprehensions and Dict Comprehensions

Compact syntax for building lists and dicts from iterables.

### Filtering

```python
# axis.py — line 216
filtered = [v for name, v in vectors.items() if name not in exclude_roles]
# Takes all values from the dict EXCEPT those whose key is in exclude_roles
```

### Type conversion

```python
# steering.py — line 166
return [float(c) for c in coefficients]
# Converts every element to float
```

### Measurement

```python
# activations.py — line 363
'actual_lengths': [len(ids) for ids in batch_full_ids]
# Gets the length of each conversation's token list
```

### Dict comprehension

```python
# activations.py — line 230
layer_activations = {layer_idx: [] for layer_idx in layer}
# Creates {"layer_20": [], "layer_22": [], "layer_24": []}
```

### Conditional expression inside comprehension

```python
# activations.py — line 364
'truncated_lengths': [min(len(ids), max_seq_len) for ids in batch_full_ids]
# Each value is the minimum of the actual length and the cap
```

### Equivalent loop form

Every comprehension can be written as a loop:

```python
# Comprehension:
result = [f(x) for x in items if condition(x)]

# Equivalent loop:
result = []
for x in items:
    if condition(x):
        result.append(f(x))
```

Comprehensions are preferred in Python when the logic is simple — they're more concise and marginally faster (the loop runs in C internally).
