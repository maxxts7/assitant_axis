# assistant_axis/tests/test_generation.py

## Overview

This file contains a pytest test suite for the generation utilities defined in `assistant_axis.generation`. The tests load real Hugging Face tokenizers (from non-gated, publicly available models) to verify that two functions -- `supports_system_prompt` and `format_conversation` -- behave correctly with actual chat templates. A third test class validates that Qwen3's "thinking mode" can be disabled via the tokenizer's `apply_chat_template` method.

---

## Line-by-Line Explanation

### Lines 1--6: Module Docstring

```python
"""
Tests for generation utilities.

These tests load real tokenizers to verify behavior with actual chat templates.
Uses non-gated models only.
"""
```

The module-level docstring describes the purpose of the file. It notes that real tokenizers are used (as opposed to mocks) and that only non-gated models are chosen so the tests can run without requiring special Hugging Face access tokens.

---

### Lines 8--9: Imports

```python
import pytest
from transformers import AutoTokenizer
```

- `pytest` is imported to use its test fixtures and assertion introspection.
- `AutoTokenizer` from the `transformers` library is the Hugging Face utility that automatically loads the correct tokenizer class for a given pretrained model identifier.

---

### Line 11: Project Imports

```python
from assistant_axis.generation import supports_system_prompt, format_conversation
```

Imports the two functions under test from the project's own `generation` module:
- `supports_system_prompt` -- determines whether a given tokenizer's chat template supports a system-role message.
- `format_conversation` -- builds a list of chat messages (dicts with `role` and `content` keys) from an instruction and a user question, adapting the format based on whether the tokenizer supports system prompts.

---

### Lines 14--16: `qwen_tokenizer` Fixture

```python
@pytest.fixture(scope="module")
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

- `@pytest.fixture(scope="module")` declares a pytest fixture whose return value is cached and reused across all tests in this module. This avoids downloading/loading the tokenizer multiple times.
- The function loads the tokenizer for the `Qwen/Qwen2.5-0.5B-Instruct` model. This is a small Qwen 2.5 instruction-tuned model whose chat template is known to support system prompts.

---

### Lines 19--21: `gemma_tokenizer` Fixture

```python
@pytest.fixture(scope="module")
def gemma_tokenizer():
    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
```

- Same pattern as above but for Google's Gemma 2 2B instruction-tuned model.
- Gemma 2's chat template notably does **not** support system-role messages, making it a useful counterpart to the Qwen tokenizer for testing both branches of the system-prompt logic.

---

### Lines 24--25: `TestSupportsSystemPrompt` Class Header

```python
class TestSupportsSystemPrompt:
    """Tests for supports_system_prompt function."""
```

Groups the tests for the `supports_system_prompt` function into a single class. The docstring documents the purpose.

---

### Lines 27--29: `test_qwen_supports_system`

```python
    def test_qwen_supports_system(self, qwen_tokenizer):
        """Qwen models support system prompts."""
        assert supports_system_prompt(qwen_tokenizer) is True
```

- Receives the `qwen_tokenizer` fixture via dependency injection.
- Asserts that calling `supports_system_prompt` with the Qwen tokenizer returns exactly `True` (using `is True` for a strict boolean identity check, not just truthiness).
- Validates the positive case: a tokenizer whose chat template supports system messages.

---

### Lines 31--33: `test_gemma_no_system`

```python
    def test_gemma_no_system(self, gemma_tokenizer):
        """Gemma 2 models do not support system prompts."""
        assert supports_system_prompt(gemma_tokenizer) is False
```

- Receives the `gemma_tokenizer` fixture.
- Asserts the function returns exactly `False` for Gemma 2, confirming the negative case: a tokenizer whose chat template does not support system messages.

---

### Lines 36--37: `TestFormatConversation` Class Header

```python
class TestFormatConversation:
    """Tests for format_conversation function."""
```

Groups the tests for `format_conversation`.

---

### Lines 39--49: `test_with_system_support`

```python
    def test_with_system_support(self, qwen_tokenizer):
        """When system is supported, instruction becomes system message."""
        result = format_conversation(
            instruction="You are a pirate.",
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "You are a pirate."}
        assert result[1] == {"role": "user", "content": "Hello!"}
```

- Calls `format_conversation` with a system instruction string, a user question, and the Qwen tokenizer (which supports system prompts).
- Asserts the result is a two-element list:
  - The first element is a system message dict containing the instruction.
  - The second element is a user message dict containing the question.
- This confirms that when the tokenizer supports system prompts, the instruction is placed into a dedicated system-role message rather than being merged with the user content.

---

### Lines 51--62: `test_without_system_support`

```python
    def test_without_system_support(self, gemma_tokenizer):
        """When system not supported, instruction is prepended to user message."""
        result = format_conversation(
            instruction="You are a pirate.",
            question="Hello!",
            tokenizer=gemma_tokenizer,
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "You are a pirate." in result[0]["content"]
        assert "Hello!" in result[0]["content"]
```

- Uses the Gemma tokenizer which does **not** support system prompts.
- Asserts the result is a single-element list with role `"user"`.
- Instead of checking for an exact string match, it uses `in` to verify both the instruction text and the question text are present within the user message content. This allows the function implementation flexibility in how it concatenates them (e.g., with a newline separator or other formatting).

---

### Lines 64--73: `test_no_instruction`

```python
    def test_no_instruction(self, qwen_tokenizer):
        """When no instruction, only user message is returned."""
        result = format_conversation(
            instruction=None,
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello!"}
```

- Passes `None` as the instruction to `format_conversation`.
- Even though the Qwen tokenizer supports system prompts, no system message should be created when there is no instruction.
- Asserts the result is a single user message containing only the question.

---

### Lines 75--85: `test_empty_instruction`

```python
    def test_empty_instruction(self, qwen_tokenizer):
        """Empty instruction is treated as no instruction."""
        result = format_conversation(
            instruction="",
            question="Hello!",
            tokenizer=qwen_tokenizer,
        )

        # Empty string is falsy, so should just have user message
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello!"}
```

- Passes an empty string `""` as the instruction.
- The comment on line 83 explains the reasoning: an empty string is falsy in Python, so `format_conversation` treats it the same as `None`.
- Asserts the result is again a single user message with no system message, confirming the edge case is handled.

---

### Lines 88--96: `TestQwenThinkingDisabled` Class Header and Docstring

```python
class TestQwenThinkingDisabled:
    """Tests that thinking mode is disabled for Qwen models.

    Qwen3 models have a thinking mode that can be controlled via enable_thinking:
    - enable_thinking=False: Adds empty <think></think> block to force model to skip thinking
    - enable_thinking=True (default): No think block, model can choose to think

    We want enable_thinking=False to prevent thinking tokens in responses.
    """
```

This class tests the Qwen3-specific "thinking mode" behavior. The docstring explains the mechanics:
- When `enable_thinking=False`, the tokenizer inserts an empty `<think></think>` block into the prompt, which signals the model to skip its internal chain-of-thought reasoning.
- When `enable_thinking=True` (the default), no such block is added, leaving the model free to produce thinking tokens.
- The project wants thinking disabled to keep responses clean.

---

### Lines 98--101: `qwen3_tokenizer` Fixture

```python
    @pytest.fixture(scope="class")
    def qwen3_tokenizer(self):
        """Load Qwen3 tokenizer which supports thinking mode."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

- A class-scoped fixture (shared across all tests in `TestQwenThinkingDisabled`) that loads the tokenizer for Qwen3-0.6B, a model whose chat template includes the `enable_thinking` parameter.
- Using `scope="class"` ensures the tokenizer is loaded once per class rather than once per test method.

---

### Lines 103--122: `test_qwen3_thinking_disabled_adds_empty_think_block`

```python
    def test_qwen3_thinking_disabled_adds_empty_think_block(self, qwen3_tokenizer):
        """enable_thinking=False adds empty think block to force skipping thinking."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # With thinking disabled (as we do in generation.py)
        prompt_no_thinking = qwen3_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # enable_thinking=False adds empty think block to force model to skip thinking
        assert "<think>" in prompt_no_thinking
        assert "</think>" in prompt_no_thinking
        # The think block should be empty (just whitespace between tags)
        assert "<think>\n\n</think>" in prompt_no_thinking
```

- Constructs a minimal conversation with a system message and a user message.
- Calls `apply_chat_template` on the Qwen3 tokenizer with:
  - `tokenize=False` -- returns the raw string instead of token IDs, so the test can inspect the text content.
  - `add_generation_prompt=True` -- appends the assistant turn prefix so the model knows to start generating.
  - `enable_thinking=False` -- the key parameter being tested; disables thinking mode.
- Asserts that the resulting prompt string contains `<think>` and `</think>` tags.
- Asserts that the think block is empty (contains only whitespace: `<think>\n\n</think>`), confirming the tokenizer pre-fills an empty thinking section to suppress the model's chain-of-thought output.

---

### Lines 124--141: `test_qwen3_thinking_enabled_no_think_block`

```python
    def test_qwen3_thinking_enabled_no_think_block(self, qwen3_tokenizer):
        """enable_thinking=True (default) does not add think block - model decides."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # With thinking enabled (the default we want to avoid for generation)
        prompt_with_thinking = qwen3_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # enable_thinking=True means no pre-filled think block - model can generate thinking
        assert "<think>" not in prompt_with_thinking
        assert "</think>" not in prompt_with_thinking
```

- Same conversation setup as the previous test.
- This time `enable_thinking=True` is passed, which is the default/opt-in behavior for thinking.
- Asserts that neither `<think>` nor `</think>` appears in the prompt. When thinking is enabled, the template does not pre-fill any think block -- the model itself decides whether to produce thinking tokens during generation.
- This test serves as the control case, confirming that the presence of the empty think block in the previous test is specifically caused by `enable_thinking=False`.
