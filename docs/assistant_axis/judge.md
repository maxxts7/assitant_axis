# `assistant_axis/judge.py`

## Overview

This file provides utilities for scoring how well an LLM's responses adhere to a specified role, using a separate LLM as a "judge." It sends each response to a judge model (e.g., GPT-4) via the OpenAI API and parses the returned score. The module supports both asynchronous and synchronous workflows, includes a token-bucket rate limiter to stay within API rate limits, and processes prompts in configurable concurrent batches.

The scoring scale used throughout is:

| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer |
| 1 | Model says it cannot be the role, but can help with related tasks |
| 2 | Model identifies as AI/LLM but exhibits some role attributes |
| 3 | Model is fully playing the role |

---

## Line-by-Line Explanation

### Module Docstring (Lines 1-21)

```python
"""
Judge LLM utilities for scoring role responses.

This module provides functions for scoring how well model responses
adhere to their assigned roles using an LLM judge (e.g., GPT-4).

Score Scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Example:
    from assistant_axis.judge import score_responses

    scores = await score_responses(
        responses=[{"question": "...", "response": "..."}],
        eval_prompt_template="Rate how well...",
        judge_model="gpt-4.1-mini"
    )
"""
```

The module-level docstring describes the purpose of the file, lays out the 0-3 scoring scale, and gives a quick usage example showing how to call the main `score_responses` function.

---

### Imports (Lines 23-31)

```python
import asyncio
import os
import re
import time
import logging
from typing import Dict, List, Optional, Any
```

- `asyncio` -- provides the asynchronous event loop, `gather`, `Lock`, and `sleep` used by the rate limiter and batch caller.
- `os` -- used to read the `OPENAI_API_KEY` environment variable.
- `re` -- used for regular-expression parsing when extracting numerical scores from the judge's text response.
- `time` -- provides `time.time()` for timestamps inside the rate limiter.
- `logging` -- standard library logging; a module-level logger is created below.
- `typing` imports -- `Dict`, `List`, `Optional`, and `Any` are used for type annotations throughout the file. (`Any` is imported but not used in the current code.)

```python
import openai
from dotenv import load_dotenv
```

- `openai` -- the official OpenAI Python client library; used to create an `AsyncOpenAI` client and call the chat completions API.
- `dotenv.load_dotenv` -- loads key-value pairs from a `.env` file into environment variables so that `OPENAI_API_KEY` (and any other secrets) can be read with `os.getenv`.

---

### Environment Setup and Logger (Lines 33-36)

```python
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
```

- `load_dotenv()` reads the `.env` file (if present) at import time and injects its variables into `os.environ`.
- `logging.getLogger(__name__)` creates a logger whose name matches the fully qualified module name (`assistant_axis.judge`). All log messages emitted later use this logger.

---

### `RateLimiter` Class (Lines 39-65)

```python
class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""
```

Declares a class that implements a **token bucket** rate limiter. A token bucket starts full (capacity = `rate`) and refills at `rate` tokens per second. Each API call consumes one token; if no tokens are available the caller sleeps until one regenerates.

#### `__init__` (Lines 42-50)

```python
    def __init__(self, rate: float):
        """
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()
```

- `self.rate` -- stores the maximum requests-per-second ceiling. This value also acts as the bucket capacity.
- `self.tokens` -- the current number of available tokens; initialised to `rate` so the bucket starts full.
- `self.last_update` -- records the last time the token count was recalculated, initialised to "now."
- `self.lock` -- an `asyncio.Lock` that serialises access to the mutable state (`tokens` and `last_update`) so concurrent coroutines do not race.

#### `acquire` (Lines 52-65)

```python
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
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

- `async with self.lock:` -- acquires the lock so only one coroutine mutates the bucket at a time.
- `now = time.time()` -- captures the current wall-clock time.
- `self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)` -- refills the bucket proportionally to the elapsed time since the last update, capping at the maximum capacity (`self.rate`).
- `self.last_update = now` -- updates the timestamp to "now."
- If at least one full token is available (`self.tokens >= 1`), the method decrements the count by 1 and returns immediately -- no waiting.
- Otherwise it calculates how long it must sleep for the token count to reach 1 (`wait_time = (1 - self.tokens) / self.rate`), sleeps for that duration, and then resets tokens to 0 (the one regenerated token was just consumed).

---

### `parse_judge_score` Function (Lines 68-93)

```python
def parse_judge_score(response_text: str) -> Optional[int]:
    """
    Parse the judge's response to extract the numerical score.

    Args:
        response_text: The judge model's response

    Returns:
        Integer score between 0-3, or None if parsing fails
    """
```

Declares a synchronous function that takes the judge model's raw text response and extracts a valid score from it.

```python
    if not response_text:
        return None
```

If the response is empty or falsy, there is nothing to parse -- return `None`.

```python
    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())
```

Uses a regular expression to find all whole numbers (sequences of digits bounded by word boundaries) in the stripped response text. The result is a list of string representations of those numbers.

```python
    if not numbers:
        return None
```

If no numbers were found at all, parsing has failed -- return `None`.

```python
    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        return None
    except ValueError:
        return None
```

Takes the **first** number found, converts it to an integer, and checks that it falls within the valid range 0-3. If it does, that integer is returned as the score. If the number is out of range or conversion raises a `ValueError`, `None` is returned.

---

### `call_judge_single` Function (Lines 96-120)

```python
async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single prompt."""
```

An async function that sends a single evaluation prompt to the judge model and returns the model's text response (or `None` on failure).

```python
    await rate_limiter.acquire()
```

Waits until the rate limiter grants a token, ensuring the request does not exceed the configured requests-per-second limit.

```python
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1
        )
```

Makes an asynchronous call to the OpenAI Chat Completions API:
- `model` -- the judge model name (e.g., `"gpt-4.1-mini"`).
- `messages` -- a single-message conversation containing just the user prompt.
- `max_completion_tokens` -- caps the judge's reply length (default 10 tokens, since only a short numeric score is expected).
- `temperature=1` -- uses the default sampling temperature.

```python
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None
```

If the API returned at least one choice with non-empty content, that content string is returned. Otherwise `None` is returned.

```python
    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None
```

Catches any exception (network errors, API errors, etc.), logs it at the `ERROR` level, and returns `None` so that the caller can continue processing the remaining prompts.

---

### `call_judge_batch` Function (Lines 123-154)

```python
async def call_judge_batch(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50
) -> List[Optional[str]]:
    """Call the judge model with multiple prompts concurrently."""
```

An async function that evaluates many prompts against the judge model, processing them in concurrent batches to balance throughput with resource usage.

```python
    results = []
```

Initialises an accumulator list that will hold the ordered results across all batches.

```python
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
```

Iterates over the prompts in chunks of `batch_size` (default 50). Each chunk is a slice of the full prompt list.

```python
        tasks = [
            call_judge_single(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
        ]
```

Creates a list of coroutine objects -- one `call_judge_single` call per prompt in the current batch. These have not started executing yet.

```python
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
```

Runs all the coroutines in the batch concurrently using `asyncio.gather`. The `return_exceptions=True` flag means that if any individual call raises an exception, the exception object is placed in the results list rather than propagating and cancelling the other tasks.

```python
        processed = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                processed.append(None)
            else:
                processed.append(result)
```

Iterates over the batch results. If a result is an `Exception` instance (meaning `call_judge_single` raised), it logs the error and substitutes `None`. Otherwise the actual string response (or `None` returned normally) is kept.

```python
        results.extend(processed)
```

Appends the processed batch results to the master list, preserving the original ordering.

```python
    return results
```

Returns the full list of judge responses (strings or `None`), one per input prompt.

---

### `score_responses` Function (Lines 157-211)

```python
async def score_responses(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
```

The main public async function. It takes a list of question/response pairs, formats them into evaluation prompts, sends them to the judge in batches, and returns parsed integer scores.

**Parameters:**

- `responses` -- a list of dictionaries, each containing a `"question"` key and a `"response"` key.
- `eval_prompt_template` -- a Python format string containing `{question}` and `{answer}` placeholders that will be filled per response.
- `judge_model` -- the OpenAI model to use as the judge (default `"gpt-4.1-mini"`).
- `max_tokens` -- maximum tokens the judge may generate (default 10).
- `requests_per_second` -- rate limit ceiling (default 100).
- `batch_size` -- how many prompts to run concurrently (default 50).

```python
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
```

Guards against a missing API key by raising a `ValueError` early, before any network calls are attempted.

```python
    # Build prompts
    prompts = []
    for resp in responses:
        prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=resp["response"]
        )
        prompts.append(prompt)
```

Iterates over the input responses and substitutes each question/response pair into the template string. Note that the template placeholder `{answer}` is filled with the dict key `"response"`, not `"answer"`. The resulting prompt strings are collected into the `prompts` list.

```python
    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)
```

- Creates an `AsyncOpenAI` client. The client reads `OPENAI_API_KEY` from the environment automatically.
- Creates a `RateLimiter` instance configured with the desired requests-per-second cap.

```python
    # Call judge
    judge_responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )
```

Delegates the actual API calling to `call_judge_batch`, passing all configuration. The result is a list of raw text responses (or `None` values).

```python
    # Parse scores
    scores = []
    for response_text in judge_responses:
        score = parse_judge_score(response_text) if response_text else None
        scores.append(score)

    return scores
```

Iterates over the judge's raw text responses. For each non-`None` response, calls `parse_judge_score` to extract the integer score. If the response was `None` (API failure), the score is also set to `None`. The final list of scores is returned to the caller.

---

### `score_responses_sync` Function (Lines 214-243)

```python
def score_responses_sync(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Synchronous wrapper for score_responses.

    Args:
        responses: List of dicts with 'question' and 'response' keys
        eval_prompt_template: Template string with {question} and {answer} placeholders
        judge_model: OpenAI model to use as judge
        max_tokens: Max tokens for judge response
        requests_per_second: Rate limit for API calls
        batch_size: Concurrent batch size

    Returns:
        List of scores (0-3) or None for failed parsing
    """
```

A synchronous convenience wrapper with the exact same signature and defaults as `score_responses`. This allows callers that are not running inside an async event loop to use the scoring functionality directly.

```python
    return asyncio.run(score_responses(
        responses=responses,
        eval_prompt_template=eval_prompt_template,
        judge_model=judge_model,
        max_tokens=max_tokens,
        requests_per_second=requests_per_second,
        batch_size=batch_size
    ))
```

`asyncio.run()` creates a new event loop, runs the `score_responses` coroutine to completion, closes the loop, and returns the result. All keyword arguments are forwarded unchanged.
