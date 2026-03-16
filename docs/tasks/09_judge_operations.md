# Task 9: Judge Operations (judge.py)

## Overview

The `judge.py` module provides utilities for scoring how well model responses adhere to their assigned roles using an LLM judge (e.g., GPT-4). It implements asynchronous, rate-limited, batched calls to the OpenAI API, parses numerical scores from judge responses, and exposes both async and synchronous entry points.

**Score Scale:**

| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer |
| 1 | Model says it can't be the role, but can help with related tasks |
| 2 | Model identifies as AI/LLM but has some role attributes |
| 3 | Model is fully playing the role |

**Module-level setup:**

```python
import asyncio
import os
import re
import time
import logging
from typing import Dict, List, Optional, Any

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
```

Environment variables (specifically `OPENAI_API_KEY`) are loaded from a `.env` file at import time via `dotenv`. A module-level logger is configured under the module's `__name__`.

---

## Sub-Tasks

### Sub-Task 9.1: Rate Limiting (RateLimiter class)

#### Description

A token-bucket rate limiter that gates async requests to the judge API. It ensures that API calls do not exceed a configurable requests-per-second threshold. The bucket refills continuously based on elapsed time and drains one token per `acquire()` call.

#### Input

**Constructor `__init__`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | `float` | *(required)* | Maximum requests per second |

**Method `acquire`:**

No parameters. Called by each request before it proceeds.

#### Processing

**Initialization** stores the rate, seeds the token bucket to full capacity, records the current wall-clock time, and creates an `asyncio.Lock` for thread-safe access:

```python
class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

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

**Token acquisition** (`acquire`) operates under the async lock. It:

1. Computes elapsed time since the last update.
2. Refills the bucket proportionally (`elapsed * rate`), capping at `rate`.
3. If at least 1 token is available, consumes it and returns immediately.
4. Otherwise, computes the wait time needed for 1 token to become available, sleeps for that duration, and sets tokens to 0.

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

#### Output

`acquire()` returns `None`. Its effect is purely temporal -- it delays execution until a token is available, ensuring downstream API calls respect the rate limit.

**Instance attributes after construction:**

| Attribute | Type | Value |
|-----------|------|-------|
| `self.rate` | `float` | The configured rate |
| `self.tokens` | `float` | Current token count (starts equal to `rate`) |
| `self.last_update` | `float` | Unix timestamp of last token refresh |
| `self.lock` | `asyncio.Lock` | Mutex for concurrent access |

---

### Sub-Task 9.2: Parse Judge Score (parse_judge_score)

#### Description

Extracts a numerical score (0--3) from the free-text response returned by the judge model. Uses regex to find the first integer token in the text and validates it falls within the expected range.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_text` | `str` | *(required)* | The raw text response from the judge model |

#### Processing

1. **Guard against empty input.** If `response_text` is falsy (empty string, `None`), return `None` immediately.
2. **Extract all integer tokens** from the stripped text using the regex `\b(\d+)\b`, which matches whole-word digit sequences.
3. **If no numbers are found**, return `None`.
4. **Take the first number**, convert it to `int`, and validate that it falls in `[0, 3]`. Return the score if valid, `None` otherwise.
5. **Catch `ValueError`** (should not normally occur given the regex, but acts as a safety net) and return `None`.

```python
def parse_judge_score(response_text: str) -> Optional[int]:
    """
    Parse the judge's response to extract the numerical score.

    Args:
        response_text: The judge model's response

    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        return None

    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())

    if not numbers:
        return None

    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        return None
    except ValueError:
        return None
```

#### Output

| Return Type | Description |
|-------------|-------------|
| `Optional[int]` | An integer in `{0, 1, 2, 3}` if parsing succeeds, or `None` if the text contains no valid score |

---

### Sub-Task 9.3: Single Judge Call (call_judge_single)

#### Description

Makes a single rate-limited, asynchronous call to the OpenAI chat completions API with the given prompt and returns the judge's raw text response. This is the atomic unit of judge evaluation -- all higher-level functions ultimately delegate to this.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `openai.AsyncOpenAI` | *(required)* | An initialized async OpenAI client |
| `prompt` | `str` | *(required)* | The fully-formatted evaluation prompt to send to the judge |
| `model` | `str` | *(required)* | The model identifier (e.g., `"gpt-4.1-mini"`) |
| `max_tokens` | `int` | *(required)* | Maximum completion tokens for the judge's reply |
| `rate_limiter` | `RateLimiter` | *(required)* | A `RateLimiter` instance controlling request throughput |

#### Processing

1. **Acquire a rate-limiter token** by awaiting `rate_limiter.acquire()`. This blocks until the request is permitted.
2. **Call the OpenAI API** using `client.chat.completions.create` with:
   - The prompt placed as a single `"user"` message.
   - `temperature=1` (default sampling temperature).
   - `max_completion_tokens` set to the provided `max_tokens`.
3. **Extract the response content.** If `response.choices` is non-empty and the first choice's message has content, return that content string.
4. **On any exception**, log the error and return `None`.

```python
async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single prompt."""
    await rate_limiter.acquire()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None

    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None
```

#### Output

| Return Type | Description |
|-------------|-------------|
| `Optional[str]` | The judge model's raw text response, or `None` if the call failed or returned no content |

---

### Sub-Task 9.4: Batch Judge Calls (call_judge_batch)

#### Description

Sends multiple evaluation prompts to the judge model concurrently, processing them in configurable batch sizes. Within each batch, all calls run as concurrent asyncio tasks via `asyncio.gather`. Batches are processed sequentially to avoid overwhelming the API beyond the rate limiter's control.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `openai.AsyncOpenAI` | *(required)* | An initialized async OpenAI client |
| `prompts` | `List[str]` | *(required)* | List of fully-formatted evaluation prompts |
| `model` | `str` | *(required)* | The model identifier |
| `max_tokens` | `int` | *(required)* | Maximum completion tokens per judge response |
| `rate_limiter` | `RateLimiter` | *(required)* | Rate limiter instance |
| `batch_size` | `int` | `50` | Number of concurrent requests per batch |

#### Processing

1. **Initialize an empty results list.**
2. **Iterate over prompts in slices of `batch_size`** using `range(0, len(prompts), batch_size)`.
3. **For each batch**, create a list of asyncio tasks, one per prompt, each calling `call_judge_single`.
4. **Await all tasks concurrently** with `asyncio.gather(*tasks, return_exceptions=True)`. The `return_exceptions=True` flag means exceptions are returned as values rather than being raised, so one failure does not abort the entire batch.
5. **Post-process the batch results**: any result that is an `Exception` instance is logged and replaced with `None`; successful results pass through.
6. **Extend the master results list** with the processed batch.

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
                logger.error(f"Exception in batch: {result}")
                processed.append(None)
            else:
                processed.append(result)

        results.extend(processed)

    return results
```

#### Output

| Return Type | Description |
|-------------|-------------|
| `List[Optional[str]]` | A list of raw judge response strings, positionally aligned with the input `prompts`. Failed or errored entries are `None`. Length equals `len(prompts)`. |

---

### Sub-Task 9.5: Score Responses -- Async (score_responses)

#### Description

The primary high-level async entry point. Takes a list of question/response pairs and an evaluation prompt template, constructs per-item prompts, sends them through the batched judge pipeline, and parses the results into integer scores. This function orchestrates the full end-to-end scoring workflow.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Dict[str, str]]` | *(required)* | List of dicts, each containing `"question"` and `"response"` keys |
| `eval_prompt_template` | `str` | *(required)* | A format string with `{question}` and `{answer}` placeholders |
| `judge_model` | `str` | `"gpt-4.1-mini"` | OpenAI model to use as the judge |
| `max_tokens` | `int` | `10` | Maximum tokens for each judge response |
| `requests_per_second` | `int` | `100` | Rate limit for API calls |
| `batch_size` | `int` | `50` | Concurrent batch size |

#### Processing

1. **Validate the API key.** Check that `OPENAI_API_KEY` is present in the environment; raise `ValueError` if missing.
2. **Build per-item prompts.** Iterate over `responses` and format each entry into `eval_prompt_template` using `resp["question"]` for `{question}` and `resp["response"]` for `{answer}`.
3. **Initialize the OpenAI client** (`openai.AsyncOpenAI()`) and a `RateLimiter` with the configured `requests_per_second`.
4. **Call the judge batch pipeline** via `call_judge_batch`, passing all prompts, model config, and rate limiter.
5. **Parse each raw response** into an integer score using `parse_judge_score`. If the response text is falsy, the score is `None`.

```python
async def score_responses(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Score a list of responses using an LLM judge.

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
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Build prompts
    prompts = []
    for resp in responses:
        prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=resp["response"]
        )
        prompts.append(prompt)

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)

    # Call judge
    judge_responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = []
    for response_text in judge_responses:
        score = parse_judge_score(response_text) if response_text else None
        scores.append(score)

    return scores
```

#### Output

| Return Type | Description |
|-------------|-------------|
| `List[Optional[int]]` | A list of integer scores in `{0, 1, 2, 3}` or `None` for entries where parsing failed. Length equals `len(responses)`. Positionally aligned with the input list. |

---

### Sub-Task 9.6: Score Responses -- Synchronous Wrapper (score_responses_sync)

#### Description

A synchronous convenience wrapper around `score_responses`. Uses `asyncio.run()` to execute the async scoring pipeline from synchronous calling code (e.g., scripts, notebooks, or non-async application layers). Accepts the exact same parameters and returns the exact same output.

#### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Dict[str, str]]` | *(required)* | List of dicts, each containing `"question"` and `"response"` keys |
| `eval_prompt_template` | `str` | *(required)* | A format string with `{question}` and `{answer}` placeholders |
| `judge_model` | `str` | `"gpt-4.1-mini"` | OpenAI model to use as the judge |
| `max_tokens` | `int` | `10` | Maximum tokens for each judge response |
| `requests_per_second` | `int` | `100` | Rate limit for API calls |
| `batch_size` | `int` | `50` | Concurrent batch size |

#### Processing

Delegates entirely to `asyncio.run()`, which creates a new event loop, runs the `score_responses` coroutine to completion, and returns the result. All parameters are forwarded verbatim.

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
    return asyncio.run(score_responses(
        responses=responses,
        eval_prompt_template=eval_prompt_template,
        judge_model=judge_model,
        max_tokens=max_tokens,
        requests_per_second=requests_per_second,
        batch_size=batch_size
    ))
```

**Note:** `asyncio.run()` cannot be called from within an already-running event loop. If you are inside an async context (e.g., a Jupyter notebook with an active loop), use `score_responses` directly with `await` instead.

#### Output

| Return Type | Description |
|-------------|-------------|
| `List[Optional[int]]` | Identical to `score_responses` -- a list of integer scores in `{0, 1, 2, 3}` or `None`. |

---

## Data Flow Summary

```
responses (List[Dict])          eval_prompt_template (str)
        \                              /
         \                            /
    score_responses / score_responses_sync
                    |
          [1] Validate OPENAI_API_KEY
          [2] Format prompts via template
          [3] Create AsyncOpenAI client + RateLimiter
                    |
            call_judge_batch
                    |
        [Split into batches of batch_size]
                    |
            call_judge_single  (x N, concurrent per batch)
                    |
              RateLimiter.acquire()  -->  OpenAI API
                    |
          Raw text responses (List[Optional[str]])
                    |
            parse_judge_score  (per response)
                    |
          Scores: List[Optional[int]]  (0-3 or None)
```
